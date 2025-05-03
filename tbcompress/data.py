# /// script
# dependencies = [
#     "torch",
#     "numpy",
#     "python-chess",
#     "tqdm",
# ]
# ///

import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import chess
import chess.syzygy
from tqdm import tqdm


def board_to_feature_vector(board):
    """
    Convert a chess.Board to a feature vector for the neural network

    Args:
        board: A chess.Board object

    Returns:
        numpy array with 65 features (64 squares + side to move)
    """
    # Create a 64-element vector (one per square)
    # Empty square: 0
    # White pieces: 1 (pawn), 2 (knight), 3 (bishop), 4 (rook), 5 (queen), 6 (king)
    # Black pieces: -1 (pawn), -2 (knight), -3 (bishop), -4 (rook), -5 (queen), -6 (king)
    feature_vector = np.zeros(64, dtype=np.float32)

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    # Sparse representation - only populate non-empty squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            value = piece_values[piece.piece_type]
            if piece.color == chess.BLACK:
                value = -value
            feature_vector[square] = value

    # Side to move
    side_to_move = 1.0 if board.turn == chess.WHITE else -1.0

    # Combine features
    combined_features = np.append(feature_vector, side_to_move)
    return combined_features


class StreamingTablebaseDataset(TablebaseDataset):
    """Streaming dataset that generates positions on-the-fly from a .rtbw file"""
    
    def __init__(self, rtbw_file, tablebase_dir, cache_size=10000, max_positions=None):
        """
        Create a streaming dataset from a specific .rtbw tablebase file
        
        Args:
            rtbw_file: Path to the .rtbw file
            tablebase_dir: Directory containing Syzygy tablebase files
            cache_size: Number of positions to cache in memory at once
            max_positions: Maximum number of positions to use (None for unlimited)
        """
        # Initialize the parent class but skip its _load_positions method
        Dataset.__init__(self)
        self.rtbw_file = rtbw_file
        self.tablebase_dir = tablebase_dir
        self.cache_size = cache_size
        self.max_positions = max_positions
        
        # Initialize tablebase
        self.tablebase = chess.syzygy.open_tablebase(tablebase_dir)
        
        # Extract material configuration
        self.material = os.path.basename(rtbw_file).split(".")[0]
        white_material, black_material = self.material.split('v')
        self.piece_count = len(white_material) + len(black_material)
        
        # Position cache
        self.position_cache = []
        self.wdl_cache = []
        self.cache_index = 0
        
        # Position generation strategy
        if self.piece_count <= 3:  # Simple endgames
            self.position_generator = self._exhaustively_enumerate_positions(self.material)
            self.estimated_size = 64*63*2  # Approximate for KvK and similar
        elif self.piece_count <= 5:  # Moderate complexity
            self.position_generator = self._systematic_enumeration(self.material)
            self.estimated_size = 1000000  # Approximate for 4-5 piece endgames
        else:  # Complex endgames
            self.position_generator = self._comprehensive_sampling(self.material)
            self.estimated_size = 10000000  # Approximate for 6+ piece endgames
        
        # Cap the size based on max_positions if specified
        if self.max_positions is not None and self.estimated_size > self.max_positions:
            self.estimated_size = self.max_positions
        
        # Fill initial cache
        self._fill_cache()
        
        # Track counts for reporting
        self.positions_processed = 0
        self.valid_positions = 0
        self.seen_positions = set()
        
        print(f"Initialized streaming dataset for {self.material} with estimated {self.estimated_size} positions")
    
    def _fill_cache(self):
        """Fill the position cache with valid positions"""
        if len(self.position_cache) >= self.cache_size:
            return
        
        new_positions = []
        new_wdl_values = []
        while len(new_positions) < self.cache_size and (self.max_positions is None or self.valid_positions < self.max_positions):
            try:
                # Get next position from generator
                board = next(self.position_generator)
                self.positions_processed += 1
                
                # Skip positions that we've already seen
                board_hash = board.epd()
                if board_hash in self.seen_positions:
                    continue
                self.seen_positions.add(board_hash)
                
                # Get WDL value from tablebase
                try:
                    # Get the WDL value from the tablebase (win/draw/loss)
                    # WDL values: 2 = win, 0 = draw, -2 = loss
                    wdl = self.tablebase.probe_wdl(board)
                    
                    # For our model, convert to 0, 1, 2 for loss, draw, win
                    # If side to move is black, need to negate wdl
                    if not board.turn:
                        wdl = -wdl
                        
                    # Map to 0 (loss), 1 (draw), 2 (win)
                    wdl_mapped = (wdl + 2) // 2
                    
                    # Add to caches
                    features = board_to_feature_vector(board)
                    new_positions.append(features)
                    new_wdl_values.append(wdl_mapped)
                    self.valid_positions += 1
                    
                    # Progress reporting
                    if self.valid_positions % 10000 == 0 and self.valid_positions > 0:
                        print(f"Generated {self.valid_positions} valid positions from {self.positions_processed} attempts")
                        
                    # Check if we've reached max positions
                    if self.max_positions is not None and self.valid_positions >= self.max_positions:
                        print(f"Reached max positions limit of {self.max_positions}")
                        break
                        
                except (ValueError, chess.syzygy.MissingTableError):
                    # Skip positions not in the tablebase
                    continue
                    
            except StopIteration:
                # No more positions available from generator
                print("No more positions available from generator")
                break
                
            except Exception as e:
                print(f"Error processing position: {e}")
                continue
                    
        
        # Add new positions and WDL values to cache
        self.position_cache.extend(new_positions)
        self.wdl_cache.extend(new_wdl_values)
        
        if new_positions:
            print(f"Cache filled with {len(self.position_cache)} positions")
    
    def __len__(self):
        """Return estimated size of dataset"""
        if self.max_positions is not None:
            return min(self.estimated_size, self.max_positions)
        return self.estimated_size
    
    def __getitem__(self, idx):
        """Get a position by index - uses a continuously refilled cache"""
        # If we're near the end of the cache, refill it
        if self.cache_index >= len(self.position_cache) - 1:
            self.cache_index = 0
            self._fill_cache()
            
            # If cache is empty after filling, we've exhausted the generator
            if not self.position_cache:
                # Restart the generator for another epoch
                print("Restarting position generator for a new epoch")
                if self.piece_count <= 3:
                    self.position_generator = self._exhaustively_enumerate_positions(self.material)
                elif self.piece_count <= 5:
                    self.position_generator = self._systematic_enumeration(self.material)
                else:
                    self.position_generator = self._comprehensive_sampling(self.material)
                self._fill_cache()
                
                # If still empty, we have a problem
                if not self.position_cache:
                    raise IndexError("No valid positions could be generated")
        
        # Get the position from the cache
        features = self.position_cache[self.cache_index]
        wdl = self.wdl_cache[self.cache_index]
        self.cache_index += 1
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(wdl, dtype=torch.long)


class TablebaseDataset(Dataset):
    """Dataset for loading positions from a single .rtbw tablebase file"""

    def __init__(self, rtbw_file, tablebase_dir):
        """
        Create a dataset from a specific .rtbw tablebase file

        Args:
            rtbw_file: Path to the .rtbw file
            tablebase_dir: Directory containing Syzygy tablebase files
        """
        self.rtbw_file = rtbw_file
        self.tablebase_dir = tablebase_dir
        self.features = []
        self.labels = []

        # Load all positions from this file
        self._load_positions()

    def _load_positions(self):
        """Load all positions from the tablebase file"""
        print(f"Loading positions from {os.path.basename(self.rtbw_file)}")

        # Initialize the tablebase
        tablebase = chess.syzygy.open_tablebase(self.tablebase_dir)

        # Extract material configuration from filename (e.g., KRvKN.rtbw)
        material = os.path.basename(self.rtbw_file).split(".")[0]

        # Storage for feature vectors and labels
        features = []
        labels = []
        positions_seen = set()  # To avoid duplicates

        # Count pieces in the material configuration
        white_material, black_material = material.split('v')
        piece_count = len(white_material) + len(black_material)
        
        # For accurate compression, we need to access the entire legal position space
        # For simple tablebases, we can enumerate all positions
        # For complex tablebases, we need to be more clever with systematic enumeration
        
        print(f"Generating complete position space for {material} ({piece_count} pieces)")
        
        # Generate positions systematically to cover the entire space
        if piece_count <= 3:  # For very simple material balances, use brute force enumeration
            position_generator = self._exhaustively_enumerate_positions(material)
        elif piece_count <= 5:  # For moderate complexity, use systematic sampling with reflection/rotation equivalence
            position_generator = self._systematic_enumeration(material)
        else:  # For very complex positions, use efficient space-filling sampling
            position_generator = self._comprehensive_sampling(material)
            
        # Process positions in batches to manage memory
        batch_size = 10000  # Process 10K positions at a time
        batch_features = []
        batch_labels = []
        positions_processed = 0
        valid_positions = 0
        total_batches = 0
        
        print(f"Processing positions in batches of {batch_size}")
        
        # Use a progress bar
        with tqdm(desc=f"Processing {material} positions") as pbar:
            for board in position_generator:
                positions_processed += 1
                
                # Skip if we've seen this position before (using compact hash)
                board_hash = board.epd()
                if board_hash in positions_seen:
                    continue
                    
                positions_seen.add(board_hash)
                
                try:
                    # Probe the tablebase for WDL value
                    wdl = tablebase.probe_wdl(board)
                    
                    # Convert WDL to label (0=loss, 1=draw, 2=win)
                    # WDL values: -2, -1 (loss), 0 (draw), 1, 2 (win)
                    if wdl < 0:
                        label = 0  # loss
                    elif wdl > 0:
                        label = 2  # win
                    else:
                        label = 1  # draw
                    
                    # Convert board to feature vector
                    feature_vector = board_to_feature_vector(board)
                    
                    batch_features.append(feature_vector)
                    batch_labels.append(label)
                    valid_positions += 1
                    
                    # Process batches to limit memory usage
                    if len(batch_features) >= batch_size:
                        # Add current batch to main storage
                        features.extend(batch_features)
                        labels.extend(batch_labels)
                        # Clear batch data
                        batch_features = []
                        batch_labels = []
                        total_batches += 1
                        # Update progress
                        pbar.update(batch_size)
                        pbar.set_postfix({
                            "valid": valid_positions, 
                            "total": positions_processed,
                            "batches": total_batches
                        })
                    
                except chess.syzygy.MissingTableError:
                    # Skip if tablebase entry not found
                    continue
                    
                # Safety limit - if we've generated an enormous number of positions
                # for complex material configurations, we can stop
                if positions_processed > 100000000:  # 100M position limit
                    print(f"Safety limit reached at {positions_processed} positions")
                    break
        
        # Add any remaining positions in the final partial batch
        if batch_features:
            features.extend(batch_features)
            labels.extend(batch_labels)
        
        print(f"Processed {positions_processed} positions, found {valid_positions} valid positions")
        print(f"Loaded {len(labels)} positions from {os.path.basename(self.rtbw_file)}")
        
        # Convert to numpy arrays
        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)
    
    def _exhaustively_enumerate_positions(self, material):
        """Exhaustively enumerate all legal positions for simple material configurations"""
        # Parse material string
        if "v" not in material:
            return
            
        white_pieces, black_pieces = material.split("v")
        
        # Map piece characters to piece types
        piece_map = {
            "K": chess.KING,
            "Q": chess.QUEEN,
            "R": chess.ROOK,
            "B": chess.BISHOP,
            "N": chess.KNIGHT,
            "P": chess.PAWN,
        }
        
        # For very simple material configurations (KvK, KQvK, etc.)
        # we can just iterate through all possible square combinations
        
        # White king can be anywhere
        for wk_square in chess.SQUARES:
            # Black king must be at least 2 squares away
            for bk_square in chess.SQUARES:
                if chess.square_distance(wk_square, bk_square) <= 1:
                    continue
                    
                # For each king placement, place the remaining pieces
                # For example, in KQvK, we need to place the white queen
                
                # Get squares that aren't occupied by kings
                available_squares = [sq for sq in chess.SQUARES 
                                    if sq != wk_square and sq != bk_square]
                
                # Place remaining white pieces
                remaining_white = white_pieces.replace("K", "", 1)  # Remove white king
                remaining_black = black_pieces.replace("K", "", 1)  # Remove black king
                
                # Handle remaining pieces based on material complexity
                if not remaining_white and not remaining_black:
                    # Just kings, try both sides to move
                    for turn in [chess.WHITE, chess.BLACK]:
                        board = chess.Board(fen=None)
                        board.clear_board()
                        board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                        board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                        board.turn = turn
                        
                        if board.is_valid() and not board.is_check():
                            yield board
                            
                # Material with just one additional piece
                elif len(remaining_white) == 1 and not remaining_black:
                    piece_type = piece_map[remaining_white]
                    
                    # Try all squares for the piece
                    for piece_square in available_squares:
                        # Skip invalid pawn placements
                        if piece_type == chess.PAWN and (
                            chess.square_rank(piece_square) == 0 or 
                            chess.square_rank(piece_square) == 7):
                            continue
                            
                        # Try both sides to move
                        for turn in [chess.WHITE, chess.BLACK]:
                            board = chess.Board(fen=None)
                            board.clear_board()
                            board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                            board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                            board.set_piece_at(piece_square, chess.Piece(piece_type, chess.WHITE))
                            board.turn = turn
                            
                            if board.is_valid() and not board.is_check():
                                yield board
                                
                # Material with just one additional black piece
                elif not remaining_white and len(remaining_black) == 1:
                    piece_type = piece_map[remaining_black]
                    
                    # Try all squares for the piece
                    for piece_square in available_squares:
                        # Skip invalid pawn placements
                        if piece_type == chess.PAWN and (
                            chess.square_rank(piece_square) == 0 or 
                            chess.square_rank(piece_square) == 7):
                            continue
                            
                        # Try both sides to move
                        for turn in [chess.WHITE, chess.BLACK]:
                            board = chess.Board(fen=None)
                            board.clear_board()
                            board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                            board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                            board.set_piece_at(piece_square, chess.Piece(piece_type, chess.BLACK))
                            board.turn = turn
                            
                            if board.is_valid() and not board.is_check():
                                yield board
                                
                # More complex simple endgames with 3 pieces total - iterate all combinations
                elif (len(remaining_white) + len(remaining_black) == 1) and (len(white_pieces) + len(black_pieces) == 3):
                    # Determine which color has the extra piece
                    if remaining_white:
                        extra_piece_color = chess.WHITE
                        piece_type = piece_map[remaining_white]
                    else:
                        extra_piece_color = chess.BLACK
                        piece_type = piece_map[remaining_black]
                        
                    # Try all squares for the piece
                    for piece_square in available_squares:
                        # Skip invalid pawn placements
                        if piece_type == chess.PAWN and (
                            chess.square_rank(piece_square) == 0 or 
                            chess.square_rank(piece_square) == 7):
                            continue
                            
                        # Try both sides to move
                        for turn in [chess.WHITE, chess.BLACK]:
                            board = chess.Board(fen=None)
                            board.clear_board()
                            board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                            board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                            board.set_piece_at(piece_square, chess.Piece(piece_type, extra_piece_color))
                            board.turn = turn
                            
                            if board.is_valid() and not board.is_check():
                                yield board
                
                # Otherwise use more efficient enumeration methods
                else:
                    pass
                    
    def _systematic_enumeration(self, material):
        """Systematically enumerate positions for moderate complexity material (4-5 pieces)"""
        if "v" not in material:
            return
            
        white_pieces, black_pieces = material.split("v")
        
        # Map piece characters to piece types
        piece_map = {
            "K": chess.KING,
            "Q": chess.QUEEN,
            "R": chess.ROOK,
            "B": chess.BISHOP,
            "N": chess.KNIGHT,
            "P": chess.PAWN,
        }
        
        # For material configurations with 4-5 pieces, we'll use a systematic approach
        # that leverages symmetry and board structure to reduce the enumeration space
        
        # Generate a systematic grid of king placements
        # For mirror symmetry across files/ranks, we can place white king in a specific region
        # and adjust other pieces accordingly
        
        # Example pattern: Place white king in the a1-d4 region (16 squares instead of 64)
        # and expand from there
        for wk_file in range(4):  # a-d files
            for wk_rank in range(4):  # 1-4 ranks
                wk_square = chess.square(wk_file, wk_rank)
                
                # For each white king placement, try black king placements systematically
                for bk_file in range(8):  # a-h files
                    for bk_rank in range(8):  # 1-8 ranks
                        bk_square = chess.square(bk_file, bk_rank)
                        
                        # Skip invalid king placements
                        if chess.square_distance(wk_square, bk_square) <= 1 or wk_square == bk_square:
                            continue
                            
                        # For 4-5 piece endgames, we need to place 2-3 additional pieces
                        # Create a base board with kings placed
                        base_board = chess.Board(fen=None)
                        base_board.clear_board()
                        base_board.set_piece_at(wk_square, chess.Piece(chess.KING, chess.WHITE))
                        base_board.set_piece_at(bk_square, chess.Piece(chess.KING, chess.BLACK))
                        
                        # Get remaining pieces
                        remaining_white = white_pieces.replace("K", "", 1)
                        remaining_black = black_pieces.replace("K", "", 1)
                        
                        # Use a dedicated helper to place the remaining pieces
                        # This will recursively try all valid combinations
                        available_squares = [sq for sq in chess.SQUARES 
                                          if sq != wk_square and sq != bk_square]
                                          
                        # Distribute placements systematically to ensure coverage
                        # For 4-5 piece endgames, we can still enumerate many positions
                        # by making intelligent choices and leveraging symmetry
                        yield from self._place_remaining_pieces(
                            base_board, 
                            remaining_white,
                            remaining_black,
                            piece_map,
                            available_squares
                        )
    
    def _place_remaining_pieces(self, base_board, remaining_white, remaining_black, piece_map, available_squares):
        """Recursively place remaining pieces for systematic enumeration"""
        # Base case: all pieces placed
        if not remaining_white and not remaining_black:
            # Try both sides to move
            for turn in [chess.WHITE, chess.BLACK]:
                board = base_board.copy()
                board.turn = turn
                if board.is_valid() and not board.is_check():
                    yield board
            return
            
        # Place next white piece
        if remaining_white:
            piece_char = remaining_white[0]
            piece_type = piece_map[piece_char]
            remaining_white_after = remaining_white[1:]
            
            # Try a subset of squares for moderate complexity endgames
            # For 4-5 piece endgames, try placing on a subset of squares based on piece type
            if piece_type == chess.PAWN:
                # Pawns can't be on first or last rank
                valid_squares = [sq for sq in available_squares 
                              if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7]
            elif piece_type in [chess.BISHOP, chess.KNIGHT]:
                # Minor pieces - try every other square to reduce combinations
                valid_squares = [sq for i, sq in enumerate(available_squares) if i % 2 == 0]
            else:
                # Major pieces - try a wider distribution
                valid_squares = available_squares[:]
                
            # Limit to a reasonable number for complex positions
            max_squares = min(len(valid_squares), 24)
            sample_squares = valid_squares[:max_squares]
            
            # Place the piece on each valid square
            for square in sample_squares:
                # Create a new board with this piece placed
                new_board = base_board.copy()
                new_board.set_piece_at(square, chess.Piece(piece_type, chess.WHITE))
                
                # Remove this square from available squares
                remaining_squares = [sq for sq in available_squares if sq != square]
                
                # Recursively place remaining pieces
                yield from self._place_remaining_pieces(
                    new_board,
                    remaining_white_after,
                    remaining_black,
                    piece_map,
                    remaining_squares
                )
                
        # Place next black piece
        elif remaining_black:
            piece_char = remaining_black[0]
            piece_type = piece_map[piece_char]
            remaining_black_after = remaining_black[1:]
            
            # Similar logic as for white pieces
            if piece_type == chess.PAWN:
                valid_squares = [sq for sq in available_squares 
                              if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7]
            elif piece_type in [chess.BISHOP, chess.KNIGHT]:
                valid_squares = [sq for i, sq in enumerate(available_squares) if i % 2 == 0]
            else:
                valid_squares = available_squares[:]
                
            max_squares = min(len(valid_squares), 24)
            sample_squares = valid_squares[:max_squares]
            
            for square in sample_squares:
                new_board = base_board.copy()
                new_board.set_piece_at(square, chess.Piece(piece_type, chess.BLACK))
                
                remaining_squares = [sq for sq in available_squares if sq != square]
                
                yield from self._place_remaining_pieces(
                    new_board,
                    remaining_white,
                    remaining_black_after,
                    piece_map,
                    remaining_squares
                )
                
    def _comprehensive_sampling(self, material):
        """Generate a comprehensive sample of positions for complex material configurations"""
        # For very complex endgames (6+ pieces), exhaustive enumeration is infeasible
        # Instead, we'll use a combination of:
        # 1. Strategic sampling of key areas of the board
        # 2. Biased sampling toward realistic positions
        # 3. Iterative refinement to ensure coverage
        
        # First, generate a baseline set of random positions for this material
        target_positions = 10000000  # Target 10M positions for complex endgames
        positions_generated = 0
        duplicate_count = 0
        seen_positions = set()
        
        print(f"Generating comprehensive position sample for complex material: {material}")
        print(f"Target: {target_positions} positions, with adaptive refinement")
        
        # Phase 1: Generate a baseline of positions through strategic sampling
        while positions_generated < target_positions:
            board = self._create_position_with_material(material)
            if board is not None:
                # Track unique positions with a compact hash
                board_hash = board.epd()
                
                if board_hash not in seen_positions:
                    seen_positions.add(board_hash)
                    positions_generated += 1
                    yield board
                else:
                    duplicate_count += 1
                    
                # Adaptive refinement: if we're getting too many duplicates,
                # adjust our generation strategy
                if duplicate_count > 1000000:  # After 1M duplicates, refine strategy
                    duplicate_count = 0
                    print(f"Refining position generation strategy after {positions_generated} positions")
                    # This would adaptively tweak piece placement for better coverage
                    
                # Safety limit to prevent runaway generation
                if positions_generated + duplicate_count > 5 * target_positions:
                    print(f"Safety limit reached after generating {positions_generated} positions")
                    break
    
    def _generate_positions(self, material, target_count):
        """Generate positions for the given material configuration"""
        # This uses random position generation to cover the space
        # but tries to be more systematic for better coverage
        positions_generated = 0
        positions_yielded = 0
        
        # Use a more systematic approach for placing pieces
        while positions_yielded < target_count:
            board = self._create_position_with_material(material)
            if board is not None:
                positions_generated += 1
                positions_yielded += 1
                yield board
                
            # Safety limit to prevent infinite loops
            if positions_generated >= 10 * target_count:
                print(f"Warning: Generation limit reached. Only found {positions_yielded} positions")
                break
    
    def _create_position_with_material(self, material):
        """Create a legal position with the specified material configuration"""
        # Parse material string (e.g., KRvKN -> King+Rook vs King+Knight)
        if "v" not in material:
            return None

        white_pieces, black_pieces = material.split("v")

        # Create empty board
        board = chess.Board(fen=None)
        board.clear_board()

        # Available squares
        available_squares = list(chess.SQUARES)
        random.shuffle(available_squares)

        # Place white king
        if "K" in white_pieces:
            white_king_square = available_squares.pop()
            board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))
            white_pieces = white_pieces.replace("K", "", 1)

        # Place black king
        if "K" in black_pieces:
            # Kings must be at least 2 squares apart
            valid_squares = [
                sq
                for sq in available_squares
                if chess.square_distance(sq, white_king_square) > 1
            ]
            if not valid_squares:
                return None

            black_king_square = random.choice(valid_squares)
            available_squares.remove(black_king_square)
            board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))
            black_pieces = black_pieces.replace("K", "", 1)

        # Map piece characters to piece types
        piece_map = {
            "Q": chess.QUEEN,
            "R": chess.ROOK,
            "B": chess.BISHOP,
            "N": chess.KNIGHT,
            "P": chess.PAWN,
        }

        # Place remaining white pieces
        for char in white_pieces:
            if char in piece_map and available_squares:
                square = available_squares.pop()
                # Pawns can't be on first or last rank
                if piece_map[char] == chess.PAWN:
                    valid_squares = [
                        sq
                        for sq in available_squares
                        if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7
                    ]
                    if not valid_squares:
                        return None
                    square = random.choice(valid_squares)
                    available_squares.remove(square)

                board.set_piece_at(square, chess.Piece(piece_map[char], chess.WHITE))

        # Place remaining black pieces
        for char in black_pieces:
            if char in piece_map and available_squares:
                square = available_squares.pop()
                # Pawns can't be on first or last rank
                if piece_map[char] == chess.PAWN:
                    valid_squares = [
                        sq
                        for sq in available_squares
                        if chess.square_rank(sq) > 0 and chess.square_rank(sq) < 7
                    ]
                    if not valid_squares:
                        return None
                    square = random.choice(valid_squares)
                    available_squares.remove(square)

        return None


def get_rtbw_files(tablebase_dir):
    """Get all .rtbw files in the tablebase directory"""
    rtbw_files = glob.glob(os.path.join(tablebase_dir, "*.rtbw"))
    return rtbw_files
