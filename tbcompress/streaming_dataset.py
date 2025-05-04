"""
Streaming dataset implementation for Syzygy tablebase positions
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import chess
import chess.syzygy


def board_to_feature_vector(board):
    """
    Convert a chess.Board to a feature vector for the neural network

    Args:
        board: A chess.Board object

    Returns:
        numpy array with 769 features (12 * 64 one-hot piece planes + side to move)
    """
    # 12 × 64 one-hot planes: 6 white piece types followed by 6 black
    feature_vector = np.zeros(12 * 64, dtype=np.float32)

    # Populate planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # Plane index: 0-5 for white pieces, 6-11 for black
        plane_idx = piece.piece_type - 1  # pawn=0 … king=5
        if piece.color == chess.BLACK:
            plane_idx += 6

        feature_vector[plane_idx * 64 + square] = 1.0

    # Side-to-move bit (1 if white to move else 0)
    side_to_move = 1.0 if board.turn == chess.WHITE else 0.0

    return np.append(feature_vector, side_to_move)


class StreamingTablebaseDataset(Dataset):
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
        self.rtbw_file = rtbw_file
        self.tablebase_dir = tablebase_dir
        self.cache_size = cache_size
        self.max_positions = max_positions

        # Initialize tablebase
        self.tablebase = chess.syzygy.open_tablebase(tablebase_dir)

        # Extract material configuration
        self.material = os.path.basename(rtbw_file).split(".")[0]
        white_material, black_material = self.material.split("v")
        self.piece_count = len(white_material) + len(black_material)

        # Position cache
        self.position_cache = []
        self.wdl_cache = []
        self.cache_index = 0

        # Track counts for reporting
        self.positions_processed = 0
        self.valid_positions = 0
        self.seen_positions = set()

        # Position generation strategy and estimate size
        if self.piece_count <= 3:  # Simple endgames
            self.position_generator = self._exhaustively_enumerate_positions()
            self.estimated_size = 64 * 63 * 2  # Approximate for KvK and similar
        else:  # Complex endgames
            # TODO: Implement exhaustive enumeration for complex games, or generalize the current function.
            pass

        # Cap the size based on max_positions if specified
        if self.max_positions is not None and self.estimated_size > self.max_positions:
            self.estimated_size = self.max_positions

        # Fill initial cache
        self._fill_cache()

        print(
            f"Initialized streaming dataset for {self.material} with estimated {self.estimated_size} positions"
        )

    def _fill_cache(self):
        """Fill the position cache with valid positions"""
        if len(self.position_cache) >= self.cache_size:
            return

        new_positions = []
        new_wdl_values = []
        while len(new_positions) < self.cache_size and (
            self.max_positions is None or self.valid_positions < self.max_positions
        ):
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
                        print(
                            f"Generated {self.valid_positions} valid positions from {self.positions_processed} attempts"
                        )

                    # Check if we've reached max positions
                    if (
                        self.max_positions is not None
                        and self.valid_positions >= self.max_positions
                    ):
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
                    self.position_generator = self._exhaustively_enumerate_positions()
                elif self.piece_count <= 5:
                    self.position_generator = self._systematic_enumeration()
                else:
                    self.position_generator = self._comprehensive_sampling()
                self._fill_cache()

                # If still empty, we have a problem
                if not self.position_cache:
                    raise IndexError("No valid positions could be generated")

        # Get the position from the cache
        features = self.position_cache[self.cache_index]
        wdl = self.wdl_cache[self.cache_index]
        self.cache_index += 1

        return torch.tensor(features, dtype=torch.float32), torch.tensor(
            wdl, dtype=torch.long
        )

    def _exhaustively_enumerate_positions(self):
        """Exhaustively enumerate all legal positions for simple material configurations"""
        # Parse material string
        if "v" not in self.material:
            return

        white_pieces, black_pieces = self.material.split("v")

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
                available_squares = [
                    sq for sq in chess.SQUARES if sq != wk_square and sq != bk_square
                ]

                # Place remaining white pieces
                remaining_white = white_pieces.replace("K", "", 1)  # Remove white king
                remaining_black = black_pieces.replace("K", "", 1)  # Remove black king

                # Handle remaining pieces based on material complexity
                if not remaining_white and not remaining_black:
                    # Just kings, try both sides to move
                    for turn in [chess.WHITE, chess.BLACK]:
                        board = chess.Board(fen=None)
                        board.clear_board()
                        board.set_piece_at(
                            wk_square, chess.Piece(chess.KING, chess.WHITE)
                        )
                        board.set_piece_at(
                            bk_square, chess.Piece(chess.KING, chess.BLACK)
                        )
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
                            chess.square_rank(piece_square) == 0
                            or chess.square_rank(piece_square) == 7
                        ):
                            continue

                        # Try both sides to move
                        for turn in [chess.WHITE, chess.BLACK]:
                            board = chess.Board(fen=None)
                            board.clear_board()
                            board.set_piece_at(
                                wk_square, chess.Piece(chess.KING, chess.WHITE)
                            )
                            board.set_piece_at(
                                bk_square, chess.Piece(chess.KING, chess.BLACK)
                            )
                            board.set_piece_at(
                                piece_square, chess.Piece(piece_type, chess.WHITE)
                            )
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
                            chess.square_rank(piece_square) == 0
                            or chess.square_rank(piece_square) == 7
                        ):
                            continue

                        # Try both sides to move
                        for turn in [chess.WHITE, chess.BLACK]:
                            board = chess.Board(fen=None)
                            board.clear_board()
                            board.set_piece_at(
                                wk_square, chess.Piece(chess.KING, chess.WHITE)
                            )
                            board.set_piece_at(
                                bk_square, chess.Piece(chess.KING, chess.BLACK)
                            )
                            board.set_piece_at(
                                piece_square, chess.Piece(piece_type, chess.BLACK)
                            )
                            board.turn = turn

                            if board.is_valid() and not board.is_check():
                                yield board
