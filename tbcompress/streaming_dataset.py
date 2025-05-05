"""
Streaming dataset implementation for Syzygy tablebase positions
"""

import os
import random
import itertools
import threading
import queue
import time
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
    # Pre-allocate the complete feature vector (768 piece features + 1 side to move)
    feature_vector = np.zeros(769, dtype=np.float32)

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
    feature_vector[768] = 1.0 if board.turn == chess.WHITE else 0.0

    return feature_vector


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

        # Thread-safe queue for position cache
        self.cache = queue.Queue(maxsize=cache_size)

        # Thread control
        self.stop_event = threading.Event()
        self.generator_exhausted = threading.Event()

        # Track counts for reporting
        self.positions_processed = 0
        self.valid_positions = 0
        self.positions_lock = threading.Lock()  # For thread-safe counter updates

        # Position generation strategy and estimate size
        if self.piece_count <= 5:  # Up to five-piece TBs
            # Use full exhaustive enumeration (lazy generator) regardless of
            # combinatorial explosion. We only materialise one position at a
            # time, so memory usage stays bounded.
            self.position_generator = self._exhaustively_enumerate_positions_general()

            # Provide a generous upper bound so training logic can compute
            # epochs. Five-piece TBs contain < 500M positions.
            self.estimated_size = int(5e8)
        else:
            raise ValueError(
                "This dataset currently supports up to 5-piece tablebases."
            )

        # Start the background producer thread
        self.producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,  # Make thread a daemon so it exits when the main thread exits
            name="TB-Producer",
        )
        self.producer_thread.start()

        # Wait for initial cache filling to ensure data is available immediately
        while self.cache.empty() and not self.generator_exhausted.is_set():
            time.sleep(0.1)  # Small wait to avoid busy waiting

        print(
            f"Initialized streaming dataset for {self.material} with estimated {self.estimated_size} positions"
        )

    def _producer_loop(self):
        """Background thread that continuously fills the queue with positions"""
        try:
            print(f"Starting position generator thread for {self.material}")
            
            # Continue until stop event is set or max positions reached
            while not self.stop_event.is_set():
                # Check if we've reached max positions (if specified)
                with self.positions_lock:
                    if (
                        self.max_positions is not None
                        and self.valid_positions >= self.max_positions
                    ):
                        print(f"Reached max positions limit of {self.max_positions}")
                        self.generator_exhausted.set()
                        break
                
                try:
                    # Get next position from generator
                    board = next(self.position_generator)
                    
                    with self.positions_lock:
                        self.positions_processed += 1
                    
                    # Get WDL value from tablebase
                    try:
                        # Get the WDL value from the tablebase (win/draw/loss)
                        # WDL values: 2 = win, 0 = draw, -2 = loss
                        wdl = self.tablebase.probe_wdl(board)
                        
                        # For our model, convert to 0, 1, 2 for loss, draw, win
                        # Map to 0 (loss), 1 (draw), 2 (win)
                        if wdl < 0:  # Loss
                            classification = 0
                        elif wdl > 0:  # Win
                            classification = 2
                        else:  # Draw
                            classification = 1
                        
                        # Convert board to features
                        features = board_to_feature_vector(board)
                        
                        # Add to queue - this will block if queue is full
                        self.cache.put((features, classification))
                        
                        with self.positions_lock:
                            self.valid_positions += 1
                            
                    except (ValueError, chess.syzygy.MissingTableError):
                        # Skip positions not in the tablebase
                        continue
                        
                except StopIteration:
                    # No more positions available from generator
                    print("Position generator exhausted - restarting generator")
                    # Restart the generator for another epoch
                    self.position_generator = self._exhaustively_enumerate_positions_general()
                    continue
                    
                except Exception as e:
                    print(f"Error processing position: {e}")
                    time.sleep(0.01)  # Brief pause to avoid tight loop on errors
                    continue
                    
        except Exception as e:
            # Log any errors but try to continue
            print(f"Error in position generator thread: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Signal that the generator is done
            self.generator_exhausted.set()
            print(f"Position generator thread for {self.material} stopped")

    def __len__(self):
        """Return estimated size of dataset"""
        if self.max_positions is not None:
            return min(self.estimated_size, self.max_positions)
        return self.estimated_size

    def __getitem__(self, idx):
        """Get a position by index - pulls from the thread-safe queue"""
        # For streaming datasets, the idx isn't used directly since we're pulling
        # from the cache queue. But we still need to implement __getitem__
        # according to the Dataset interface.
        if self.stop_event.is_set():
            raise IndexError("Dataset is shutting down")

        # Get batches of items when possible to reduce queue contention
        try:
            # Get next item from the queue with a timeout
            # This avoids hanging indefinitely if the producer thread is dead
            features, wdl = self.cache.get(block=True, timeout=2.0)  # Reduced timeout
            self.cache.task_done()  # Mark task as done
            
            # Convert to tensors - feature vector and classification label
            return torch.tensor(features, dtype=torch.float32), torch.tensor(wdl, dtype=torch.long)

        except queue.Empty:
            if self.generator_exhausted.is_set() and self.cache.empty():
                # Generator is done and queue is empty - no more data
                raise IndexError("No more positions available - dataset exhausted")
            else:
                # Short sleep to avoid CPU thrashing and reduce contention
                time.sleep(0.001)
                # Try again
                return self.__getitem__(idx)

    def shutdown(self):
        """Gracefully shut down the producer thread"""
        print(f"Shutting down producer thread for {self.material}")
        self.stop_event.set()

        # Wait for the thread to finish (with timeout)
        if self.producer_thread.is_alive():
            self.producer_thread.join(timeout=5.0)

        # Clear the queue to free memory
        while not self.cache.empty():
            try:
                self.cache.get_nowait()
                self.cache.task_done()
            except queue.Empty:
                break

        print(f"Producer thread for {self.material} shut down")

    def _exhaustively_enumerate_positions_general(self):
        """Exhaustively enumerate **all** legal positions for up to 5 pieces.

        This is a lazy generator. It yields one position at a time so memory
        usage remains low even for enormous state spaces.
        """
        # Parse material strings
        white_str, black_str = self.material.split("v")

        # Build list of (piece_type, color) including BOTH kings first removed
        white_list = [ch for ch in white_str if ch != "K"]
        black_list = [ch for ch in black_str if ch != "K"]

        remaining_pieces = [
            (
                chess.Piece(
                    {
                        "Q": chess.QUEEN,
                        "R": chess.ROOK,
                        "B": chess.BISHOP,
                        "N": chess.KNIGHT,
                        "P": chess.PAWN,
                    }[p],
                    chess.WHITE,
                )
            )
            for p in white_list
        ] + [
            chess.Piece(
                {
                    "Q": chess.QUEEN,
                    "R": chess.ROOK,
                    "B": chess.BISHOP,
                    "N": chess.KNIGHT,
                    "P": chess.PAWN,
                }[p],
                chess.BLACK,
            )
            for p in black_list
        ]

        # Precompute all square combinations for kings
        for wk_sq in chess.SQUARES:
            for bk_sq in chess.SQUARES:
                if wk_sq == bk_sq or chess.square_distance(wk_sq, bk_sq) <= 1:
                    continue

                occupied = {wk_sq, bk_sq}

                # Choose squares for the remaining pieces (combination order
                # does not matter yet).
                for squares in itertools.permutations(
                    [s for s in chess.SQUARES if s not in occupied],
                    len(remaining_pieces),
                ):
                    skip = False
                    for sq, piece in zip(squares, remaining_pieces):
                        if piece.piece_type == chess.PAWN and chess.square_rank(sq) in (
                            0,
                            7,
                        ):
                            skip = True
                            break
                    if skip:
                        continue

                    board = chess.Board(fen=None)
                    board.clear_board()
                    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
                    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))

                    for sq, piece in zip(squares, remaining_pieces):
                        board.set_piece_at(sq, piece)

                    if not board.is_valid():
                        continue

                    # Yield both sides to move
                    for turn in [chess.WHITE, chess.BLACK]:
                        board.turn = turn
                        yield board.copy()
