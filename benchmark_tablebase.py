#!/usr/bin/env python
"""
Benchmark script to evaluate the performance of Syzygy tablebase probing.
This script measures how fast we can read WDL values from a tablebase for ~1 million random boards.
"""

import os
import time
import random
import argparse
import chess
import chess.syzygy
import itertools
from tqdm import tqdm
import numpy as np


def generate_random_board(piece_count=5):
    """Generate a random legal chess position with the given number of pieces."""
    board = chess.Board(fen=None)
    board.clear_board()
    
    # Always include both kings
    squares = list(chess.SQUARES)
    random.shuffle(squares)
    
    # Place white king
    wk_sq = squares.pop()
    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
    
    # Place black king (ensuring kings are not adjacent)
    valid_bk_squares = [sq for sq in squares if chess.square_distance(sq, wk_sq) > 1]
    if not valid_bk_squares:
        return None  # Unlikely but possible
    
    bk_sq = random.choice(valid_bk_squares)
    squares.remove(bk_sq)
    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
    
    # Add remaining pieces (if any)
    remaining_pieces = piece_count - 2  # subtract the two kings
    if remaining_pieces > 0:
        # Available piece types (no kings)
        piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        colors = [chess.WHITE, chess.BLACK]
        
        for _ in range(remaining_pieces):
            if not squares:  # No more available squares
                break
                
            sq = squares.pop()
            piece_type = random.choice(piece_types)
            color = random.choice(colors)
            
            # Avoid pawns on first/last rank
            if piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
                
            board.set_piece_at(sq, chess.Piece(piece_type, color))
    
    # Randomly set turn
    board.turn = random.choice([chess.WHITE, chess.BLACK])
    
    if not board.is_valid():
        return None
        
    return board


def benchmark_tablebase_probing(tablebase_dir, num_positions=1000000, piece_count=5, batch_size=10000):
    """
    Benchmark the speed of tablebase probing for a large number of random positions.
    
    Args:
        tablebase_dir: Directory containing Syzygy tablebase files
        num_positions: Number of positions to probe
        piece_count: Number of pieces on the board (including kings)
        batch_size: Number of positions to generate and probe in each batch
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"Opening tablebase from {tablebase_dir}")
    tablebase = chess.syzygy.open_tablebase(tablebase_dir)
    
    total_positions = 0
    valid_positions = 0
    probe_times = []
    
    start_time = time.time()
    
    # Process in batches to show progress
    num_batches = (num_positions + batch_size - 1) // batch_size
    
    for batch in tqdm(range(num_batches), desc="Probing positions"):
        batch_start_time = time.time()
        batch_valid = 0
        
        for _ in range(batch_size):
            if total_positions >= num_positions:
                break
                
            board = generate_random_board(piece_count)
            total_positions += 1
            
            if board is None:
                continue
                
            try:
                probe_start = time.time()
                wdl = tablebase.probe_wdl(board)
                probe_end = time.time()
                
                probe_times.append(probe_end - probe_start)
                valid_positions += 1
                batch_valid += 1
                
            except (ValueError, chess.syzygy.MissingTableError):
                # Position not in tablebase
                continue
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        # Print batch statistics
        if batch_valid > 0:
            avg_probe_time = batch_time / batch_valid
            tqdm.write(f"Batch {batch+1}/{num_batches}: {batch_valid} valid positions, "
                      f"{avg_probe_time*1e6:.2f} μs per position")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    if valid_positions > 0:
        avg_probe_time = sum(probe_times) / valid_positions
        median_probe_time = np.median(probe_times)
        p95_probe_time = np.percentile(probe_times, 95)
        p99_probe_time = np.percentile(probe_times, 99)
        
        positions_per_second = valid_positions / total_time
        
        results = {
            "total_positions": total_positions,
            "valid_positions": valid_positions,
            "total_time_seconds": total_time,
            "avg_probe_time_us": avg_probe_time * 1e6,  # Convert to microseconds
            "median_probe_time_us": median_probe_time * 1e6,
            "p95_probe_time_us": p95_probe_time * 1e6,
            "p99_probe_time_us": p99_probe_time * 1e6,
            "positions_per_second": positions_per_second
        }
        
        return results
    else:
        print("No valid positions found in tablebase")
        return None


def print_results(results):
    """Print benchmark results in a formatted way."""
    if not results:
        return
        
    print("\n===== Tablebase Probing Benchmark Results =====")
    print(f"Total positions generated: {results['total_positions']:,}")
    print(f"Valid positions probed: {results['valid_positions']:,}")
    print(f"Total benchmark time: {results['total_time_seconds']:.2f} seconds")
    print(f"Positions per second: {results['positions_per_second']:.2f}")
    print("\nProbe timing statistics:")
    print(f"  Average: {results['avg_probe_time_us']:.2f} μs")
    print(f"  Median:  {results['median_probe_time_us']:.2f} μs")
    print(f"  P95:     {results['p95_probe_time_us']:.2f} μs")
    print(f"  P99:     {results['p99_probe_time_us']:.2f} μs")
    print("==============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Syzygy tablebase probing speed")
    parser.add_argument("tablebase_dir", help="Directory containing Syzygy tablebase files")
    parser.add_argument("--positions", type=int, default=1000000, 
                        help="Number of positions to probe (default: 1,000,000)")
    parser.add_argument("--pieces", type=int, default=5, choices=range(3, 8),
                        help="Number of pieces on the board (default: 5)")
    parser.add_argument("--batch-size", type=int, default=10000,
                        help="Batch size for progress reporting (default: 10,000)")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.tablebase_dir):
        print(f"Error: Tablebase directory '{args.tablebase_dir}' does not exist")
        exit(1)
    
    results = benchmark_tablebase_probing(
        args.tablebase_dir, 
        num_positions=args.positions,
        piece_count=args.pieces,
        batch_size=args.batch_size
    )
    
    print_results(results)
