# ADD THESE TWO IMPORTS NEAR THE TOP OF THE FILE
import os
import multiprocessing
import random
import chess
import chess.syzygy
import json
import numpy as np
import sys
import time
import random
import chess
import chess.syzygy
import json
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# ---------------------------------------------------------------------------
# OPTIMISED HELPERS – replace the originals with everything in this block
# ---------------------------------------------------------------------------


def generate_valid_5piece_boards(n: int):
    """
    Generate *n* random, valid 5‑piece positions **as FEN strings**.
    Returning FENs keeps memory footprint tiny and makes the list cheap to
    pass between processes.
    """
    piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    fens = []

    # local bindings for tighter inner loop
    sample = random.sample
    choice = random.choice
    Piece = chess.Piece
    Board = chess.Board

    while len(fens) < n:
        b = Board.empty()

        # Pick 5 distinct squares
        sqs = sample(range(64), 5)

        # Kings
        b.set_piece_at(sqs[0], Piece(chess.KING, chess.WHITE))
        b.set_piece_at(sqs[1], Piece(chess.KING, chess.BLACK))

        # Three other pieces
        for sq in sqs[2:]:
            pt = choice(piece_types)
            # Keep pawns off first/last rank to avoid auto‑rejection
            if pt == chess.PAWN and (sq >> 3) in (0, 7):
                pt = choice(piece_types[:-1])
            b.set_piece_at(sq, Piece(pt, choice((True, False))))

        # Misc position fields
        b.turn = choice((True, False))
        b.castling_rights = 0
        b.ep_square = None
        b.halfmove_clock = 0
        b.fullmove_number = 1

        if b.is_valid():
            fens.append(b.fen())

    return fens


# ---------------------------------------------------------------------------
# Parallel probing helpers
# ---------------------------------------------------------------------------

_worker_tb = None  # each worker keeps its own tablebase handle


def _pool_init(tb_dir: str):
    """Pool initialiser – open the Syzygy tablebase once per process."""
    global _worker_tb
    _worker_tb = chess.syzygy.open_tablebase(tb_dir)


def _probe_fen(fen: str):
    """Worker task: build the board from FEN and probe WDL."""
    board = chess.Board(fen)
    _worker_tb.probe_wdl(board)
    return None


def probe_positions(tb, boards):
    """
    Probe every position in *boards* (list of FENs).

    • Uses up to min(8, os.cpu_count()) worker processes.
    • Falls back to the original single‑thread loop if we can’t retrieve the
      tablebase directory or if only one CPU core is available.
    """
    # Best‑effort extraction of the directory used to open *tb*
    tb_dir = (
        getattr(tb, "path", None)
        or getattr(tb, "directory", None)
        or (getattr(tb, "paths", [None])[0] if hasattr(tb, "paths") else None)
    )

    workers = min(8, os.cpu_count() or 1)

    if tb_dir and workers > 1:
        with multiprocessing.Pool(
            workers, initializer=_pool_init, initargs=(tb_dir,)
        ) as pool:
            pool.map(_probe_fen, boards, chunksize=256)
    else:
        # Single‑thread fallback
        for fen in boards:
            tb.probe_wdl(chess.Board(fen))


def benchmark_random_positions(tablebase_dir, num_positions=10000):
    all_stats = []
    for _ in range(3):
        stats = {}
        tb = chess.syzygy.open_tablebase(tablebase_dir)

        print(f"Generating {num_positions} valid 5-piece boards")
        start_time = time.perf_counter()
        boards = generate_valid_5piece_boards(num_positions)
        end_time = time.perf_counter()
        print(
            f"Generated {num_positions} valid 5-piece boards in {end_time - start_time:.3f} s"
        )
        # num boards / second
        stats["boards_per_second"] = num_positions / (end_time - start_time)
        print(f"{stats['boards_per_second']:.3f} boards/s")

        # how much memory is used?
        print(f"Memory used: {sys.getsizeof(boards) / 1e6:.3f} MB")
        stats["memory_used"] = sys.getsizeof(boards) / 1e6

        # Benchmark loop
        print(f"Benchmarking {num_positions} probes")
        start_time = time.perf_counter()
        probe_positions(tb, boards)
        end_time = time.perf_counter()

        total = end_time - start_time
        avg_us = (total / num_positions) * 1e6
        throughput = num_positions / total

        stats["total_time"] = total
        stats["avg_us"] = avg_us
        stats["throughput"] = throughput

        print(f"Positions: {num_positions}")
        print(f"Total time: {total:.3f} s")
        print(f"Avg per probe: {avg_us:.2f} µs")
        print(f"Throughput: {throughput:.0f} probes/sec")
        all_stats.append(stats)

    # construct a baseline_stats_avg.json with averages and standard deviations
    baseline_stats_avg = {
        "boards_per_second": sum([s["boards_per_second"] for s in all_stats])
        / len(all_stats),
        "boards_per_second_std": np.std([s["boards_per_second"] for s in all_stats]),
        "throughput": sum([s["throughput"] for s in all_stats]) / len(all_stats),
        "throughput_std": np.std([s["throughput"] for s in all_stats]),
    }

    with open(f"o3_stats_{num_positions}.json", "w") as f:
        json.dump(baseline_stats_avg, f, indent=4)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "positions",
        type=int,
        default=1000,
        help="Number of positions to probe (default: 1000)",
    )
    p.add_argument(
        "--tablebase_dir",
        help="Path to Syzygy WDL tablebase dir",
        default="Syzygy345_WDL",
    )
    args = p.parse_args()
    benchmark_random_positions(args.tablebase_dir, num_positions=args.positions)
