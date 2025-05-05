"""
Benchmarking script for StreamingTablebaseDataset performance.
Measures speed of position generator and cache filling.
"""

import os
import sys
import time
import argparse

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = script_dir
sys.path.insert(0, project_root)

from tbcompress.streaming_dataset import StreamingTablebaseDataset


def benchmark_generator(rtbw_file, tablebase_dir, n):
    """Benchmark raw position generation via the lazy generator."""
    ds = StreamingTablebaseDataset(
        rtbw_file, tablebase_dir, cache_size=1, max_positions=None
    )
    gen = ds._exhaustively_enumerate_positions_general()
    start = time.perf_counter()
    count = 0
    try:
        for _ in range(n):
            next(gen)
            count += 1
    except StopIteration:
        pass
    duration = time.perf_counter() - start
    print(
        f"Generated {count} positions in {duration:.4f}s ({count/duration:.2f} pos/s)"
    )


def benchmark_fill_cache(rtbw_file, tablebase_dir, cache_size):
    """Benchmark the _fill_cache method."""
    ds = StreamingTablebaseDataset(
        rtbw_file, tablebase_dir, cache_size=cache_size, max_positions=cache_size
    )
    # Clear any initial cache
    ds.position_cache.clear()
    ds.wdl_cache.clear()
    ds.valid_positions = 0
    ds.positions_processed = 0
    start = time.perf_counter()
    ds._fill_cache()
    duration = time.perf_counter() - start
    filled = len(ds.position_cache)
    print(
        f"Filled cache with {filled} positions in {duration:.4f}s ({filled/duration:.2f} pos/s)"
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark StreamingTablebaseDataset")
    parser.add_argument("rtbw_file", help="Path to .rtbw tablebase file")
    parser.add_argument("tablebase_dir", help="Directory of Syzygy tablebases")
    parser.add_argument(
        "--gen_positions",
        "-g",
        type=int,
        default=10000,
        help="Number of positions to generate in generator benchmark",
    )
    parser.add_argument(
        "--cache_size",
        "-c",
        type=int,
        default=10000,
        help="Cache size for fill_cache benchmark",
    )
    args = parser.parse_args()

    print("\nBenchmarking position generator...")
    benchmark_generator(args.rtbw_file, args.tablebase_dir, args.gen_positions)
    print("\nBenchmarking cache filling...")
    benchmark_fill_cache(args.rtbw_file, args.tablebase_dir, args.cache_size)


if __name__ == "__main__":
    main()
