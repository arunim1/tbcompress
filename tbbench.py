import sys
import time
import random
import chess
import chess.syzygy
import json


def generate_valid_5piece_boards(n):
    """
    Generate n random 5-piece boards that:
      - Are valid per python-chess
    """
    piece_types = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    boards = []
    while len(boards) < n:
        # Build an empty board and place pieces
        board = chess.Board.empty()
        sqs = random.sample(range(64), 5)
        # place kings
        board.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        # place 3 other pieces
        for sq in sqs[2:]:
            pt = random.choice(piece_types)
            col = random.choice([True, False])
            # avoid pawns on first or eighth rank
            if pt == chess.PAWN and sq // 8 in (0, 7):
                pt = random.choice([t for t in piece_types if t != chess.PAWN])
            board.set_piece_at(sq, chess.Piece(pt, col))
        # random side to move, clear castling/en passant
        board.turn = random.choice([True, False])
        board.castling_rights = 0
        board.ep_square = None
        board.halfmove_clock = 0
        board.fullmove_number = 1

        # filter for validity and tablebase support
        if not board.is_valid():
            continue

        boards.append(board)
    return boards


def probe_positions(tb, boards):
    for board in boards:
        tb.probe_wdl(board)


# %%


def benchmark_random_positions(tablebase_dir, num_positions=10000):
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

    with open("stats.json", "w") as f:
        json.dump(stats, f, indent=4)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("tablebase_dir", help="Path to Syzygy WDL tablebase dir")
    args = p.parse_args()
    benchmark_random_positions(args.tablebase_dir, num_positions=100000)
