import sys
import time
import random
import chess
import chess.syzygy
import json
import numpy as np
import torch


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


def board_to_tensor(board):
    # convert a chess.Board to a torch tensor of shape (769,)
    feature_vector = torch.zeros(769, dtype=torch.uint8)

    # Populate planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue

        # Plane index: 0-5 for white pieces, 6-11 for black
        plane_idx = piece.piece_type - 1  # pawn=0 … king=5
        if piece.color == chess.BLACK:
            plane_idx += 6

        feature_vector[plane_idx * 64 + square] = 1

    # Side-to-move bit (1 if white to move else 0)
    feature_vector[768] = 1 if board.turn == chess.WHITE else 0

    return feature_vector


def probe_positions(tb, boards):
    # save a torch tensor of inputs (769-vector capturing the board state) and outputs (-2, 0, 2)
    inputs = torch.zeros((len(boards), 769), dtype=torch.uint8)
    outputs = torch.zeros(len(boards), dtype=torch.int8)
    for i, board in enumerate(boards):
        inputs[i] = board_to_tensor(board)
        out = tb.probe_wdl(board)
        # out is -2, 0, 2
        outputs[i] = out
    return inputs, outputs


# DO NOT EDIT BELOW THIS POINT


def benchmark_random_positions(tablebase_dir, num_positions=10000):
    all_stats = []
    for _ in range(3):
        stats = {}
        tb = chess.syzygy.open_tablebase(tablebase_dir, max_fds=1024)

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
        inputs, outputs = probe_positions(tb, boards)
        end_time = time.perf_counter()

        # save
        torch.save(inputs, "inputs.pt")
        torch.save(outputs, "outputs.pt")

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

    with open(f"new_stats_{num_positions}1.json", "w") as f:
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
