# compressor.py

import torch
import torch.nn.functional as F
import chess
from tbbench import generate_valid_5piece_boards, board_to_tensor
import os
import sys
import pickle

import torch
import torch.nn.functional as F
import chess


# single‐board bit‐vector
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    v = torch.zeros(769, dtype=torch.uint8)
    for sq, p in board.piece_map().items():
        plane = (p.piece_type - 1) + (6 if p.color == chess.BLACK else 0)
        v[plane * 64 + sq] = 1
    v[768] = board.turn == chess.WHITE
    return v


# manual packbits
def packbits_torch(bits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # bits: uint8 {0,1}, last-dim length = L
    L = bits.size(dim)
    pad = (-L) % 8
    if pad:
        # pad zeros on the right along dim
        pad_dims = [(0, 0)] * bits.dim()
        pad_dims[dim] = (0, pad)
        # F.pad wants padding as (last_dim_before, last_dim_after, second_before, second_after,...)
        # so we must flatten pad_dims
        flat = []
        for before, after in reversed(pad_dims):
            flat.extend([before, after])
        bits = F.pad(bits, flat, value=0)

    # reshape to (..., num_bytes, 8)
    new_shape = list(bits.shape)
    L2 = new_shape[dim]
    nb = L2 // 8
    new_shape[dim : dim + 1] = [nb, 8]
    bits = bits.reshape(new_shape)

    # prepare weights [1,2,4,...,128] on same device/dtype
    w = 1 << torch.arange(8, device=bits.device, dtype=torch.uint8)
    # move w into shape broadcastable: (1,...,1,8)
    shape_w = [1] * bits.dim()
    shape_w[dim + 1] = 8
    w = w.view(shape_w)

    # multiply & sum along bit-axis
    packed = (bits * w).sum(dim=dim + 1, dtype=torch.uint8)
    return packed  # dtype uint8, shape with that dim replaced by nb


# manual unpackbits
def unpackbits_torch(packed: torch.Tensor, out_len: int, dim: int = -1) -> torch.Tensor:
    # packed: uint8 bytes, we want out_len bits back along dim
    # expand byte values to bits
    # shape: (..., num_bytes) → (..., num_bytes, 8)
    pb = packed.unsqueeze(dim + 1)  # new byte-axis is dim+1
    # create shifts 0..7
    shifts = torch.arange(8, device=packed.device)
    # torch.right_shift and bitwise_and
    bits8 = ((pb.to(torch.int32) >> shifts) & 1).to(torch.uint8)
    # now flatten the last two dims (num_bytes,8) back to bits
    new_shape = list(packed.shape)
    nb = new_shape[dim]
    new_shape[dim : dim + 2] = [nb * 8]
    bits = bits8.reshape(new_shape)
    # trim to out_len
    return torch.narrow(bits, dim, 0, out_len)


# batch conversion
def boards_to_packed(boards: list[chess.Board]) -> torch.Tensor:
    N = len(boards)
    bits = torch.zeros((N, 769), dtype=torch.uint8)
    for i, b in enumerate(boards):
        tb = board_to_tensor(b)
        bits[i] = tb
    # pack along dim=1 → shape (N, 97)
    return packbits_torch(bits, dim=1)


def packed_to_bits(packed: torch.Tensor) -> torch.Tensor:
    # unpack back to (N,769)
    return unpackbits_torch(packed, out_len=769, dim=1)


# Example
if __name__ == "__main__":
    import time

    # Timing: Board generation
    t0 = time.time()
    boards = generate_valid_5piece_boards(100000)
    t1 = time.time()
    print(f"Generated {len(boards)} boards in {t1-t0:.2f} sec")

    # get size of boards variable
    boards_size = sys.getsizeof(boards)
    print(f"Boards size: {boards_size / 1e6:.3f} MB")
    # save boards variable but not as torch
    with open("boards.pkl", "wb") as f:
        pickle.dump(boards, f)
    boards_size = os.path.getsize("boards.pkl")
    print(f"Boards size (pkl): {boards_size / 1e6:.3f} MB")

    # Timing: Baseline tensor creation
    t2 = time.time()
    baseline = torch.zeros((len(boards), 769), dtype=torch.uint8)
    for i, board in enumerate(boards):
        baseline[i] = board_to_tensor(board)
    t3 = time.time()
    print(f"Converted boards to baseline tensor in {t3-t2:.2f} sec")

    # Timing: Packing
    t4 = time.time()
    packed = boards_to_packed(boards)  # → (N,97), ~9.4 MB for N=100k
    t5 = time.time()
    print(f"Packed boards in {t5-t4:.2f} sec")

    # Timing: Unpacking
    t6 = time.time()
    recovered = packed_to_bits(packed)  # → (N,769)
    t7 = time.time()
    print(f"Unpacked boards in {t7-t6:.2f} sec")

    # Sanity check on the first board:
    assert torch.equal(
        recovered[0],
        board_to_tensor(boards[0]),
    )

    # Timing: Saving files
    t8 = time.time()
    torch.save(baseline, "baseline_boards.pt")
    torch.save(packed, "packed_boards.pt")
    torch.save(recovered, "recovered_boards.pt")
    t9 = time.time()
    print(f"Saved all files in {t9-t8:.2f} sec")

    # File sizes
    baseline_size = os.path.getsize("baseline_boards.pt")
    packed_size = os.path.getsize("packed_boards.pt")
    recovered_size = os.path.getsize("recovered_boards.pt")

    assert torch.equal(baseline, recovered)
    print(f"Baseline boards file size: {baseline_size / 1e6:.3f} MB")
    print(f"Packed boards file size: {packed_size / 1e6:.3f} MB")
    print(f"Recovered boards file size: {recovered_size / 1e6:.3f} MB")

    print("Compression ratio: {:.2f}".format(baseline_size / packed_size))
