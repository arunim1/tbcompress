#!/usr/bin/env python
"""
End-to-end demo:
  1. imports tb_stream (Rust wheel)
  2. builds dataset & model
  3. trains for a few thousand steps
"""
from dataset import SyzygyIterable
from model import SmallCNN
import time
import torch
from collections import defaultdict
import os

TB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Syzygy345_WDL"))
BATCH = 4096
STEPS = 1000

data = SyzygyIterable(TB_PATH, seed=42, batch_size=BATCH)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = SmallCNN().to(device)
opt = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = torch.nn.MSELoss()

data_iter = iter(data)
device_is_cuda = device == "cuda"

# accumulators for total time spent in each stage
times = defaultdict(float)

for step in range(STEPS):
    # 1) data-loading
    t0 = time.perf_counter()
    x, y = next(data_iter)
    if device_is_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    times["load"] += t1 - t0

    # 2) to device
    x, y = x.to(device), y.to(device).unsqueeze(1)
    if device_is_cuda:
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    times["to_device"] += t2 - t1

    # 3) forward
    pred = model(x)
    if device_is_cuda:
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    times["forward"] += t3 - t2

    # 4) backward
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    if device_is_cuda:
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    times["backward"] += t4 - t3

    # 5) optimizer step
    opt.step()
    if device_is_cuda:
        torch.cuda.synchronize()
    t5 = time.perf_counter()
    times["step"] += t5 - t4

    # print running averages every 100 steps
    if step and step % 100 == 0:
        avg = {k: (v / step) * 1e3 for k, v in times.items()}  # ms
        print(
            f"Step {step:4d}  avg times (ms):",
            f"load={avg['load']:.2f}",
            f"to_dev={avg['to_device']:.2f}",
            f"fwd={avg['forward']:.2f}",
            f"bwd={avg['backward']:.2f}",
            f"opt={avg['step']:.2f}",
            f"loss={loss.item():.3f}",
        )

print("✔ done")
