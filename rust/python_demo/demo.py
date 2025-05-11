#!/usr/bin/env python
"""
End-to-end demo:
  1. imports tb_stream (Rust wheel)
  2. builds dataset & model
  3. trains for a few thousand steps
"""
import time, itertools
import torch
from torch.utils.data import DataLoader
from dataset import SyzygyIterable
from model import SmallCNN

import os

TB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../Syzygy345_WDL"))
BATCH = 4096  # Increase batch size
STEPS = 1000

data = SyzygyIterable(TB_PATH)
loader = DataLoader(data, batch_size=BATCH, pin_memory=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
model = SmallCNN().to(device)
opt = torch.optim.Adam(model.parameters(), 1e-3)
loss_fn = torch.nn.MSELoss()

start = time.time()
for step, (x, y) in enumerate(itertools.islice(loader, STEPS)):
    x, y = x.to(device), y.to(device).unsqueeze(1)
    pred = model(x)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        elapsed = time.time() - start
        rate = BATCH * (step + 1) / elapsed / 1e3
        print(f"step {step:5d}  loss={loss.item():5.3f}  {rate:7.1f} kpos/s")
print("✔ done")
