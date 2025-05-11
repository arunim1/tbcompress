### rust/python_demo/__init__.py

```
from .dataset import SyzygyIterable
from .model import SmallCNN

```


### rust/python_demo/model.py

```
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(12, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.out(x))  # outputs ∈ (-1,1)


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(769, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

```


### rust/python_demo/dataset.py

```
import numpy as np
import torch
from torch.utils.data import IterableDataset
import tb_stream  # the wheel built by maturin


class SyzygyIterable(IterableDataset):
    def __init__(self, tb_path: str, seed: int = 42):
        super().__init__()
        self._it = tb_stream.make_tb_stream(tb_path, seed)

    def __iter__(self):
        for planes, wdl in self._it:
            # np.uint8 → float32 torch tensor in [0,1]
            yield (
                torch.from_numpy(np.asarray(planes, dtype=np.float32)),
                torch.tensor(wdl, dtype=torch.float32),
            )

```


### rust/python_demo/test_tb_stream.py

```
import tb_stream, numpy as np


def test_shape_and_wdl():
    it = tb_stream.make_tb_stream("../Syzygy345_WDL", seed=1234)
    for _ in range(1000):
        planes, wdl = next(it)
        assert planes.shape == (12, 8, 8)
        assert planes.dtype == np.uint8
        assert wdl in (-1, 0, 1)

```


### rust/python_demo/demo.py

```
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
    if step == 0:
        print(x.shape, y.shape)
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

```


### rust/rust_src/Cargo.toml

```
[package]
name = "tb_stream"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"

[lib]
name = "tb_stream"
crate-type = ["cdylib"]

[dependencies]
crossbeam-channel = "0.5"
serde = { version = "1", features = ["derive"] }
rand = { version = "0.8", features = ["small_rng"] }   # gives SmallRng
shakmaty = "0.27"                                      # modern API
shakmaty-syzygy = { version = "0.25.3", features = ["mmap"] }


# PyO3 bindings
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"

[features]
# no crate-local features required
default = []

[profile.release]
opt-level = 3
lto = true
codegen-units = 1

```


### rust/rust_src/target/release/build/target-lexicon-802e8d47b55a463d/out/host.rs

```

#[allow(unused_imports)]
use crate::Aarch64Architecture::*;
#[allow(unused_imports)]
use crate::ArmArchitecture::*;
#[allow(unused_imports)]
use crate::CustomVendor;
#[allow(unused_imports)]
use crate::Mips32Architecture::*;
#[allow(unused_imports)]
use crate::Mips64Architecture::*;
#[allow(unused_imports)]
use crate::Riscv32Architecture::*;
#[allow(unused_imports)]
use crate::Riscv64Architecture::*;
#[allow(unused_imports)]
use crate::X86_32Architecture::*;

/// The `Triple` of the current host.
pub const HOST: Triple = Triple {
    architecture: Architecture::Aarch64(Aarch64),
    vendor: Vendor::Apple,
    operating_system: OperatingSystem::Darwin,
    environment: Environment::Unknown,
    binary_format: BinaryFormat::Macho,
};

impl Architecture {
    /// Return the architecture for the current host.
    pub const fn host() -> Self {
        Architecture::Aarch64(Aarch64)
    }
}

impl Vendor {
    /// Return the vendor for the current host.
    pub const fn host() -> Self {
        Vendor::Apple
    }
}

impl OperatingSystem {
    /// Return the operating system for the current host.
    pub const fn host() -> Self {
        OperatingSystem::Darwin
    }
}

impl Environment {
    /// Return the environment for the current host.
    pub const fn host() -> Self {
        Environment::Unknown
    }
}

impl BinaryFormat {
    /// Return the binary format for the current host.
    pub const fn host() -> Self {
        BinaryFormat::Macho
    }
}

impl Triple {
    /// Return the triple for the current host.
    pub const fn host() -> Self {
        Self {
            architecture: Architecture::Aarch64(Aarch64),
            vendor: Vendor::Apple,
            operating_system: OperatingSystem::Darwin,
            environment: Environment::Unknown,
            binary_format: BinaryFormat::Macho,
        }
    }
}

```


### rust/rust_src/src/lib.rs

```
use crossbeam_channel::{bounded, Receiver};
use numpy::PyArray1;
use pyo3::{
    iter::PyIterNextOutput,   // ← HERE  (was callback::…)
    prelude::*,
    wrap_pyfunction,
};
use rand::{rngs::SmallRng, SeedableRng};
use shakmaty::{bitboard::Bitboard, Color, Position, Role};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase};
use std::thread;

mod board_gen;

// —— helpers ———————————————————————————————————————————————————————

fn encode(pos: &shakmaty::Chess) -> [u8; 768] {
    let mut p = [0u8; 768];
    for sq in Bitboard::FULL {
        if let Some(pc) = pos.board().piece_at(sq) {
            let r = match pc.role {
                Role::Pawn => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook => 3,
                Role::Queen => 4,
                Role::King => 5,
            };
            let off = if pc.color == Color::Black { 6 } else { 0 };
            p[(r + off) * 64 + (sq as usize)] = 1;
        }
    }
    p
}

#[inline] fn wdl_sign(w: AmbiguousWdl) -> i8 { w.signum() as i8 }

// —— Python iterator ———————————————————————————————————————————

#[pyclass]
struct StreamIter { rx: Receiver<(Vec<u8>, i8)> }

#[pymethods]
impl StreamIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&self, py: Python<'_>)
        -> PyResult<PyIterNextOutput>
    {
        match self.rx.recv() {
            Ok((bytes, wdl)) => {
                let arr: Py<PyAny> = PyArray1::<u8>::from_slice(py, &bytes)
                            .reshape([12, 8, 8])?
                            .into_py(py);
                let tup = (arr, wdl).into_py(py);
                Ok(PyIterNextOutput::Yield(tup))
            }
            Err(_) => Ok(PyIterNextOutput::Return(py.None())),
        }
    }
}

// —— factory ————————————————————————————————————————————————

#[pyfunction]
fn make_tb_stream(path: &str, seed: u64) -> PyResult<StreamIter> {
    let (tx, rx) = bounded(1024);
    let dir = path.to_owned();

    thread::spawn(move || {
        let mut tb = unsafe { Tablebase::with_mmap_filesystem() };
        tb.add_directory(dir).unwrap();

        let mut rng = SmallRng::seed_from_u64(seed);

        Python::with_gil(|py| {
            py.allow_threads(|| loop {
                let pos = board_gen::random_board(&mut rng);
                if let Ok(w) = tb.probe_wdl(&pos) {
                    let _ = tx.send((encode(&pos).to_vec(), wdl_sign(w)));
                }
            })
        });
    });

    Ok(StreamIter { rx })
}

// —— module init ————————————————————————————————————————————

#[pymodule]
fn tb_stream(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_tb_stream, m)?)?;
    Ok(())
}

```


### rust/rust_src/src/board_gen.rs

```
//! ≤ 5-piece random legal position generator (shakmaty 0.27).

use rand::{
    prelude::{SliceRandom, SmallRng},
    Rng,
};
use shakmaty::{
    bitboard::Bitboard, board::Board, CastlingMode, Color, FromSetup, Role, Setup,
};
use std::num::NonZeroU32;

pub fn random_board(rng: &mut SmallRng) -> shakmaty::Chess {
    loop {
        // —— place pieces ————————————————————————————————————————————
        let mut board = Board::empty();
        let mut sqs: Vec<_> = Bitboard::FULL.into_iter().collect();
        sqs.shuffle(rng);

        board.set_piece_at(sqs.pop().unwrap(), Role::King.of(Color::White));
        board.set_piece_at(sqs.pop().unwrap(), Role::King.of(Color::Black));

        const ROLES: &[Role] =
            &[Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen];
        for _ in 0..rng.gen_range(0..=3) {
            let role = *ROLES.choose(rng).unwrap();
            let col  = if rng.gen_bool(0.5) { Color::White } else { Color::Black };
            board.set_piece_at(sqs.pop().unwrap(), role.of(col));
        }

        // —— build Setup ————————————————————————————————————————————
        let stm = if rng.gen_bool(0.5) { Color::White } else { Color::Black };

        let setup = Setup {
            board,
            turn: stm,
            castling_rights: Bitboard::EMPTY,
            ep_square: None,
            fullmoves: NonZeroU32::new(1).unwrap(),
            halfmoves: 0,
            promoted: Bitboard::EMPTY,
            pockets: None,
            remaining_checks: None,
        };

        if let Ok(pos) = shakmaty::Chess::from_setup(setup, CastlingMode::Standard) {
            return pos;
        }
    }
}

```


