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
