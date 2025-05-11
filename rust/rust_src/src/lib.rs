use crossbeam_channel::{bounded, Receiver};
use numpy::{PyArray1, PyArray4};
use pyo3::{
    iter::PyIterNextOutput,
    prelude::*,
    wrap_pyfunction,
};
use rand::{rngs::SmallRng, SeedableRng};
use shakmaty::{bitboard::Bitboard, Color, Role};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase};
use std::thread;

mod board_gen;

use shakmaty::Position;
// encode one position into 768 bytes
fn encode(pos: &shakmaty::Chess) -> [u8; 768] {
    let mut p = [0u8; 768];
    for sq in Bitboard::FULL {
        if let Some(pc) = pos.board().piece_at(sq) {
            let r = match pc.role {
                Role::Pawn   => 0,
                Role::Knight => 1,
                Role::Bishop => 2,
                Role::Rook   => 3,
                Role::Queen  => 4,
                Role::King   => 5,
            };
            let off = if pc.color == Color::Black { 6 } else { 0 };
            p[(r + off) * 64 + (sq as usize)] = 1;
        }
    }
    p
}

#[inline]
fn wdl_sign(w: AmbiguousWdl) -> i8 {
    w.signum() as i8
}

#[pyclass]
struct StreamIter {
    rx: Receiver<(Vec<u8>, Vec<i8>)>,
    batch_size: usize,
}

#[pymethods]
impl StreamIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<PyIterNextOutput> {
        match self.rx.recv() {
            Ok((bytes, wdls)) => {
                // planes: reshape 1-D slice into (batch,12,8,8)
                let planes: &PyArray4<u8> = PyArray1::<u8>::from_slice(py, &bytes)
                    .reshape([self.batch_size, 12, 8, 8])?;
                let planes: Py<PyAny> = planes.into_py(py);
                // labels: (batch,)
                let labels: &PyArray1<i8> = PyArray1::<i8>::from_slice(py, &wdls)
                    .reshape([self.batch_size])?;
                let labels: Py<PyAny> = labels.into_py(py);
                let tup = (planes, labels).into_py(py);
                Ok(PyIterNextOutput::Yield(tup))
            }
            Err(_) => Ok(PyIterNextOutput::Return(py.None())),
        }
    }
}

#[pyfunction]
fn make_tb_stream(path: &str, seed: u64, batch_size: usize) -> PyResult<StreamIter> {
    let (tx, rx) = bounded(1024);
    let dir = path.to_owned();

    thread::spawn(move || {
        let mut tb = unsafe { Tablebase::with_mmap_filesystem() };
        tb.add_directory(dir).unwrap();
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut plane_bytes = Vec::with_capacity(batch_size * 768);
        let mut wdls        = Vec::with_capacity(batch_size);

        loop {
            plane_bytes.clear();
            wdls.clear();
            // keep generating until we have exactly batch_size valid WDLs
            while wdls.len() < batch_size {
                let pos = board_gen::random_board(&mut rng);
                if let Ok(w) = tb.probe_wdl(&pos) {
                    plane_bytes.extend_from_slice(&encode(&pos));
                    wdls.push(wdl_sign(w));
                }
            }

            // one send per full batch
            let _ = tx.send((plane_bytes.clone(), wdls.clone()));
        }
    });

    Ok(StreamIter { rx, batch_size })
}

#[pymodule]
fn tb_stream(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(make_tb_stream, m)?)?;
    Ok(())
}
