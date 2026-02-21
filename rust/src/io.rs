use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy)]
pub struct SaveCfg {
    pub save_states: bool,
    pub save_every: usize,
}

#[derive(Serialize)]
struct StateDump<'a> {
    iter: usize,
    n: usize,
    // Flattened row-major A
    a: &'a [f64],
    b: &'a [f64],
}

pub fn ensure_dir<P: AsRef<Path>>(p: P) -> std::io::Result<()> {
    fs::create_dir_all(p)
}

pub fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

pub fn load_system_npz(path: &str) -> anyhow::Result<(Array2<f64>, Array1<f64>)> {
    let f = File::open(path)?;
    let mut npz = NpzReader::new(f)?;

    // Python np.savez_compressed -> inside npz: "A.npy" and "b.npy"
    let a: Array2<f64> = npz.by_name("A.npy")?;
    let b: Array1<f64> = npz.by_name("b.npy")?;
    Ok((a, b))
}

pub fn save_state_json(out_dir: &str, iter: usize, n: usize, a_flat: &[f64], b: &[f64]) -> anyhow::Result<()> {
    ensure_dir(out_dir)?;
    let path = format!("{}/state_{:04}.json", out_dir, iter);
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let dump = StateDump {
        iter,
        n,
        a: a_flat,
        b,
    };
    serde_json::to_writer(&mut w, &dump)?;
    w.write_all(b"\n")?;
    Ok(())
}

pub fn save_meta_json(out_dir: &str, meta: &serde_json::Value) -> anyhow::Result<()> {
    ensure_dir(out_dir)?;
    let path = format!("{}/meta.json", out_dir);
    let mut f = File::create(path)?;
    f.write_all(serde_json::to_string_pretty(meta)?.as_bytes())?;
    f.write_all(b"\n")?;
    Ok(())
}