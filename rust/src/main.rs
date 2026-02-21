mod io;
mod solver;

use clap::Parser;
use io::{load_system_npz, now_ms, save_meta_json, SaveCfg};
use serde_json::json;
use solver::{residual_norm, solve_seq, solve_threads};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)]
    system: String,

    #[arg(long, default_value = "outputs/rust_run")]
    out: String,

    #[arg(long)]
    save_states: bool,

    #[arg(long, default_value_t = 1)]
    save_every: usize,

    #[arg(long, default_value_t = 0.0)]
    eps: f64,

    /// "seq" or "thr"
    #[arg(long, default_value = "seq")]
    mode: String,

    /// counts threads for mode=thr
    #[arg(long, default_value_t = 4)]
    threads: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let (a_nd, b_nd) = load_system_npz(&args.system)?;
    let n = a_nd.nrows();

    let a_flat: Vec<f64> = a_nd.iter().copied().collect();
    let b_vec: Vec<f64> = b_nd.iter().copied().collect();

    let save = SaveCfg {
        save_states: args.save_states,
        save_every: args.save_every,
    };

    let (x, elapsed, impl_name, nthreads) = if args.mode == "thr" {
        let (x, t) = solve_threads(
            n,
            &a_flat,
            &b_vec,
            &args.out,
            save,
            args.eps,
            args.threads,
            4,
        )?;
        (x, t, "rust_threads", args.threads)
    } else {
        let (x, t) = solve_seq(n, &a_flat, &b_vec, &args.out, save, args.eps)?;
        (x, t, "rust_seq", 1usize)
    };

    let res = residual_norm(n, &a_flat, &x, &b_vec);

    let meta = json!({
        "impl": impl_name,
        "n": n,
        "threads": nthreads,
        "save_states": args.save_states,
        "save_every": args.save_every,
        "elapsed_s": elapsed,
        "residual_norm": res,
        "timestamp_ms": now_ms()
    });

    save_meta_json(&args.out, &meta)?;

    println!(
        "[{}] n={} threads={} elapsed={:.6}s residual={:.3e} x[0..3]={:?}",
        impl_name,
        n,
        nthreads,
        elapsed,
        res,
        &x[0..x.len().min(4)]
    );

    Ok(())
}
