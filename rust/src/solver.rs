use crate::io::{save_state_json, SaveCfg};
use std::time::Instant;
use anyhow::{Result, bail};

#[inline]
fn idx(n: usize, i: usize, j: usize) -> usize {
    i * n + j
}

pub fn residual_norm(n: usize, a: &[f64], x: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for i in 0..n {
        let mut ax = 0.0;
        for j in 0..n {
            ax += a[idx(n, i, j)] * x[j];
        }
        let r = ax - b[i];
        sum_sq += r * r;
    }
    sum_sq.sqrt()
}

pub fn back_substitution(n: usize, a: &[f64], b: &[f64], eps: f64) -> Result<Vec<f64>> {
    let mut x = vec![0.0f64; n];
    for ii in 0..n {
        let i = n - 1 - ii;
        let piv = a[idx(n, i, i)];
        if piv.abs() <= eps {
            bail!("Zero/near-zero pivot at row {} during back-substitution.", i);
        }
        let mut s = 0.0;
        for j in (i + 1)..n {
            s += a[idx(n, i, j)] * x[j];
        }
        x[i] = (b[i] - s) / piv;
    }
    Ok(x)
}


fn pivot_row(n: usize, a: &[f64], k: usize) -> usize {
    let mut best_i = k;
    let mut best_val = a[idx(n, k, k)].abs();
    for i in (k + 1)..n {
        let v = a[idx(n, i, k)].abs();
        if v > best_val {
            best_val = v;
            best_i = i;
        }
    }
    best_i
}

fn swap_rows(n: usize, a: &mut [f64], b: &mut [f64], r1: usize, r2: usize) {
    if r1 == r2 { return; }
    for j in 0..n {
        a.swap(idx(n, r1, j), idx(n, r2, j));
    }
    b.swap(r1, r2);
}

pub fn solve_seq(
    n: usize,
    a_in: &[f64],
    b_in: &[f64],
    out_dir: &str,
    save: SaveCfg,
    eps: f64,
) -> Result<(Vec<f64>, f64)> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();

    if save.save_states {
        save_state_json(out_dir, 0, n, &a, &b)?;
    }

    let t0 = Instant::now();

    for k in 0..(n - 1) {
        let pr = pivot_row(n, &a, k);
        let piv_test = a[idx(n, pr, k)];
        if piv_test.abs() <= eps {
            bail!("Singular/near-singular: zero pivot at col {}", k);
        }
        swap_rows(n, &mut a, &mut b, k, pr);

        let piv = a[idx(n, k, k)];

        for i in (k + 1)..n {
            let factor = a[idx(n, i, k)] / piv;
            if factor != 0.0 {
                for j in k..n {
                    a[idx(n, i, j)] -= factor * a[idx(n, k, j)];
                }
                b[i] -= factor * b[k];
                a[idx(n, i, k)] = 0.0;
            }
        }

        if save.save_states && ((k + 1) % save.save_every == 0) {
            save_state_json(out_dir, k + 1, n, &a, &b)?;
        }
    }

    let x = back_substitution(n, &a, &b, eps)?;
    Ok((x, t0.elapsed().as_secs_f64()))
}

pub fn solve_threads(
    n: usize,
    a_in: &[f64],
    b_in: &[f64],
    out_dir: &str,
    save: SaveCfg,
    eps: f64,
    nthreads: usize,
    tail_seq_mult: usize,
) -> Result<(Vec<f64>, f64)> {
    let mut a = a_in.to_vec();
    let mut b = b_in.to_vec();

    if save.save_states {
        save_state_json(out_dir, 0, n, &a, &b)?;
    }

    let t0 = Instant::now();
    let threads = nthreads.max(1);

    for k in 0..(n - 1) {
        let pr = pivot_row(n, &a, k);
        let piv_test = a[idx(n, pr, k)];
        if piv_test.abs() <= eps {
            bail!("Singular/near-singular: zero pivot at col {}", k);
        }
        swap_rows(n, &mut a, &mut b, k, pr);

        let piv = a[idx(n, k, k)];
        if piv.abs() <= eps {
            bail!("Singular/near-singular: zero pivot at col {}", k);
        }

        let pivot_row_vec: Vec<f64> = (0..n).map(|j| a[idx(n, k, j)]).collect();
        let pivot_b = b[k];

        let start_row = k + 1;
        let total = n - start_row;
        if total == 0 { break; }

        if total < tail_seq_mult * threads {
            for i in start_row..n {
                let factor = a[idx(n, i, k)] / piv;
                if factor != 0.0 {
                    for j in k..n {
                        a[idx(n, i, j)] -= factor * pivot_row_vec[j];
                    }
                    b[i] -= factor * pivot_b;
                    a[idx(n, i, k)] = 0.0;
                }
            }
        } else {
            let a_start = start_row * n;
            let (_a_head, mut a_tail) = a.split_at_mut(a_start);
            let (_b_head, mut b_tail) = b.split_at_mut(start_row);

            let rows_per = (total + threads - 1) / threads;

            std::thread::scope(|scope| {
                let mut a_rem: &mut [f64] = &mut a_tail;
                let mut b_rem: &mut [f64] = &mut b_tail;

                for t in 0..threads {
                    let row0 = t * rows_per;
                    let row1 = ((t + 1) * rows_per).min(total);
                    if row0 >= row1 { continue; }

                    let rows = row1 - row0;
                    let a_len = rows * n;
                    let b_len = rows;

                    let (a_chunk, a_rest) = a_rem.split_at_mut(a_len);
                    let (b_chunk, b_rest) = b_rem.split_at_mut(b_len);
                    a_rem = a_rest;
                    b_rem = b_rest;

                    let pivot_ref = &pivot_row_vec;

                    scope.spawn(move || {
                        for rr in 0..rows {
                            let base = rr * n;
                            let factor = a_chunk[base + k] / piv;
                            if factor != 0.0 {
                                for j in k..n {
                                    a_chunk[base + j] -= factor * pivot_ref[j];
                                }
                                b_chunk[rr] -= factor * pivot_b;
                                a_chunk[base + k] = 0.0;
                            }
                        }
                    });
                }
            });
        }

        if save.save_states && ((k + 1) % save.save_every == 0) {
            save_state_json(out_dir, k + 1, n, &a, &b)?;
        }
    }

    let x = back_substitution(n, &a, &b, eps)?;
    Ok((x, t0.elapsed().as_secs_f64()))
}