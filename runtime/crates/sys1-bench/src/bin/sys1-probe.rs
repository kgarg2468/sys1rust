//! Steady-state latency per request shape, for comparing backend settings quickly.
//!
//! sys1-probe --workload FILE [--model M] [--variant V] [--warmup N] [--iters N] [--order grouped|mixed]
//!
//! Takes the first row of each shape (`shape.state_tokens`, `shape.n_questions`). `grouped` runs
//! each shape `iters` times in a row after `warmup` runs; `mixed` cycles through the shapes
//! `iters` times, so every request follows a different shape. Prints a header with the model,
//! the settings and the precision the timings were taken under, then min, p50 and max ms.
//! Backend settings come from `SYS1_MLX` (see laya-mlx `Knobs`) and are checked at startup;
//! `--variant` only picks the precision (`mlx-fp16`, `mlx-fp32`), any other name is an error.

use anyhow::{Context, Result};
use laya_core::{Agent, BackendOptions};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    let mut workload = String::new();
    let mut model = "typed-decisions".to_string();
    let mut f32 = false;
    let mut warmup = 3usize;
    let mut iters = 10usize;
    let mut mixed = false;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--workload" => workload = val()?,
            "--model" => model = val()?,
            "--variant" => f32 = parse_variant(&val()?)?,
            "--warmup" => warmup = val()?.parse()?,
            "--iters" => iters = val()?.parse()?,
            "--order" => mixed = parse_order(&val()?)?,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    if iters == 0 {
        anyhow::bail!("--iters must be at least 1 (there is no p50 of no runs)");
    }
    // The settings the run is reported under; a bad one is an error, not a silent default.
    let spec = std::env::var("SYS1_MLX").unwrap_or_default();
    laya_mlx::check_settings(&spec).context("SYS1_MLX")?;
    if std::env::var_os("SYS1_MICRO").is_some() {
        return micro();
    }
    let bench = std::env::var("BENCH_ROOT").context("source bench/env.sh")?;
    let (dir, sha) = sys1_bench::pinned_model_dir(Path::new(&bench), &model)?;

    let mut picked: Vec<(String, Value)> = Vec::new();
    let file = std::fs::File::open(&workload).with_context(|| format!("open workload {workload}"))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        let shape = format!("s{}_q{}", row["shape"]["state_tokens"], row["shape"]["n_questions"]);
        if !picked.iter().any(|(s, _)| *s == shape) {
            picked.push((shape, row));
        }
    }
    if picked.is_empty() {
        anyhow::bail!("{workload}: no requests, nothing to time");
    }

    let opts = BackendOptions { f32, tuning: Some(spec.clone()), ..Default::default() };
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
    print!("{}", header(&model, &sha, &spec, &agent.backend_name()));
    let run = |row: &Value| -> Result<(f64, u64)> {
        let t0 = Instant::now();
        let (out, _) = agent.predict_batch_timed(std::slice::from_ref(&row["body"]["state"]), &row["body"]["questions"], None)?;
        Ok((t0.elapsed().as_secs_f64() * 1000.0, out[0]["usage"]["input_tokens"].as_u64().unwrap_or(0)))
    };

    let mut times: Vec<Vec<f64>> = vec![Vec::new(); picked.len()];
    let mut tokens = vec![0u64; picked.len()];
    if mixed {
        for (_, row) in &picked {
            for _ in 0..warmup {
                run(row)?;
            }
        }
        for _ in 0..iters {
            for (i, (_, row)) in picked.iter().enumerate() {
                let (ms, tok) = run(row)?;
                times[i].push(ms);
                tokens[i] = tok;
            }
        }
    } else {
        for (i, (_, row)) in picked.iter().enumerate() {
            for _ in 0..warmup {
                run(row)?;
            }
            for _ in 0..iters {
                let (ms, tok) = run(row)?;
                times[i].push(ms);
                tokens[i] = tok;
            }
        }
    }
    println!("shape\ttokens\tmin\tp50\tmax");
    let mut logsum = 0.0;
    for (i, (shape, _)) in picked.iter().enumerate() {
        let mut t = times[i].clone();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = t[t.len() / 2];
        logsum += p50.ln();
        println!("{shape}\t{}\t{:.1}\t{:.1}\t{:.1}", tokens[i], t[0], p50, t[t.len() - 1]);
    }
    println!("geo_p50\t{:.1}", (logsum / picked.len() as f64).exp());
    let mb = |r: mlx_rs::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    println!(
        "mlx_mb\tactive {}\tcache {}\tpeak {}",
        mb(mlx_rs::memory::active_memory()),
        mb(mlx_rs::memory::cache_memory()),
        mb(mlx_rs::memory::peak_memory())
    );
    Ok(())
}

/// The lines above the timings that say what they were taken under: the model and its pinned
/// sha, the settings spec as applied (`(none)` for the upstream defaults) and the engine name,
/// which carries the device and the precision (`mlx(gpu,f16)`). Saved runs with different
/// tuning are told apart by this header alone.
fn header(model: &str, sha: &str, spec: &str, engine: &str) -> String {
    let spec = if spec.is_empty() { "(none)" } else { spec };
    format!("model\t{model}\t{sha}\nsettings\t{spec}\nengine\t{engine}\n")
}

/// `--variant`: the probe knows the two precisions of the sys1-bench variants, `mlx-fp16` (the
/// default) and `mlx-fp32`. Backend settings come from `SYS1_MLX`, so a tuned variant name such
/// as `mlx-fp16-fast` is an error here rather than a run under other settings than its label.
fn parse_variant(name: &str) -> Result<bool> {
    match name {
        "mlx-fp16" => Ok(false),
        "mlx-fp32" => Ok(true),
        other => anyhow::bail!(
            "unknown variant {other}: the probe takes mlx-fp16 or mlx-fp32 and reads backend settings from SYS1_MLX"
        ),
    }
}

/// `--order`: `mixed` cycles through the shapes, `grouped` runs each shape's iterations in a row.
fn parse_order(order: &str) -> Result<bool> {
    match order {
        "mixed" => Ok(true),
        "grouped" => Ok(false),
        other => anyhow::bail!("unknown order {other}: use mixed or grouped"),
    }
}

/// Raw MLX timings to compare with the same ops from Python (`SYS1_MICRO=1`).
fn micro() -> Result<()> {
    use mlx_rs::{ops, random, transforms, Array, Dtype};
    let f16 = |shape: &[i32]| -> Result<Array> {
        Ok(random::normal::<f32>(shape, None, None, None)?.as_dtype(Dtype::Float16)?)
    };
    let time = |name: &str, n: usize, f: &dyn Fn() -> Result<Array>| -> Result<()> {
        for _ in 0..3 {
            f()?.eval()?;
        }
        let mut ts = Vec::new();
        for _ in 0..n {
            let t0 = Instant::now();
            f()?.eval()?;
            ts.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{name}\tmin {:.3}\tp50 {:.3} ms", ts[0], ts[n / 2]);
        Ok(())
    };
    let a = f16(&[2048, 2048])?;
    let b = f16(&[2048, 2048])?;
    transforms::eval([&a, &b])?;
    time("gemm 2048^3", 20, &|| Ok(ops::matmul(&a, &b)?))?;
    let x = f16(&[1, 89, 768])?;
    let w = f16(&[2304, 768])?;
    let wt = ops::transpose(&w)?;
    // Every input of the timed closures is generated and evaluated here, so the timings hold
    // the chained ops alone.
    let w2 = ops::transpose(&f16(&[768, 768])?)?;
    transforms::eval([&x, &w, &wt, &w2])?;
    time("x[1,89,768] @ W^T[768,2304]", 50, &|| Ok(ops::matmul(&x, &wt)?))?;
    time("22 x chained small gemm+add", 30, &|| {
        let mut h = x.clone();
        for _ in 0..22 {
            h = ops::add(&ops::matmul(&h, &w2)?, &h)?;
        }
        Ok(h)
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The header names the model, the settings and the precision; an empty spec is `(none)`.
    #[test]
    fn header_says_what_the_timings_were_taken_under() {
        let h = header("typed-decisions", "1a793eb", "f16gelu,cache=512", "mlx(gpu,f16)");
        assert_eq!(h, "model\ttyped-decisions\t1a793eb\nsettings\tf16gelu,cache=512\nengine\tmlx(gpu,f16)\n");
        let h = header("multilingual", "e4e9ddf", "", "mlx(gpu,f32)");
        assert!(h.contains("settings\t(none)\n") && h.contains("engine\tmlx(gpu,f32)\n"), "{h}");
    }

    #[test]
    fn order_is_mixed_or_grouped() {
        assert!(parse_order("mixed").unwrap());
        assert!(!parse_order("grouped").unwrap());
        for bad in ["Mixed", "mixd", "random", ""] {
            assert!(parse_order(bad).is_err(), "{bad:?} was accepted");
        }
    }

    #[test]
    fn variant_is_one_of_the_two_precisions() {
        assert!(!parse_variant("mlx-fp16").unwrap());
        assert!(parse_variant("mlx-fp32").unwrap());
        for bad in ["mlx-fp16-fast", "mlx-env", "fp32", ""] {
            assert!(parse_variant(bad).is_err(), "{bad:?} was accepted");
        }
    }
}
