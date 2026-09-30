//! Bench adapter for the sys1rust runtime (bench/PLAN.md, "Adapter interface").
//!
//! run --list-variants
//! run --variant V --model M --workload FILE --out FILE [--warmup N] [--repeats R] [--duration S]
//!
//! In-process only. `latency_ms` covers `Agent::predict_batch_timed` for one request: sequence
//! building, tokenization, the forward pass with its GPU sync, and answer decoding. Each result
//! also records the time spent in each of those phases (`phase_us`).

use anyhow::{bail, Context, Result};
use laya_core::{Agent, BackendOptions};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const CONTENDER: &str = "sys1rust";
const MODELS: [&str; 2] = ["typed-decisions", "multilingual"];

struct Variant {
    name: &'static str,
    notes: &'static str,
    opts: fn() -> BackendOptions,
}

fn tuned(tuning: &str) -> BackendOptions {
    BackendOptions { tuning: Some(tuning.into()), ..Default::default() }
}

const VARIANTS: [Variant; 5] = [
    Variant {
        name: "mlx-fp16",
        notes: "laya-r-mlx 914c9a7 as forked, unchanged: fp16 weights, questions in one batch padded to the \
                longest row. Its GELU promotes activations to fp32 from the first MLP on.",
        opts: || tuned(""),
    },
    Variant {
        name: "mlx-fp16-fix",
        notes: "mlx-fp16 with GELU kept in fp16, so activations and gemms stay fp16.",
        opts: || tuned("f16gelu"),
    },
    Variant {
        name: "mlx-fp16-fast",
        notes: "mlx-fp16-fix with MLX's buffer cache capped at 512 MiB (the default lets freed buffers \
                grow to about the size of RAM when request lengths vary) and a 2 GiB wired limit so \
                the weights stay resident.",
        opts: || tuned("f16gelu,cache=512,wired=2048"),
    },
    Variant {
        name: "mlx-env",
        notes: "Exploration only: backend settings from the SYS1_MLX environment variable.",
        opts: || tuned(&std::env::var("SYS1_MLX").unwrap_or_default()),
    },
    Variant {
        name: "mlx-fp32",
        notes: "Same as mlx-fp16 with the transformer in fp32.",
        opts: || BackendOptions { f32: true, ..tuned("") },
    },
];

/// MLX allocator state after a request: active, buffer cache and peak, in MiB. Metal buffers
/// do not show up in RSS, so this is the only view of the GPU-side memory.
fn mlx_mb() -> Value {
    let mb = |r: mlx_rs::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    json!({
        "active": mb(mlx_rs::memory::active_memory()),
        "cache": mb(mlx_rs::memory::cache_memory()),
        "peak": mb(mlx_rs::memory::peak_memory()),
    })
}

fn now_unix() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[derive(Default)]
struct Args {
    list: bool,
    variant: String,
    model: String,
    workload: String,
    out: String,
    warmup: usize,
    repeats: usize,
    duration: Option<f64>,
    concurrency: usize,
}

fn parse_args() -> Result<Args> {
    let mut a = Args { warmup: 5, repeats: 1, concurrency: 1, ..Default::default() };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--list-variants" => a.list = true,
            "--variant" => a.variant = val()?,
            "--model" => a.model = val()?,
            "--workload" => a.workload = val()?,
            "--out" => a.out = val()?,
            "--warmup" => a.warmup = val()?.parse()?,
            "--repeats" => a.repeats = val()?.parse()?,
            "--duration" => a.duration = Some(val()?.parse()?),
            "--concurrency" => a.concurrency = val()?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(a)
}

/// `$HF_HOME/hub/models--org--name/snapshots/<sha>` for the model's pin in `models.lock.json`.
fn model_dir(bench: &str, model: &str) -> Result<(PathBuf, String)> {
    let lock: Value = serde_json::from_str(&std::fs::read_to_string(format!("{bench}/models.lock.json"))?)?;
    let pin = lock.get(model).with_context(|| format!("{model} is not in models.lock.json"))?;
    let repo = pin["repo"].as_str().context("repo")?;
    let sha = pin["sha"].as_str().context("sha")?;
    let hf = std::env::var("HF_HOME").context("HF_HOME is not set (source bench/env.sh)")?;
    let dir = PathBuf::from(hf)
        .join("hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(sha);
    if !dir.exists() {
        bail!("{} is not downloaded", dir.display());
    }
    Ok((dir, sha.to_string()))
}

fn main() -> Result<()> {
    let t_process_start = now_unix();
    let args = parse_args()?;
    if args.list {
        let v: Vec<Value> = VARIANTS
            .iter()
            .map(|v| json!({"variant": v.name, "models": MODELS, "mode": "inproc", "max_state_tokens": null, "notes": v.notes}))
            .collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    if args.concurrency > 1 {
        bail!("--concurrency is for http mode; this adapter is in-process");
    }
    let variant = VARIANTS
        .iter()
        .find(|v| v.name == args.variant)
        .with_context(|| format!("unknown variant {}", args.variant))?;
    let bench = std::env::var("BENCH_ROOT").context("BENCH_ROOT is not set (source bench/env.sh)")?;
    let (dir, sha) = model_dir(&bench, &args.model)?;

    let rows: Vec<Value> = BufReader::new(std::fs::File::open(&args.workload)?)
        .lines()
        .filter_map(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l))
        .collect::<std::result::Result<_, _>>()?;

    let t_load = Instant::now();
    let opts = (variant.opts)();
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let mut out = std::io::BufWriter::new(std::fs::File::create(&args.out)?);
    let meta = json!({
        "type": "meta", "contender": CONTENDER, "variant": variant.name, "model": args.model,
        "model_sha": sha, "code_version": std::env::var("SYS1_CODE_VERSION").unwrap_or_else(|_| "unknown".into()),
        "backend": "mlx", "mode": "inproc", "load_ms": (load_ms * 10.0).round() / 10.0,
        "pid": std::process::id(), "t_process_start": t_process_start,
        "engine": agent.backend_name(), "tuning": opts.tuning, "warmup": args.warmup, "repeats": args.repeats, "duration": args.duration,
    });
    writeln!(out, "{meta}")?;

    let run_one = |row: &Value| -> (f64, f64, Result<(Value, laya_core::agent::Timing)>) {
        let body = &row["body"];
        let t_start = now_unix();
        let t0 = Instant::now();
        let r = agent
            .predict_batch_timed(std::slice::from_ref(&body["state"]), &body["questions"], None)
            .map(|(mut v, t)| (v.remove(0), t))
            .map_err(anyhow::Error::from);
        (t_start, t0.elapsed().as_secs_f64() * 1000.0, r)
    };

    for row in rows.iter().take(args.warmup) {
        if let (_, _, Err(e)) = run_one(row) {
            eprintln!("warmup {}: {e:#}", row["id"]);
        }
    }

    let t_measure = Instant::now();
    let mut repeat = 0usize;
    'outer: loop {
        for row in &rows {
            if let Some(d) = args.duration {
                if t_measure.elapsed().as_secs_f64() >= d {
                    break 'outer;
                }
            }
            let (t_start, latency_ms, r) = run_one(row);
            let line = match r {
                Ok((res, t)) => json!({
                    "type": "result", "id": row["id"], "repeat": repeat, "t_start": t_start,
                    "latency_ms": (latency_ms * 1000.0).round() / 1000.0,
                    "answers": res["answers"],
                    "input_tokens": res["usage"]["input_tokens"],
                    "phase_us": {"encode": t.encode_us, "forward": t.forward_us, "decode": t.decode_us},
                    "batch": {"rows": t.batch_rows, "len": t.batch_len},
                    "mlx_mb": mlx_mb(),
                }),
                Err(e) => json!({"type": "result", "id": row["id"], "repeat": repeat, "error": format!("{e:#}")}),
            };
            writeln!(out, "{line}")?;
            out.flush()?;
        }
        repeat += 1;
        if args.duration.is_none() && repeat >= args.repeats {
            break;
        }
    }
    out.flush()?;
    Ok(())
}
