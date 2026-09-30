//! Repeated-forward timing of the MLX backend on the fixture cases (ignored by default; a
//! missing fixture or checkpoint fails loudly when run with `--ignored`).
//! Run with: `cargo test -p laya-mlx --release --test bench -- --ignored --nocapture`.

use laya_core::testing::fixtures_path;
use laya_core::{parse_questions, Agent, BackendOptions};
use serde_json::Value;

fn bench(fixture: &str, subfolder: Option<&str>, opts: &BackendOptions) {
    let path = fixtures_path(fixture);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()));
    let fx: Value = serde_json::from_slice(&bytes).unwrap();
    let dir = laya_core::resolve::resolve_model_dir("convaiinnovations/laya", subfolder)
        .unwrap_or_else(|e| panic!("checkpoint convaiinnovations/laya {subfolder:?}: {e}"));
    let agent = Agent::load(&dir, opts, laya_mlx::factory()).unwrap();
    println!("{} [{}]", fixture, agent.backend_name());
    for case in fx["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let qs = parse_questions(&case["questions"]).unwrap();
        let items = agent.encode(&case["state"], &qs).unwrap();
        let batch = agent.collate(&items);
        for _ in 0..3 {
            agent.backend().forward(&batch).unwrap();
        }
        let iters = 20;
        let mut times: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = std::time::Instant::now();
            agent.backend().forward(&batch).unwrap();
            times.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean: f64 = times.iter().sum::<f64>() / iters as f64;
        println!(
            "  {:<12} n {} len {:<4} min {:6.1} ms  median {:6.1} ms  mean {:6.1} ms",
            name,
            batch.n,
            batch.len,
            times[0],
            times[iters / 2],
            mean
        );
    }
}

#[test]
#[ignore]
fn bench_laya() {
    bench("laya", None, &BackendOptions::default());
}

#[test]
#[ignore]
fn bench_multilingual() {
    bench(
        "multilingual",
        Some("multilingual"),
        &BackendOptions::default(),
    );
}

/// Raw f16 gemm throughput: `[m, k] @ [k, n]` with a transposed-view weight vs a contiguous one.
#[test]
#[ignore]
fn bench_gemm() {
    use mlx_rs::{ops, transforms, Array, Dtype, Stream};
    let s = Stream::gpu();
    mlx_rs::with_stream(&s, || {
        for &(m, k, n) in &[
            (2048i32, 1024i32, 3072i32),
            (384, 1024, 3072),
            (2048, 1024, 5248),
            (2048, 2624, 1024),
        ] {
            let x = mlx_rs::random::normal::<f32>(&[m, k], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let w = mlx_rs::random::normal::<f32>(&[n, k], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap();
            let wt_view = ops::transpose(&w).unwrap();
            let wt_contig = wt_view.contiguous().unwrap();
            transforms::eval([&x, &w, &wt_view, &wt_contig]).unwrap();
            for (label, wt) in [("view", &wt_view), ("contig", &wt_contig)] {
                for _ in 0..3 {
                    ops::matmul(&x, wt).unwrap().eval().unwrap();
                }
                let iters = 20;
                let t0 = std::time::Instant::now();
                let mut outs: Vec<Array> = Vec::new();
                for _ in 0..iters {
                    outs.push(ops::matmul(&x, wt).unwrap());
                }
                transforms::eval(outs.iter()).unwrap();
                let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
                let tflops = 2.0 * m as f64 * k as f64 * n as f64 / (ms * 1e-3) / 1e12;
                println!("  gemm {m}x{k}x{n} {label:<6} {ms:6.2} ms  {tflops:5.1} TFLOPS");
            }
        }
    });
}
