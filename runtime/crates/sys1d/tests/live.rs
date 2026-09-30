//! Live end-to-end test (ignored by default; needs the typed-decisions checkpoint in the HF
//! cache, `source bench/env.sh` first). Starts the real `sys1d` binary on port 0, posts every
//! request in `bench/workloads/smoke.jsonl`, checks the answers against an in-process
//! `Agent::predict` and against the fp32 CPU reference (key sets and order, probabilities and
//! `answer_confidence` within a tolerance, argmax agreement), then stops it with SIGINT.
//!
//! Run with: `cargo test -p sys1d --release --test live -- --ignored --nocapture`

mod common;

use common::{build_request, read_response, send};
use laya_core::{Agent, BackendOptions};
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use tokio::net::TcpStream;

/// Served (f16 on the GPU) against in-process with the same engine: the same numbers.
const SAME_ENGINE_TOL: f64 = 1e-3;
/// Served (f16 GPU) against the fp32 CPU reference. The smoke set measures 0.0008 at most;
/// this leaves room for kernel changes without hiding a real precision regression.
const REFERENCE_TOL: f64 = 0.01;

fn bench_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../bench")
        .canonicalize()
        .unwrap()
}

fn read_jsonl(path: &PathBuf) -> Vec<Value> {
    BufReader::new(std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(&l).unwrap())
        .collect()
}

fn keys(v: &Value) -> Vec<String> {
    v.as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

/// Argmax choice / rounded score / noul side of one answer, the categorical part the bench
/// harness compares.
fn categorical(answer: &Value) -> Value {
    match answer["type"].as_str() {
        Some("choice") => answer["choice"].clone(),
        Some("noul") => Value::from(answer["noul"].as_f64().unwrap_or(0.0) >= 0.5),
        Some("score") => Value::from(answer["score"].as_f64().unwrap_or(0.0).round()),
        _ => Value::Null,
    }
}

/// Largest absolute difference over `probabilities` (or `noul` for noul answers).
fn max_prob_diff(a: &Value, b: &Value) -> f64 {
    let (Some(pa), Some(pb)) = (
        a["probabilities"].as_object(),
        b["probabilities"].as_object(),
    ) else {
        return match (a["noul"].as_f64(), b["noul"].as_f64()) {
            (Some(x), Some(y)) => (x - y).abs(),
            _ => f64::NAN,
        };
    };
    pa.iter()
        .map(|(k, v)| {
            (v.as_f64().unwrap_or(0.0) - pb.get(k).and_then(Value::as_f64).unwrap_or(f64::NAN))
                .abs()
        })
        .fold(0.0, f64::max)
}

fn field_diff(a: &Value, b: &Value, key: &str) -> f64 {
    match (a[key].as_f64(), b[key].as_f64()) {
        (Some(x), Some(y)) => (x - y).abs(),
        _ => f64::NAN,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn smoke_workload_through_the_real_server() {
    let bench = bench_root();
    let workload = read_jsonl(&bench.join("workloads/smoke.jsonl"));
    let reference = read_jsonl(&bench.join("reference/typed-decisions/smoke.jsonl"));
    assert!(!workload.is_empty());

    // Start the binary on a free port and parse its ready line.
    let t0 = Instant::now();
    let mut child = Command::new(env!("CARGO_BIN_EXE_sys1d"))
        .args(["--port", "0", "--model", "typed-decisions"])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn sys1d");
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    stdout.read_line(&mut line).unwrap();
    let ready: Value =
        serde_json::from_str(line.trim()).unwrap_or_else(|e| panic!("ready line {line:?}: {e}"));
    eprintln!(
        "ready after {:.0} ms: {}",
        t0.elapsed().as_secs_f64() * 1000.0,
        line.trim()
    );
    assert_eq!(ready["event"], "listening");
    assert_eq!(ready["model"], "typed-decisions");
    assert!(ready["engine"].as_str().unwrap().starts_with("mlx("));
    assert!(ready["load_ms"].as_f64().unwrap() > 0.0);
    let addr = ready["addr"].as_str().unwrap().to_string();

    let mut conn = TcpStream::connect(&addr).await.unwrap();
    let h = {
        send(&mut conn, &build_request("GET", "/health", &[], None)).await;
        read_response(&mut conn).await
    };
    assert_eq!(h.status, 200);
    let hv = h.json();
    assert_eq!(hv["status"], "ok");
    assert_eq!(hv["loaded"], serde_json::json!(["typed-decisions"]));
    eprintln!("health: {hv}");

    // Post every smoke request over one kept-alive connection.
    let mut served: Vec<(String, Value)> = Vec::new();
    let mut latencies = Vec::new();
    for row in &workload {
        let body = row["body"].to_string();
        let req = build_request(
            "POST",
            "/v1/systemone",
            &[("Content-Type", "application/json")],
            Some(body.as_bytes()),
        );
        let t = Instant::now();
        send(&mut conn, &req).await;
        let r = read_response(&mut conn).await;
        latencies.push(t.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(
            r.status,
            200,
            "{}: {}",
            row["id"],
            String::from_utf8_lossy(&r.body)
        );
        let infer: f64 = r.header("x-inference-time-ms").unwrap().parse().unwrap();
        assert!(infer > 0.0);
        let v = r.json();
        assert_eq!(v["routing"]["model"], "typed-decisions");
        assert_eq!(v["model"], "laya-rl-agent");
        served.push((row["id"].as_str().unwrap().to_string(), v));
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "{} requests over http: p50 {:.1} ms, max {:.1} ms",
        latencies.len(),
        latencies[latencies.len() / 2],
        latencies[latencies.len() - 1]
    );

    // Stop the server with SIGINT and check for a clean exit before loading a second copy
    // of the weights in this process.
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    let deadline = Instant::now() + Duration::from_secs(15);
    let exit = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        assert!(
            Instant::now() < deadline,
            "sys1d did not exit within 15 s of SIGINT"
        );
        std::thread::sleep(Duration::from_millis(50));
    };
    assert_eq!(exit.code(), Some(0), "clean shutdown exit code");

    // In-process answers with the same engine settings must match the served ones.
    let dir = laya_core::resolve::resolve_model_dir("convaiinnovations/laya-typed-decisions", None)
        .unwrap();
    let opts = BackendOptions {
        tuning: Some(sys1d::config::DEFAULT_TUNING.into()),
        ..Default::default()
    };
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend)).unwrap();
    let mut exact = 0usize;
    let mut worst = 0.0f64;
    for (row, (id, over_http)) in workload.iter().zip(&served) {
        let local = agent
            .predict(&row["body"]["state"], &row["body"]["questions"])
            .unwrap();
        assert_eq!(local["usage"], over_http["usage"], "{id}");
        let (la, ha) = (
            local["answers"].as_object().unwrap(),
            over_http["answers"].as_object().unwrap(),
        );
        assert_eq!(la.len(), ha.len(), "{id}");
        if local["answers"] == over_http["answers"] {
            exact += 1;
        }
        for (qid, a) in la {
            let b = &ha[qid];
            assert_eq!(
                categorical(a),
                categorical(b),
                "{id}/{qid}: served {b} vs in-process {a}"
            );
            let d = max_prob_diff(a, b);
            assert!(d <= SAME_ENGINE_TOL, "{id}/{qid}: probability diff {d}");
            worst = worst.max(d);
        }
    }
    eprintln!(
        "in-process agreement: {}/{} requests byte-identical answers, max probability diff {worst:.2e}",
        exact,
        served.len()
    );

    // Against the fp32 CPU reference: every answer has the reference's key set in the
    // reference's order, probabilities and `answer_confidence` are within REFERENCE_TOL, and
    // argmax agreement on choice questions is at least 99%.
    let mut agree = 0usize;
    let mut total = 0usize;
    let mut answers_checked = 0usize;
    let mut worst_prob = 0.0f64;
    let mut worst_conf = 0.0f64;
    let mut disagreements = Vec::new();
    for (id, over_http) in &served {
        let reference = reference
            .iter()
            .find(|r| r["id"] == *id)
            .unwrap_or_else(|| panic!("{id} is not in the reference file"));
        let (ha, ra) = (
            over_http["answers"].as_object().unwrap(),
            reference["answers"].as_object().unwrap(),
        );
        assert_eq!(
            keys(&over_http["answers"]),
            keys(&reference["answers"]),
            "{id}: question ids"
        );
        for (qid, ans) in ha {
            let want = &ra[qid];
            assert_eq!(
                keys(ans),
                keys(want),
                "{id}/{qid}: answer keys or their order differ"
            );
            assert_eq!(ans["type"], want["type"], "{id}/{qid}");
            if ans["type"] == "score" {
                assert_eq!(ans["legend"], want["legend"], "{id}/{qid}");
            }
            if ans.get("probabilities").is_some() {
                assert_eq!(
                    keys(&ans["probabilities"]),
                    keys(&want["probabilities"]),
                    "{id}/{qid}: probability keys"
                );
            }
            let dp = max_prob_diff(ans, want);
            assert!(
                dp <= REFERENCE_TOL,
                "{id}/{qid}: probability diff {dp} vs reference"
            );
            let dc = field_diff(ans, want, "answer_confidence");
            assert!(
                dc <= REFERENCE_TOL,
                "{id}/{qid}: answer_confidence {} vs reference {}",
                ans["answer_confidence"],
                want["answer_confidence"]
            );
            worst_prob = worst_prob.max(dp);
            worst_conf = worst_conf.max(dc);
            answers_checked += 1;
            if ans["type"] == "choice" {
                total += 1;
                if ans["choice"] == want["choice"] {
                    agree += 1;
                } else {
                    disagreements.push(format!(
                        "{id}/{qid}: served {} reference {}",
                        ans["choice"], want["choice"]
                    ));
                }
            }
        }
    }
    assert!(
        total > 0,
        "reference has no choice questions for the smoke ids"
    );
    let pct = 100.0 * agree as f64 / total as f64;
    eprintln!(
        "reference agreement: {answers_checked} answers with matching key order; max probability diff {worst_prob:.4}, max answer_confidence diff {worst_conf:.4}; {agree}/{total} choice answers ({pct:.1}%) {disagreements:?}"
    );
    assert!(
        pct >= 99.0,
        "argmax agreement {pct:.1}% < 99%: {disagreements:?}"
    );
}
