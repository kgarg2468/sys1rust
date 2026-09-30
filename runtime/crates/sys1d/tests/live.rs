//! Live end-to-end test (ignored by default; needs the typed-decisions checkpoint at the
//! revision pinned in `bench/models.lock.json` in the HF cache, `source bench/env.sh` first).
//! Starts the real `sys1d` binary on port 0 at that revision with every server setting
//! pinned (the caller's environment cannot change them), posts every request in
//! `bench/workloads/smoke.jsonl`, requires each answer to equal an in-process
//! `Agent::predict` of the same revision and engine settings byte for byte, checks it against
//! the fp32 CPU reference (key sets and order, every numeric field within a tolerance,
//! categorical agreement), then stops it with SIGINT. The child is killed if the test fails
//! or hangs at any point.
//!
//! Run from the repository root with:
//! `cargo test --manifest-path runtime/Cargo.toml -p sys1d --release --test live -- --ignored --nocapture`

mod common;

use common::{build_request, read_response, send, Resp};
use laya_core::{Agent, BackendOptions};
use serde_json::Value;
use std::collections::BTreeSet;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use sys1d::config::{DEFAULT_MAX_CONCURRENT, DEFAULT_TUNING};
use tokio::net::TcpStream;

/// Upstream's name for the served checkpoint, and its key in `bench/models.lock.json`.
const MODEL: &str = "typed-decisions";
/// Rows in `bench/workloads/smoke.jsonl`, and answer rows in its reference file. Asserted, so
/// a smoke set that loses rows cannot pass by checking fewer requests.
const SMOKE_REQUESTS: usize = 24;

/// Served (f16 GPU) against the fp32 CPU reference, for `probabilities`, `noul`,
/// `confidence`, `answer_confidence` and `action.act_probability`. Measured over the 120
/// smoke answers: 0.0008, 0.0008, 0.0009 and 0.0000 (`results/SERVER.md`). 2.5x the largest
/// catches a regression of a few thousandths while leaving room for kernel-level noise.
const REFERENCE_TOL: f64 = 0.002;
/// `score` is the probability-weighted level, so its spread grows with the level count
/// (up to 32); the smoke set's score questions have 4 levels and measure 0.0013 at most.
const SCORE_TOL: f64 = 0.005;
/// Model load is about 3 s, warm-up under 1 s; a ready line later than this means a hang.
const READY_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(15);
/// One request round trip (send plus the full response) over the kept-alive connection. A
/// smoke request takes milliseconds; a stall past this fails the test instead of hanging it,
/// so the server guard still runs.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

fn bench_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../bench")
        .canonicalize()
        .unwrap()
}

/// The pinned snapshot sha of `name` in `bench/models.lock.json`, the revision the fp32
/// reference was produced from.
fn pinned_sha(bench: &Path, name: &str) -> String {
    let path = bench.join("models.lock.json");
    let lock: Value = serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    lock[name]["sha"]
        .as_str()
        .unwrap_or_else(|| panic!("{}: no sha for {name:?}", path.display()))
        .to_string()
}

/// Send `req` and read its response, within `REQUEST_TIMEOUT`.
async fn round_trip(conn: &mut TcpStream, req: &[u8], what: &str) -> Resp {
    tokio::time::timeout(REQUEST_TIMEOUT, async {
        send(conn, req).await;
        read_response(conn).await
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: no complete response within {REQUEST_TIMEOUT:?}"))
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

/// Absolute difference of a numeric field, `None` when `a` lacks it. Key sets are compared
/// before this is called, so a field missing on one side only is caught there.
fn field_diff(a: &Value, b: &Value, key: &str) -> Option<f64> {
    let x = a[key].as_f64()?;
    Some((x - b[key].as_f64().unwrap_or(f64::NAN)).abs())
}

/// Every environment variable the server's configuration reads, taken from the clap
/// definition so the list cannot fall behind `config.rs`.
fn config_env_vars() -> Vec<String> {
    use clap::CommandFactory;
    sys1d::Config::command()
        .get_arguments()
        .filter_map(|a| a.get_env().map(|e| e.to_string_lossy().into_owned()))
        .collect()
}

/// The introspection above sees the settings this test depends on.
#[test]
fn config_env_vars_are_known() {
    let vars = config_env_vars();
    for name in ["LAYA_API_KEY", "SYS1_F32", "SYS1_MLX_TUNING", "SYS1_MODEL"] {
        assert!(vars.iter().any(|v| v == name), "{name} not in {vars:?}");
    }
    assert_eq!(vars.iter().collect::<BTreeSet<_>>().len(), vars.len());
}

/// The `sys1d` child. Dropping it kills the process if it is still running, so a failed
/// assertion or a timeout never leaves a server holding the model and GPU memory.
struct Server(Child);

impl Server {
    /// Start the binary on a free port, serving `MODEL` at snapshot `revision`. Every server
    /// setting is pinned on the command line and its variable removed from the child's
    /// environment: an inherited `LAYA_API_KEY` would answer these unauthenticated requests
    /// with 401, and `SYS1_F32` or `SYS1_MLX_TUNING` would give the child an engine the
    /// in-process agent below does not have.
    fn spawn(revision: &str) -> Server {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_sys1d"));
        for var in config_env_vars() {
            cmd.env_remove(var);
        }
        Server(
            cmd.args([
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--model",
                MODEL,
                "--revision",
                revision,
                "--tuning",
                DEFAULT_TUNING,
                "--max-concurrent",
                &DEFAULT_MAX_CONCURRENT.to_string(),
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn sys1d"),
        )
    }

    /// The first stdout line, within `READY_TIMEOUT`. A reader thread keeps draining stdout
    /// afterwards so the child never blocks on a full pipe.
    fn ready_line(&mut self) -> String {
        let stdout = self.0.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut lines = BufReader::new(stdout).lines();
            if let Some(Ok(first)) = lines.next() {
                let _ = tx.send(first);
            }
            for _ in lines.by_ref() {}
        });
        match rx.recv_timeout(READY_TIMEOUT) {
            Ok(line) => line,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                panic!("no ready line within {READY_TIMEOUT:?}")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!(
                    "sys1d exited before printing a ready line: {:?}",
                    self.0.wait()
                )
            }
        }
    }

    /// SIGINT, then the exit status within `SHUTDOWN_TIMEOUT`.
    fn stop(&mut self) -> ExitStatus {
        let status = Command::new("kill")
            .args(["-INT", &self.0.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success(), "kill -INT");
        let deadline = Instant::now() + SHUTDOWN_TIMEOUT;
        loop {
            if let Some(st) = self.0.try_wait().unwrap() {
                return st;
            }
            assert!(
                Instant::now() < deadline,
                "sys1d did not exit within {SHUTDOWN_TIMEOUT:?} of SIGINT"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Ok(None) = self.0.try_wait() {
            eprintln!("killing sys1d (pid {}) that is still running", self.0.id());
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn smoke_workload_through_the_real_server() {
    let bench = bench_root();
    let workload = read_jsonl(&bench.join("workloads/smoke.jsonl"));
    assert_eq!(workload.len(), SMOKE_REQUESTS, "smoke workload rows");
    // The reference file is one `meta` row (the run that produced it) and one `result` row
    // per smoke request. The reference was produced from the pinned revision; serve and load
    // that same one.
    let sha = pinned_sha(&bench, MODEL);
    let reference = read_jsonl(&bench.join("reference/typed-decisions/smoke.jsonl"));
    let meta = reference
        .iter()
        .find(|r| r["type"] == "meta")
        .expect("reference meta row");
    assert_eq!(
        meta["model_sha"], sha,
        "reference revision is the pinned one"
    );
    let reference: Vec<&Value> = reference.iter().filter(|r| r["type"] == "result").collect();
    assert_eq!(reference.len(), SMOKE_REQUESTS, "reference answer rows");

    // Start the binary on a free port and parse its ready line.
    let t0 = Instant::now();
    let mut server = Server::spawn(&sha);
    let line = server.ready_line();
    let ready: Value =
        serde_json::from_str(line.trim()).unwrap_or_else(|e| panic!("ready line {line:?}: {e}"));
    eprintln!(
        "ready after {:.0} ms: {}",
        t0.elapsed().as_secs_f64() * 1000.0,
        line.trim()
    );
    assert_eq!(ready["event"], "listening");
    assert_eq!(ready["model"], MODEL);
    assert_eq!(ready["revision"], sha, "served revision is the pinned one");
    assert!(ready["engine"].as_str().unwrap().starts_with("mlx("));
    assert!(ready["load_ms"].as_f64().unwrap() > 0.0);
    let addr = ready["addr"].as_str().unwrap().to_string();

    let mut conn = TcpStream::connect(&addr).await.unwrap();
    let h = round_trip(
        &mut conn,
        &build_request("GET", "/health", &[], None),
        "GET /health",
    )
    .await;
    assert_eq!(h.status, 200);
    let hv = h.json();
    assert_eq!(hv["status"], "ok");
    assert_eq!(hv["loaded"], serde_json::json!([MODEL]));
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
        let r = round_trip(&mut conn, &req, &format!("POST {}", row["id"])).await;
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
        assert_eq!(v["routing"]["model"], MODEL);
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
    let exit = server.stop();
    assert_eq!(exit.code(), Some(0), "clean shutdown exit code");

    // In-process answers from the same revision with the same engine settings (the ones
    // pinned on the child's command line: the default tuning, f16) must equal the served
    // ones byte for byte. Resolve it the way the binary did, so both sides read one snapshot.
    let local_model = sys1d::config::resolve_served(MODEL, Some(&sha)).unwrap();
    assert_eq!(local_model.revision.as_deref(), Some(sha.as_str()));
    let dir = local_model.dir;
    let opts = BackendOptions {
        f32: false,
        tuning: Some(DEFAULT_TUNING.into()),
        ..Default::default()
    };
    let agent = Agent::load(&dir, &opts, Box::new(laya_mlx::make_backend)).unwrap();
    for (row, (id, over_http)) in workload.iter().zip(&served) {
        let local = agent
            .predict(&row["body"]["state"], &row["body"]["questions"])
            .unwrap();
        assert_eq!(local["usage"], over_http["usage"], "{id}");
        if local["answers"] != over_http["answers"] {
            // Name the question and the size of the gap before failing.
            let (la, ha) = (
                local["answers"].as_object().unwrap(),
                over_http["answers"].as_object().unwrap(),
            );
            for (qid, a) in la {
                let b = &ha[qid];
                if a != b {
                    panic!(
                        "{id}/{qid}: served answer differs from Agent::predict (probability diff {:.2e}): served {b} in-process {a}",
                        max_prob_diff(a, b)
                    );
                }
            }
            panic!(
                "{id}: answers differ: served {} in-process {}",
                over_http["answers"], local["answers"]
            );
        }
    }
    eprintln!(
        "in-process agreement: {}/{} requests byte-identical answers",
        served.len(),
        served.len()
    );

    // Against the fp32 CPU reference: every reference request was served, every answer has
    // the reference's key set in the reference's order, every numeric field is within its
    // tolerance, and the categorical answer (argmax choice, rounded score, noul side) agrees
    // on at least 99% of answers.
    let served_ids: BTreeSet<&str> = served.iter().map(|(id, _)| id.as_str()).collect();
    let reference_ids: BTreeSet<&str> = reference
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        served_ids, reference_ids,
        "every reference request is served, once"
    );
    assert_eq!(served_ids.len(), SMOKE_REQUESTS);
    let mut agree = 0usize;
    let mut total = 0usize;
    let mut disagreements = Vec::new();
    // (field, tolerance, largest difference seen)
    let mut spread = [
        ("probabilities", REFERENCE_TOL, 0.0f64),
        ("answer_confidence", REFERENCE_TOL, 0.0),
        ("confidence", REFERENCE_TOL, 0.0),
        ("action.act_probability", REFERENCE_TOL, 0.0),
        ("score", SCORE_TOL, 0.0),
    ];
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
            let diffs = [
                Some(max_prob_diff(ans, want)),
                field_diff(ans, want, "answer_confidence"),
                field_diff(ans, want, "confidence"),
                field_diff(&ans["action"], &want["action"], "act_probability"),
                field_diff(ans, want, "score"),
            ];
            for ((field, tol, worst), d) in spread.iter_mut().zip(diffs) {
                let Some(d) = d else { continue };
                assert!(
                    d <= *tol,
                    "{id}/{qid}: {field} differs by {d} from the reference (tolerance {tol}): served {ans} reference {want}"
                );
                *worst = worst.max(d);
            }
            total += 1;
            if categorical(ans) == categorical(want) {
                agree += 1;
            } else {
                disagreements.push(format!(
                    "{id}/{qid} ({}): served {} reference {}",
                    ans["type"],
                    categorical(ans),
                    categorical(want)
                ));
            }
        }
    }
    assert!(total > 0, "reference has no answers for the smoke ids");
    let pct = 100.0 * agree as f64 / total as f64;
    let worst: Vec<String> = spread
        .iter()
        .map(|(f, _, w)| format!("{f} {w:.4}"))
        .collect();
    eprintln!(
        "reference agreement: {total} answers with matching key order; largest differences {}; {agree}/{total} categorical answers agree ({pct:.1}%) {disagreements:?}",
        worst.join(", ")
    );
    assert!(
        pct >= 99.0,
        "categorical agreement {pct:.1}% < 99%: {disagreements:?}"
    );
}
