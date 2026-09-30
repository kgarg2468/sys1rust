#!/usr/bin/env python3
"""powermetrics logger for the bench.

Starts exactly
    sudo -n /usr/bin/powermetrics --samplers cpu_power,gpu_power,ane_power,thermal -i 1000 -n 21600 -f plist
parses its NUL-separated plist stream into timestamped samples, and stops it cleanly.
If `sudo -n` is refused, it logs that and exits 0 so a run carries on without energy.

Usage:
  power.py check                          3-second check that sudo -n powermetrics works
  power.py start --out SAMPLES.jsonl      start a background logger (pidfile SAMPLES.jsonl.pid)
  power.py stop  --out SAMPLES.jsonl      stop it
  power.py run   --out SAMPLES.jsonl -- CMD ARGS...   log while CMD runs
  power.py parse RAW.plist SAMPLES.jsonl  re-parse a saved raw stream

Sample line (powers in milliwatts; a sample covers [t_start, t]; t = arrival time on the pipe):
  {"t": 1790000000.1, "t_start": t - elapsed, "elapsed_ms": 1003.2, "t_plist": <1 s resolution>,
   "cpu_mw": ..., "gpu_mw": ..., "ane_mw": ..., "combined_mw": ..., "thermal_pressure": "Nominal"}
Plus a first line {"type": "power_meta", ...} and, on failure, {"type": "power_error", ...}.
Energy for a window = sum over samples of combined_mw * overlap_seconds (millijoules).
"""
import argparse
import datetime as dt
import json
import os
import plistlib
import signal
import subprocess
import sys
import time

CMD = ["sudo", "-n", "/usr/bin/powermetrics", "--samplers", "cpu_power,gpu_power,ane_power,thermal",
       "-i", "1000", "-n", "21600", "-f", "plist"]


def _num(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    return None


def sample_from_plist(doc, t_recv, live=True):
    proc = doc.get("processor", {}) if isinstance(doc, dict) else {}
    gpu = doc.get("gpu", {}) if isinstance(doc, dict) else {}
    # The plist `timestamp` has 1-second resolution, so the sample end is taken as the time the
    # sample arrived on the pipe (powermetrics writes each sample when its interval ends).
    ts = doc.get("timestamp")
    t_plist = (ts if ts.tzinfo else ts.replace(tzinfo=dt.timezone.utc)).timestamp() \
        if isinstance(ts, dt.datetime) else None
    elapsed_ms = (doc.get("elapsed_ns") or 0) / 1e6
    # Re-parsing a saved stream has no arrival times; fall back to the plist stamp (+-1 s).
    t = t_recv if live or t_plist is None else t_plist + elapsed_ms / 1000.0
    cpu = _num(proc, "cpu_power")
    gpu_mw = _num(proc, "gpu_power")
    if gpu_mw is None:
        gpu_mw = _num(gpu, "gpu_power")
    ane = _num(proc, "ane_power")
    comb = _num(proc, "combined_power")
    if comb is None and None not in (cpu, gpu_mw, ane):
        comb = cpu + gpu_mw + ane
    return {"t": t, "t_start": t - elapsed_ms / 1000.0, "elapsed_ms": round(elapsed_ms, 3),
            "t_plist": t_plist, "cpu_mw": cpu, "gpu_mw": gpu_mw, "ane_mw": ane, "combined_mw": comb,
            "thermal_pressure": doc.get("thermal_pressure")}


def iter_docs(stream):
    """Yield (plist dict, receive time) from a NUL-separated plist byte stream."""
    buf = b""
    while True:
        chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
        if not chunk:
            break
        buf += chunk
        while b"\x00" in buf:
            part, buf = buf.split(b"\x00", 1)
            part = part.strip()
            if part:
                try:
                    yield plistlib.loads(part), time.time()
                except Exception:
                    pass
    if buf.strip():
        try:
            yield plistlib.loads(buf.strip()), time.time()
        except Exception:
            pass


def log(out, obj):
    out.write(json.dumps(obj) + "\n")
    out.flush()


def logger(out_path):
    """Foreground logger: run powermetrics, write samples until stopped (SIGTERM/SIGINT)."""
    raw_path = out_path + ".raw.plist"
    with open(out_path, "a") as out:
        try:
            proc = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        except Exception as e:
            log(out, {"type": "power_error", "t": time.time(), "error": "spawn failed: %s" % e})
            return 0
        log(out, {"type": "power_meta", "t": time.time(), "cmd": " ".join(CMD), "logger_pid": os.getpid(),
                  "sudo_pid": proc.pid})
        stopping = {"flag": False}

        def on_signal(*_):
            if stopping["flag"]:
                return
            stopping["flag"] = True
            # sudo relays SIGINT/SIGTERM to powermetrics, which then exits after its current sample.
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    os.kill(proc.pid, sig)
                    break
                except (PermissionError, ProcessLookupError):
                    continue
            # Fallback that needs no privileges: close our end of the pipe; powermetrics gets
            # SIGPIPE on its next write (<= 1 s) and exits.
            def closer():
                time.sleep(3)
                if proc.poll() is None:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
            import threading
            threading.Thread(target=closer, daemon=True).start()

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        n = 0
        with open(raw_path, "ab") as raw:
            class Tee:
                def read1(self, n_):
                    if stopping["flag"]:
                        return b""
                    try:
                        b = proc.stdout.read1(n_)
                    except (ValueError, OSError):
                        return b""
                    raw.write(b)
                    return b
            for doc, t_recv in iter_docs(Tee()):
                log(out, sample_from_plist(doc, t_recv))
                n += 1
        try:
            proc.stdout.close()  # powermetrics exits on SIGPIPE at its next write
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        rc = proc.returncode
        if n == 0:
            log(out, {"type": "power_error", "t": time.time(), "returncode": rc,
                      "error": err[:500] or "no samples (sudo -n refused or powermetrics failed)"})
        else:
            log(out, {"type": "power_end", "t": time.time(), "samples": n, "returncode": rc, "stderr": err[:300]})
    return 0


def start(out_path):
    pidfile = out_path + ".pid"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "_logger", "--out", out_path],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
    open(pidfile, "w").write(str(p.pid))
    # Wait for the first sample or an error so callers know whether energy is available.
    t0 = time.time()
    while time.time() - t0 < 5:
        status = read_status(out_path)
        if status != "starting":
            break
        if p.poll() is not None:
            break
        time.sleep(0.2)
    status = read_status(out_path)
    print("power: %s (%s)" % (status, out_path), file=sys.stderr)
    return p.pid, status


def read_status(out_path):
    if not os.path.exists(out_path):
        return "starting"
    kinds = [json.loads(l).get("type", "sample") for l in open(out_path) if l.strip()]
    if "power_error" in kinds:
        return "unavailable"
    if "sample" in kinds:
        return "logging"
    return "starting"


def stop(out_path):
    pidfile = out_path + ".pid"
    if not os.path.exists(pidfile):
        return
    pid = int(open(pidfile).read().strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    t0 = time.time()
    while time.time() - t0 < 15:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    os.remove(pidfile)
    print("power: stopped, status %s" % read_status(out_path), file=sys.stderr)


def load_samples(path):
    """Samples sorted by time; ignores meta/error lines."""
    if not path or not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        if not l.strip():
            continue
        d = json.loads(l)
        if "type" not in d and d.get("combined_mw") is not None:
            out.append(d)
    return sorted(out, key=lambda s: s["t"])


def energy_mj(samples, t0, t1, key="combined_mw"):
    """Integrate power over [t0, t1] (seconds) -> millijoules. None if the window is not covered."""
    if not samples or t1 <= t0:
        return None
    if t0 < samples[0]["t_start"] - 0.5 or t1 > samples[-1]["t"] + 0.5:
        return None
    e = 0.0
    for s in samples:
        a, b = max(t0, s["t_start"]), min(t1, s["t"])
        if b > a and s.get(key) is not None:
            e += s[key] * (b - a)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "start", "stop", "run", "parse", "_logger"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--out")
    args, extra = ap.parse_known_args()
    if args.cmd == "_logger":
        return logger(args.out)
    if args.cmd == "check":
        out = args.out or "/tmp/bench_power_check.jsonl"
        if os.path.exists(out):
            os.remove(out)
        start(out)
        time.sleep(3)
        stop(out)
        print(open(out).read())
        return 0
    if args.cmd == "start":
        start(args.out)
        return 0
    if args.cmd == "stop":
        stop(args.out)
        return 0
    if args.cmd == "parse":
        raw, out = args.rest
        with open(raw, "rb") as f, open(out, "w") as o:
            for doc, t_recv in iter_docs(f):
                log(o, sample_from_plist(doc, t_recv, live=False))
        return 0
    if args.cmd == "run":
        cmd = args.rest + extra
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        start(args.out)
        try:
            rc = subprocess.call(cmd)
        finally:
            stop(args.out)
        return rc


if __name__ == "__main__":
    sys.exit(main())
