"""Python laya-mlx forward timing on the first short-workload row, whole and per layer."""
import json, sys, time, statistics as st
import mlx.core as mx
sys.path.insert(0, "contenders/laya-mlx/src")
from laya_mlx.agent import Agent
lock = json.load(open("models.lock.json"))["typed-decisions"]
import os
d = os.path.join(os.environ["HF_HOME"], "hub", "models--" + lock["repo"].replace("/", "--"), "snapshots", lock["sha"])
ag = Agent(d, dtype="float16")
row = json.loads(open("workloads/short.jsonl").readline())
st_, qs = row["body"]["state"], row["body"]["questions"]
def once():
    t = time.perf_counter(); ag.system_one(st_, qs); return (time.perf_counter() - t) * 1000
for _ in range(3): once()
ts = sorted(once() for _ in range(15)); print("system_one min %.1f p50 %.1f" % (ts[0], ts[len(ts)//2]))
enc = ag.model.encoder
orig = [l.__call__ for l in enc.layers]
times = []
class Timed:
    def __init__(self, l): self.l = l; self.attention_type = l.attention_type
    def __call__(self, x, m):
        t = time.perf_counter(); y = self.l(x, m); mx.eval(y); times.append(int((time.perf_counter() - t) * 1e6)); return y
enc.layers = [Timed(l) for l in enc.layers]
for _ in range(3):
    times.clear(); ag.system_one(st_, qs)
print("layers_us", times)
