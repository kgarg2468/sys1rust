"""Per-op eval timing of Python laya-mlx encoder layers (same split as the Rust `ops` knob).

Run with the laya-mlx venv after `source bench/env.sh` (for HF_HOME), from any directory.
"""
import json, os, sys, time
import mlx.core as mx, mlx.nn as nn
BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BENCH, "contenders", "laya-mlx", "src"))
from laya_mlx.agent import Agent
from laya_mlx import model as M
lock = json.load(open(os.path.join(BENCH, "models.lock.json")))["typed-decisions"]
d = os.path.join(os.environ["HF_HOME"], "hub", "models--" + lock["repo"].replace("/", "--"), "snapshots", lock["sha"])
ag = Agent(d, dtype="float16")
row = json.loads(open(os.path.join(BENCH, "workloads", "short.jsonl")).readline())
acc = {}
t = [0.0]
def mark(name, *a):
    mx.eval(*a); now = time.perf_counter(); acc[name] = acc.get(name, 0) + int((now - t[0]) * 1e6); t[0] = now
def layer_call(self, x, mask):
    t[0] = time.perf_counter()
    a = self.attn_norm(x); mark("attn_norm", a)
    at = self.attn; b, L, _ = a.shape
    qkv = at.Wqkv(a); mark("wqkv", qkv)
    qkv = qkv.reshape(b, L, 3, at.num_heads, at.head_dim)
    q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
    q = mx.fast.rope(q, at.head_dim, traditional=False, base=at.base, scale=1.0, offset=0)
    k = mx.fast.rope(k, at.head_dim, traditional=False, base=at.base, scale=1.0, offset=0)
    mark("split+rope", q, k, v)
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=at.head_dim**-0.5, mask=mask)
    mark("sdpa_" + ("local" if self.attention_type == "sliding_attention" else "global"), o)
    x = x + at.Wo(o.transpose(0, 2, 1, 3).reshape(b, L, -1)); mark("merge+wo", x)
    m = self.mlp_norm(x); mark("mlp_norm", m)
    wi = self.mlp.Wi(m); mark("wi", wi)
    val, gate = mx.split(wi, 2, axis=-1); act = nn.gelu(val) * gate; mark("geglu", act)
    x = x + self.mlp.Wo(act); mark("wo2", x)
    return x
M.EncoderLayer.__call__ = layer_call
for i in range(4):
    acc.clear(); ag.system_one(row["body"]["state"], row["body"]["questions"])
print("ops_us", sorted(acc.items(), key=lambda kv: kv[0]))
