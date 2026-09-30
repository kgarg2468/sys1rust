"""laya-mlx bench adapter with simulated int8 matmuls (quantize, then dequantize, in fp16).

Tells whether W8A8 int8 GEMMs could pass the 99% agreement gate before anyone writes int8
Metal kernels. FAKEQ picks what to simulate:
  w8       weights int8, symmetric, one scale per output channel
  w8a8     plus activations int8, symmetric, one scale per token (row)
  w8a8mlp  w8a8 on the encoder MLP GEMMs only (Wi, Wo), everything else fp16
Only the encoder and decision-head GEMMs change; the scorer and act head stay fp16.
Usage: FAKEQ=w8a8 python fakeq_adapter.py <laya-mlx adapter args>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER_DIR = os.path.join(os.path.dirname(HERE), "contenders", "laya-mlx")
sys.path.insert(0, ADAPTER_DIR)

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import laya_mlx  # noqa: E402

MODES = ("w8", "w8a8", "w8a8mlp")
MODE = os.environ.get("FAKEQ", "w8a8")
if MODE not in MODES:
    sys.exit(f"fakeq: unknown FAKEQ mode {MODE!r}; use one of {', '.join(MODES)}")


def q8_rows(x, axis):
    """Symmetric int8 quantize-dequantize with one scale per slice along `axis`, in fp32."""
    xf = x.astype(mx.float32)
    s = mx.maximum(mx.abs(xf).max(axis=axis, keepdims=True) / 127.0, 1e-12)
    return mx.clip(mx.round(xf / s), -127, 127) * s


class QLinear(nn.Linear):
    """nn.Linear whose weight was quantized at patch time; optionally quantizes its input."""

    def __call__(self, x):
        if self._fq_act:
            x = q8_rows(x, -1).astype(x.dtype)
        y = x @ self.weight.T
        if "bias" in self:
            y = y + self.bias
        return y


def patch(model):
    mlp_only = MODE == "w8a8mlp"
    act = MODE in ("w8a8", "w8a8mlp")
    targets = []
    for layer in model.encoder.layers:
        if not mlp_only:
            targets += [layer.attn.Wqkv, layer.attn.Wo]
        targets += [layer.mlp.Wi, layer.mlp.Wo]
    if not mlp_only:
        for layer in model.head.layers:
            targets += [layer.self_attn.in_proj, layer.self_attn.out_proj, layer.linear1, layer.linear2]
    for lin in targets:
        lin.weight = q8_rows(lin.weight, 1).astype(lin.weight.dtype)
        lin.__class__ = QLinear
        object.__setattr__(lin, "_fq_act", act)
    mx.eval(model.parameters())
    print(f"fakeq: mode {MODE}, {len(targets)} GEMMs patched", file=sys.stderr)


_load = laya_mlx.load


def load(*a, **kw):
    agent = _load(*a, **kw)
    patch(agent.model)
    return agent


laya_mlx.load = load
import adapter  # noqa: E402

if __name__ == "__main__":
    adapter.main()
