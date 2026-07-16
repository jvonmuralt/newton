# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce (and verify the fix for) zero input gradients from warp-nn's ONNX runtime.

Three cases — single-layer MLP, multi-layer MLP, LSTM — each runs one forward +
autodiff backward through ``warp_nn.runtime.onnx_runtime.OnnxRuntime`` and
compares the input gradient to a finite-difference reference.

    uv run python warp_nn_grad_repro/repro.py

Against UNPATCHED warp-nn:
    single-layer MLP : PASS   (no intermediate tensor, so grads flow)
    multi-layer  MLP : FAIL   (grad == 0; intermediate lacks requires_grad)
    LSTM             : FAIL   (grad == 0; op-cache tensors lack requires_grad)

After applying onnx_runtime_requires_grad.patch, construct with
``OnnxRuntime(path, requires_grad=True)`` and all three PASS. The script detects
the opt-in automatically and uses it when present.
"""

from __future__ import annotations

import inspect
import sys
import tempfile

import numpy as np
import warp as wp

try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    print("This repro needs `onnx` (pip install onnx).")
    sys.exit(1)

from warp_nn.runtime.onnx_runtime import OnnxRuntime

wp.init()
DEV = "cuda:0" if wp.get_cuda_devices() else "cpu"
RNG = np.random.default_rng(0)
K, H, I = 3, 4, 3  # MLP input dim, hidden dim, LSTM input dim

# Is the requires_grad opt-in present (i.e. is the patch applied)?
HAS_OPT_IN = "requires_grad" in inspect.signature(OnnxRuntime.__init__).parameters


def _make_runtime(path):
    if HAS_OPT_IN:
        return OnnxRuntime(path, device=DEV, batch_size=1, requires_grad=True)
    return OnnxRuntime(path, device=DEV, batch_size=1)


def _save(path, nodes, inits, in_shape, out_shape, in_name, out_name):
    x = helper.make_tensor_value_info(in_name, TensorProto.FLOAT, list(in_shape))
    y = helper.make_tensor_value_info(out_name, TensorProto.FLOAT, list(out_shape))
    graph = helper.make_graph(nodes, "g", [x], [y], initializer=inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def _tape_grad(rt, in_name, out_name, in_shape, x0, out_shape, seed_idx):
    xin = wp.zeros(in_shape, dtype=wp.float32, device=DEV, requires_grad=True)
    xin.assign(x0.reshape(in_shape))
    seed = wp.zeros(out_shape, dtype=wp.float32, device=DEV)
    s = seed.numpy()
    s[seed_idx] = 1.0
    seed.assign(s)
    xin.grad.zero_()
    tape = wp.Tape()
    with tape:
        out = rt({in_name: xin})[out_name]
    tape.backward(grads={out: seed})
    return xin.grad.numpy().reshape(-1).copy()


def _fd(fwd, x0, n, eps=1e-3):
    g = np.zeros(n, dtype=np.float32)
    for i in range(n):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (fwd(xp) - fwd(xm)) / (2 * eps)
    return g


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def case_single(tmp):
    w = (RNG.standard_normal((1, K)) * 1.5).astype(np.float32)
    b = np.array([0.3], dtype=np.float32)
    path = f"{tmp}/single.onnx"
    _save(
        path,
        [helper.make_node("Gemm", ["input", "W", "b"], ["output"], alpha=1.0, beta=1.0, transB=1)],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "b")],
        (1, K),
        (1, 1),
        "input",
        "output",
    )
    x0 = RNG.standard_normal(K).astype(np.float32)
    g = _tape_grad(_make_runtime(path), "input", "output", (1, K), x0, (1, 1), (0, 0))
    return g, _fd(lambda x: float((w @ x + b)[0]), x0, K)


def case_multi(tmp):
    w1 = (RNG.standard_normal((H, K)) * 1.2).astype(np.float32)
    b1 = (RNG.standard_normal(H) * 0.3).astype(np.float32)
    w2 = (RNG.standard_normal((1, H)) * 1.2).astype(np.float32)
    b2 = np.array([0.1], dtype=np.float32)
    path = f"{tmp}/multi.onnx"
    _save(
        path,
        [
            helper.make_node("Gemm", ["input", "W1", "b1"], ["h"], alpha=1.0, beta=1.0, transB=1),
            helper.make_node("Elu", ["h"], ["a"], alpha=1.0),
            helper.make_node("Gemm", ["a", "W2", "b2"], ["output"], alpha=1.0, beta=1.0, transB=1),
        ],
        [numpy_helper.from_array(a, n) for a, n in ((w1, "W1"), (b1, "b1"), (w2, "W2"), (b2, "b2"))],
        (1, K),
        (1, 1),
        "input",
        "output",
    )
    x0 = RNG.standard_normal(K).astype(np.float32)

    def fwd(x):
        hl = w1 @ x + b1
        a = np.where(hl >= 0.0, hl, np.exp(hl) - 1.0)
        return float((w2 @ a + b2)[0])

    g = _tape_grad(_make_runtime(path), "input", "output", (1, K), x0, (1, 1), (0, 0))
    return g, _fd(fwd, x0, K)


def case_lstm(tmp):
    w = (RNG.standard_normal((1, 4 * H, I)) * 0.8).astype(np.float32)
    r = (RNG.standard_normal((1, 4 * H, H)) * 0.8).astype(np.float32)
    bb = (RNG.standard_normal((1, 8 * H)) * 0.2).astype(np.float32)
    path = f"{tmp}/lstm.onnx"
    _save(
        path,
        [helper.make_node("LSTM", ["X", "W", "R", "B"], ["Y", "Y_h", "Y_c"], hidden_size=H, direction="forward")],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(r, "R"), numpy_helper.from_array(bb, "B")],
        (1, 1, I),
        (1, 1, H),
        "X",
        "Y_h",
    )
    x0 = RNG.standard_normal(I).astype(np.float32)

    def fwd(x):
        g = w[0] @ x + bb[0, : 4 * H] + bb[0, 4 * H :]  # h_prev = c_prev = 0
        gi, go, _gf, gc = _sig(g[0:H]), _sig(g[H : 2 * H]), _sig(g[2 * H : 3 * H]), np.tanh(g[3 * H : 4 * H])
        c = gi * gc
        return float((go * np.tanh(c))[0])

    g = _tape_grad(_make_runtime(path), "X", "Y_h", (1, 1, I), x0, (1, 1, H), (0, 0, 0))
    return g, _fd(fwd, x0, I)


def main():
    print(f"device: {DEV}   requires_grad opt-in available (patched): {HAS_OPT_IN}\n")
    print(f"{'case':<18}{'result':<8}{'max|warp - FD|':<16}warp-nn grad / finite-difference")
    print("-" * 92)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in (("single-layer MLP", case_single), ("multi-layer MLP", case_multi), ("LSTM", case_lstm)):
            gw, gf = fn(tmp)
            err = float(np.max(np.abs(gw - gf)))
            passed = np.allclose(gw, gf, atol=1e-2)
            ok = ok and passed
            print(f"{name:<18}{'PASS' if passed else 'FAIL':<8}{err:<16.2e}{np.round(gw, 3)}  vs  {np.round(gf, 3)}")
    print("\n" + ("ALL PASS — gradients flow." if ok else "FAILURES — apply onnx_runtime_requires_grad.patch."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
