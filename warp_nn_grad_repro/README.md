# warp-nn ONNX runtime: zero input gradients (repro + patch)

`warp_nn.runtime.onnx_runtime.OnnxRuntime` runs ONNX policies forward and is
meant to be autodiff-able (wrap the call in a `wp.Tape`, seed the output grad,
read the input grad). For anything past a single layer the **input gradient
comes back all zeros**, even though finite differences show it is nonzero.

Tested with **warp-nn 0.1.0**.

## Root cause

The ONNX runtime allocates *every* tensor with `requires_grad=False` (the Warp
default), so the tape has no gradient buffers to chain through:

| allocation | file / line (warp-nn 0.1.0) |
|---|---|
| weights / initializers | `_np_to_warp` → `wp.array(...)` (l. 197, 234) |
| MLP op outputs (intermediates) | `_shape_gemm` / `_shape_elementwise_unary` → `wp.zeros(...)` (l. 359, 369) |
| LSTM gates / h / c / biases | `_shape_lstm` → `wp.zeros(...)` (l. 460–473) |

- **Single-layer MLP** works by accident: there is no intermediate tensor, so
  the adjoint of the one Gemm writes straight into `input.grad`.
- **Multi-layer MLP** breaks: the intermediate between layers has no `.grad`
  buffer, so the chain snaps and `input.grad == 0`.
- **LSTM** breaks: its `gates` / `h` / `c` tensors live in the op **cache**, not
  in `runtime._tensors`.

## Why the monkeypatch workaround is not enough (important)

A tempting workaround is to flip `requires_grad` *after* construction:

```python
for t in runtime._tensors.values():
    t.requires_grad = True
```

This makes **multi-layer MLP** work, but **not LSTM**, for two reasons:

1. The LSTM intermediates are in `op.attrs["_cache"]`, not `runtime._tensors`,
   so the loop never reaches them.
2. Even reaching them is too late: the LSTM output is a `reshape` **view** of
   `h_buf` created at *preallocation* time. A view made from a non-grad array
   has no grad buffer, and flipping `requires_grad` on the parent afterwards
   does not retro-attach one to the existing view. Seeding that view's grad
   therefore never reaches `h_buf.grad`.

**Conclusion:** `requires_grad` must be set at **allocation time**, before the
views are taken — i.e. inside the runtime, not from outside.

## The fix

Add an **opt-in** `requires_grad: bool = False` to `OnnxRuntime.__init__` and
thread it into every allocation. Default `False` → replay/inference is
completely unchanged (no grad buffers allocated); pass `True` when you need
backward. See `onnx_runtime_requires_grad.patch`.

```python
rt = OnnxRuntime(path, device=dev, requires_grad=True)   # opt in for backward
```

The patch is 15 small insertions: the `__init__` flag, `_np_to_warp`, the four
`_shape_*` handler signatures + their `wp.zeros`, and the `_preallocate_buffers`
call site. The two lazy zero *initial states* in `_exec_lstm`
(`h_prev_zero` / `c_prev_zero`) are deliberately left as constants.

## Reproduce

```bash
# 1) against the current (unpatched) warp-nn -> multi + LSTM FAIL
python repro.py

# 2) apply the patch
python -c "import warp_nn,os; print(os.path.dirname(warp_nn.__file__))"   # find the install
cd <site-packages>                          # dir that contains warp_nn/
patch -p1 < <this-dir>/onnx_runtime_requires_grad.patch

# 3) re-run -> all three PASS
python repro.py
```

`repro.py` auto-detects whether the `requires_grad` opt-in exists and uses it,
so the same script reports the bug before the patch and the fix after.

### Results

Each case: one forward + `tape.backward`, input gradient vs central finite
differences (`atol=1e-2`).

| case | unpatched | patched |
|---|---|---|
| single-layer MLP | PASS | PASS |
| multi-layer MLP (Gemm→Elu→Gemm) | **FAIL** (grad = 0) | PASS |
| LSTM (single step) | **FAIL** (grad = 0) | PASS |

Example patched run (`max\|warp − FD\|`): single 4.6e-05, multi 1.2e-04,
LSTM 6.9e-06.

## Files

- `repro.py` — the three self-checking cases (single MLP, multi MLP, LSTM).
- `onnx_runtime_requires_grad.patch` — the proposed fix (`patch -p1`).
