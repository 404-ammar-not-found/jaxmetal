# Double-single (df64) extended precision

**Since v0.6.0** · `kernels/df64.metal` · `src/ops/df64.mm` · `tests/cpp/df64_test.cpp` ·
`benchmarks/bench_df64.py`

## Read this first: it is slower than the CPU

**df64 is a precision feature, not a performance feature.** It is slower than doing
the same work in `float64` on the CPU, at every size measured.

The rationale originally given for building it was that the GPU's higher memory
bandwidth would pay for the emulation on bandwidth-bound kernels. **That was wrong.**
Apple Silicon is *unified memory*: the CPU and GPU share one memory controller, so
there is no GPU bandwidth advantage to exploit for bandwidth-bound work. The error was
comparing a GPU figure (~273 GB/s) against a discrete-GPU intuition about CPU
bandwidth (~120 GB/s) that does not apply on this architecture.

It ships anyway, deliberately, because **Metal has no `double` type at all** — so a
pipeline whose data is already GPU-resident otherwise has no way to exceed f32 without
round-tripping to the host.

## The problem

f32 carries 24 bits of significand. Anywhere a computation cancels — a difference of
nearly-equal quantities, a finite-difference stencil, a long accumulation — the
relative error is amplified by the cancellation ratio, and 24 bits runs out fast. The
standard fix is `float64`, which this hardware does not have.

df64 represents a value as an unevaluated sum of two f32 limbs (`hi + lo`, with
`|lo| ≤ ulp(hi)/2`), giving **~48 bits** of significand.

**This is not IEEE float64.** It has f32's exponent range (overflows near 3.4e38, not
1.8e308), no guaranteed correct rounding, and weaker denormal behaviour. ~48 bits, not
53.

## How it works

Three error-free transformations, each computing a quantity that is algebraically
zero and recovering the bits a single f32 operation threw away:

```metal
inline float two_sum(float a, float b, thread float& e) {   // exact sum, any order
    float s = a + b, bb = s - a;
    e = (a - (s - bb)) + (b - bb);
    return s;
}
inline float two_prod(float a, float b, thread float& e) {  // exact product
    float p = a * b;
    e = fma(a, b, -p);
    return p;
}
```

`two_prod` is why `fma` matters here beyond speed: without a fused multiply-add there
is no cheap way to recover the low half of a product. `df_add`/`df_mul` compose these;
`df_div` takes one Newton step from an f32 reciprocal, which doubles the correct
digits — exactly the step from 24 bits to ~48.

Ops provided: elementwise add/mul/div, and a 3-point stencil (the representative PDE
kernel), plus a plain-f32 stencil with an identical access pattern for comparison.

## Load-bearing invariants

> **Requires safe math** (`MTLMathModeSafe` / `fastMathEnabled = NO`, set in
> `src/metal/kernel_library.mm`). Every transformation above is algebraically zero —
> `(a - (a + b)) + b` and `fma(a, b, -a*b)` — and a fast-math compiler is entitled to
> fold each to `0`. Under fast math every operation here silently degrades to plain
> f32: same speed, none of the precision, **no error**. `DF64AddBeatsFloat32` is the
> regression test, and it asserts f32 *does* fail the case first, so it cannot pass
> vacuously if both paths become the same thing.

> **The df64 and f32 kernels take different coefficient layouts** — 3 `(hi, lo)` pairs
> versus 3 plain floats. Feeding the df64 layout to the f32 kernel hands it
> `[hi0, lo0, hi1]`, i.e. the wrong coefficients, and makes the comparison flatter
> df64 by orders of magnitude. This happened during development and produced a bogus
> "107 trillion×" figure before it was caught.

**Cancellation amplifies relative error at any precision.** If operands are stored to
relative accuracy `eps` and the result is a factor `R` smaller, the result carries
`~eps·R`. At `R = 1e5`, df64 (eps ~3.6e-15) can only reach ~3.6e-10 — so a test
asserting 1e-13 there would be asserting something no finite precision can deliver.
The test tolerances are derived from this, not tuned to pass.

## Measurements

`.venv/bin/python benchmarks/bench_df64.py` — M4 Pro, relative error vs `float64`:

| case | GPU df64 | GPU f32 | improvement |
|---|---:|---:|---:|
| `a + b` with `b ≈ -a` (cancels 1e5) | 1.78e-10 | 5.95e-03 | 3.4e7× |
| `a * b` | 7.77e-15 | 8.34e-08 | 1.1e7× |
| 3-pt stencil (1,−2,1), h=0.01 | 1.87e-10 | 3.40e-03 | 1.8e7× |

### And the cost

| n | GPU df64 | CPU f64 | GPU f32 | df64 vs CPU f64 |
|---:|---:|---:|---:|---:|
| 1,048,576 | 5.50 ms | 0.20 ms | 0.10 ms | **0.04×** |
| 4,194,304 | 21.96 ms | 0.86 ms | 0.35 ms | **0.04×** |
| 16,777,216 | 89.19 ms | 3.49 ms | 1.58 ms | **0.04×** |

The GPU column includes the host round trip, which dominates at these sizes — the
kernel alone is far closer to parity (an independent measurement of a resident 64M
elementwise add put it at 7.53 ms against 7.08 ms for an 8-thread CPU, i.e. ~0.94×).
Either way the CPU wins, for two structural reasons: unified memory removes the
bandwidth advantage, and df64 moves 2× the bytes of f32 for the same element count
while these kernels are bandwidth-bound.

## When to use it

- **Yes:** data is already GPU-resident, and a few steps of the pipeline need more
  than 24 bits — a cancelling difference, a stencil, an accumulation. Staying on
  device beats round-tripping to the host for f64.
- **No:** the data is on the host. `numpy` in `float64` is faster *and* more accurate.
- **No:** for GEMM. Measured separately: emulated f64 GEMM lands at 270–540 GFLOP/s
  against the CPU's 729 GFLOP/s native `dgemm`. That is why no df64 matmul exists here.
- **Consider instead:** [COMPENSATED_REDUCTIONS.md](COMPENSATED_REDUCTIONS.md) if the
  problem is specifically *summation* accuracy. Compensated summation gets near-f64
  accuracy at **no** measurable cost, where df64 costs 2× the bytes and ~10–20× the
  arithmetic.

## Limits and things left out

- **No transcendentals** (`exp`, `log`, `sin`). Each needs its own df64 argument
  reduction and polynomial; substantial work, no caller yet.
- **No resident entry point.** The C ABI copies host arrays in and out, which is what
  makes the measured cost 0.04× rather than ~0.94×. Since the whole justification is
  "data already on the GPU", this is the gap most worth closing.
- **No df64 matmul**, by design — see above.
- **f32 exponent range.** Values beyond ~3.4e38 overflow where real f64 would not.
