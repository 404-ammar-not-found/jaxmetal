# Double-single (df64) extended precision

**Since v0.6.0** · `kernels/df64.metal` · `src/ops/df64.mm` · `tests/cpp/df64_test.cpp` ·
`benchmarks/bench_df64.py`

## Read this first: residency decides whether it is fast

df64 gives ~48 bits of significand on a GPU that has **no `double` type at all**. Its
speed depends entirely on whether the data is already on the device:

| | vs CPU `float64` |
|---|---|
| **Resident** (`df64_binop_resident`) | **1.65–1.88× FASTER** above ~4M elements |
| Host operands (`df64_binop`) | ~0.04× — the copies dominate completely |

**An earlier version of this document said df64 was slower than the CPU at every
size. That was measured on the host path and was wrong as a statement about the
feature.** The copies, not the arithmetic, were the cost: at 16.7M elements the host
path takes 74.8 ms and the resident path 2.07 ms — **36× apart**.

The reasoning that produced the wrong claim is worth keeping. Apple Silicon is unified
memory, so the CPU and GPU share one memory controller and there is no raw bandwidth
*ratio* to exploit; that part is true. What it missed is that the GPU still achieves
higher *streaming* bandwidth on this access pattern, and that the emulation arithmetic
hides under memory latency almost entirely — df64 costs only **12% more than plain f32
on the GPU** despite moving twice the bytes.

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

### And the cost — resident

| n | df64 resident | CPU f64 | GPU f32 resident | df64 vs CPU f64 |
|---:|---:|---:|---:|---:|
| 1,048,576 | 0.41 ms | 0.19 ms | 0.10 ms | 0.47× |
| 4,194,304 | 0.79 ms | 0.78 ms | 0.39 ms | 0.98× |
| 16,777,216 | 1.88 ms | 3.10 ms | 1.55 ms | **1.65×** |
| 67,108,864 | 7.01 ms | 13.17 ms | 6.27 ms | **1.88×** |

**The crossover is ~4M elements.** Below it the command-buffer round trip dominates;
above it df64 beats CPU `float64` outright while carrying ~48 bits.

Note the `GPU f32` column: df64 costs only **12%** more than plain f32 on the GPU at
67M, despite moving 2× the bytes. Larger accesses use the memory system better, so the
emulation is close to free once you are bandwidth-bound.

### The host path, for comparison

| n | df64 host | CPU f64 | ratio |
|---:|---:|---:|---:|
| 16,777,216 | 74.82 ms | 3.09 ms | 0.04× |

Same kernel, 36× slower, entirely because of copying 2×n floats in and out.

## When to use it

- **Yes:** data is already GPU-resident and there are ≥ ~4M elements. It is both more
  accurate than f32 and *faster than the CPU in float64*.
- **No:** the data is on the host and used once. The copies cost 36× the kernel;
  `numpy` in `float64` is faster and more accurate.
- **No:** for GEMM. Measured separately: emulated f64 GEMM lands at 270–540 GFLOP/s
  against the CPU's 729 GFLOP/s native `dgemm`. That is why no df64 matmul exists here.
- **Consider instead:** [COMPENSATED_REDUCTIONS.md](COMPENSATED_REDUCTIONS.md) if the
  problem is specifically *summation* accuracy. Compensated summation gets near-f64
  accuracy at **no** measurable cost, where df64 costs 2× the bytes and ~10–20× the
  arithmetic.

## Limits and things left out

- **No transcendentals** (`exp`, `log`, `sin`). Each needs its own df64 argument
  reduction and polynomial; substantial work, no caller yet.
- **`df64_binop_resident` covers elementwise ops only.** The stencil has no resident
  entry point yet and still pays the copies.
- **No df64 matmul**, by design — see above.
- **f32 exponent range.** Values beyond ~3.4e38 overflow where real f64 would not.
