# Compensated f32 reductions

**Since v0.3.0** · `kernels/reduce.metal` · `src/ops/reduce.mm` · `tests/cpp/reduce_test.cpp` ·
`benchmarks/bench_reduce.py`

## The problem

Summing a large `float32` array loses precision as the running total outgrows the
addends. Once `|sum| >> |x|`, `sum + x` rounds `x` away entirely and the error grows
like `O(n · eps · |sum|)`.

The standard fix is to accumulate in `float64`. **Apple GPUs do not have it.** Metal
Shading Language has no `double` type and the hardware has no fp64 units, so on this
platform the usual choice is between an inaccurate GPU sum and shipping the data to
the CPU. Compensated summation is the third option, and it is the only one that keeps
the work on-device.

This is a genuine gap in the ecosystem, not just in this repo:
[jax-mps](https://github.com/tillahoffmann/jax-mps) (the mature MLX-backed JAX
backend for Apple Silicon) documents float64 as a permanent exclusion, and neither it
nor MLX offers compensated reductions.

## How it works

Neumaier summation. The accumulator is a `(sum, compensation)` pair whose true value
is `s + c`; only the very last step folds them together.

```metal
struct Acc { float s; float c; };

inline void acc_add(thread Acc& a, float x) {      // add a plain value
    float t = a.s + x;
    if (fabs(a.s) >= fabs(x)) a.c += (a.s - t) + x;   // s larger: recover x's low bits
    else                      a.c += (x - t) + a.s;   // x larger: recover s's low bits
    a.s = t;
}
```

The branch is what makes this *Neumaier* rather than classic Kahan, and it matters:
classic Kahan is wrong when a large value arrives late into a sum that started small.
`TEST(ReduceCompensatedHandlesLateLargeValues)` covers exactly that layout.

### The part that is easy to get wrong

A per-thread Kahan loop is the obvious parallel implementation and it is **wrong
here** — each thread's compensation is silently discarded the moment the partial sums
are added together, so the result degrades to a tree sum with extra steps.

Accumulators must therefore merge *associatively*, carrying the exact rounding error
of the merge itself:

```metal
inline Acc acc_merge(Acc a, Acc b) {
    float s = a.s + b.s;
    float bv = s - a.s;
    float err = (a.s - (s - bv)) + (b.s - bv);   // two_sum: exact error of s = a.s + b.s
    return Acc{ s, a.c + b.c + err };
}
```

With that, compensation survives every level of the hierarchy: within a thread's
grid-stride loop, across threads in a threadgroup tree reduction, across
threadgroups, and across the two passes.

### Launch structure

Two passes, because one threadgroup cannot saturate the GPU on a large array:

1. `reduce_sum_comp_partials` — `kReduceGroups` (512) threadgroups × `kReduceTG` (256)
   threads grid-stride the input; each threadgroup emits one `float2(s, c)`.
2. `reduce_sum_comp_finish` — one threadgroup merges the 512 partials and writes
   `s + c`.

`reduce_sum_tree_partials` / `_finish` are the uncompensated twins, with **identical
memory traffic and launch shape**, so benchmarking the pair isolates the cost of the
arithmetic rather than of the reduction strategy.

`reduce_sum_axis0_comp` is the accuracy-preserving counterpart of
`nn_reduce_sum_axis0`: one threadgroup per output column, compensated down the rows.

## Load-bearing invariants

> **Safe math is required.** Every compensation term has the form `(a - (a + b)) + b`,
> which is algebraically zero and which a fast-math compiler is entitled to fold away.
> Under fast math these kernels silently degrade to a plain tree sum: same speed, no
> accuracy, **no error message**. `src/metal/kernel_library.mm` sets `MTLMathModeSafe`
> / `fastMathEnabled = NO`; do not change it.
>
> `TEST(ReduceCompensatedBeatsNaive)` is the regression test. It asserts that the tree
> sum *does* lose accuracy before asserting the compensated one does not, so it cannot
> pass vacuously if both silently become the same kernel.

- `RED_TG` in `kernels/reduce.metal` must equal `kReduceTG` in `src/ops/reduce.mm`.
  The kernel sizes its threadgroup array and its tree reduction to exactly that width.
- `kReduceGroups × kReduceTG` is the grid stride. Tests that construct adversarial
  inputs (one large value per thread's first slot) depend on that product.

## Measurements

`.venv/bin/python benchmarks/bench_reduce.py` — Apple M4 Pro, n = 16,777,216,
relative error against a `float64` reference:

| Input | GPU compensated | GPU tree | `numpy` f32 (pairwise) |
|---|---:|---:|---:|
| `uniform [-1,1)` | 3.3e-08 | 6.8e-08 | 3.3e-08 |
| `uniform [0,1)` | 1.6e-08 | 1.6e-08 | 4.3e-08 |
| **1e8 prefix, then values below its ulp** | **1.0e-08** | 1.3e-06 | 7.0e-08 |
| `log-uniform 1e-5..1e5` | 3.9e-08 | 7.4e-08 | 3.9e-08 |

**127× better on the adversarial row**, and better than `numpy`'s f32 sum on every
row. f32 machine epsilon is 1.19e-07, so the compensated column is at the limit of
what an f32 result can represent.

The baseline is deliberately strong: `numpy.sum` on f32 is *pairwise*, not a naive
loop. Beating a naive loop would prove nothing.

### Cost: none

| Elements | Compensated | Tree | Overhead |
|---:|---:|---:|---:|
| 1.0 M (4 MB) | 22 GB/s | 26 GB/s | 1.18× |
| 4.2 M (17 MB) | 92 GB/s | 111 GB/s | 1.22× |
| 16.8 M (67 MB) | 171 GB/s | 175 GB/s | 1.02× |
| 67.1 M (268 MB) | 214 GB/s | 212 GB/s | **0.99×** |

Both kernels are bandwidth-bound against the M4 Pro's ~273 GB/s, so the extra 3–4
fp32 ops per element hide entirely under memory latency. The original estimate for
this feature was "~2× cost"; the measurement says free at the sizes it is for.

## Limits and things left out

- **Small arrays are dominated by submission overhead, not bandwidth.** The 22 GB/s
  at 4 MB is two command-buffer round trips (~140 µs each — see
  [CHUNKED_TRAINING.md](CHUNKED_TRAINING.md)), not a property of the reduction.
  Fusing the two passes into one command buffer would fix it. Not done: the sizes
  this feature targets are large.
- **`reduce_sum_f32` allocates scratch per call** (512 `float2` + one `float`).
  Negligible beside a multi-megabyte scan, but real at small n.
- **The MLP still uses the uncompensated `nn_reduce_sum_axis0`.** Its batch
  reductions are short, the golden-reference gate already bounds the error, and
  switching would change MLP numerics for no measured benefit. `reduce_sum_axis0_comp`
  exists and is tested if that changes.
- **Only `sum`.** No compensated `mean`, `var`, or dot product yet. `mean` is a
  division away; `var` needs a compensated two-pass or Welford formulation to be
  worth having.
