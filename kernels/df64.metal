// Double-single (df64) arithmetic: extended precision on a GPU that has no f64 at all.
//
// WHAT THIS IS FOR, AND WHAT IT IS NOT FOR.
//
// Metal Shading Language has no `double` type and Apple GPUs have no fp64 units, so
// "just accumulate in float64" — the standard answer to precision loss — is simply
// unavailable on this hardware. df64 represents a number as an unevaluated sum of two
// f32 limbs (hi + lo, with |lo| <= ulp(hi)/2), giving ~48 bits of significand against
// f32's 24 and f64's 53.
//
// THIS IS A PRECISION FEATURE, NOT A PERFORMANCE FEATURE. It is SLOWER than doing the
// same work in f64 on the CPU, at every size measured. The original rationale for it
// — that the GPU's higher memory bandwidth would pay for the emulation on
// bandwidth-bound kernels — was WRONG, because Apple Silicon is unified memory: the
// CPU and GPU share one memory controller, so there is no bandwidth advantage to
// exploit. See docs/features/DF64_PRECISION.md for the numbers. It exists so that a
// pipeline already resident on the GPU can take a few high-precision steps without
// round-tripping to the CPU, and because ~48 bits beats 24 when that is what you need.
//
// ACCURACY CLAIM, STATED CAREFULLY: ~48 bits of significand. This is NOT IEEE
// float64. The exponent range is f32's (so it overflows around 3.4e38, not 1.8e308),
// there is no guaranteed correct rounding, and denormal handling is weaker.
//
// ---------------------------------------------------------------------------
// LOAD-BEARING: REQUIRES SAFE MATH (MTLMathModeSafe / fastMathEnabled=NO, set in
// src/metal/kernel_library.mm). Every error-free transformation below computes a
// quantity that is algebraically zero — `(a - (a + b)) + b` and `fma(a, b, -a*b)` —
// and a fast-math compiler is entitled to fold each to 0. Under fast math every
// operation here silently degrades to plain f32: same speed, none of the precision,
// no error. DF64AddBeatsFloat32 is the regression test.
// ---------------------------------------------------------------------------

#include <metal_stdlib>
using namespace metal;

struct DFDims { uint n; };

// A df64 value: the real number is exactly hi + lo. Stored as float2 so host arrays
// stay a single contiguous buffer and each thread's load is one 8-byte access.
struct df64 { float hi; float lo; };

// ---- error-free transformations ------------------------------------------------

// Exact sum of two floats: s = a+b rounded, e = the part that did not fit.
// Requires |a| >= |b| (hence "quick"); the caller must guarantee it.
inline float quick_two_sum(float a, float b, thread float& e) {
    float s = a + b;
    e = b - (s - a);
    return s;
}

// Exact sum with no ordering requirement. Two more operations than quick_two_sum.
inline float two_sum(float a, float b, thread float& e) {
    float s = a + b;
    float bb = s - a;
    e = (a - (s - bb)) + (b - bb);
    return s;
}

// Exact product: p = a*b rounded, e = the rounding error, recovered exactly by fma.
// This is why fma() matters here beyond speed — without a fused multiply-add there is
// no cheap way to get the low half of a product.
inline float two_prod(float a, float b, thread float& e) {
    float p = a * b;
    e = fma(a, b, -p);
    return p;
}

// ---- df64 arithmetic ------------------------------------------------------------

inline df64 df_add(df64 a, df64 b) {
    float s2, t2, t1;
    float s1 = two_sum(a.hi, b.hi, s2);
    t1 = two_sum(a.lo, b.lo, t2);
    s2 += t1;
    s1 = quick_two_sum(s1, s2, s2);
    s2 += t2;
    s1 = quick_two_sum(s1, s2, s2);
    return df64{s1, s2};
}

inline df64 df_neg(df64 a) { return df64{-a.hi, -a.lo}; }
inline df64 df_sub(df64 a, df64 b) { return df_add(a, df_neg(b)); }

inline df64 df_mul(df64 a, df64 b) {
    float p2;
    float p1 = two_prod(a.hi, b.hi, p2);
    // The cross terms are ~ulp of p1, so a plain f32 sum of them is enough.
    p2 += a.hi * b.lo + a.lo * b.hi;
    p1 = quick_two_sum(p1, p2, p2);
    return df64{p1, p2};
}

// Newton refinement from an f32 reciprocal: one step doubles the correct digits,
// which is exactly what is needed to go from 24 bits to ~48.
inline df64 df_div(df64 a, df64 b) {
    float q1 = a.hi / b.hi;
    df64 r = df_sub(a, df_mul(df64{q1, 0.0f}, b));
    float q2 = r.hi / b.hi;
    float e;
    float s = quick_two_sum(q1, q2, e);
    return df64{s, e};
}

inline df64 df_from_f32(float x) { return df64{x, 0.0f}; }

// ---- kernels --------------------------------------------------------------------

kernel void df64_add(device const float2* a [[buffer(0)]],
                     device const float2* b [[buffer(1)]],
                     device float2*       o [[buffer(2)]],
                     constant DFDims&     d [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    float2 av = a[gid], bv = b[gid];
    df64 r = df_add(df64{av.x, av.y}, df64{bv.x, bv.y});
    o[gid] = float2(r.hi, r.lo);
}

kernel void df64_mul(device const float2* a [[buffer(0)]],
                     device const float2* b [[buffer(1)]],
                     device float2*       o [[buffer(2)]],
                     constant DFDims&     d [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    float2 av = a[gid], bv = b[gid];
    df64 r = df_mul(df64{av.x, av.y}, df64{bv.x, bv.y});
    o[gid] = float2(r.hi, r.lo);
}

kernel void df64_div(device const float2* a [[buffer(0)]],
                     device const float2* b [[buffer(1)]],
                     device float2*       o [[buffer(2)]],
                     constant DFDims&     d [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    float2 av = a[gid], bv = b[gid];
    df64 r = df_div(df64{av.x, av.y}, df64{bv.x, bv.y});
    o[gid] = float2(r.hi, r.lo);
}

// 3-point stencil out[i] = c0*x[i-1] + c1*x[i] + c2*x[i+1], the representative
// bandwidth-bound PDE kernel. Dirichlet (zero) boundaries.
kernel void df64_stencil3(device const float2* x  [[buffer(0)]],
                          device float2*       o  [[buffer(1)]],
                          device const float2* c  [[buffer(2)]],   // 3 coefficients
                          constant DFDims&     d  [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    const df64 c0{c[0].x, c[0].y}, c1{c[1].x, c[1].y}, c2{c[2].x, c[2].y};
    const df64 zero{0.0f, 0.0f};

    float2 lv = (gid > 0) ? x[gid - 1] : float2(0.0f);
    float2 mv = x[gid];
    float2 rv = (gid + 1 < d.n) ? x[gid + 1] : float2(0.0f);

    df64 acc = df_mul(c1, df64{mv.x, mv.y});
    acc = df_add(acc, (gid > 0) ? df_mul(c0, df64{lv.x, lv.y}) : zero);
    acc = df_add(acc, (gid + 1 < d.n) ? df_mul(c2, df64{rv.x, rv.y}) : zero);
    o[gid] = float2(acc.hi, acc.lo);
}

// Same stencil in plain f32, for the accuracy comparison. Identical memory access
// pattern per element so the benchmark isolates arithmetic, not layout.
kernel void f32_stencil3(device const float* x [[buffer(0)]],
                         device float*       o [[buffer(1)]],
                         device const float* c [[buffer(2)]],
                         constant DFDims&    d [[buffer(3)]],
                         uint gid [[thread_position_in_grid]]) {
    if (gid >= d.n) return;
    float l = (gid > 0) ? x[gid - 1] : 0.0f;
    float r = (gid + 1 < d.n) ? x[gid + 1] : 0.0f;
    o[gid] = c[0] * l + c[1] * x[gid] + c[2] * r;
}
