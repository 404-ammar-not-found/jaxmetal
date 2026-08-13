// Compensated (Neumaier) f32 summation on the GPU.
//
// WHY. Summing a large f32 array in the obvious way loses precision as the running
// total grows: once |sum| >> |x|, `sum + x` rounds x away entirely and the error
// grows like O(n*eps*|sum|). A pairwise/tree reduction (what nn_reduce_sum_axis0
// already does, and what numpy does on the CPU) cuts that to O(log n * eps), which
// is usually enough. Compensated summation goes further and tracks the rounding
// error explicitly, giving a result correct to ~1 ulp of the exact sum almost
// regardless of n — near-f64 accuracy while every value stays f32.
//
// This matters on Apple Silicon specifically because the GPU has no f64 at all
// (Metal has no `double` type), so "just use float64" is not available here the way
// it is on CUDA. Compensation is the only way to get trustworthy large sums on this
// hardware, and it costs ~2-4 fp32 ops per element rather than the ~10-20x of
// emulated double-double arithmetic.
//
// ---------------------------------------------------------------------------
// LOAD-BEARING: THIS FILE REQUIRES SAFE MATH (MTLMathModeSafe / fastMathEnabled=NO,
// set in src/metal/kernel_library.mm). Every compensation term below has the form
// `(a - (a + b)) + b`, which is algebraically zero and which a fast-math compiler is
// free to fold away. Under fast math these kernels silently degrade to a plain
// tree sum -- same speed, none of the accuracy, no error message. The
// ReduceCompensatedBeatsNaive test is what would catch that regression.
// ---------------------------------------------------------------------------
//
// The accumulator is a (sum, compensation) pair; the true value is sum + c, and
// only the final combine adds them. Partial sums combine associatively via
// two_sum, so this composes correctly across threads, threadgroups, and the
// two-pass structure below -- unlike a naive per-thread Kahan loop, whose
// compensation would simply be discarded when the partials are added together.

#include <metal_stdlib>
using namespace metal;

struct ReduceDims { uint n; uint stride; };   // element count, row stride (axis0)

constant constexpr uint RED_TG = 256;         // threads per threadgroup; must match
                                              // kReduceTG in src/ops/reduce.mm

// A compensated accumulator. Represents the real number (s + c) with |c| << |s|.
struct Acc { float s; float c; };

// Neumaier's variant of Kahan summation. Unlike classic Kahan it is also correct
// when |x| > |s| (the branch), which happens whenever a large element arrives late
// in a sum that started small -- exactly the case classic Kahan gets wrong.
inline void acc_add(thread Acc& a, float x) {
    float t = a.s + x;
    if (fabs(a.s) >= fabs(x))
        a.c += (a.s - t) + x;      // s is larger: recover the low bits of x
    else
        a.c += (x - t) + a.s;      // x is larger: recover the low bits of s
    a.s = t;
}

// Combine two accumulators. two_sum gives the exact rounding error of s1 + s2, so
// no information is lost when partial results meet; the compensations simply add.
inline Acc acc_merge(Acc a, Acc b) {
    float s = a.s + b.s;
    float bv = s - a.s;
    float err = (a.s - (s - bv)) + (b.s - bv);   // exact error of the addition
    Acc r;
    r.s = s;
    r.c = a.c + b.c + err;
    return r;
}

// Tree-reduce `part` (RED_TG entries) into part[0]. Every thread reaches every
// barrier; only the low half merges.
inline void tg_reduce(threadgroup Acc* part, uint lid) {
    for (uint stride = RED_TG / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) part[lid] = acc_merge(part[lid], part[lid + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// ---- pass 1: grid-stride partial sums, one (s,c) pair per threadgroup ---------
// Launched as G threadgroups x RED_TG threads. Writes partials[g] = float2(s, c).
kernel void reduce_sum_comp_partials(device const float* x       [[buffer(0)]],
                                     device float2*      partials[[buffer(1)]],
                                     constant ReduceDims& d      [[buffer(2)]],
                                     uint gid  [[thread_position_in_grid]],
                                     uint lid  [[thread_position_in_threadgroup]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint gsz  [[threads_per_grid]]) {
    threadgroup Acc part[RED_TG];

    Acc a{0.0f, 0.0f};
    for (uint i = gid; i < d.n; i += gsz) acc_add(a, x[i]);
    part[lid] = a;

    tg_reduce(part, lid);
    if (lid == 0) partials[tgid] = float2(part[0].s, part[0].c);
}

// ---- pass 2: reduce the G partials to one scalar -----------------------------
// Launched as ONE threadgroup. out[0] = the compensated total, flattened to f32.
kernel void reduce_sum_comp_finish(device const float2* partials [[buffer(0)]],
                                   device float*        out      [[buffer(1)]],
                                   constant ReduceDims& d        [[buffer(2)]],
                                   uint lid [[thread_position_in_threadgroup]]) {
    threadgroup Acc part[RED_TG];

    Acc a{0.0f, 0.0f};
    for (uint i = lid; i < d.n; i += RED_TG) {
        // Each partial is already a (sum, compensation) pair: merge, don't acc_add.
        Acc p{partials[i].x, partials[i].y};
        a = acc_merge(a, p);
    }
    part[lid] = a;

    tg_reduce(part, lid);
    if (lid == 0) out[0] = part[0].s + part[0].c;   // fold the compensation in last
}

// ---- uncompensated tree sum, same launch shape -------------------------------
// The baseline the compensated version is measured against: identical memory
// traffic and parallel structure, so a benchmark of the two isolates the cost of
// compensation rather than of the reduction strategy.
kernel void reduce_sum_tree_partials(device const float* x       [[buffer(0)]],
                                     device float*       partials[[buffer(1)]],
                                     constant ReduceDims& d      [[buffer(2)]],
                                     uint gid  [[thread_position_in_grid]],
                                     uint lid  [[thread_position_in_threadgroup]],
                                     uint tgid [[threadgroup_position_in_grid]],
                                     uint gsz  [[threads_per_grid]]) {
    threadgroup float part[RED_TG];

    float s = 0.0f;
    for (uint i = gid; i < d.n; i += gsz) s += x[i];
    part[lid] = s;

    for (uint stride = RED_TG / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) part[lid] += part[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) partials[tgid] = part[0];
}

kernel void reduce_sum_tree_finish(device const float* partials [[buffer(0)]],
                                   device float*       out      [[buffer(1)]],
                                   constant ReduceDims& d       [[buffer(2)]],
                                   uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float part[RED_TG];

    float s = 0.0f;
    for (uint i = lid; i < d.n; i += RED_TG) s += partials[i];
    part[lid] = s;

    for (uint stride = RED_TG / 2; stride > 0; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < stride) part[lid] += part[lid + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) out[0] = part[0];
}

// ---- compensated axis-0 reduction: out[j] = sum_i a[i,j] over [M,N] ----------
// The accuracy-preserving counterpart of nn_reduce_sum_axis0, for bias gradients
// and column statistics over long batches. One threadgroup per output column;
// d.n is M (rows) and d.stride is N (the row stride).
kernel void reduce_sum_axis0_comp(device const float* a   [[buffer(0)]],
                                  device float*       out [[buffer(1)]],
                                  constant ReduceDims& d  [[buffer(2)]],
                                  uint col [[threadgroup_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]]) {
    threadgroup Acc part[RED_TG];

    Acc a_acc{0.0f, 0.0f};
    for (uint i = lid; i < d.n; i += RED_TG) acc_add(a_acc, a[i * d.stride + col]);
    part[lid] = a_acc;

    tg_reduce(part, lid);
    if (lid == 0) out[col] = part[0].s + part[0].c;
}
