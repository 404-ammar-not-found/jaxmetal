// Batched dense solve for thousands of TINY systems: A[b]·x[b] = rhs[b], one thread
// per system, LU with partial pivoting, everything in registers.
//
// WHY THIS SHAPE. Scientific code is full of workloads that are millions of
// independent 3x3 or 6x6 solves — per-element constitutive updates, per-particle
// frames, per-voxel fits, small Kalman updates. LAPACK is built for one large matrix,
// so a batch becomes millions of library calls whose per-call overhead dwarfs the
// ~50 FLOPs of actual work. One GPU thread per system inverts that: the batch axis
// IS the parallelism, and the matrix never leaves registers.
//
// WHY ONLY n <= 8. MPSMatrixDecompositionLU and MPSMatrixSolveLU inherit
// batchStart/batchSize, so Apple already ships a batched LU — and measured, it beats
// a hand-written thread-per-system kernel from about n=12 (2.4x at n=12, 54x at
// n=32). Hand-writing those sizes would be reimplementing something slower. The
// register-resident approach wins decisively at n<=8, where MPS's per-matrix setup
// dominates its own work, so that is exactly the range shipped here.
//
// EVERY INDEX MUST BE COMPILE-TIME CONSTANT. `a` and `b` live in registers; a single
// dynamic index into them (a pivot row chosen at runtime, say) forces the whole array
// to thread-local memory and costs roughly 10x. That is why N is a template
// parameter, every loop is unrolled, and the pivot row interchange below is written
// as a sequence of *conditional* swaps with static indices rather than an indexed
// assignment.

#include <metal_stdlib>
using namespace metal;

struct BSolveDims { uint batch; };

// ---- shared implementation, instantiated per size ------------------------------
template <uint N>
inline void lu_solve_impl(device const float* A,
                          device const float* Rhs,
                          device float*       X,
                          device float*       pivmin,
                          uint sys) {
    float a[N * N];
    float b[N];

    #pragma unroll
    for (uint i = 0; i < N; ++i) {
        #pragma unroll
        for (uint j = 0; j < N; ++j) a[i * N + j] = A[(ulong)sys * N * N + i * N + j];
        b[i] = Rhs[(ulong)sys * N + i];
    }

    // Gaussian elimination with partial pivoting.
    #pragma unroll
    for (uint k = 0; k < N; ++k) {
        // Which row below k has the largest |a[i][k]|? Tracked as a value, not used
        // as an index, so nothing here indexes a register array dynamically.
        float best = fabs(a[k * N + k]);
        uint piv = k;
        #pragma unroll
        for (uint i = k + 1; i < N; ++i) {
            float v = fabs(a[i * N + k]);
            if (v > best) { best = v; piv = i; }
        }

        // Row interchange as N-k-1 CONDITIONAL swaps, all indices static. Writing
        // `a[k*N+j] = a[piv*N+j]` instead would be a dynamic register index and would
        // spill the entire array.
        #pragma unroll
        for (uint i = k + 1; i < N; ++i) {
            const bool sw = (i == piv);
            #pragma unroll
            for (uint j = 0; j < N; ++j) {
                const float t = a[k * N + j];
                if (sw) { a[k * N + j] = a[i * N + j]; a[i * N + j] = t; }
            }
            const float tb = b[k];
            if (sw) { b[k] = b[i]; b[i] = tb; }
        }

        // Eliminate. A zero pivot yields inf/NaN and propagates; it is REPORTED via
        // pivmin below rather than branched on, because an early return would
        // diverge the SIMD group on the rare bad system and slow the common case.
        const float akk = a[k * N + k];
        #pragma unroll
        for (uint i = k + 1; i < N; ++i) {
            const float f = a[i * N + k] / akk;
            #pragma unroll
            for (uint j = k + 1; j < N; ++j) a[i * N + j] -= f * a[k * N + j];
            b[i] -= f * b[k];
        }
    }

    // Back substitution.
    #pragma unroll
    for (int i = int(N) - 1; i >= 0; --i) {
        float s = b[i];
        #pragma unroll
        for (uint j = 0; j < N; ++j)
            if (j > uint(i)) s -= a[uint(i) * N + j] * b[j];
        b[i] = s / a[uint(i) * N + i];
    }

    #pragma unroll
    for (uint i = 0; i < N; ++i) X[(ulong)sys * N + i] = b[i];

    // Singularity flag: min|U_kk| / max|U_kk|, or exactly 0.0f when the system is
    // singular or the input contained NaN/Inf.
    //
    // NOT built from min()/max(): those are fmin/fmax, which DROP a NaN operand, so a
    // NaN input would sail through and report the healthiest possible value while X
    // is entirely NaN. `!(u > 0.0f)` is true for NaN, zero and negative alike, which
    // is the only form that catches all three. Branchless: two selects.
    //
    // This detects SINGULARITY and pivot scaling. It is NOT a condition estimator —
    // a well-scaled but ill-conditioned system (every |U_kk| equal, cond ~1e5) scores
    // a perfect 1.0 while its answer is wrong in the 4th digit. Threshold it for
    // "did this fail", never for "is this accurate".
    bool bad = false;
    float umin = INFINITY, umax = 0.0f;
    #pragma unroll
    for (uint k = 0; k < N; ++k) {
        const float u = fabs(a[k * N + k]);
        bad = bad || !(u > 0.0f);
        umin = (u < umin) ? u : umin;
        umax = (u > umax) ? u : umax;
    }
    pivmin[sys] = (bad || !(umax > 0.0f)) ? 0.0f : (umin / umax);
}

// ---- SPD variant: Cholesky instead of LU ---------------------------------------
//
// For symmetric positive definite systems this is strictly less work than the LU path
// above: n^3/6 for the factorisation against n^3/3, and no pivot search or row
// interchange at all -- which matters more than the FLOP count here, because the
// conditional-swap sequence is a large part of the LU kernel's instruction stream.
//
// Only the LOWER triangle of A is read. That is deliberate: it means a caller can pass
// a matrix whose upper triangle holds anything, and it is what BatchedCholeskyIgnores
// UpperTriangle pins down.
template <uint N>
inline void chol_solve_impl(device const float* A,
                            device const float* Rhs,
                            device float*       X,
                            device float*       pivmin,
                            uint sys) {
    float a[N * N];
    float b[N];

    #pragma unroll
    for (uint i = 0; i < N; ++i) {
        #pragma unroll
        for (uint j = 0; j < N; ++j)
            a[i * N + j] = (j <= i) ? A[(ulong)sys * N * N + i * N + j] : 0.0f;
        b[i] = Rhs[(ulong)sys * N + i];
    }

    // Factor A = L*L^T in place, lower triangle.
    bool bad = false;
    float dmin = INFINITY, dmax = 0.0f;
    #pragma unroll
    for (uint k = 0; k < N; ++k) {
        float d = a[k * N + k];
        #pragma unroll
        for (uint p = 0; p < N; ++p)
            if (p < k) d -= a[k * N + p] * a[k * N + p];

        // !(d > 0) is true for NaN, zero and negative alike -- the only form that
        // catches all three. min()/max() would be fmin/fmax and would DROP a NaN,
        // reporting perfect health for an all-NaN answer.
        bad = bad || !(d > 0.0f);
        dmin = (d < dmin) ? d : dmin;
        dmax = (d > dmax) ? d : dmax;
        const float lkk = sqrt(bad ? 1.0f : d);
        a[k * N + k] = lkk;

        #pragma unroll
        for (uint i = 0; i < N; ++i) {
            if (i > k) {
                float s = a[i * N + k];
                #pragma unroll
                for (uint p = 0; p < N; ++p)
                    if (p < k) s -= a[i * N + p] * a[k * N + p];
                a[i * N + k] = s / lkk;
            }
        }
    }

    // Forward: L y = b.
    #pragma unroll
    for (uint i = 0; i < N; ++i) {
        float s = b[i];
        #pragma unroll
        for (uint p = 0; p < N; ++p)
            if (p < i) s -= a[i * N + p] * b[p];
        b[i] = s / a[i * N + i];
    }
    // Backward: L^T x = y.
    #pragma unroll
    for (int i = int(N) - 1; i >= 0; --i) {
        float s = b[uint(i)];
        #pragma unroll
        for (uint p = 0; p < N; ++p)
            if (p > uint(i)) s -= a[p * N + uint(i)] * b[p];
        b[uint(i)] = s / a[uint(i) * N + uint(i)];
    }

    #pragma unroll
    for (uint i = 0; i < N; ++i) X[(ulong)sys * N + i] = b[i];

    // Same contract as the LU path: ratio of extreme pivots, or exactly 0 for a
    // non-SPD or NaN/Inf input. Accumulated pre-sqrt so a negative pivot is visible.
    pivmin[sys] = (bad || !(dmax > 0.0f)) ? 0.0f : (dmin / dmax);
}

#define DEFINE_BATCHED_CHOL(N)                                               \
kernel void batched_chol_##N(device const float* A      [[buffer(0)]],       \
                             device const float* Rhs    [[buffer(1)]],       \
                             device float*       X      [[buffer(2)]],       \
                             device float*       pivmin [[buffer(3)]],       \
                             constant BSolveDims& d     [[buffer(4)]],       \
                             uint gid [[thread_position_in_grid]]) {         \
    if (gid >= d.batch) return;                                              \
    chol_solve_impl<N>(A, Rhs, X, pivmin, gid);                              \
}

DEFINE_BATCHED_CHOL(2)
DEFINE_BATCHED_CHOL(3)
DEFINE_BATCHED_CHOL(4)
DEFINE_BATCHED_CHOL(5)
DEFINE_BATCHED_CHOL(6)
DEFINE_BATCHED_CHOL(7)
DEFINE_BATCHED_CHOL(8)

#define DEFINE_BATCHED_SOLVE(N)                                              \
kernel void batched_solve_##N(device const float* A      [[buffer(0)]],      \
                              device const float* Rhs    [[buffer(1)]],      \
                              device float*       X      [[buffer(2)]],      \
                              device float*       pivmin [[buffer(3)]],      \
                              constant BSolveDims& d     [[buffer(4)]],      \
                              uint gid [[thread_position_in_grid]]) {        \
    if (gid >= d.batch) return;                                              \
    lu_solve_impl<N>(A, Rhs, X, pivmin, gid);                                \
}

DEFINE_BATCHED_SOLVE(2)
DEFINE_BATCHED_SOLVE(3)
DEFINE_BATCHED_SOLVE(4)
DEFINE_BATCHED_SOLVE(5)
DEFINE_BATCHED_SOLVE(6)
DEFINE_BATCHED_SOLVE(7)
DEFINE_BATCHED_SOLVE(8)
