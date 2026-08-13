// Blocked right-looking Cholesky factorisation: A = L·Lᵀ, lower triangular, f32.
//
// WHY HAND-WRITTEN. MPS ships MPSMatrixDecompositionCholesky, and it is unusable:
// measured on an M4 Pro it takes 620 ms at N=4096 against LAPACK's 45 ms — 14x
// SLOWER than the CPU, and the gap is worse at small N (0.01x at N=512). Its timing
// scales like one dispatch per column. So "just call MPS" is not an option here, and
// jax-mps punting all decompositions to the CPU was the correct call rather than a
// missed opportunity.
//
// THE STRATEGY. Only the O(NB³) panel work is hand-written; the O(N³) trailing
// update is an MPS GEMM, which is where essentially all the FLOPs are and where the
// GPU actually wins (a rank-256 update at M=4096 measured 3297 GFLOP/s). Per block
// step k:
//
//   1. chol_panel      factor the NB×NB diagonal block   A11 = L11·L11ᵀ
//   2. chol_trsm_right solve for the off-diagonal panel  L21 = A21·L11⁻ᵀ
//   3. (host, MPS)     trailing update                   A22 -= L21·L21ᵀ
//
// Every step of every iteration is encoded into ONE command buffer, so the ~140 us
// driver round trip is paid once for the whole factorisation rather than once per
// block step — the same lesson as docs/features/CHUNKED_TRAINING.md. Correctness
// across steps relies on Metal's intra-command-buffer hazard tracking.
//
// NB=64 IS A MEASURED OPTIMUM, AND THE PANEL PHASES ARE THE BOTTLENECK. Larger NB
// makes the trailing GEMM more efficient (rank-64 at M=4096 runs at 1985 GFLOP/s,
// rank-256 at 3297), so raising it looks like an obvious win. Measured at N=4096 it is
// the opposite: NB=64 -> 46 ms, NB=128 -> 64 ms, NB=256 -> 111 ms, NB=512 -> 228 ms.
// That inversion is the diagnosis — if the GEMM dominated, bigger NB would help. It
// does not, because chol_panel is ONE threadgroup doing O(NB^3) work and
// chol_trsm_right is one thread per row doing O(NB^2) SEQUENTIAL work, both of which
// grow faster than the GEMM saving. Sweep it yourself with JAXMETAL_CHOL_NB.
//
// The real fix is a blocked TRSM that expresses the panel solve as GEMMs too, i.e.
// genuine two-level blocking. That is a different algorithm, not a tuning constant.
//
// Only the lower triangle is read and written. The strictly upper triangle of the
// result is left exactly as the caller supplied it — callers that want a clean L
// must zero it themselves (chol_zero_upper does this).

#include <metal_stdlib>
using namespace metal;

struct CholDims {
    uint n;    // full matrix order (row stride, elements)
    uint k;    // first row/col of the current block
    uint nb;   // block size for this step (host-chosen; see kCholNB)
    uint m;    // rows below the diagonal block (0 for the last block)
};

constant constexpr uint CHOL_TG = 256;  // threads per threadgroup for the panel kernels

// ---- 1. factor the NB x NB diagonal block, in place, with ONE threadgroup --------
//
// Unblocked right-looking Cholesky on one threadgroup. The column loop is inherently
// sequential (column j needs every column before it), so there is a barrier per
// column: NB barriers over O(NB^3/6) work. At NB=64 that is ~44 kFLOP per panel, and
// the NB sweep above shows this phase — not the trailing GEMM — is what stops NB from
// being raised.
kernel void chol_panel(device float*       A [[buffer(0)]],
                       constant CholDims&  d [[buffer(1)]],
                       device uint*        status [[buffer(2)]],
                       uint lid [[thread_position_in_threadgroup]]) {
    const uint nb = d.nb, n = d.n;
    device float* A11 = A + (ulong)d.k * n + d.k;

    // Operates directly on device memory rather than staging the block in threadgroup
    // memory. Staging is faster per panel, but it caps NB at 64 (64x64 f32 = 16 KB
    // against Apple's 32 KB threadgroup limit) and NB turns out to be the dominant
    // performance lever: rank-64 trailing GEMM runs at 502 GFLOP/s where rank-256 runs
    // at 3297. Panel work is O(NB^3) against O(N^3) of trailing update, so paying more
    // here to allow a larger NB is the right trade.
    for (uint j = 0; j < nb; ++j) {
        if (lid == 0) {
            float s = A11[(ulong)j * n + j];
            for (uint p = 0; p < j; ++p) {
                float v = A11[(ulong)j * n + p];
                s -= v * v;
            }
            // !(s > 0) rather than s <= 0: the latter is FALSE for NaN and would let a
            // NaN input through as a successful factorisation. Record the first failing
            // column (LAPACK `info`, 1-based) and substitute a benign pivot so the
            // trailing GEMM cannot fill the matrix with NaN and destroy the evidence.
            if (!(s > 0.0f)) {
                if (status[0] == 0u) status[0] = d.k + j + 1u;
                s = 1.0f;
            }
            A11[(ulong)j * n + j] = sqrt(s);
        }
        threadgroup_barrier(mem_flags::mem_device);

        const float djj = A11[(ulong)j * n + j];
        for (uint i = j + 1 + lid; i < nb; i += CHOL_TG) {
            float s = A11[(ulong)i * n + j];
            for (uint p = 0; p < j; ++p)
                s -= A11[(ulong)i * n + p] * A11[(ulong)j * n + p];
            A11[(ulong)i * n + j] = s / djj;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// ---- 2. off-diagonal panel: solve L21 · L11^T = A21 for L21 ----------------------
//
// One thread per row of A21, so the m rows are fully independent — this is where the
// panel-phase parallelism lives (m is up to N-NB). L11 is shared by every row, so it
// is read straight from device memory and served by cache.
kernel void chol_trsm_right(device float*      A [[buffer(0)]],
                            constant CholDims& d [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= d.m) return;
    const uint nb = d.nb, n = d.n;
    device const float* L11 = A + (ulong)d.k * n + d.k;
    device float* row = A + (ulong)(d.k + nb + gid) * n + d.k;

    // Forward substitution along the row. Every thread reads the same L11, so it is
    // left in device memory and served by cache rather than staged per threadgroup —
    // which also removes the NB cap that staging would impose.
    for (uint j = 0; j < nb; ++j) {
        float s = row[j];
        for (uint p = 0; p < j; ++p) s -= row[p] * L11[(ulong)j * n + p];
        row[j] = s / L11[(ulong)j * n + j];
    }
}

// ---- zero the strictly upper triangle, so the result is a clean L ---------------
kernel void chol_zero_upper(device float*      A [[buffer(0)]],
                            constant CholDims& d [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint n = d.n;
    if (gid >= n * n) return;
    uint i = gid / n, j = gid % n;
    if (j > i) A[gid] = 0.0f;
}

// ---- triangular solves against an already-computed L ----------------------------
//
// Both solve for all `nrhs` right-hand sides at once, one thread per column of B, so
// the parallelism is nrhs-wide. Each thread walks the triangle sequentially, which is
// the inherent dependency of a triangular solve.
//
// B is [n, nrhs] row-major with row stride `nrhs` (packed).

// Forward: solve L·X = B (lower, non-transposed).
kernel void chol_solve_fwd(device const float* L [[buffer(0)]],
                           device float*       B [[buffer(1)]],
                           constant CholDims&  d [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint n = d.n, nrhs = d.nb;   // nb field carries nrhs here
    if (gid >= nrhs) return;
    for (uint i = 0; i < n; ++i) {
        float s = B[(ulong)i * nrhs + gid];
        for (uint p = 0; p < i; ++p) s -= L[(ulong)i * n + p] * B[(ulong)p * nrhs + gid];
        B[(ulong)i * nrhs + gid] = s / L[(ulong)i * n + i];
    }
}

// Backward: solve L^T·X = B (lower stored, transposed).
kernel void chol_solve_bwd(device const float* L [[buffer(0)]],
                           device float*       B [[buffer(1)]],
                           constant CholDims&  d [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
    const uint n = d.n, nrhs = d.nb;
    if (gid >= nrhs) return;
    for (int i = int(n) - 1; i >= 0; --i) {
        float s = B[(ulong)i * nrhs + gid];
        for (uint p = uint(i) + 1; p < n; ++p) s -= L[(ulong)p * n + i] * B[(ulong)p * nrhs + gid];
        B[(ulong)i * nrhs + gid] = s / L[(ulong)i * n + i];
    }
}
