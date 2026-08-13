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
// NB IS CAPPED BY THREADGROUP MEMORY, NOT BY GEMM EFFICIENCY. The panel kernels hold
// the NB×NB diagonal block in threadgroup memory: NB=64 is 16 KB, and Apple GPUs
// give 32 KB per threadgroup. NB=128 would need 64 KB and does not fit, which is why
// NB is 64 even though larger blocks would make the trailing GEMM more efficient.
//
// Only the lower triangle is read and written. The strictly upper triangle of the
// result is left exactly as the caller supplied it — callers that want a clean L
// must zero it themselves (chol_zero_upper does this).

#include <metal_stdlib>
using namespace metal;

struct CholDims {
    uint n;    // full matrix order (row stride, elements)
    uint k;    // first row/col of the current block
    uint nb;   // block size (<= CHOL_NB)
    uint m;    // rows below the diagonal block (0 for the last block)
};

constant constexpr uint CHOL_NB = 64;   // must match kCholNB in src/ops/cholesky.mm
constant constexpr uint CHOL_TG = 256;  // threads per threadgroup for the panel kernels

// ---- 1. factor the NB x NB diagonal block, in place, with ONE threadgroup --------
//
// Unblocked right-looking Cholesky over the block held in threadgroup memory. The
// column loop is inherently sequential (column j needs every column before it), so
// there is a barrier per column: NB barriers over O(NB^3/6) work. For NB=64 that is
// ~44 kFLOP per panel and ~2.8 MFLOP over a whole N=4096 factorisation — utterly
// negligible beside the 46 GFLOP of trailing updates, which is exactly why it is
// acceptable to leave this part under-parallelised.
kernel void chol_panel(device float*       A [[buffer(0)]],
                       constant CholDims&  d [[buffer(1)]],
                       device uint*        status [[buffer(2)]],
                       uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float T[CHOL_NB * CHOL_NB];
    const uint nb = d.nb;
    device float* A11 = A + (ulong)d.k * d.n + d.k;

    // Load the block (lower triangle is all we touch; load it all for simplicity).
    for (uint idx = lid; idx < nb * nb; idx += CHOL_TG)
        T[idx] = A11[(idx / nb) * (ulong)d.n + (idx % nb)];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < nb; ++j) {
        // Diagonal: L[j][j] = sqrt(A[j][j] - sum_{p<j} L[j][p]^2).
        if (lid == 0) {
            float s = T[j * nb + j];
            for (uint p = 0; p < j; ++p) s -= T[j * nb + p] * T[j * nb + p];
            // Not positive definite (or NaN — note the !(s > 0) form, which catches
            // NaN where s <= 0 would not). Record the FIRST failing column, 1-based,
            // LAPACK `info` convention. Only lid==0 writes and panels run sequentially
            // in the command buffer, so the read-then-write needs no atomic. Substitute
            // a benign pivot so the trailing GEMM cannot fill the matrix with NaN and
            // destroy the evidence of where the failure actually started.
            if (!(s > 0.0f)) {
                if (status[0] == 0u) status[0] = d.k + j + 1u;
                s = 1.0f;
            }
            T[j * nb + j] = sqrt(s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Column below the diagonal: L[i][j] = (A[i][j] - sum_{p<j} L[i][p]L[j][p]) / L[j][j].
        const float djj = T[j * nb + j];
        for (uint i = j + 1 + lid; i < nb; i += CHOL_TG) {
            float s = T[i * nb + j];
            for (uint p = 0; p < j; ++p) s -= T[i * nb + p] * T[j * nb + p];
            T[i * nb + j] = s / djj;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store back the lower triangle only.
    for (uint idx = lid; idx < nb * nb; idx += CHOL_TG) {
        uint i = idx / nb, j = idx % nb;
        if (i >= j) A11[i * (ulong)d.n + j] = T[idx];
    }
}

// ---- 2. off-diagonal panel: solve L21 · L11^T = A21 for L21 ----------------------
//
// One thread per row of A21, so the m rows are fully independent — this is where the
// panel-phase parallelism lives (m is up to N-NB). L11 is shared by every row, so it
// is staged once into threadgroup memory and read from there.
kernel void chol_trsm_right(device float*      A [[buffer(0)]],
                            constant CholDims& d [[buffer(1)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float L11[CHOL_NB * CHOL_NB];
    const uint nb = d.nb;
    device const float* A11 = A + (ulong)d.k * d.n + d.k;

    for (uint idx = lid; idx < nb * nb; idx += CHOL_TG) {
        uint i = idx / nb, j = idx % nb;
        L11[idx] = (i >= j) ? A11[i * (ulong)d.n + j] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid >= d.m) return;
    device float* row = A + (ulong)(d.k + nb + gid) * d.n + d.k;

    // Forward substitution along the row: row[j] = (row[j] - sum_{p<j} row[p]·L11[j][p]) / L11[j][j]
    for (uint j = 0; j < nb; ++j) {
        float s = row[j];
        for (uint p = 0; p < j; ++p) s -= row[p] * L11[j * nb + p];
        row[j] = s / L11[j * nb + j];
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
