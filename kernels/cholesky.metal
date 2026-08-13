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
// THE PANEL IS THE LARGEST PHASE, AND IT RESISTS OPTIMISATION. Measured with GPU
// timestamps at N=4096: panel 27.1 ms, TRSM 15.2 ms, trailing GEMM 26.1 ms. It does
// only ~2.8 MMAC over the whole factorisation, i.e. ~0.2 GFLOP/s, so it is entirely
// latency: 64 columns x 64 blocks = 4096 sequential steps, each with a serial section
// (the diagonal) and a barrier. That chain is Cholesky's inherent dependency.
//
// Tried and measured, none of which helped (all in the feature doc):
//   * threadgroup staging of the block: panel 28.9 ms (no better; and it caps NB=64)
//   * one SIMD group with simdgroup_barrier instead of 256 threads with
//     threadgroup_barrier: panel 32.1 ms (lost column parallelism costs more than the
//     cheaper barrier saves)
//
// Left operating directly on device memory: simplest, marginally fastest, and it does
// not cap NB, which keeps JAXMETAL_CHOL_NB able to sweep.
kernel void chol_panel(device float*       A [[buffer(0)]],
                       constant CholDims&  d [[buffer(1)]],
                       device uint*        status [[buffer(2)]],
                       uint lid [[thread_position_in_threadgroup]]) {
    const uint nb = d.nb, n = d.n;
    device float* A11 = A + (ulong)d.k * n + d.k;

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
constant constexpr uint CHOL_NB = 64;   // fast-path block width; matches kCholNB

kernel void chol_trsm_right(device float*      A [[buffer(0)]],
                            constant CholDims& d [[buffer(1)]],
                            uint gid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float L11[CHOL_NB * CHOL_NB];
    const uint nb = d.nb, n = d.n;
    device const float* src = A + (ulong)d.k * n + d.k;

    // L11 is read by every one of the m rows, so stage it once per threadgroup.
    if (nb == CHOL_NB) {
        for (uint idx = lid; idx < CHOL_NB * CHOL_NB; idx += CHOL_TG) {
            uint i = idx / CHOL_NB, j = idx % CHOL_NB;
            L11[idx] = (i >= j) ? src[(ulong)i * n + j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid >= d.m) return;
    device float* row = A + (ulong)(d.k + nb + gid) * n + d.k;

    // FAST PATH: hold this thread's row in REGISTERS across the whole substitution.
    // The obvious version keeps the row in device memory and reads row[p] inside the
    // inner loop, which makes every one of the nb^2/2 inner iterations a dependent
    // device load. That version measured 31.6 ms of a 46.6 ms N=4096 factorisation --
    // 68% of total runtime, against ~0.5% for the trailing GEMM. nb must be a
    // COMPILE-TIME constant and both loops fully unrolled, or `r[p]` becomes a dynamic
    // index into a register array and spills straight back to thread-local memory.
    if (nb == CHOL_NB) {
        float r[CHOL_NB];
        #pragma unroll
        for (uint j = 0; j < CHOL_NB; ++j) r[j] = row[j];

        // Single accumulator. A four-way split was tried to break the dependent FMA
        // chain (safe math forbids the compiler reassociating it) and measured NO
        // improvement: 29.13 -> 29.47 ms. The chain is not what binds this loop; at 64
        // floats per thread `r` is almost certainly spilling, which is what a real fix
        // has to address. Reverted rather than left in as unearned complexity.
        #pragma unroll
        for (uint j = 0; j < CHOL_NB; ++j) {
            float s = r[j];
            #pragma unroll
            for (uint p = 0; p < CHOL_NB; ++p)
                if (p < j) s -= r[p] * L11[j * CHOL_NB + p];
            r[j] = s / L11[j * CHOL_NB + j];
        }

        #pragma unroll
        for (uint j = 0; j < CHOL_NB; ++j) row[j] = r[j];
        return;
    }

    // Ragged final block (nb < CHOL_NB): scalar fallback straight out of device
    // memory. It runs once per factorisation on at most CHOL_NB-1 columns.
    for (uint j = 0; j < nb; ++j) {
        float s = row[j];
        for (uint p = 0; p < j; ++p) s -= row[p] * src[(ulong)j * n + p];
        row[j] = s / src[(ulong)j * n + j];
    }
}

// ---- 2b. chunked variant: fewer registers, higher occupancy ---------------------
//
// The full-row variant above holds all CHOL_NB=64 floats of a row in registers. That
// is ~64 registers per thread before anything else, which on Apple GPUs collapses
// occupancy -- and occupancy, not arithmetic, is what limits this kernel: it does only
// ~273 MMAC in total across an N=4096 factorisation yet runs at ~19 GFLOP/s against a
// ~5000 GFLOP/s machine.
//
// This variant keeps only CHOL_CH columns live at a time. Columns already solved are
// re-read from device memory (this thread's own row, 256 contiguous bytes, so it stays
// in cache). Trades a little extra traffic for a lot more threads in flight.
constant constexpr uint CHOL_CH = 16;

kernel void chol_trsm_right_chunked(device float*      A [[buffer(0)]],
                                    constant CholDims& d [[buffer(1)]],
                                    uint gid [[thread_position_in_grid]],
                                    uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float L11[CHOL_NB * CHOL_NB];
    const uint nb = d.nb, n = d.n;
    device const float* src = A + (ulong)d.k * n + d.k;

    if (nb == CHOL_NB) {
        for (uint idx = lid; idx < CHOL_NB * CHOL_NB; idx += CHOL_TG) {
            uint i = idx / CHOL_NB, j = idx % CHOL_NB;
            L11[idx] = (i >= j) ? src[(ulong)i * n + j] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid >= d.m) return;
    device float* row = A + (ulong)(d.k + nb + gid) * n + d.k;

    if (nb != CHOL_NB) {
        for (uint j = 0; j < nb; ++j) {
            float s = row[j];
            for (uint p = 0; p < j; ++p) s -= row[p] * src[(ulong)j * n + p];
            row[j] = s / src[(ulong)j * n + j];
        }
        return;
    }

    #pragma unroll
    for (uint jb = 0; jb < CHOL_NB; jb += CHOL_CH) {
        float r[CHOL_CH];
        #pragma unroll
        for (uint t = 0; t < CHOL_CH; ++t) r[t] = row[jb + t];

        #pragma unroll
        for (uint t = 0; t < CHOL_CH; ++t) {
            const uint j = jb + t;
            float s = r[t];
            for (uint p = 0; p < jb; ++p) s -= row[p] * L11[j * CHOL_NB + p];
            #pragma unroll
            for (uint u = 0; u < CHOL_CH; ++u)
                if (u < t) s -= r[u] * L11[j * CHOL_NB + jb + u];
            r[t] = s / L11[j * CHOL_NB + j];
        }

        #pragma unroll
        for (uint t = 0; t < CHOL_CH; ++t) row[jb + t] = r[t];
    }
}

// ---- 2c. invert the diagonal block, so the panel solve becomes a GEMM -----------
//
// THE POINT. The panel solve L21 = A21 * L11^-T is 65% of this factorisation and runs
// at ~19 GFLOP/s on a ~5000 GFLOP/s machine. Five attempts to make the solve itself
// faster all measured flat or worse (see the feature doc). So instead: form L11^-1
// explicitly, once per block step, and the solve becomes L21 = A21 * (L11^-1)^T --
// a pure GEMM, which the phase breakdown shows costs essentially nothing here.
//
// The inverse costs O(nb^3/6) ~ 44 kMAC per step and ~2.8 MMAC over an N=4096
// factorisation, against 273 MMAC for the solve it replaces.
//
// NUMERICAL NOTE: explicit inversion is less backward-stable than a triangular solve.
// It is acceptable here for the same reason MAGMA does it -- L11 is the Cholesky
// factor of a diagonal block of an SPD matrix, so it is well conditioned -- but it is
// a real trade, and CholeskyMatchesLapack's residual check is what bounds it.
//
// One thread per COLUMN of the inverse: column j is the solution of L*x = e_j, found
// by forward substitution. Columns are independent.
kernel void chol_invert_panel(device const float* A    [[buffer(0)]],
                              device float*       Linv [[buffer(1)]],
                              constant CholDims&  d    [[buffer(2)]],
                              uint lid [[thread_position_in_threadgroup]]) {
    threadgroup float L[CHOL_NB * CHOL_NB];
    const uint nb = d.nb, n = d.n;
    device const float* src = A + (ulong)d.k * n + d.k;

    for (uint idx = lid; idx < nb * nb; idx += CHOL_TG) {
        uint i = idx / nb, j = idx % nb;
        L[idx] = (i >= j) ? src[(ulong)i * n + j] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Zero the whole tile first: the strict upper triangle of Linv must be 0 for the
    // GEMM that consumes it to be correct.
    for (uint idx = lid; idx < nb * nb; idx += CHOL_TG) Linv[idx] = 0.0f;
    threadgroup_barrier(mem_flags::mem_device);

    if (lid >= nb) return;
    const uint j = lid;
    // x = L^-1 e_j : x[i] = 0 for i<j, then forward substitution.
    for (uint i = j; i < nb; ++i) {
        float s = (i == j) ? 1.0f : 0.0f;
        for (uint p = j; p < i; ++p) s -= L[i * nb + p] * Linv[p * nb + j];
        Linv[i * nb + j] = s / L[i * nb + i];
    }
}

// out[m, nb] -> A[k+nb.., k..], copying the GEMM result back into the matrix.
kernel void chol_copy_panel(device const float* src [[buffer(0)]],
                            device float*       A   [[buffer(1)]],
                            constant CholDims&  d   [[buffer(2)]],
                            uint gid [[thread_position_in_grid]]) {
    const uint nb = d.nb, n = d.n;
    if (gid >= d.m * nb) return;
    const uint i = gid / nb, j = gid % nb;
    A[(ulong)(d.k + nb + i) * n + d.k + j] = src[gid];
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
