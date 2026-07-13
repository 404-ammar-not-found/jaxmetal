// Matrix multiply kernels: C[M,N] = A[M,K] * B[K,N], all row-major f32.
//
// Two kernels:
//   matmul_tiled   — simple 16x16 shared-memory tiling (reference / fallback).
//   matmul_blocked — classic register-tiled GEMM: 128x128 threadgroup tiles,
//                    each thread computes an 8x8 micro-tile of C in registers.
//                    This maximizes ALU utilization (Apple GPUs have no
//                    dedicated f32 matrix unit, so raw FMA throughput is the
//                    lever) and is the fast path.
//
// The host launches these as *uniform* threadgroups (see dispatch_threadgroups);
// out-of-range global indices are guarded here.

#include <metal_stdlib>
using namespace metal;

struct MatMulDims { uint M; uint N; uint K; };

// ------------------------------------------------------------------ reference
constant constexpr uint TILE = 16;

kernel void matmul_tiled(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float*       C [[buffer(2)]],
                         constant MatMulDims& d [[buffer(3)]],
                         uint2 tid [[thread_position_in_threadgroup]],
                         uint2 gid [[thread_position_in_grid]]) {
  threadgroup float As[TILE][TILE];
  threadgroup float Bs[TILE][TILE];

  const uint row = gid.y;
  const uint col = gid.x;

  float acc = 0.0f;
  const uint num_tiles = (d.K + TILE - 1) / TILE;
  for (uint t = 0; t < num_tiles; ++t) {
    const uint a_col = t * TILE + tid.x;
    const uint b_row = t * TILE + tid.y;
    As[tid.y][tid.x] = (row < d.M && a_col < d.K) ? A[row * d.K + a_col] : 0.0f;
    Bs[tid.y][tid.x] = (b_row < d.K && col < d.N) ? B[b_row * d.N + col] : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < TILE; ++k) acc += As[tid.y][k] * Bs[k][tid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (row < d.M && col < d.N) C[row * d.N + col] = acc;
}

// ------------------------------------------------------------ register-tiled
// Threadgroup computes a BM x BN output tile; the threadgroup is a TX x TY grid
// of threads and each thread owns a TM x TN micro-tile of C in registers. Per
// K-step, each thread reads TM values of A and TN of B from threadgroup memory
// and does TM*TN FMAs — TM*TN work per (TM+TN) loads, i.e. high arithmetic
// intensity. As is stored K-major-transposed so the inner reads are contiguous.
// Must stay consistent with kBlock*/kThreads* in src/ops/matmul.mm.
constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 8;
constant constexpr uint TM = 4;              // rows of C per thread
constant constexpr uint TN = 4;              // cols of C per thread
constant constexpr uint TX = BN / TN;        // 16 threads across
constant constexpr uint TY = BM / TM;        // 16 threads down
constant constexpr uint NTHREADS = TX * TY;  // 256

kernel void matmul_blocked(device const float* A [[buffer(0)]],
                           device const float* B [[buffer(1)]],
                           device float*       C [[buffer(2)]],
                           constant MatMulDims& d [[buffer(3)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           uint2 ltid [[thread_position_in_threadgroup]]) {
  threadgroup float As[BK][BM];  // transposed: As[k][m]
  threadgroup float Bs[BK][BN];  // Bs[k][n]

  const uint tg_row = tgid.y * BM;
  const uint tg_col = tgid.x * BN;
  const uint tid = ltid.y * TX + ltid.x;  // flat 0..NTHREADS-1
  const uint row0 = ltid.y * TM;          // this thread's first row in the tile
  const uint col0 = ltid.x * TN;          // ... first col

  // TM x TN accumulators, one float4 per row (TN == 4).
  float4 acc[TM];
  for (uint i = 0; i < TM; ++i) acc[i] = float4(0.0f);

  const uint ktiles = (d.K + BK - 1) / BK;
  for (uint kt = 0; kt < ktiles; ++kt) {
    const uint kbase = kt * BK;

    // Cooperative, bounds-padded loads (each thread moves BM*BK/NTHREADS elems).
    for (uint idx = tid; idx < BK * BM; idx += NTHREADS) {
      const uint k = idx / BM, m = idx % BM;
      const uint gr = tg_row + m, gc = kbase + k;
      As[k][m] = (gr < d.M && gc < d.K) ? A[gr * d.K + gc] : 0.0f;
    }
    for (uint idx = tid; idx < BK * BN; idx += NTHREADS) {
      const uint k = idx / BN, n = idx % BN;
      const uint gr = kbase + k, gc = tg_col + n;
      Bs[k][n] = (gr < d.K && gc < d.N) ? B[gr * d.N + gc] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Vectorized inner product: row0/col0 are multiples of 4, so the threadgroup
    // rows are 16-byte aligned for float4 access.
    for (uint k = 0; k < BK; ++k) {
      const float4 a4 = *(threadgroup const float4*)&As[k][row0];
      const float4 b4 = *(threadgroup const float4*)&Bs[k][col0];
      acc[0] += a4.x * b4;
      acc[1] += a4.y * b4;
      acc[2] += a4.z * b4;
      acc[3] += a4.w * b4;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = 0; i < TM; ++i) {
    const uint gr = tg_row + row0 + i;
    if (gr >= d.M) continue;
    for (uint j = 0; j < TN; ++j) {
      const uint gc = tg_col + col0 + j;
      if (gc < d.N) C[gr * d.N + gc] = acc[i][j];
    }
  }
}
