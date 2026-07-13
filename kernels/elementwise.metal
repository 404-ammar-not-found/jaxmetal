// Hand-written Metal Shading Language (MSL) elementwise arithmetic kernels.
//
// These are compiled at runtime (newLibraryWithSource:) by KernelLibrary and
// embedded into the binary as a string by cmake/EmbedMetal.cmake. All compute
// is f32 for now (see CLAUDE.md). Buffer binding convention:
//   binary op:  a=[[buffer(0)]] b=[[buffer(1)]] out=[[buffer(2)]] n=[[buffer(3)]]
//   unary op:   a=[[buffer(0)]] out=[[buffer(1)]] n=[[buffer(2)]]
// The trailing `n` is the element count, passed as a push constant so kernels
// can guard against over-dispatch.

#include <metal_stdlib>
using namespace metal;

// ---- binary ----
kernel void ew_add(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] + b[gid];
}

kernel void ew_sub(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] - b[gid];
}

kernel void ew_mul(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] * b[gid];
}

kernel void ew_div(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] / b[gid];
}

kernel void ew_max(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = max(a[gid], b[gid]);
}

kernel void ew_min(device const float* a   [[buffer(0)]],
                   device const float* b   [[buffer(1)]],
                   device float*       out [[buffer(2)]],
                   constant uint&      n   [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = min(a[gid], b[gid]);
}

// ---- unary ----
kernel void ew_neg(device const float* a   [[buffer(0)]],
                   device float*       out [[buffer(1)]],
                   constant uint&      n   [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = -a[gid];
}

kernel void ew_abs(device const float* a   [[buffer(0)]],
                   device float*       out [[buffer(1)]],
                   constant uint&      n   [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = fabs(a[gid]);
}

kernel void ew_exp(device const float* a   [[buffer(0)]],
                   device float*       out [[buffer(1)]],
                   constant uint&      n   [[buffer(2)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = exp(a[gid]);
}
