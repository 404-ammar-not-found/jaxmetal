// Tests for the neural-net kernels (bias_add, relu, relu_grad, reduce_sum_axis0,
// transpose2d, sgd_update, softmax_xent, argmax) vs double-precision CPU
// references, plus a full forward+backward+SGD integration step that mirrors the
// exact op chaining the resident MLP uses.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"
#include "jaxmetal/metal/kernel_library.h"
#include "jaxmetal/runtime/dispatcher.h"
#include "jaxmetal/ops/nn.h"
#include "jaxmetal/ops/matmul.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

// nn + matmul kernels registered once into the shared test library.
KernelLibrary& nn_lib() {
  KernelLibrary& l = testutil::lib();
  static int once = (register_nn_kernels(l), register_matmul_kernel(l), 0);
  (void)once;
  return l;
}

std::vector<float> randf(int64_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

const float* host(const std::shared_ptr<MetalBuffer>& b) {
  return static_cast<const float*>(b->contents());
}

}  // namespace

TEST(NNBiasAddExact) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t M = 3, N = 4;
  std::vector<float> a(M * N), bias(N);
  for (int64_t i = 0; i < M * N; ++i) a[i] = (float)i;
  for (int64_t j = 0; j < N; ++j) bias[j] = (float)(10 * (j + 1));
  auto da = c.from_host(a.data(), {M, N}, DType::F32);
  auto db = c.from_host(bias.data(), {N}, DType::F32);
  auto out = c.alloc({M, N}, DType::F32);
  bias_add_into(nn_lib(), disp, *da, *db, *out, M, N);
  disp.wait();
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      CHECK_NEAR(host(out)[i * N + j], a[i * N + j] + bias[j], 1e-6);
}

TEST(NNReluExact) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  std::vector<float> a = {-2.0f, -0.0f, 0.0f, 1.5f, -1e9f, 3.0f};
  auto da = c.from_host(a.data(), {(int64_t)a.size()}, DType::F32);
  auto out = c.alloc({(int64_t)a.size()}, DType::F32);
  relu_into(nn_lib(), disp, *da, *out, (int64_t)a.size());
  disp.wait();
  for (size_t i = 0; i < a.size(); ++i) CHECK_NEAR(host(out)[i], std::fmax(a[i], 0.0f), 1e-6);
}

TEST(NNReluGradExact) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  std::vector<float> pre = {-1.0f, 0.0f, 2.0f, -0.5f, 5.0f};
  std::vector<float> g = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  auto dp = c.from_host(pre.data(), {5}, DType::F32);
  auto dg = c.from_host(g.data(), {5}, DType::F32);
  auto out = c.alloc({5}, DType::F32);
  relu_grad_into(nn_lib(), disp, *dp, *dg, *out, 5);
  disp.wait();
  for (int i = 0; i < 5; ++i) CHECK_NEAR(host(out)[i], pre[i] > 0.0f ? g[i] : 0.0f, 1e-6);
}

TEST(NNReduceSumAxis0) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t M = 128, N = 64;
  auto a = randf(M * N, -3, 3, 7);
  auto da = c.from_host(a.data(), {M, N}, DType::F32);
  auto out = c.alloc({N}, DType::F32);
  reduce_sum_axis0_into(nn_lib(), disp, *da, *out, M, N);
  disp.wait();
  for (int64_t j = 0; j < N; ++j) {
    float s = 0.0f;  // match the kernel's sequential f32 accumulation order
    for (int64_t i = 0; i < M; ++i) s += a[i * N + j];
    CHECK_NEAR(host(out)[j], s, 1e-4);
  }
}

TEST(NNTranspose2d) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t M = 17, N = 31;
  auto a = randf(M * N, -5, 5, 9);
  auto da = c.from_host(a.data(), {M, N}, DType::F32);
  auto out = c.alloc({N, M}, DType::F32);
  transpose2d_into(nn_lib(), disp, *da, *out, M, N);
  disp.wait();
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      CHECK_NEAR(host(out)[j * M + i], a[i * N + j], 1e-6);
}

TEST(NNSgdUpdate) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  std::vector<float> w = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> dw = {0.5f, 1.0f, -2.0f, 0.0f};
  const float lr = 0.1f;
  auto dW = c.from_host(w.data(), {4}, DType::F32);
  auto dD = c.from_host(dw.data(), {4}, DType::F32);
  sgd_update_into(nn_lib(), disp, *dW, *dD, lr, 4);
  disp.wait();
  for (int i = 0; i < 4; ++i) CHECK_NEAR(host(dW)[i], w[i] - lr * dw[i], 1e-6);
}

TEST(NNSoftmaxXent) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t B = 8, C = 10;
  auto logits = randf(B * C, -4, 4, 11);
  std::vector<int32_t> labels(B);
  std::mt19937 rng(12);
  std::uniform_int_distribution<int> ld(0, (int)C - 1);
  for (auto& y : labels) y = ld(rng);

  auto dl = c.from_host(logits.data(), {B, C}, DType::F32);
  auto dy = c.from_host(labels.data(), {B}, DType::I32);
  auto loss = c.alloc({B}, DType::F32);
  auto dlog = c.alloc({B, C}, DType::F32);
  auto probs = c.alloc({B, C}, DType::F32);
  softmax_xent_into(nn_lib(), disp, *dl, *dy, *loss, *dlog, *probs, B, C);
  disp.wait();

  for (int64_t b = 0; b < B; ++b) {
    double m = -1e300;
    for (int64_t j = 0; j < C; ++j) m = std::max(m, (double)logits[b * C + j]);
    double s = 0.0;
    for (int64_t j = 0; j < C; ++j) s += std::exp(logits[b * C + j] - m);
    double psum = 0.0, gsum = 0.0;
    for (int64_t j = 0; j < C; ++j) {
      double p = std::exp(logits[b * C + j] - m) / s;
      double oh = (j == labels[b]) ? 1.0 : 0.0;
      CHECK_NEAR(host(probs)[b * C + j], p, 1e-4);
      CHECK_NEAR(host(dlog)[b * C + j], (p - oh) / B, 1e-4);
      psum += host(probs)[b * C + j];
      gsum += host(dlog)[b * C + j];
    }
    double refloss = -(logits[b * C + labels[b]] - m) + std::log(s);
    CHECK_NEAR(host(loss)[b], refloss, 1e-4);
    CHECK_NEAR(psum, 1.0, 1e-4);   // probs sum to 1
    CHECK_NEAR(gsum, 0.0, 1e-4);   // grad rows sum to 0
  }
}

TEST(NNSoftmaxXentStability) {
  // A huge logit must not overflow (proves the max-subtraction).
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t B = 1, C = 4;
  std::vector<float> logits = {1e4f, 0.0f, -3.0f, 2.0f};
  std::vector<int32_t> labels = {0};
  auto dl = c.from_host(logits.data(), {B, C}, DType::F32);
  auto dy = c.from_host(labels.data(), {B}, DType::I32);
  auto loss = c.alloc({B}, DType::F32);
  auto dlog = c.alloc({B, C}, DType::F32);
  auto probs = c.alloc({B, C}, DType::F32);
  softmax_xent_into(nn_lib(), disp, *dl, *dy, *loss, *dlog, *probs, B, C);
  disp.wait();
  CHECK(std::isfinite(host(loss)[0]));
  CHECK_NEAR(host(loss)[0], 0.0, 1e-4);           // label is the argmax spike
  CHECK_NEAR(host(probs)[0], 1.0, 1e-4);
}

// Full forward+backward+SGD on a tiny net using the _into ops, compared to a
// double-precision CPU reference. This exercises the op set exactly as the MLP
// scheduler chains it (resident buffers, one wait at the end).
TEST(NNMlpForwardBackwardStep) {
  auto& c = testutil::ctx();
  Dispatcher disp(c);
  const int64_t Din = 6, H = 4, Dout = 3, B = 5;
  const float lr = 0.2f;

  auto x = randf(B * Din, -1, 1, 20);
  auto W1 = randf(Din * H, -0.5, 0.5, 21);
  auto b1 = randf(H, -0.1, 0.1, 22);
  auto W2 = randf(H * Dout, -0.5, 0.5, 23);
  auto b2 = randf(Dout, -0.1, 0.1, 24);
  std::vector<int32_t> y(B);
  std::mt19937 rng(25);
  std::uniform_int_distribution<int> ld(0, (int)Dout - 1);
  for (auto& v : y) v = ld(rng);

  // ---- GPU path (resident buffers, _into ops) ----
  auto dx = c.from_host(x.data(), {B, Din}, DType::F32);
  auto dW1 = c.from_host(W1.data(), {Din, H}, DType::F32);
  auto db1 = c.from_host(b1.data(), {H}, DType::F32);
  auto dW2 = c.from_host(W2.data(), {H, Dout}, DType::F32);
  auto db2 = c.from_host(b2.data(), {Dout}, DType::F32);
  auto dy = c.from_host(y.data(), {B}, DType::I32);
  auto z1 = c.alloc({B, H}, DType::F32);
  auto h1 = c.alloc({B, H}, DType::F32);
  auto logits = c.alloc({B, Dout}, DType::F32);
  auto probs = c.alloc({B, Dout}, DType::F32);
  auto loss = c.alloc({B}, DType::F32);
  auto dlogits = c.alloc({B, Dout}, DType::F32);
  auto dh1 = c.alloc({B, H}, DType::F32);
  auto drelu = c.alloc({B, H}, DType::F32);
  auto h1T = c.alloc({H, B}, DType::F32);
  auto W2T = c.alloc({Dout, H}, DType::F32);
  auto xT = c.alloc({Din, B}, DType::F32);
  auto gW1 = c.alloc({Din, H}, DType::F32);
  auto gb1 = c.alloc({H}, DType::F32);
  auto gW2 = c.alloc({H, Dout}, DType::F32);
  auto gb2 = c.alloc({Dout}, DType::F32);

  KernelLibrary& lib = nn_lib();
  matmul_into(lib, disp, *dx, *dW1, *z1, B, Din, H);        // z1 = x@W1
  bias_add_into(lib, disp, *z1, *db1, *z1, B, H);
  relu_into(lib, disp, *z1, *h1, B * H);
  matmul_into(lib, disp, *h1, *dW2, *logits, B, H, Dout);   // logits = h1@W2
  bias_add_into(lib, disp, *logits, *db2, *logits, B, Dout);
  softmax_xent_into(lib, disp, *logits, *dy, *loss, *dlogits, *probs, B, Dout);
  transpose2d_into(lib, disp, *h1, *h1T, B, H);
  matmul_into(lib, disp, *h1T, *dlogits, *gW2, H, B, Dout);  // dW2
  reduce_sum_axis0_into(lib, disp, *dlogits, *gb2, B, Dout);  // db2
  transpose2d_into(lib, disp, *dW2, *W2T, H, Dout);
  matmul_into(lib, disp, *dlogits, *W2T, *dh1, B, Dout, H);   // dh1
  relu_grad_into(lib, disp, *z1, *dh1, *drelu, B * H);
  transpose2d_into(lib, disp, *dx, *xT, B, Din);
  matmul_into(lib, disp, *xT, *drelu, *gW1, Din, B, H);       // dW1
  reduce_sum_axis0_into(lib, disp, *drelu, *gb1, B, H);       // db1
  sgd_update_into(lib, disp, *dW1, *gW1, lr, Din * H);
  sgd_update_into(lib, disp, *db1, *gb1, lr, H);
  sgd_update_into(lib, disp, *dW2, *gW2, lr, H * Dout);
  sgd_update_into(lib, disp, *db2, *gb2, lr, Dout);
  disp.wait();

  // ---- double CPU reference ----
  auto mm = [](const std::vector<double>& A, const std::vector<double>& Bm,
               int64_t M, int64_t K, int64_t N) {
    std::vector<double> Cc(M * N, 0.0);
    for (int64_t i = 0; i < M; ++i)
      for (int64_t k = 0; k < K; ++k)
        for (int64_t j = 0; j < N; ++j) Cc[i * N + j] += A[i * K + k] * Bm[k * N + j];
    return Cc;
  };
  std::vector<double> xd(x.begin(), x.end()), W1d(W1.begin(), W1.end());
  std::vector<double> b1d(b1.begin(), b1.end()), W2d(W2.begin(), W2.end()), b2d(b2.begin(), b2.end());
  auto z1r = mm(xd, W1d, B, Din, H);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < H; ++j) z1r[i * H + j] += b1d[j];
  std::vector<double> h1r(B * H);
  for (int64_t i = 0; i < B * H; ++i) h1r[i] = std::max(z1r[i], 0.0);
  auto lgr = mm(h1r, W2d, B, H, Dout);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < Dout; ++j) lgr[i * Dout + j] += b2d[j];
  std::vector<double> dlr(B * Dout);
  for (int64_t i = 0; i < B; ++i) {
    double m = -1e300; for (int64_t j = 0; j < Dout; ++j) m = std::max(m, lgr[i * Dout + j]);
    double s = 0.0; for (int64_t j = 0; j < Dout; ++j) s += std::exp(lgr[i * Dout + j] - m);
    for (int64_t j = 0; j < Dout; ++j) {
      double p = std::exp(lgr[i * Dout + j] - m) / s;
      dlr[i * Dout + j] = (p - ((j == y[i]) ? 1.0 : 0.0)) / B;
    }
  }
  // dW2 = h1^T @ dlogits
  std::vector<double> h1Td(H * B);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < H; ++j) h1Td[j * B + i] = h1r[i * H + j];
  auto gW2r = mm(h1Td, dlr, H, B, Dout);
  std::vector<double> gb2r(Dout, 0.0);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < Dout; ++j) gb2r[j] += dlr[i * Dout + j];
  // dh1 = dlogits @ W2^T
  std::vector<double> W2Td(Dout * H);
  for (int64_t i = 0; i < H; ++i) for (int64_t j = 0; j < Dout; ++j) W2Td[j * H + i] = W2d[i * Dout + j];
  auto dh1r = mm(dlr, W2Td, B, Dout, H);
  std::vector<double> drelur(B * H);
  for (int64_t i = 0; i < B * H; ++i) drelur[i] = z1r[i] > 0.0 ? dh1r[i] : 0.0;
  // dW1 = x^T @ drelu
  std::vector<double> xTd(Din * B);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < Din; ++j) xTd[j * B + i] = xd[i * Din + j];
  auto gW1r = mm(xTd, drelur, Din, B, H);
  std::vector<double> gb1r(H, 0.0);
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < H; ++j) gb1r[j] += drelur[i * H + j];

  // Compare updated params: p' = p - lr*grad.
  for (int64_t i = 0; i < Din * H; ++i)
    CHECK_NEAR(host(dW1)[i], W1[i] - lr * gW1r[i], 1e-4);
  for (int64_t i = 0; i < H; ++i)
    CHECK_NEAR(host(db1)[i], b1[i] - lr * gb1r[i], 1e-4);
  for (int64_t i = 0; i < H * Dout; ++i)
    CHECK_NEAR(host(dW2)[i], W2[i] - lr * gW2r[i], 1e-4);
  for (int64_t i = 0; i < Dout; ++i)
    CHECK_NEAR(host(db2)[i], b2[i] - lr * gb2r[i], 1e-4);
}
