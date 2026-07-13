// Tests for the resident MLP (src/ops/mlp): forward parity, loss value, gradient
// parity (recovered from the SGD update), and a tiny-batch overfit smoke test.
// Constructs jaxmetal::MLP directly (it owns its own KernelLibrary).

#include "test_framework.h"

#include "jaxmetal/ops/mlp.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace jaxmetal;

namespace {

const int64_t Din = 6, H = 5, Dout = 3, B = 4;

std::vector<float> randf(int64_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

struct Fixture {
  std::vector<float> x, W1, b1, W2, b2;
  std::vector<int32_t> y;
  Fixture() {
    x = randf(B * Din, -1, 1, 40);
    W1 = randf(Din * H, -0.5, 0.5, 41);
    b1 = randf(H, -0.1, 0.1, 42);
    W2 = randf(H * Dout, -0.5, 0.5, 43);
    b2 = randf(Dout, -0.1, 0.1, 44);
    y.resize(B);
    std::mt19937 rng(45);
    std::uniform_int_distribution<int> ld(0, (int)Dout - 1);
    for (auto& v : y) v = ld(rng);
  }
};

// double CPU forward -> logits.
std::vector<double> cpu_logits(const Fixture& f) {
  std::vector<double> z1(B * H, 0.0);
  for (int64_t i = 0; i < B; ++i)
    for (int64_t k = 0; k < Din; ++k)
      for (int64_t j = 0; j < H; ++j) z1[i * H + j] += (double)f.x[i * Din + k] * f.W1[k * H + j];
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < H; ++j) z1[i * H + j] += f.b1[j];
  std::vector<double> h1(B * H);
  for (int64_t i = 0; i < B * H; ++i) h1[i] = std::max(z1[i], 0.0);
  std::vector<double> lg(B * Dout, 0.0);
  for (int64_t i = 0; i < B; ++i)
    for (int64_t k = 0; k < H; ++k)
      for (int64_t j = 0; j < Dout; ++j) lg[i * Dout + j] += h1[i * H + k] * f.W2[k * Dout + j];
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < Dout; ++j) lg[i * Dout + j] += f.b2[j];
  return lg;
}

}  // namespace

TEST(MlpForwardParity) {
  Fixture f;
  MLP m(Din, H, Dout, B);
  m.set_params(f.W1.data(), f.b1.data(), f.W2.data(), f.b2.data());
  m.upload_batch(f.x.data(), f.y.data(), B);
  std::vector<float> logits(B * Dout);
  m.forward(B, logits.data());
  auto ref = cpu_logits(f);
  for (int64_t i = 0; i < B * Dout; ++i) CHECK_NEAR(logits[i], ref[i], 1e-4);
}

TEST(MlpLossValue) {
  Fixture f;
  MLP m(Din, H, Dout, B);
  m.set_params(f.W1.data(), f.b1.data(), f.W2.data(), f.b2.data());
  m.upload_batch(f.x.data(), f.y.data(), B);
  float loss = m.train_step(B, 0.0f);  // lr=0 -> params unchanged, just read loss
  auto lg = cpu_logits(f);
  double ref = 0.0;
  for (int64_t i = 0; i < B; ++i) {
    double mmx = -1e300; for (int64_t j = 0; j < Dout; ++j) mmx = std::max(mmx, lg[i * Dout + j]);
    double s = 0.0; for (int64_t j = 0; j < Dout; ++j) s += std::exp(lg[i * Dout + j] - mmx);
    ref += -(lg[i * Dout + f.y[i]] - mmx) + std::log(s);
  }
  ref /= B;
  CHECK_NEAR(loss, ref, 1e-4);
}

TEST(MlpGradParity) {
  Fixture f;
  const float lr = 0.1f;
  MLP m(Din, H, Dout, B);
  m.set_params(f.W1.data(), f.b1.data(), f.W2.data(), f.b2.data());
  m.upload_batch(f.x.data(), f.y.data(), B);
  m.train_step(B, lr);
  std::vector<float> W1(Din * H), b1(H), W2(H * Dout), b2(Dout);
  m.get_params(W1.data(), b1.data(), W2.data(), b2.data());
  // recovered grad = (theta0 - theta1)/lr
  // CPU reference backprop
  auto lg = cpu_logits(f);
  std::vector<double> z1(B * H, 0.0);
  for (int64_t i = 0; i < B; ++i)
    for (int64_t k = 0; k < Din; ++k)
      for (int64_t j = 0; j < H; ++j) z1[i * H + j] += (double)f.x[i * Din + k] * f.W1[k * H + j];
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < H; ++j) z1[i * H + j] += f.b1[j];
  std::vector<double> h1(B * H);
  for (int64_t i = 0; i < B * H; ++i) h1[i] = std::max(z1[i], 0.0);
  std::vector<double> dl(B * Dout);
  for (int64_t i = 0; i < B; ++i) {
    double mmx = -1e300; for (int64_t j = 0; j < Dout; ++j) mmx = std::max(mmx, lg[i * Dout + j]);
    double s = 0.0; for (int64_t j = 0; j < Dout; ++j) s += std::exp(lg[i * Dout + j] - mmx);
    for (int64_t j = 0; j < Dout; ++j) {
      double p = std::exp(lg[i * Dout + j] - mmx) / s;
      dl[i * Dout + j] = (p - ((j == f.y[i]) ? 1.0 : 0.0)) / B;
    }
  }
  std::vector<double> gW2(H * Dout, 0.0), gb2(Dout, 0.0);
  for (int64_t k = 0; k < H; ++k)
    for (int64_t i = 0; i < B; ++i)
      for (int64_t j = 0; j < Dout; ++j) gW2[k * Dout + j] += h1[i * H + k] * dl[i * Dout + j];
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < Dout; ++j) gb2[j] += dl[i * Dout + j];
  std::vector<double> dh1(B * H, 0.0);
  for (int64_t i = 0; i < B; ++i)
    for (int64_t j = 0; j < Dout; ++j)
      for (int64_t k = 0; k < H; ++k) dh1[i * H + k] += dl[i * Dout + j] * f.W2[k * Dout + j];
  std::vector<double> dz1(B * H);
  for (int64_t i = 0; i < B * H; ++i) dz1[i] = z1[i] > 0.0 ? dh1[i] : 0.0;
  std::vector<double> gW1(Din * H, 0.0), gb1(H, 0.0);
  for (int64_t k = 0; k < Din; ++k)
    for (int64_t i = 0; i < B; ++i)
      for (int64_t j = 0; j < H; ++j) gW1[k * H + j] += (double)f.x[i * Din + k] * dz1[i * H + j];
  for (int64_t i = 0; i < B; ++i) for (int64_t j = 0; j < H; ++j) gb1[j] += dz1[i * H + j];

  for (int64_t i = 0; i < Din * H; ++i) CHECK_NEAR((f.W1[i] - W1[i]) / lr, gW1[i], 2e-4);
  for (int64_t i = 0; i < H; ++i) CHECK_NEAR((f.b1[i] - b1[i]) / lr, gb1[i], 2e-4);
  for (int64_t i = 0; i < H * Dout; ++i) CHECK_NEAR((f.W2[i] - W2[i]) / lr, gW2[i], 2e-4);
  for (int64_t i = 0; i < Dout; ++i) CHECK_NEAR((f.b2[i] - b2[i]) / lr, gb2[i], 2e-4);
}

TEST(MlpOverfitsTinyBatch) {
  Fixture f;
  MLP m(Din, H, Dout, B);
  m.set_params(f.W1.data(), f.b1.data(), f.W2.data(), f.b2.data());
  m.upload_batch(f.x.data(), f.y.data(), B);
  float first = m.train_step(B, 0.5f);
  float last = first;
  for (int i = 0; i < 200; ++i) last = m.train_step(B, 0.5f);
  CHECK(last < first);      // learning signal
  CHECK(last < 0.05);       // memorizes the tiny batch
}
