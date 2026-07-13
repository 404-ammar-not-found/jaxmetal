// Tests for the low-level Metal runtime: device, buffer allocation, host<->device
// round-trip, and shape/dtype bookkeeping.

#include "test_framework.h"
#include "gpu_test_util.h"

#include "jaxmetal/metal/metal_context.h"
#include "jaxmetal/metal/metal_buffer.h"

#include <cstdint>
#include <vector>

using namespace jaxmetal;

TEST(DeviceAvailable) {
  CHECK(!testutil::ctx().device_name().empty());
  CHECK(testutil::ctx().device_handle() != nullptr);
  CHECK(testutil::ctx().queue_handle() != nullptr);
}

TEST(AllocShapeAndSize) {
  auto buf = testutil::ctx().alloc({3, 4}, DType::F32);
  CHECK(buf->num_elements() == 12);
  CHECK(buf->size_bytes() == 12 * sizeof(float));
  CHECK(buf->shape().size() == 2);
  CHECK(buf->shape()[0] == 3 && buf->shape()[1] == 4);
  CHECK(buf->dtype() == DType::F32);
  CHECK(buf->mtl_handle() != nullptr);
  CHECK(buf->contents() != nullptr);
}

TEST(ScalarShapeIsOneElement) {
  auto buf = testutil::ctx().alloc({}, DType::F32);
  CHECK(buf->num_elements() == 1);
  CHECK(buf->size_bytes() == sizeof(float));
}

TEST(HostRoundTrip) {
  const int64_t n = 257;  // deliberately not a round number
  std::vector<float> host(n);
  for (int64_t i = 0; i < n; ++i) host[i] = static_cast<float>(i) * 1.5f - 3.0f;

  auto buf = testutil::ctx().from_host(host.data(), {n}, DType::F32);
  const float* dev = static_cast<const float*>(buf->contents());
  for (int64_t i = 0; i < n; ++i) CHECK(dev[i] == host[i]);
}

TEST(I32BufferSize) {
  auto buf = testutil::ctx().alloc({5}, DType::I32);
  CHECK(buf->size_bytes() == 5 * sizeof(int32_t));
  CHECK(buf->dtype() == DType::I32);
}
