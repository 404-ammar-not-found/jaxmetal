#include "test_framework.h"

#include "jaxmetal/metal/metal_context.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

// Return code that signals "test skipped" to CTest (matched by the
// SKIP_RETURN_CODE property set on each test in CMakeLists.txt).
static constexpr int kSkipReturnCode = 125;

namespace testing {
namespace {

struct Case {
  std::string name;
  std::function<void()> fn;
};

std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Failure {
  std::string msg;
};

// Per-case outcome, distinct from the process exit code.
enum class Status { kPass, kFail, kSkip };

Status run_one(const Case& c) {
  try {
    c.fn();
    std::printf("[ PASS ] %s\n", c.name.c_str());
    return Status::kPass;
  } catch (const jaxmetal::MetalUnavailable& e) {
    // No GPU on this host (e.g. a headless CI runner) — not a failure.
    std::printf("[ SKIP ] %s\n         %s\n", c.name.c_str(), e.what());
    return Status::kSkip;
  } catch (const Failure& f) {
    std::printf("[ FAIL ] %s\n         %s\n", c.name.c_str(), f.msg.c_str());
  } catch (const std::exception& e) {
    std::printf("[ FAIL ] %s\n         unexpected exception: %s\n", c.name.c_str(), e.what());
  } catch (...) {
    std::printf("[ FAIL ] %s\n         unknown exception\n", c.name.c_str());
  }
  return Status::kFail;
}

}  // namespace

int register_test(const char* name, std::function<void()> fn) {
  registry().push_back({name, std::move(fn)});
  return 0;
}

void fail(const std::string& msg) { throw Failure{msg}; }

void check(bool cond, const char* expr, const char* file, int line) {
  if (!cond)
    fail(std::string("CHECK failed: ") + expr + " (" + file + ":" + std::to_string(line) + ")");
}

void check_near(double a, double b, double tol, const char* ea, const char* eb,
                const char* file, int line) {
  double d = std::fabs(a - b);
  if (!(d <= tol))
    fail(std::string("CHECK_NEAR failed: |") + ea + " - " + eb + "| = " + std::to_string(d) +
         " > " + std::to_string(tol) + " (" + file + ":" + std::to_string(line) + ")");
}

int run(int argc, char** argv) {
  std::string filter = (argc > 1) ? argv[1] : "";
  int failures = 0, skipped = 0, ran = 0;
  for (const auto& c : registry()) {
    if (!filter.empty() && c.name != filter) continue;
    ++ran;
    switch (run_one(c)) {
      case Status::kFail: ++failures; break;
      case Status::kSkip: ++skipped; break;
      case Status::kPass: break;
    }
  }
  if (ran == 0) {
    std::printf("No tests matched '%s'\n", filter.c_str());
    return 2;
  }
  std::printf("\n%d/%d passed", ran - failures - skipped, ran);
  if (skipped) std::printf(" (%d skipped)", skipped);
  std::printf("\n");
  if (failures) return 1;
  // Nothing failed but nothing ran for real (no GPU) -> tell CTest to skip.
  if (skipped == ran) return kSkipReturnCode;
  return 0;
}

}  // namespace testing

int main(int argc, char** argv) { return ::testing::run(argc, argv); }
