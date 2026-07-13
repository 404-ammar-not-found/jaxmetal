#include "test_framework.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

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

int run_one(const Case& c) {
  try {
    c.fn();
    std::printf("[ PASS ] %s\n", c.name.c_str());
    return 0;
  } catch (const Failure& f) {
    std::printf("[ FAIL ] %s\n         %s\n", c.name.c_str(), f.msg.c_str());
  } catch (const std::exception& e) {
    std::printf("[ FAIL ] %s\n         unexpected exception: %s\n", c.name.c_str(), e.what());
  } catch (...) {
    std::printf("[ FAIL ] %s\n         unknown exception\n", c.name.c_str());
  }
  return 1;
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
  int failures = 0, ran = 0;
  for (const auto& c : registry()) {
    if (!filter.empty() && c.name != filter) continue;
    ++ran;
    failures += run_one(c);
  }
  if (ran == 0) {
    std::printf("No tests matched '%s'\n", filter.c_str());
    return 2;
  }
  std::printf("\n%d/%d passed\n", ran - failures, ran);
  return failures ? 1 : 0;
}

}  // namespace testing

int main(int argc, char** argv) { return ::testing::run(argc, argv); }
