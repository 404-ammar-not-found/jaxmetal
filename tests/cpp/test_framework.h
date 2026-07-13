#pragma once
// Minimal dependency-free test framework with CTest integration.
//
//   TEST(Name) { CHECK(cond); CHECK_NEAR(a, b, tol); CHECK_THROWS(stmt); }
//
// Each TEST auto-registers at static-init time. main() (in test_framework.cpp)
// runs all tests, or just the one named in argv[1] (how CTest invokes each case).

#include <functional>
#include <string>

namespace testing {

int register_test(const char* name, std::function<void()> fn);
int run(int argc, char** argv);

[[noreturn]] void fail(const std::string& msg);
void check(bool cond, const char* expr, const char* file, int line);
void check_near(double a, double b, double tol, const char* ea, const char* eb,
                const char* file, int line);

}  // namespace testing

#define TEST(name)                                                        \
  static void name();                                                     \
  static int name##_registered = ::testing::register_test(#name, name);   \
  static void name()

#define CHECK(cond) ::testing::check((cond), #cond, __FILE__, __LINE__)

#define CHECK_NEAR(a, b, tol) \
  ::testing::check_near((a), (b), (tol), #a, #b, __FILE__, __LINE__)

#define CHECK_THROWS(stmt)                                                \
  do {                                                                    \
    bool _threw = false;                                                  \
    try {                                                                 \
      stmt;                                                               \
    } catch (...) {                                                       \
      _threw = true;                                                      \
    }                                                                     \
    ::testing::check(_threw, "expected exception from: " #stmt, __FILE__, \
                     __LINE__);                                           \
  } while (0)
