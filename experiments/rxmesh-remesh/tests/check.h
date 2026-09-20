#pragma once
#include <cmath>
#include <iostream>
#include <string>

inline int g_fails = 0;

#define CHECK(cond)                                                                            \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #cond << '\n';             \
      ++g_fails;                                                                               \
    }                                                                                          \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                                                  \
  do {                                                                                         \
    const double _va = double(a), _vb = double(b);                                             \
    if (!(std::fabs(_va - _vb) <= double(eps))) {                                              \
      std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << " " << #a << "=" << _va << " "    \
                << #b << "=" << _vb << " eps=" << (eps) << '\n';                               \
      ++g_fails;                                                                               \
    }                                                                                          \
  } while (0)

inline int test_result(const char *name) {
  if (g_fails) {
    std::cerr << name << " failed: " << g_fails << " checks\n";
    return 1;
  }
  std::cout << name << " passed\n";
  return 0;
}
