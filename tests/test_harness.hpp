#pragma once

// Minimal dependency-free test harness. Registers checks, runs them, and
// exits nonzero on any failure so CTest sees a real failure.

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace ft_test {

struct Check {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Check>& registry() {
    static std::vector<Check> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back(Check{name, std::move(fn)});
    }
};

struct Failure {
    std::string message;
};

inline int run_all(const char* suite) {
    int failures = 0;
    for (const auto& c : registry()) {
        try {
            c.fn();
            std::printf("[PASS] %s\n", c.name);
        } catch (const Failure& f) {
            std::printf("[FAIL] %s: %s\n", c.name, f.message.c_str());
            ++failures;
        } catch (const std::exception& e) {
            std::printf("[FAIL] %s: unexpected exception: %s\n", c.name, e.what());
            ++failures;
        } catch (...) {
            std::printf("[FAIL] %s: unknown exception\n", c.name);
            ++failures;
        }
    }
    std::printf("suite %s: %zu checks, %d failures\n", suite, registry().size(), failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace ft_test

#define FT_TEST(name)                                                     \
    static void ft_check_##name();                                        \
    static ::ft_test::Registrar ft_reg_##name(#name, ft_check_##name);    \
    static void ft_check_##name()

#define FT_ASSERT(cond)                                                     \
    do {                                                                    \
        if (!(cond)) {                                                      \
            throw ::ft_test::Failure{std::string("assertion failed: ") +   \
                                     #cond " at " __FILE__ ":" +            \
                                     std::to_string(__LINE__)};             \
        }                                                                   \
    } while (0)

#define FT_ASSERT_EQ(a, b)                                                  \
    do {                                                                    \
        auto va = (a);                                                      \
        auto vb = (b);                                                      \
        if (!(va == vb)) {                                                  \
            throw ::ft_test::Failure{std::string("assertion failed: ") +   \
                                     #a " == " #b " at " __FILE__ ":" +     \
                                     std::to_string(__LINE__)};             \
        }                                                                   \
    } while (0)

#define FT_ASSERT_THROWS(expr, errcode)                                     \
    do {                                                                    \
        bool threw = false;                                                 \
        try {                                                               \
            (void)(expr);                                                   \
        } catch (const ::flashtier::Error& e) {                             \
            threw = (e.code() == (errcode));                                \
        } catch (...) {                                                     \
            threw = false;                                                  \
        }                                                                   \
        if (!threw) {                                                       \
            throw ::ft_test::Failure{std::string("expected ") + #errcode + \
                                     " to throw at " __FILE__ ":" +         \
                                     std::to_string(__LINE__)};             \
        }                                                                   \
    } while (0)
