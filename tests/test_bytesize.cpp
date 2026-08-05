#include "test_harness.hpp"

#include "flashtier/config.hpp"
#include "flashtier/error.hpp"

using namespace flashtier;

FT_TEST(bytesize_parses_binary_suffixes) {
    FT_ASSERT_EQ(parse_bytesize("1KiB"), 1024ull);
    FT_ASSERT_EQ(parse_bytesize("2MiB"), 2ull * 1024 * 1024);
    FT_ASSERT_EQ(parse_bytesize("3GiB"), 3ull * 1024 * 1024 * 1024);
    FT_ASSERT_EQ(parse_bytesize("4TiB"), 4ull * 1024 * 1024 * 1024 * 1024);
    FT_ASSERT_EQ(parse_bytesize("4096"), 4096ull);
    FT_ASSERT_EQ(parse_bytesize("16 KiB"), 16ull * 1024);
    FT_ASSERT_EQ(parse_bytesize("1kib"), 1024ull);
    FT_ASSERT_EQ(parse_bytesize("0"), 0ull);
}

FT_TEST(bytesize_rejects_malformed) {
    FT_ASSERT_THROWS(parse_bytesize(""), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("   "), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("abc"), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("1.5GiB"), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("-1"), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("1XB"), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("18446744073709551616"), ErrorCode::Config);
    FT_ASSERT_THROWS(parse_bytesize("99999999999999999999999999999"), ErrorCode::Config);
}

FT_TEST(bytesize_to_string_roundtrip) {
    FT_ASSERT_EQ(bytesize_to_string(1024ull), "1.00 KiB");
    FT_ASSERT_EQ(bytesize_to_string(2ull * 1024 * 1024), "2.00 MiB");
}

int main() { return ft_test::run_all("test_bytesize"); }
