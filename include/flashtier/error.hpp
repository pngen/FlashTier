#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace flashtier {

// Typed error codes used across the runtime. Every failure maps to one of
// these; nothing is silently ignored.
enum class ErrorCode : int {
    None = 0,
    Config = 1,
    InvalidArgument = 2,
    NotFound = 3,
    State = 4,
    Invariant = 5,
    Budget = 6,
    Cuda = 7,       // NVIDIA CUDA backend errors (historical name, retained)
    Io = 8,
    StoreCorrupt = 9,
    Unsupported = 10,
    Cancelled = 11,
    Integrity = 12,
    Internal = 13,
    Device = 14,    // vendor-neutral device-backend errors (HIP, Level Zero,
                    // Vulkan, Metal, ...)
};

const char* error_code_name(ErrorCode code) noexcept;

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, std::string message);
    Error(ErrorCode code, std::string message, std::string detail);

    ErrorCode code() const noexcept { return code_; }
    const std::string& detail() const noexcept { return detail_; }

private:
    ErrorCode code_;
    std::string detail_;
};

// Convenience factories.
Error make_error(ErrorCode code, const std::string& message);
Error make_error(ErrorCode code, const std::string& message, const std::string& detail);

}  // namespace flashtier
