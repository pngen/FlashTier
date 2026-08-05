#include "flashtier/error.hpp"

namespace flashtier {

const char* error_code_name(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::None: return "none";
        case ErrorCode::Config: return "config";
        case ErrorCode::InvalidArgument: return "invalid_argument";
        case ErrorCode::NotFound: return "not_found";
        case ErrorCode::State: return "state";
        case ErrorCode::Invariant: return "invariant";
        case ErrorCode::Budget: return "budget";
        case ErrorCode::Cuda: return "cuda";
        case ErrorCode::Io: return "io";
        case ErrorCode::StoreCorrupt: return "store_corrupt";
        case ErrorCode::Unsupported: return "unsupported";
        case ErrorCode::Cancelled: return "cancelled";
        case ErrorCode::Integrity: return "integrity";
        case ErrorCode::Internal: return "internal";
    }
    return "unknown";
}

Error::Error(ErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

Error::Error(ErrorCode code, std::string message, std::string detail)
    : std::runtime_error(std::move(message)),
      code_(code),
      detail_(std::move(detail)) {}

Error make_error(ErrorCode code, const std::string& message) {
    return Error(code, message);
}

Error make_error(ErrorCode code, const std::string& message, const std::string& detail) {
    return Error(code, message, detail);
}

}  // namespace flashtier
