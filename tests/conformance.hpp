#pragma once

// Generic device-backend conformance battery. Every real vendor backend
// must pass this suite on its hardware; the mock and CPU backends run it on
// every platform. A mock passing never counts as validating a vendor
// backend.

#include "flashtier/backends/device_backend.hpp"

namespace flashtier {
namespace conformance {

// Runs the full battery against `backend` (which must not be open yet).
// `allow_exhaustion` enables budget-exhaustion checks that need a bounded
// fake limit (mock/cpu); hardware backends keep limit checks to the
// max_allocation_size guard, which never touches real device memory.
// Throws on the first failing check with a descriptive message.
void run_battery(DeviceBackend& backend, const char* label, bool allow_exhaustion);

}  // namespace conformance
}  // namespace flashtier
