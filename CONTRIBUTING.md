# Contributing to FlashTier

Thanks for contributing. FlashTier is Apache-2.0 licensed and accepts
contributions **without a Contributor License Agreement** — by submitting a
pull request you agree that your contribution is offered under the Apache
License 2.0, and you warrant you have the right to do so.

## Ground rules

- Be technically rigorous. No unsupported performance claims, no
  advertising language ("unlimited VRAM", "SSD becomes VRAM"), no claims
  about hardware that has not been implemented and measured.
- Preserve the central invariant: explicit, observable, deterministic
  degradation instead of fatal GPU out-of-memory.
- Keep Windows and Linux first-class. Optional backends (DirectStorage,
  GDS, HBF, AMD) are optional; they must never become build requirements.
- All telemetry stays local.
- No new mandatory dependencies without discussion. Prefer the standard
  library and pinned, minimal dependencies.

## Getting started

1. Read [ARCHITECTURE.md](ARCHITECTURE.md) and
   [BENCHMARKS.md](BENCHMARKS.md).
2. Build and test:
   `cmake --preset windows-cpu-release && cmake --build build/windows-cpu-release --config Release --parallel`
   then `ctest --test-dir build/windows-cpu-release -C Release --output-on-failure`.
3. Pick an issue, or open one describing what you plan to change.

## Code style

- C++20, matching the surrounding style; no comments unless they add
  value that the code cannot express.
- Every CUDA call and every file operation is checked and mapped to a
  typed `flashtier::Error`.
- State transitions go through the page state machine; illegal transitions
  are rejected with `ErrorCode::State`.
- New behavior ships with tests in `tests/` and is registered with CTest.
- Project code must compile warning-clean with
  `FLASHTIER_WARNINGS_AS_ERRORS=ON`.

## Commit messages

Describe the completed work accurately in the imperative tense; keep
history linear and do not squash away implementation history.

## Review

Maintainers review for correctness, determinism, safety of large
allocations, and honest reporting. Benchmark changes must update
BENCHMARKS.md methodology where they affect metric definitions.
