# FlashTier

FlashTier is an open-source heterogeneous-memory runtime that governs AI
workload residency across accelerator-local device memory, pinned system
RAM, and NVMe storage.

The CUDA backend is **hardware-validated on an NVIDIA GeForce RTX 5090**
(Blackwell, compute capability 12.0, Windows). The runtime has demonstrated
real device-memory oversubscription: a logical working set exceeding both
the configured device-memory and pinned-host budgets was forced across all
three tiers — CUDA device memory, pinned system RAM, and NVMe — and exact
integrity was preserved through tens of thousands of transfers, evictions,
and writebacks. The CPU backend is validated as an emulation path; other
accelerator backends are either compile-gated or fail closed as noted below.

**The central invariant:** when a working set exceeds physical accelerator
memory, FlashTier replaces abrupt out-of-memory failure with explicit,
observable, policy-governed degradation.

## Hardware-validated proof

The proof below was executed on the validation machine (RTX 5090, driver
610.88, CUDA toolkit 12.9.86, build compiled for `sm_120` via native
architecture detection) using the governed three-tier oversubscription
test with 64 KiB pages:

| Backend | Hardware | Device budget | Host budget | NVMe budget | Working set | Peak device usage | Peak host usage | Evictions | Writebacks | Promotions | Integrity failures |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CUDA | RTX 5090, sm_120, Windows | 256 MiB | 128 MiB | 1 GiB | 512 MiB | 230.38 MiB | 127.94 MiB | 62,672 | 36,871 | 38,096 | 0 |

Result: a working set exceeding the configured device-memory and
host-memory budgets was governed across CUDA device memory, pinned system
RAM, and NVMe **without OOM, budget overshoot, corruption, or shutdown
failure**. Peak device usage stayed below the 256 MiB budget (the 10%
reserve margin is enforced), peak host usage stayed below 128 MiB, and all
8192 pages verified with zero integrity failures after repeated
promote/demote/reload cycles.

## Measured transfer paths

64 MiB tier bandwidth benchmark (best of 8), measured on the RTX 5090 with
CUDA event timing where supported:

| Path | Throughput |
|---|---|
| host memcpy | 27.90 GB/s |
| pinned host → CUDA device | 28.92 GB/s |
| CUDA device → pinned host | 28.69 GB/s |
| host → NVMe | 3.23 GB/s |
| NVMe → host | 9.60 GB/s |
| NVMe → host → CUDA device | 7.13 GB/s |
| CUDA device → host → NVMe | 2.78 GB/s |

NVMe paths are explicitly staged through host memory — there is no
direct NVMe-to-GPU DMA path in v0.1. These measurements are specific to
the validated machine, driver, and backing volume; they are reference
points, not guarantees.

## Cornerstone target

FlashTier targets high-compute GPUs stranded between consumer memory
limits and datacenter capacity. The NVIDIA GeForce RTX 5090 is the first
**validation device**, not a hardcoded product boundary: it pairs
frontier-class compute with a 32 GB VRAM ceiling, making it a compelling
surface for proving governed oversubscription. The architecture is
capability-driven and intended for CUDA GPUs generally; other Blackwell
products and datacenter GPUs are potentially applicable targets, but no
untested GPU is described as validated.

## What FlashTier is and is not

- FlashTier does not make SSD storage as fast as device memory. NVMe
  bandwidth, latency, and endurance differ fundamentally from
  accelerator-local memory, and FlashTier never claims otherwise.
- FlashTier does not "turn SSD into VRAM." It is an explicit placement,
  paging, prefetching, transfer, eviction, and telemetry runtime: every
  decision is made by its own planner and recorded in telemetry.
- FlashTier is not a thin wrapper around `cudaMallocManaged`. The
  governed runtime uses explicit device allocations; Unified Memory
  appears only in a bounded comparison benchmark.
- DirectStorage and GPUDirect Storage are not implemented in the
  validated path. Current CUDA storage movement is
  **NVMe ↔ explicitly managed host memory ↔ CUDA device memory**.
- FlashTier does not claim performance superiority where measurements do
  not prove it. In the bounded Unified Memory comparison, the explicit
  path measured slower than the driver-managed path, and that result is
  reported as-is.

## Architecture overview

```
                        +---------------------------+
                        |      Planner + Policy      |
                        |   (LRU / predictive)       |
                        +-------------+-------------+
                                      |
    page table / state machine  +-----+-----+   telemetry (JSONL + human)
    (authority for residency)   |  Runtime  |   (local only, schema v1.0)
                                      |
                          +-----------+-----------+
                          |   DeviceBackend contract |
                          | (opaque streams/events, |
                          |  capability discovery,   |
                          |  typed errors)           |
          +----------------+----+--------+---------+------+
          |        |             |        |         |      |
       cuda     hip      level_zero   vulkan     metal    cpu
   (validated) (gated)  (disabled)   (disabled) (disabled) (emulation)
          +----------------+-------------+----------------+
          |                |             |
    Host backend      NVMe backend   Tier::Vram /
    (pinned via the   (IOCP async     Tier::HostPinned / Tier::Nvme
    device backend)    file store)
```

Governed tiers:

- **accelerator-local device memory** — represented publicly by
  `Tier::Vram` for compatibility; the name is historical and now denotes
  device memory across backend implementations that satisfy the backend
  contract;
- **pinned host memory**;
- **NVMe**.

Runtime policy, page-table logic, integrity, telemetry, and NVMe behavior
are vendor-neutral: the runtime core contains no vendor API calls. Pages
(or tensor-like regions) carry stable IDs, residency state, dirty/pin
metadata, access statistics, semantic class, execution hints, and
checksums, and every decision is recorded. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the full model and the
vendor-neutral contract in
[`include/flashtier/backends/device_backend.hpp`](include/flashtier/backends/device_backend.hpp).

## Backend and validation matrix

| Backend | Vendor | Build status | Runtime status | Hardware validation | Current claim |
|---|---|---|---|---|---|
| cuda | NVIDIA | compiled | available | **validated** on RTX 5090 / sm_120 / Windows | conformance, transfer, oversubscription, integrity, and shutdown passed |
| hip | AMD | source implemented, compile-gated | n/a (no ROCm toolchain locally) | not validated | written, not compiled locally |
| level_zero | Intel | containment stub | unavailable | not validated | fails closed; prior SDK-free ABI implementation was unsafe |
| vulkan | cross-vendor | containment stub | unavailable | not validated | fails closed; no conforming host-memory contract is implemented |
| metal | Apple | containment stub | unavailable | not validated | fails closed; no conforming raw-host-pointer contract is implemented |
| cpu | portable | compiled | available | validated | contract conformance and CPU/RAM/NVMe runtime tests passed; no GPU execution claimed |

The detailed authority is
[docs/validation-matrix.md](docs/validation-matrix.md), which records the
proof sequence every backend must pass before it is called hardware-
validated.

## Supported platforms

| Platform | Status | Notes |
|---|---|---|
| Windows | **first-class validated** | CPU and CUDA builds validated; NVMe backend uses overlapped I/O via I/O completion ports |
| Linux | supported source/build target | CPU and CUDA architecture present; no hardware validation recorded yet on this repository's validation machine |
| NVIDIA CUDA | RTX 5090 / Blackwell sm_120 **validated** | other CUDA architectures are capability-driven and expected to build with appropriate architecture flags (`CMAKE_CUDA_ARCHITECTURES`); untested architectures remain unvalidated |
| AMD | HIP backend implemented, compile-gated | no local toolchain or hardware validation |
| Intel | Level Zero backend fails closed | requires a specification-header-based implementation and hardware validation |
| Apple | Metal backend fails closed | requires a conforming implementation and macOS hardware validation |

CPU-only builds (`FLASHTIER_ENABLE_CUDA=OFF` or `--no-cuda`) build, test,
and run the core runtime, planner, eviction, NVMe store, telemetry, and
integrity layers with no accelerator dependency.

## Prerequisites

Core:

- CMake ≥ 3.24
- A C++20 compiler: MSVC (Visual Studio 2022) or GCC/Clang
- An NVMe or any filesystem path for the storage tier (a fast NVMe volume
  makes NVMe-tier results meaningful)

CUDA (optional, for the CUDA backend):

- CUDA toolkit ≥ 12.0 (12.8+ recommended for Blackwell `sm_120`; the
  installed toolkit is detected, never hardcoded)
- NVIDIA driver
- A supported NVIDIA GPU

Other backends (all optional):

- HIP: a ROCm/HIP toolchain (`FLASHTIER_ENABLE_HIP=ON`)

Level Zero, Vulkan, and Metal options currently build containment stubs that
report `Unsupported`; they are not usable accelerator backends.

Nothing beyond the core prerequisites is mandatory.

## Build

CMake presets are defined in `CMakePresets.json`
(`windows-cuda-debug`, `windows-cuda-release`, `windows-cpu-release`,
`linux-cuda-release`, `linux-cpu-release`). The Windows presets target the
Visual Studio 17 2022 generator; the Windows CUDA preset additionally
requires the CUDA MSBuild integration, which is part of the CUDA Visual
Studio integration package. Where that integration is absent (as on the
validation machine), the validated CUDA build used Ninja with an explicit
`nvcc` under the Visual Studio `vcvars64` environment:

```powershell
# Windows, CUDA (Ninja + explicit nvcc; run inside the VS x64 developer prompt)
cmake -S . -B build/windows-cuda-release -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_CUDA_COMPILER=nvcc -DFLASHTIER_ENABLE_CUDA=ON `
      -DFLASHTIER_BUILD_TESTS=ON -DFLASHTIER_BUILD_BENCHMARKS=ON `
      -DFLASHTIER_BUILD_EXAMPLES=ON
cmake --build build/windows-cuda-release --parallel

# Windows, CPU-only
cmake --preset windows-cpu-release
cmake --build build/windows-cpu-release --config Release --parallel

# Linux
cmake --preset linux-cuda-release
cmake --build build/linux-cuda-release
```

Options (all optional): `FLASHTIER_ENABLE_CUDA`, `FLASHTIER_ENABLE_HIP`,
`FLASHTIER_ENABLE_LEVEL_ZERO`, `FLASHTIER_ENABLE_VULKAN`,
`FLASHTIER_ENABLE_METAL`, `FLASHTIER_ENABLE_CPU_BACKEND`,
`FLASHTIER_ENABLE_DIRECTSTORAGE`, `FLASHTIER_ENABLE_GDS`,
`FLASHTIER_BUILD_TESTS`, `FLASHTIER_BUILD_BENCHMARKS`,
`FLASHTIER_BUILD_EXAMPLES`, `FLASHTIER_WARNINGS_AS_ERRORS`. A build
succeeds with any subset of accelerator backends enabled; CPU-only builds
remain fully supported.

## Test

Current validated record:

```powershell
# CPU correctness suite (no benchmarks)
ctest --test-dir build/windows-cpu-release -C Release -V -LE benchmark

# CUDA/GPU validation suite (labeled gpu)
ctest --test-dir build/windows-cuda-release -C Release -V -L gpu --timeout 180
```

Validated results (current suite):

- CPU Release CTest: **17 passed, 1 intentionally skipped** (the CUDA
  Unified Memory benchmark), 0 failures
- Generic backend contract: mock and CPU backends passed
- CUDA device-backend contract: **16/16 checks** on the RTX 5090
- Tiny CUDA round trip: 16 MiB, zero mismatches
- CUDA runtime oversubscription: three-tier residency with integrity
- CUDA shutdown: repeated cycles, reopen, no tracked allocation leaks
- Final focused GPU regression: **4/4 tests passed**, including runtime,
  oversubscription, shutdown, and Unified Memory comparison

Test counts describe the current suite and will grow with the project.

## Quick-start validation

The fastest validated demonstration is the bounded three-tier
oversubscription test. The validated record from the RTX 5090 run:

```text
three-tier oversubscription geometry:
  page size: 64.00 KiB
  device budget: 256.00 MiB
  device reserve margin: 0.10 (effective 230.40 MiB usable)
  host budget: 128.00 MiB
  nvme budget: 1.00 GiB
  working set: 512.00 MiB (8192 pages)
  oversubscription ratio vs device budget: 2.00x
peak residency: device=3686 pages (230.38 MiB) host=2047 pages (127.94 MiB)
                nvme=8192 pages (512.00 MiB)
aggregates: evictions=62672 writebacks=36871 demand_faults=0
            promotions=38096 integrity_failures=0
```

Reproduce it with:

```powershell
ctest --test-dir build/windows-cuda-release -C Release -R "^test_cuda_oversubscription$" -V --timeout 180
```

or explore the tiers directly:

```powershell
flashtier inspect --backend cuda --device 0
flashtier benchmark tiers --backend cuda --device 0 --working-set 64MiB
flashtier benchmark oversubscription --backend cuda --vram-budget 256MiB --host-budget 128MiB
```

Benchmark defaults are intentionally bounded for local validation; larger
runs require explicit sizes and are validated against detected free
capacity before any allocation (see [BENCHMARKS.md](BENCHMARKS.md)).

## Interpreting telemetry

- `demand faults` and per-tier hit rates show how much of the working set
  actually misses device memory.
- `evictions` vs `writebacks`: writebacks are dirty pages that had to be
  persisted to NVMe — minimize them with clean demotion or write-behind.
- `stalled` is time threads spent waiting on transfers; prefetch is
  worthwhile when it lowers stalls without inflating `prefetch waste`.
- `transfer volume per path` shows where bandwidth actually goes.
  Transfer paths are backend-aware, e.g. `cuda_device->host_pinned`,
  `host_pinned->cuda_device`, `nvme->host_pinned`, `host_pinned->nvme`.
  Compound paths such as NVMe → host → CUDA device are staged sequences
  composed of recorded transfers.

JSONL telemetry uses the format defined by
[`schemas/telemetry.schema.json`](schemas/telemetry.schema.json) (schema
version 1.0). All telemetry is local; nothing leaves the machine.

## Unified Memory comparison

A bounded 512 MiB comparison benchmark measures FlashTier's explicit
placement against CUDA Unified Memory (`cudaMallocManaged`) on the same
machine, working-set size, and page size. Each leg uses a deterministic
verification workload, but their access operations are not identical, so
the result is not presented as an apples-to-apples throughput comparison:

- the driver-managed path completed with **zero mismatches**;
- this WDDM driver rejects `cudaMemPrefetchAsync`/`cudaMemAdvise` at
  runtime, and FlashTier's capability reporting reflects that truthfully
  instead of claiming support from property bits;
- FlashTier's explicit placement measured **slower** in this comparison;

The purpose is mechanism comparison — what each strategy does under the
same bounded load — not a performance-superiority claim, and Unified
Memory is not FlashTier's implementation mechanism.

## Current limitations (v0.1)

- CUDA is the only hardware-validated GPU backend; validation covers the
  RTX 5090 (Blackwell sm_120) on Windows.
- Single GPU; no multi-GPU placement.
- NVMe transport is staged through pinned host memory; no DirectStorage
  or GPUDirect Storage data path.
- Generic byte regions only; no tensor metadata adapter, real-model
  inference adapter, or framework integration (llama.cpp, PyTorch, vLLM
  research only, see [ROADMAP.md](ROADMAP.md)).
- No hardware validation yet for AMD or Linux CUDA, or for NVIDIA
  architectures other than sm_120 on the RTX 5090. Level Zero, Vulkan, and
  Metal intentionally fail closed pending conforming implementations.
- Windows WDDM imposes implicit synchronization and disallows
  `concurrentManagedAccess` on many systems; the runtime uses explicit
  copies and reports capabilities truthfully.
- The public API and backend ABI are v0.1 and subject to change.

HIP is implemented and compile-gated (not "not implemented"): it awaits a
ROCm toolchain and AMD hardware to run the same proof sequence.

## Version status

FlashTier v0.1.0 is the first hardware-proven proof-of-concept release of
the governed heterogeneous-memory mechanism. It is not production-ready,
and it is not merely experimental scaffolding: the mechanism has survived
real device-memory oversubscription with exact integrity on actual GPU
hardware. The founder determines when development is complete.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Paul Ngen.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). No CLA is required; the
security policy is in [SECURITY.md](SECURITY.md), the conduct policy in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and methodology for benchmark
results in [BENCHMARKS.md](BENCHMARKS.md).
