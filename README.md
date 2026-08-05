# FlashTier

FlashTier is an open-source heterogeneous-memory runtime that governs AI
workload residency across GPU VRAM, system RAM, and NVMe storage.

## Cornerstone target

FlashTier targets high-compute GPUs stranded between consumer VRAM limits and
datacenter memory capacity, beginning with the NVIDIA GeForce RTX 5090. The
RTX 5090 pairs exceptional compute throughput with a 32 GB VRAM ceiling.

**The central invariant:** when a working set exceeds physical VRAM, FlashTier
replaces abrupt failure with explicit, observable, policy-governed degradation.

## What FlashTier is — and is not

- **FlashTier does not make SSD storage as fast as VRAM.** NVMe bandwidth,
  latency, and endurance characteristics are fundamentally different from
  device memory, and FlashTier never claims otherwise.
- **FlashTier v0.1 uses explicit tier management.** Every page placement,
  transfer, eviction, and prefetch decision is made by FlashTier's own
  planner and is recorded in telemetry. FlashTier is not a thin wrapper
  around `cudaMallocManaged`, and it does not claim that Windows
  DirectStorage transparently exposes NVMe as CUDA-addressable memory.
- The v0.1 transport path is **NVMe ↔ explicitly managed host buffer ↔ CUDA
  VRAM**. Future backends (DirectStorage, GPUDirect Storage, High Bandwidth
  Flash) may bypass the host staging step, but nothing is claimed until it
  is implemented and measured.

## Architecture overview

```
                      +---------------------------+
                      |      Planner + Policy      |
                      |   (LRU / predictive)       |
                      +-------------+-------------+
                                    |
  page table / state machine  +----+-----+    telemetry (JSONL + human)
  (authority for residency)   | Runtime  |    (local only, schema v1.0)
                                    |
        +---------------+ +---------+---------+ +----------------+
        | VRAM backend  | | Host backend      | | NVMe backend   |
        | (CUDA, vram)  | | (pinned host RAM) | | (IOCP async)   |
        +-------+-------+ +---------+---------+ +-------+--------+
                |                     |                   |
            Tier::Vram          Tier::HostPinned     Tier::Nvme
```

Pages (or tensor-like regions) carry stable IDs, residency state, dirty/pin
metadata, access statistics, semantic class, execution hints, and checksums.
The runtime moves them between three governed tiers, and every decision is
recorded. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full model.

## Supported platforms

| Platform        | Status          | Notes                                    |
|-----------------|-----------------|------------------------------------------|
| Windows 10/11   | **v0.1 first-class** | MSVC + CUDA; NVMe backend uses overlapped I/O via I/O completion ports |
| Linux           | supported build | CPU-only and CUDA builds; NVMe backend uses threaded pread/pwrite |
| GPU support     | NVIDIA CUDA     | sm_120 (RTX 5090) verified; older CC works with appropriate arch flags |
| DirectStorage   | experimental probe only | No v0.1 integration; see ROADMAP.md |
| GPUDirect Storage | not implemented | Linux-first future backend; see ROADMAP.md |
| AMD ROCm        | not implemented | Future backend; see ROADMAP.md |

CPU-only mode (`FLASHTIER_ENABLE_CUDA=OFF`) builds, tests, and runs the core
runtime, planner, eviction, NVMe store, telemetry, and integrity layers
without any CUDA dependency.

## Prerequisites

- CMake ≥ 3.24
- A C++20 compiler: MSVC (Visual Studio 2022) or GCC/Clang
- CUDA toolkit ≥ 12.0 (required only for CUDA mode; 12.8+ recommended for
  Blackwell `sm_120` targets). The installed toolkit is detected, never
  hardcoded.
- An NVIDIA GPU for CUDA benchmarks (e.g., RTX 5090)
- A fast NVMe volume for meaningful NVMe-tier results (any filesystem path
  works)

## Build

```powershell
# Windows, CUDA
cmake --preset windows-cuda-release
cmake --build build/windows-cuda-release --config Release --parallel

# Windows, CPU-only
cmake --preset windows-cpu-release
cmake --build build/windows-cpu-release --config Release --parallel
```

```bash
# Linux
cmake --preset linux-cuda-release
cmake --build build/linux-cuda-release
```

Presets: `windows-cuda-debug`, `windows-cuda-release`, `windows-cpu-release`,
`linux-cuda-release`, `linux-cpu-release`. Options include
`FLASHTIER_ENABLE_CUDA`, `FLASHTIER_ENABLE_DIRECTSTORAGE`,
`FLASHTIER_ENABLE_GDS`, `FLASHTIER_BUILD_TESTS`, `FLASHTIER_BUILD_BENCHMARKS`,
`FLASHTIER_BUILD_EXAMPLES`, `FLASHTIER_WARNINGS_AS_ERRORS`.

## Test

```powershell
ctest --test-dir build/windows-cuda-release -C Release --output-on-failure
```

The full unit suite (state machine, policy, eviction, NVMe store, telemetry,
integrity, runtime cycles) runs in CPU-only mode. CUDA allocation/copy tests
run only when CUDA is enabled and a GPU is present.

## Quick-start benchmark

```powershell
flashtier benchmark tiers
flashtier benchmark oversubscription
flashtier benchmark prefetch
flashtier benchmark sparse-experts
```

Example output (oversubscription run, RTX 5090):

```
=== FlashTier benchmark: oversubscription ===
device: NVIDIA GeForce RTX 5090 (cc 12.0)
working set: 4.00 GiB | vram budget: 2.00 GiB | host budget: 2.00 GiB | nvme budget: 4.00 GiB
phase ratio=1.00 logical=2.00GiB pages=1024
  demand faults=0  vram hits=100.0%  throughput=... ops/s
phase ratio=2.00 logical=4.00GiB pages=2048
  demand faults=480  vram hits=80.4%  host hits=19.6%  nvme hits=0.0%
  evictions=480  writebacks=2  stalled=31.2ms  throughput=... ops/s
integrity: 2048/2048 pages verified OK
```

Exact numbers depend on hardware and configuration.

## Interpreting telemetry

- `demand faults` and per-tier hit rates show how much of the working set
  actually misses VRAM.
- `evictions` vs `writebacks`: writebacks are dirty pages that had to be
  persisted to NVMe — minimize them with clean demotion or write-behind.
- `stalled` is time threads spent waiting on transfers; prefetch is
  worthwhile when it lowers stalls without inflating `prefetch waste`.
- `transfer volume per path` shows where bandwidth actually goes
  (NVMe↔host↔VRAM); compare it to the tier bandwidth benchmark.

JSONL telemetry is validated against
[`schemas/telemetry.schema.json`](schemas/telemetry.schema.json) (schema
version 1.0). All telemetry is local; nothing leaves the machine.

## Current limitations (v0.1)

- Generic byte regions only; no tensor metadata adapter yet.
- Windows WDDM imposes implicit sync and disallows `concurrentManagedAccess`
  on many systems; the runtime uses explicit copies and reports capability
  truthfully.
- The NVMe tier is a file-backed store; DirectStorage/GDS bypass paths do
  not exist yet.
- Single GPU; no multi-GPU placement.
- No real-model inference integration (llama.cpp, PyTorch, vLLM research
  only, see [ROADMAP.md](ROADMAP.md)).

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Copyright 2026 Paul Ngen.

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). No CLA is required; the
security policy is in [SECURITY.md](SECURITY.md), the conduct policy in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and methodology for benchmark
results in [BENCHMARKS.md](BENCHMARKS.md).
