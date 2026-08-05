# FlashTier Cross-Vendor Validation Matrix

FlashTier is designed for GPUs generally through a vendor-neutral
device-backend contract. Each vendor backend is considered **validated**
only after its conformance, transfer, oversubscription, integrity, budget,
and shutdown tests pass on real hardware. This matrix records the current
truthful status; entries change only when the proof sequence is executed
on the corresponding hardware.

Legend: **compiled** = the module builds on the platform; **runtime
detected** = the backend's loader/runtime and a device are present;
**hardware tested** = the proof sequence ran on real hardware.

| Backend | Vendor | Platform | Compiled | Runtime detected | Hardware tested | Conformance passed | Transfer benchmark | Oversubscription | Direct storage | Limitations |
|---|---|---|---|---|---|---|---|---|---|---|
| cuda | NVIDIA | Windows (validated), Linux | yes | yes | yes (RTX 5090, cc 12.0) | yes (18-point battery) | yes (64 MiB) | yes (device→host→NVMe, 2× budget) | no (v0.1) | WDDM: managed-memory prefetch/advice rejected by driver; CAM=no |
| hip | AMD | Windows/Linux when ROCm installed | gated (requires HIP toolchain) | no (no ROCm on validation machine) | no | no (pending hardware) | no | no | no | source written, not compiled locally |
| level_zero | Intel | Windows/Linux | yes (runtime-loader, no SDK) | no (no loader/driver) | no | no (pending hardware) | no | no | no | self-declared API surface; not runtime-validated |
| vulkan | cross-vendor | Windows/Linux | yes (runtime-loader, no SDK; experimental) | no (no loader/driver) | no | no | no | no | no | experimental; storage buffers are not CUDA-compatible VRAM |
| metal | Apple | macOS | gated (Apple only) | no (Windows host) | no | no | no | no | no | architecture present; not compiled/executed in this pass |
| cpu | portable | all | yes | yes | yes | yes | yes | yes (host/NVMe) | n/a | emulation; no GPU execution claimed |

## Proof sequence per backend

1. Generic DeviceBackend conformance battery (enumeration, invalid-device
   rejection, capabilities, alignment, alloc/release cycles, pinned host,
   H2D/D2H round trips, async streams, event timing, repeated open/close,
   typed errors, budget behavior, concurrent transfers, shutdown).
2. Vendor-specific discovery and tiny allocation round trip.
3. Bounded tier bandwidth benchmark.
4. Governed-runtime residency test (device oversubscription into host).
5. True three-tier oversubscription (device → host → NVMe) with peak
   residency, eviction/writeback/reload counts, and zero integrity
   failures.
6. Deterministic shutdown with in-flight work.

## CUDA validation record (this environment)

- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GiB VRAM.
- Driver 610.82 (CUDA UMD 13.3), toolkit 12.9.86.
- Conformance battery: passed on real hardware.
- Tiny round trip: 16 MiB, byte-for-byte + integrity pattern, zero
  mismatches.
- Tier benchmark (64 MiB): host memcpy 27.90 GB/s, pinned→device
  28.92 GB/s, device→pinned 28.69 GB/s, host→NVMe 3.23 GB/s, NVMe→host
  9.60 GB/s, NVMe→host→device 7.13 GB/s, device→host→NVMe 2.78 GB/s.
- Three-tier oversubscription (device 256 MiB, host 128 MiB, NVMe 1 GiB,
  working set 512 MiB, 64 KiB pages): peak device 230.38 MiB (≤ budget),
  peak host 127.94 MiB (≤ budget), peak NVMe 512 MiB; 62,672 device
  evictions; 36,871 NVMe writebacks; 38,096 promotions; 0 integrity
  failures; no budget overshoot; no CUDA OOM.
- Unified Memory comparison (512 MiB working set): driver-managed path
  completed with 0 mismatches; this WDDM driver rejects
  cudaMemPrefetchAsync/cudaMemAdvise at runtime (recorded, not claimed);
  FlashTier's explicit path measured slower on this micro-benchmark and no
  performance claim is made.

## Untested hardware paths (explicitly not claimed)

- AMD (HIP/ROCm): no ROCm toolchain or AMD GPU on the validation machine.
- Intel (Level Zero): no Level Zero loader/driver or Intel GPU.
- Cross-vendor Vulkan: experimental module compiles; no loader/driver.
- Apple Metal: architecture only; no macOS build environment in this pass.
- NVIDIA GPUs other than the RTX 5090: not executed here; behavior is
  capability-driven and expected to generalize, but each device family
  requires the same proof sequence before being called validated.
