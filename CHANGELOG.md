# Changelog

All notable changes to FlashTier are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-05

### Added

- Three-tier governed runtime: VRAM, pinned host RAM, NVMe, with explicit
  placement, paging, prefetching, transfer, eviction, and telemetry.
- Page table with explicit state machine (`Unallocated`, `ResidentNvme`,
  `LoadingToHost`, `ResidentHost`, `LoadingToVram`, `ResidentVram`,
  `EvictingToHost`, `EvictingToNvme`, `Error`, `Released`) and illegal
  transition rejection.
- Deterministic LRU and temperature-aware predictive placement policies
  with stable page-ID tie-breaking.
- Deterministic eviction with pinning, in-flight, dirty-writeback, and
  reserve-margin guarantees; typed budget errors when no victim exists.
- Explicit prefetch, sequential prefetch hints, bounded prefetch queue,
  cancellation, deduplicated demand fallback, hit/waste accounting.
- CUDA VRAM backend: device enumeration, capability reporting, explicit
  allocations, streams/events, async copies, typed error mapping, GPU fill
  and verify kernels.
- Host backend: pinned allocation (CUDA when available), pageable fallback
  only when explicitly enabled, budget/high-water accounting.
- NVMe backend: Windows overlapped I/O via I/O completion ports, Linux
  threaded `pread`/`pwrite`, preallocation, free-space map, stable
  page-to-offset mapping, clean-page write avoidance, store header
  validation, cancellation and deterministic shutdown.
- Structured telemetry: human-readable and JSON Lines (schema v1.0,
  `schemas/telemetry.schema.json`), local only, with aggregations.
- Deterministic integrity: seed+page-id-derived content, FNV-1a 64
  checksums, full verification checkpoints.
- CLI (`flashtier`): `inspect`, `capabilities`, `verify`, `benchmark`
  (`tiers`, `oversubscription`, `prefetch`, `sparse-experts`,
  `unified-memory`), `--version`, strict byte-size parsing, JSONL output,
  exit codes.
- Benchmarks: tier bandwidth, oversubscribed working sets (1.0×/1.25×/
  1.5×/2.0×), prefetch comparison, sparse-expert MoE-like trace, bounded
  Unified Memory comparison.
- Examples: `minimal_tiered_buffer`, `oversubscribed_working_set`.
- CPU-only build mode with full core unit test coverage.
- CMake presets for Windows CUDA debug/release, Windows CPU release, and
  Linux CUDA/CPU release.
- Licensing and governance: Apache License 2.0, NOTICE, CONTRIBUTING,
  CODE_OF_CONDUCT, SECURITY, README, ARCHITECTURE, ROADMAP, BENCHMARKS.

### Known limitations (v0.1)

- Generic byte regions only; no tensor metadata adapter yet.
- Single GPU; no multi-GPU placement.
- No DirectStorage, GPUDirect Storage, or HBF backends (interfaces defined;
  see ROADMAP.md).
- Windows WDDM disables `concurrentManagedAccess` on many systems; the
  runtime reports capability truthfully and uses explicit copies.
- Level Zero, Vulkan, and Metal fail closed until conforming backend
  implementations pass their hardware proof sequences.
- No real-model inference integration yet (research only).
