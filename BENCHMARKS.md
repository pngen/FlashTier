# FlashTier Benchmark Methodology

This document defines how FlashTier benchmarks are run, what they measure,
and how results may be reported. Benchmarks that do not follow this
methodology are not comparable to each other.

## 1. Methodology

Every benchmark is a deterministic, seeded run:

- A fixed seed (default `0xC0FFEE`) drives the workload generator; the same
  seed + same configuration produces the same access trace and the same
  decisions (policies are deterministic, tie-broken by page ID).
- Benchmarks print the exact configuration before running: page size,
  budgets (VRAM/host/NVMe), policy, prefetch mode and depth, queue depth,
  iterations, seed, device, and GPU.
- Sizes are validated against detected free VRAM, free RAM, and free disk
  before any allocation. Reserve margins are enforced; a benchmark refuses
  to start rather than risk system instability. Explicit larger sizes are
  allowed when the operator overrides and capacity is verified.
- Backing stores are created in a temporary directory (or `--nvme-path`)
  and removed unless `--retain-store` is given.

## 2. Metrics

| Metric | Definition |
|---|---|
| Throughput | workload operations completed per second (accesses to pages) |
| Stall time | total wall time demand accesses spent waiting for transfers |
| Demand faults | accesses that required a page load from a lower tier |
| VRAM / host / NVMe hit rate | fraction of accesses served by each tier |
| Evictions | pages removed from VRAM by policy (not frees) |
| Dirty writebacks | evictions that wrote dirty data to NVMe |
| Transfer volume | bytes moved per path (NVMe→host, host→NVMe, host→VRAM, VRAM→host) |
| Effective bandwidth | bytes / measured duration per path |
| Latency percentiles | p50/p90/p99 of individual transfer durations when sample volume permits (≥100 samples) |
| Prefetch hit rate | prefetched pages accessed after completion / prefetched pages |
| Prefetch waste | prefetched pages evicted or never accessed |
| Max residency | peak bytes resident per tier |
| Integrity failures | pages failing checksum or deterministic-content verification |

## 3. Integrity checks

- Every page content is deterministic: `f(seed, page_id, block, byte)`.
- Verification checkpoints: after upload; after each demotion/promotion;
  after repeated eviction cycles; at workload completion.
- Routine telemetry uses FNV-1a 64 checksums; checkpoints perform full
  content verification.
- A mismatch aborts the benchmark with exit code 6 and identifies page,
  expected vs actual, and latest transfer path. Never reported as success.

## 4. Safe sizing

- VRAM budget ≤ 80% of free VRAM at start (default).
- Host budget ≤ 25% of free system RAM (default; configurable).
- NVMe budget ≤ 25% of free disk on the backing volume (default;
  configurable).
- Oversubscription ratios are applied to the configured VRAM budget, and
  the resulting working set must fit host + NVMe budgets.
- Defaults never allocate dangerous proportions of RAM or disk.

## 5. Reproducibility

A report must include the "configuration block" printed at the top of the
run (or `--jsonl` output), the exact command line, the seed, and the
hardware disclosure fields below. Runs on different machines are not
comparable unless configurations match.

## 6. Hardware disclosure fields

- GPU model, compute capability, total VRAM, driver version, CUDA runtime
  version (from `flashtier inspect`)
- CPU model, cores, system RAM
- OS name/version
- Backing store filesystem and volume (e.g., `NTFS on Samsung 990 Pro
  (NVMe)`); note that Windows NTFS may report the volume generically — the
  physical device class should be stated by the reporter
- Page size, queue depth, policy, prefetch settings

## 7. Submitting results

- Report real numbers from real runs with the configuration block.
- Do not exaggerate: do not claim "VRAM at SSD prices", "unlimited VRAM",
  or that synthetic benchmarks prove complete-model inference speed.
- The sparse-expert benchmark is a synthetic MoE-like access trace — it is
  not a transformer inference engine, and results must not be presented as
  such.
- The Unified Memory comparison compares FlashTier's explicit strategy
  against driver-managed oversubscription with `cudaMallocManaged` +
  `cudaMemPrefetchAsync` on the same machine, same page size, same working
  set; it documents what it compares and does not.
