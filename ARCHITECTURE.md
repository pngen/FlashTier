# FlashTier Architecture

Version: 1.1 (v0.1.0 scope). This document describes the implemented
runtime as shipped. Future changes are tracked in ROADMAP.md.

## 0. Vendor-neutral accelerator contract

The governed runtime never calls vendor APIs directly. All accelerator
behavior sits behind `include/flashtier/backends/device_backend.hpp`:

- opaque execution queues (streams) and events;
- structured capability discovery (allocation, async copies, pinned host,
  unified/shared memory, prefetch/advice, direct storage, peer-to-peer,
  multi-device, event timing, limits, alignment, granularity);
- typed error translation (CUDA errors map to `ErrorCode::Cuda`, other
  backends to `ErrorCode::Device`);
- a backend registry with compiled-backend listing, runtime availability
  probes, explicit and automatic selection, deterministic order, and
  per-backend failure isolation.

Capabilities are discovered at runtime, never guessed from product names.
The CUDA backend live-probes managed-memory prefetch/advice because some
WDDM drivers report `managedMemory` but reject the hint APIs.

Backends shipped in v0.1: `cuda` (validated on the RTX 5090), `cpu`
(emulation), `level_zero` and `vulkan` (self-declared API surfaces resolved
from runtime loaders; compile without SDKs), `hip` and `metal`
(compile-gated). A backend is considered validated only after its
conformance, transfer, oversubscription, integrity, budget, and shutdown
tests pass on real hardware — see docs/validation-matrix.md.

`Tier::Vram` remains the user-facing name for accelerator-local device
memory across vendors (CUDA, HIP, Level Zero, Vulkan, Metal); it denotes
device memory, not NVIDIA-only memory.

## 1. Memory tiers

FlashTier governs page residency across three primary tiers plus one
diagnostic tier:

| Tier             | Backend             | Notes                                             |
|------------------|---------------------|---------------------------------------------------|
| `Tier::Vram`     | CUDA device memory  | Explicit `cudaMalloc`; never silently OOMs        |
| `Tier::HostPinned`| pinned host RAM     | `cudaMallocHost` when CUDA available; pageable `_aligned_malloc`/`aligned_alloc` otherwise |
| `Tier::Nvme`     | file-backed store    | Windows: overlapped I/O on an I/O completion port; Linux: threaded `pread`/`pwrite` |
| `Tier::HostPageable` | pageable RAM    | Explicit diagnostic/fallback only; never silently substitutes pinned memory in performance claims |

Tier selection is governed by configured budgets (VRAM, host, NVMe) with
reserve margins. The runtime is a **placement, paging, prefetching,
transfer, eviction, and telemetry** engine — it never claims SSD is
physically equivalent to VRAM.

### Transfer paths

The v0.1 transport path is:

```
NVMe  <-- async file I/O -->  pinned host staging  <-- CUDA async copy -->  VRAM
```

Every inter-tier move is composed of atomic one-hop steps. The architecture
permits future backends (DirectStorage, GPUDirect Storage, HBF) to bypass
the host staging step behind the `StorageBackend` interface; nothing bypasses
it in v0.1.

## 2. Authority and coherence

The **page table is the single authority** for residency, physical handles,
storage offsets, transfer state, access metadata, dirty state, pin state,
checksums, and memory accounting. All mutations go through the page table
under a lock.

Coherence rules (v0.1):

1. Exactly one authoritative (writable) copy of a page exists at any time.
2. The authoritative copy is identified by the page state:
   `ResidentVram` → device pointer; `ResidentHost` → pinned host pointer;
   `ResidentNvme` → store offset.
3. During a transfer (`LoadingToVram`, `LoadingToHost`, `EvictingToHost`,
   `EvictingToNvme`) the source location remains authoritative until the
   transfer completes, after which authority atomically moves to the
   destination. The page is marked in-flight and all accesses block until
   the state settles.
4. Never allow the same mutable page to have two authoritative writable
   copies. Staging buffers used during transfers are transient, are never
   writable by callers, and never become authoritative.
5. Read-only replicated clean copies are permitted only when an explicit
   coherence mode says so. v0.1 has exactly one copy per page; there is no
   replication.
6. Dirty pages are written back to NVMe before the authoritative copy is
   released. Clean pages whose NVMe copy is still valid are never rewritten.

## 3. Page state machine

```
                      +--------- Unallocated ---------+
                      |       |          |            |
                      |       |          |            v
                      |       |          |    LoadingToVram (alloc)
                      |       |          v              |
                      |       |    ResidentHost <-------+   (failed transfer: Error)
                      |       |          |            |
                      |       |          v            v
                      |       |   LoadingToVram    EvictingToNvme
                      |       |          |            |
                      |       |          v            v
                      |       |    ResidentVram    ResidentNvme
                      |       |          |            |
                      |       |          v            v
                      |       |   EvictingToHost   LoadingToHost
                      |       |          |            |
                      |       |          v            v
                      |       +------ ResidentHost  ResidentVram
                      |                    |          |
                      +--------------------+----------+
                                              |
                                           Released
```

States: `Unallocated`, `ResidentNvme`, `LoadingToHost`, `ResidentHost`,
`LoadingToVram`, `ResidentVram`, `EvictingToHost`, `EvictingToNvme`,
`Error`, `Released`.

Legal transitions (all others rejected):

| From             | To                                    |
|------------------|---------------------------------------|
| Unallocated      | LoadingToVram, ResidentHost, ResidentVram, ResidentNvme, Released |
| LoadingToVram    | ResidentVram, Error                   |
| LoadingToHost    | ResidentHost, Error                   |
| ResidentVram     | EvictingToHost, ResidentNvme, Error, Released |
| ResidentHost     | LoadingToVram, EvictingToNvme, Error, Released |
| ResidentNvme     | LoadingToHost, Error, Released        |
| EvictingToHost   | ResidentHost, Error                   |
| EvictingToNvme   | ResidentNvme, Error                   |
| Error            | Released                             |
| Released         | (terminal)                           |

Notes:

- Allocation into VRAM goes `Unallocated → LoadingToVram → ResidentVram`
  (async H2D copy of freshly written data).
- Allocation into host is synchronous: `Unallocated → ResidentHost`.
- Allocation into NVMe stages through host: `Unallocated → ResidentHost →
  EvictingToNvme → ResidentNvme`.
- `ResidentVram → ResidentNvme` is the clean drop: the page has a valid
  NVMe copy and no transfer is needed (single authority moves without a
  copy).
- There is no direct VRAM↔NVMe hop in v0.1; both directions route through
  the host staging tier by design.

## 4. Planner and policies

The planner accepts VRAM/host/NVMe budgets, reserve margins, page metadata,
current access, future-access hints, transfer costs, semantic class,
pinning constraints, dirty state, and measured history. It decides:

- whether a demand load can proceed (free VRAM minus reserve ≥ needed);
- which pages to evict and to which tier, when headroom is insufficient;
- where new allocations land.

Two policies ship in v0.1, both deterministic (tie-broken by stable page ID
so identical runs produce identical decisions):

1. **Deterministic LRU** — victims are the pages with the oldest
   last-access sequence. Baseline only; not presented as optimal.
2. **Temperature-aware predictive** — scores pages with a weighted
   combination of recency, access frequency, expected next-use distance,
   semantic class, transfer cost, page size, dirty-write penalty, explicit
   execution hints, and pin state. The same score drives victim selection
   and prefetch promotion. It is a documented heuristic, not a proof.

Constraints honored by eviction: never evict pinned pages; never evict
pages with in-flight transfers; prefer clean over dirty when scores permit;
write dirty pages before dropping authority; keep the VRAM reserve
headroom; fail with a typed budget error when no legal victim exists.

## 5. Prefetch

- Explicit prefetch by page ID (`prefetch({id...})`).
- Sequential prefetch hints (`prefetch_sequential(id, count)`).
- Next-use hints (future-access hints consumed by the predictive policy).
- Configurable prefetch depth bounds the prefetch queue; the queue is
  separate from the demand path and is filled with lower priority.
- Prefetch operations can be cancelled: queued-but-unstarted prefetches are
  dropped on eviction/free/over-capacity; in-flight ones complete and are
  accounted as waste if never used.
- Demand fallback: a demand fault for a page that is queued for prefetch
  promotes that request to demand priority (deduplicated), so a prefetch
  miss costs only the demand latency, never a duplicate transfer.
- Telemetry separates prefetch hits (page accessed after prefetch
  completed) from prefetch waste (page evicted or never accessed).

## 6. Concurrency

- One **runtime lock** protects the page table and accounting.
- A **transfer engine** owns a bounded pool of worker threads
  (`queue_depth`, default 8). Workers execute whole page-level chains
  (e.g., `ensure host space → read NVMe → H2D`) sequentially per chain, so
  a chain never waits on another chain (no deadlock: victims are chosen
  from idle pages and evicted by the same worker).
- The NVMe backend performs **asynchronous** overlapped I/O on Windows
  (I/O completion port with a bounded worker pool) with per-operation
  completion tracking; on Linux it uses threaded `pread`/`pwrite` with the
  same `AsyncOp` completion interface.
- CUDA transfers are issued on dedicated streams with events; workers wait
  on events.
- Bounded queues: demand requests are bounded by `queue_depth`, prefetch by
  `prefetch_depth`. No unbounded thread creation, no detached threads, no
  fire-and-forget `std::thread` (std::thread members are joined in
  deterministic shutdown).
- Shutdown: stop flag → workers finish current chains (NVMe ops support
  cancellation where possible; in-flight CUDA copies drain via stream
  synchronize) → join all threads → release backends. Worker exceptions are
  captured into typed errors and reported, never silently terminating the
  process.

## 7. Failure handling

- Every CUDA call is checked and mapped to `ErrorCode::Cuda`.
- Every file op is checked and mapped to `ErrorCode::Io` /
  `ErrorCode::StoreCorrupt`.
- Illegal state transitions raise `ErrorCode::State`.
- Broken budgets, no-legal-victim, and reserve violations raise
  `ErrorCode::Budget`.
- Data mismatches raise `ErrorCode::Integrity` with page ID, expected vs
  actual, and the latest transfer path.
- CLI exit codes: `0` success, `1` error, `2` usage/config, `3` corruption,
  `4` unsupported capability, `5` benchmark failure, `6` integrity failure,
  `7` I/O or CUDA fatal.

## 8. Telemetry

Every placement and transfer decision emits an event with timestamp,
monotonic sequence number, page ID, event type, source/destination tiers,
bytes, duration, bandwidth, policy, reason, queue depth, per-tier usage,
hit/miss and prefetch hit/waste flags, stall duration, checksum result, and
CUDA error when present. Human-readable and JSON Lines forms are supported;
the JSONL schema is versioned (`schemas/telemetry.schema.json`, version
1.0). Aggregation covers per-path bytes/counts, percentile latencies where
volume permits, effective per-path bandwidth, tier hit rates, demand-fault
counts, prefetch statistics, evictions, writebacks, stalled time, maximum
residency per tier, integrity failures, and workload throughput. All
telemetry is local; nothing is transmitted.

## 9. Backend interfaces

- `StorageBackend` (abstract): open/close, extent allocate/free, async
  read/write, completion tracking, cancellation, flush, integrity of the
  store metadata, recovery validation. Implemented by `NvmeBackend`;
  future `DirectStorageBackend` / `GdsBackend` / `HbfBackend` slot in here
  and may bypass host staging.
- `VramBackend` (CUDA): device enumeration/selection, attribute reporting
  (name, capability, total/free VRAM, unified memory, concurrent managed
  access), explicit allocations, streams, async copies, events, bandwidth
  measurement, GPU fill/verify kernels, typed error mapping.
- `HostBackend`: pinned allocation (CUDA when available, pageable fallback
  only when explicitly enabled), budget accounting, high-water tracking,
  telemetry distinguishing pinned vs pageable.

### DirectStorage boundary (v0.1)

- Optional compile-time probe only (`FLASHTIER_ENABLE_DIRECTSTORAGE`),
  isolated behind the storage interface, never required, never described as
  transparent CUDA NVMe paging. Production DirectStorage→CUDA would require
  deliberate D3D12/CUDA interop plus validation.

### GPUDirect Storage boundary (v0.1)

- A future-compatible backend interface exists (`StorageBackend`). Linux
  builds compile without GDS; `FLASHTIER_ENABLE_GDS` and cuFile detection
  are future work. No Windows GDS claims; integration path is Linux-first
  (see ROADMAP.md).

## 10. Integrity

- Deterministic content: each byte of a page derives from
  `f(seed, page_id, block_index, byte_index)` (splitmix64-based), so any
  run can regenerate expected data.
- Fast checksum: FNV-1a 64 over page content, maintained on write and
  verified on read/evict cycles for routine telemetry.
- Full deterministic verification at defined checkpoints: after upload,
  after every demotion/promotion, after repeated eviction cycles, and at
  workload completion.
- Any mismatch identifies the page, expected vs actual (checksum or byte
  sample), and the latest transfer path, then terminates the run with a
  failure exit code.

## 11. Hardware scope

v0.1 targets the RTX 5090 (compute capability 12.0, sm_120) but runs on any
CUDA-capable NVIDIA GPU with a suitable `CMAKE_CUDA_ARCHITECTURES` setting
(e.g., `native` detects the installed GPU at configure time). The design is
extensible to lower-VRAM GPUs, future consumer GPUs, workstation
accelerators, DGX Spark-class systems, GDS-capable Linux hosts, future HBF
hardware, and eventual AMD backends (see ROADMAP.md).
