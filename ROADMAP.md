# FlashTier Roadmap

No dates are promised. Items are ordered roughly by dependency; priorities
are set by the founder as the project evolves. The v0.1.0 milestone is the
explicit placement/paging/prefetch/eviction/telemetry runtime with the
NVMe ↔ host staging ↔ CUDA VRAM transport path.

## v0.2 — real workloads

- [ ] Tensor metadata adapter: n-dimensional logical views over byte
      regions, layout/stride metadata, typed accessors.
- [ ] GGUF / llama.cpp integration: load model tensors as governed pages,
      map weights/kv-cache into tiers by semantic class.
- [ ] PyTorch custom allocator / CUDA extension research (prototype only;
      native runtime remains the dependency-free core).
- [ ] vLLM integration research (prototype only; no vLLM build dependency
      in v0.x).

## v0.3 — workload-driven residency

- [ ] KV-cache paging: class-aware residency and reuse-distance estimation
      for generation workloads; sequence-aware eviction.
- [ ] MoE expert residency: hot/shared expert staging, expert-priority
      prefetch driven by routing hints.
- [ ] Predictive policy tuning: calibration of the temperature-aware
      heuristic on real traces; optional learned weights kept opt-in.

## v0.4 — storage acceleration paths

- [ ] Windows DirectStorage / D3D12 / CUDA interop research and prototype
      behind `StorageBackend`; validation of the host-staging bypass.
- [ ] Linux GPUDirect Storage backend (cuFile) behind `StorageBackend`;
      GDS detection at configure time, never mandatory.
- [ ] High Bandwidth Flash (HBF) backend behind `StorageBackend` when
      hardware becomes available. No claim of HBF support before it exists.

## v1.0 — scale

- [ ] Multi-GPU placement: per-device page tables, peer transfer planning.
- [ ] DGX Spark-class profiling and tuning (small-form-factor datacenter
      machines).
- [ ] AMD ROCm backend: backend abstraction audit, HIP port, capability
      parity on the governed tiers.
- [ ] Distributed storage: only as a later, separate concern, and only
      after local tiers and telemetry are stable. v0.1 deliberately has no
      networking, RPC, or daemon.

## Standing commitments

- No dates.
- No silent fallbacks: any degradation is explicit, observable, and
  policy-governed.
- No telemetry leaves the machine.
- Windows and Linux both stay first-class; optional backends never become
  requirements.
