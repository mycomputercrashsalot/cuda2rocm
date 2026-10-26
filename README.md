# rocm-cuda-compat

CUDA→ROCm compatibility tools for porting NVIDIA-only GPU libraries to AMD HIP.

## Projects

### 1. `cuda-compat-shims` — Reusable CUDA→HIP shim headers ✅ Verified

The shim layer developed for the [Sirius ROCm port](https://github.com/hamstergamerszhang-ops/sirius/tree/feat/rocm-port), extracted as a standalone header-only library usable by any CUDA→ROCm porting effort:

- `cuda_runtime.h` — 68 `cuda*`→`hip*` macro aliases + 6 type aliases (verified on gfx942)
- `cub/` → hipCUB redirect + `namespace cub = hipcub` + `warp_threads = 64`
- `cuda/std/*` — CCCL redirects with `#include_next` (defers to hipDF's CCCL when installed)
- `cuda/functional` — `cuda::minimum`/`cuda::maximum` functors
- `cuda/stream_ref`, `cuda/cmath`, `cuda/__memory/aligned_size.h`, `cuda/atomic`, `cuda/memory_resource`
- `cooperative_groups` memcpy_async/wait serial fallback
- `nvtx3` → `roctx` shim
- `curand.h` → `hiprand.h` redirect

**Status:** ✅ Verified on real gfx942/ROCm 7.2.1 (0 compile errors, 6 benign warnings).

**Usage:**
```cmake
find_package(rocm-cuda-compat CONFIG REQUIRED)
target_link_libraries(my-target PRIVATE rocm-cuda-compat::cuda-compat-shims)
```

### 2. `cuco-rocm` — cuCollections port for ROCm/HIP 🔧 In Progress

cuCollections (cuco) is NVIDIA's GPU hash-map and Bloom-filter library, used by Sirius, RAPIDS, and the broader CUDA ecosystem. No ROCm port exists.

**Scope:** Port the subset of cuco that Sirius uses:
- `cuco::bloom_filter` (blocked Bloom filter) — 🔧 Compiles in CI, runtime pending
- `cuco::static_set` (open-addressing hash set) — 🔧 Compiles in CI, runtime pending
- `cuco::xxhash_64` (device-compatible xxHash) — ✅ Implemented
- Supporting types: `cuco::extent`, `cuco::empty_key`, `cuco::arrow_filter_policy`,
  `cuco::default_filter_policy`, `cuco::double_hashing`, `cuco::default_hash_function`,
  `cuco::contains`

**Approach:** Use rocPRIM/hipCUB primitives (warp shuffles, atomicCAS) as building blocks, matching cuco's API surface exactly so downstream code compiles without changes. Target: `namespace cuco`, same headers, same templates.

**Status:** Drafted (6 headers). The test binaries (`test_bloom_filter.cu`,
`test_static_set.cu`) compile with hipcc in this repo's CI on every run and
assert 7/7 keys found / 6/6 correct. What's still missing is a real run —
GitHub-hosted runners have no /dev/kfd, so CI is compile-only there. An
earlier version of this README's "Verified on real hardware" section listed
specific throughput figures (~3M keys/s, ~3.4M keys/s) for these; those
claims are unreproduced and have been removed pending a real run. See the
"Verified on real hardware" section below for what is and isn't actually
verified.

### 3. `cucascade-rocm` — cuCascade memory reservation port 🔧 In Progress

cuCascade is NVIDIA's GPU memory-reservation and out-of-core data repository system. 63 files in Sirius depend on it. No ROCm port exists.

**Scope:** Port the core subsystem:
- `cucascade::memory::memory_space` — stream + allocator + device context
- `cucascade::memory::memory_reservation_manager` — reservation tracking
- `cucascade::memory::reservation` — reserved capacity handle
- `cucascade::memory::topology_discovery` — multi-GPU topology (rocm-smi/sysfs)
- `cucascade::data::data_batch` / `data_repository` — batch lifecycle

**Approach:** Use hipMM's `rmm::device_async_resource_ref` as the allocator interface, rocThrust/hipCUB for data movement. Match cuCascade's API exactly.

**Status:** All 32 headers are real implementations — the last two stubs (the disk-data representation and the NUMA-aware pinned allocator) are now implemented:

- `data/disk_data_representation.hpp` — `clone()` copies the backing file to a new unique path (`memory::clone_disk_file` in `disk_table.hpp`) and wraps the new allocation; `get_size_in_bytes()` reports the allocation's byte count.
- `memory/numa_region_pinned_host_allocator.hpp` — `allocate()` mmaps, binds the pages to the GPU's local NUMA node with `mbind(2)` (node resolved from sysfs by PCI bus ID, same scan as `topology_discovery`), then pins with `hipHostRegister`; `deallocate()` unpins and unmaps. If the node can't be resolved or mbind fails, the allocation still works, just not node-bound. Admission is bounded by `mem_limit` with the same single-counter CAS the fixed-size pool uses.

The full memory-reservation subsystem (`memory_space`, `reservation`, `memory_reservation_manager`, `fixed_size_host_memory_resource`, `reservation_aware_resource_adaptor`, `oom_handling_policy`, `error`, `small_pinned_host_memory_resource`), the full topology + event subsystem, the data-repository containers (`data_repository`, `data_repository_manager`), the data-batch lifecycle (`data_batch` clone/convert/set_data), the representation types (`gpu_table_representation::clone` via cudf::copy, `host_data_representation::clone` via hipMemcpyAsync), the representation converter registry, and the config/common/POD types are all ported.

Verification status, plainly: the host-logic test (`tests/test_cucascade_compile.cpp`) compiles and passes on Linux, now covering the file-clone path with real I/O and the allocator's construction/guard paths. The allocator's `hipHostRegister` path and the mbind binding still need a run on real gfx942 hardware to be called verified — the same gap as the rest of the runtime side of this port.

See `cucascade-rocm/docs/api-surface.md` for the full API mapping and which scope items remain.

## Build

```bash
# Configure — use hipcc as CXX compiler (needed for -x hip flags)
cmake -B build -S . \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_C_COMPILER=hipcc

# Build tests (requires GPU for runtime)
cmake -B build -S . \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DCMAKE_C_COMPILER=hipcc \
  -DBUILD_TESTS=ON
cmake --build build
ctest --test-dir build

# Install
cmake --install build --prefix /opt/rocm
```

### Verification status

What's actually been verified, and what hasn't (stated plainly, because this
repo is a public credibility reference and unverified claims are worse than
no claim):

- **Shim layer:** 0 compile errors on gfx942/ROCm 7.2.1 (hipcc 7.2.53211) —
  verified in sirius-rocm's CI (`.github/workflows/rocm-test.yml`, green).
- **cucascade-rocm pure-logic headers:** 4 of the 30 real headers
  (`column_metadata`, `notification_channel`, `disk_table`,
  `chunked_resource_info`) compile clean with g++ -std=c++20 (no HIP
  dependency). The remaining 26 real headers transitively include
  `hip/hip_runtime.h` via the cuda-compat-shims and require a HIP toolchain
  to compile-verify — a cross-compilation CI job
  (`.github/workflows/compile-test.yml`) now compiles the shim layer, the
  cuco-rocm tests, and the host-logic test with hipcc --offload-arch=gfx942
  on every push/PR (no GPU needed for compilation; runtime execution is
  gated on /dev/kfd).
- **cuco-rocm bloom_filter / static_set:** **NOT verified.** The test
  binaries are written and assert 7/7 keys found / 6/6 correct, but have not
  been compiled with hipcc or run on a real AMD GPU. An earlier version of
  this README listed throughput figures (~3M keys/s, ~3.4M keys/s) and a
  "0.03% FP rate" for these — those figures are unreproduced and have been
  removed. They will be filled in with real numbers once a self-hosted ROCm
  runner is registered.
- **hipMM (RMM for HIP):** built + installed to /opt/rocm on real gfx942
  (per Sirius's build_rocm_deps.sh).

### Known build fixes (incorporated into Sirius's build_rocm_deps.sh)

1. **hipcc as C/CXX compiler** — hipMM's .cpp files need `-x hip --offload-arch`
2. **`git config --global http.version HTTP/1.1`** — fixes flaky GitHub clones
3. **`ROCM_AMDGPU_TARGETS=gfx942` env** — avoids rapids_cmake arch mismatch
4. **Pre-clone all 16 CPM deps** — correct branch names from versions.json
5. **Patch get_rmm.cmake** — use find_package(rmm) instead of CPM re-fetch

## License

Apache-2.0 (matching cuco, cuCascade, and Sirius)
