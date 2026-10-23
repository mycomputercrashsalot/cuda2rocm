/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade numa_region_pinned_host_memory_resource — ROCm real implementation.
//! mmap, mbind to the gpu's numa node, hipHostRegister. deallocate is the
//! reverse. if we can't find the node, or mbind fails, we still allocate,
//! just unbound. mem_limit caps total outstanding bytes the same way the
//! fixed_size pool does it.

#pragma once
#ifndef RMM_DEVICE_ASYNC_RESOURCE_REF_STUBBED
#include <rmm/cuda_stream.hpp>
#include <rmm/resource_ref.hpp>
#endif
#include <hip/hip_runtime_api.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>
#endif

namespace cucascade::memory {

class numa_region_pinned_host_memory_resource {
 public:
  // upstream is what the resource_ref conversion hands back. mem_limit caps
  // total outstanding bytes, 0 = no cap.
  numa_region_pinned_host_memory_resource(int32_t device_id,
                                          rmm::device_async_resource_ref upstream,
                                          std::size_t mem_limit = 0,
                                          std::size_t /*slab_size*/ = 0)
    : device_id_(device_id), upstream_(upstream), mem_limit_(mem_limit) {}

  // test/benchmark ctor. no upstream, no limit. make_portable registers the
  // pages portable so every gpu in the box can dma to them, not just ours.
  numa_region_pinned_host_memory_resource(int32_t device_id, bool make_portable)
    : device_id_(device_id), portable_(make_portable) {}

  void* allocate(rmm::cuda_stream_view, std::size_t bytes,
                 std::size_t alignment = 0) {
    if (bytes == 0) return nullptr;
    if (!do_reserve(bytes)) {
      throw std::runtime_error(
        std::string("cuCascade: numa_region_pinned_host_memory_resource OOM: cannot allocate ")
        + std::to_string(bytes) + " bytes within memory limit "
        + std::to_string(mem_limit_));
    }

#ifndef __linux__
    do_release(bytes);
    throw std::runtime_error(
      "cuCascade: numa_region_pinned_host_memory_resource requires Linux "
      "(mmap + mbind + hipHostRegister path)");
#else
    long page = page_size();
    // alignment bigger than a page: over-allocate and carve an aligned
    // window out of it. we register and unmap the whole mapping (base is
    // page-aligned, length a page multiple) rather than the window, that's
    // what hipHostRegister wants anyway.
    std::size_t raw_len = bytes;
    if (alignment > static_cast<std::size_t>(page)) raw_len = bytes + alignment;
    std::size_t map_len = round_up(raw_len, static_cast<std::size_t>(page));

    void* base = mmap(nullptr, map_len, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
      do_release(bytes);
      throw std::runtime_error(
        "cuCascade: mmap failed in numa_region_pinned_host_memory_resource::allocate");
    }

    // bind before first touch so the pages fault in on the right node.
    // if mbind fails we keep going without the binding. a working alloc
    // matters more than a strictly placed one.
    int32_t node = numa_node();
    if (node >= 0) {
      std::vector<unsigned long> nodemask(static_cast<std::size_t>(node) / 64 + 1, 0UL);
      nodemask[static_cast<std::size_t>(node) / 64] = 1UL << (node % 64);
      (void)syscall(SYS_mbind, base, map_len, MPOL_BIND,
                    nodemask.data(), nodemask.size() * 64, 0);
    }

    unsigned int flags = portable_ ? hipHostRegisterPortable : hipHostRegisterDefault;
    hipError_t err = hipHostRegister(base, map_len, flags);
    if (err != hipSuccess) {
      munmap(base, map_len);
      do_release(bytes);
      throw std::runtime_error(
        std::string("cuCascade: hipHostRegister failed in "
                    "numa_region_pinned_host_memory_resource::allocate: ")
        + hipGetErrorString(err));
    }

    void* user = base;
    if (alignment > static_cast<std::size_t>(page)) {
      uintptr_t addr = reinterpret_cast<uintptr_t>(base);
      uintptr_t aligned = (addr + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
      user = reinterpret_cast<void*>(aligned);
    }
    {
      std::lock_guard<std::mutex> lk(mutex_);
      live_allocs_[user] = {base, map_len};
    }
    return user;
#endif
  }

  void deallocate(rmm::cuda_stream_view, void* ptr, std::size_t bytes,
                  std::size_t /*alignment*/ = 0) {
    if (!ptr) return;
#ifndef __linux__
    (void)bytes;
    throw std::runtime_error(
      "cuCascade: numa_region_pinned_host_memory_resource requires Linux "
      "(munmap + hipHostUnregister path)");
#else
    void* base = nullptr;
    std::size_t map_len = 0;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      auto it = live_allocs_.find(ptr);
      if (it == live_allocs_.end()) {
        throw std::runtime_error(
          "cuCascade: numa deallocate of a pointer this resource did not allocate");
      }
      base = it->second.first;
      map_len = it->second.second;
      live_allocs_.erase(it);
    }
    // hipHostUnregister can block, keep it out of the lock
    (void)hipHostUnregister(base);
    munmap(base, map_len);
    do_release(bytes);
#endif
  }

  operator rmm::device_async_resource_ref() const {
    if (!upstream_) {
      throw std::runtime_error(
        "cuCascade: numa_region_pinned_host_memory_resource has no upstream "
        "resource (constructed with the test/benchmark constructor)");
    }
    return *upstream_;
  }

  std::size_t get_total_allocated_bytes() const {
    return total_allocated_.load(std::memory_order_relaxed);
  }

  /// which numa node is this gpu on. match the drm device to the gpu by pci
  /// bus id and read its numa_node file. same trick topology_discovery uses,
  /// just not calling it directly so we don't pull in that whole header.
  /// -1 if we can't tell.
  static int32_t resolve_numa_node(int32_t device_id) {
    if (device_id < 0) return -1;
    char gpu_pci[32] = {0};
    if (hipDeviceGetPCIBusId(gpu_pci, sizeof(gpu_pci), device_id) != hipSuccess) {
      return -1;
    }
    for (int render = 128; render < 200; ++render) {
      std::string dir = "/sys/class/drm/renderD" + std::to_string(render) + "/device";
      if (!dir_pci_matches(dir, gpu_pci)) continue;
      return read_numa_file(dir + "/numa_node");
    }
    for (int card = 0; card < 32; ++card) {
      std::string dir = "/sys/class/drm/card" + std::to_string(card) + "/device";
      if (!dir_pci_matches(dir, gpu_pci)) continue;
      return read_numa_file(dir + "/numa_node");
    }
    return -1;
  }

 private:
  /// node lookup, resolved once on first allocate
  int32_t numa_node() {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!numa_resolved_) {
      numa_node_ = resolve_numa_node(device_id_);
      numa_resolved_ = true;
    }
    return numa_node_;
  }

  static long page_size() {
    long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? page : 4096;
  }

  static std::size_t round_up(std::size_t bytes, std::size_t granularity) {
    if (granularity == 0) return bytes;
    if (bytes > SIZE_MAX - (granularity - 1)) return SIZE_MAX;
    return ((bytes + granularity - 1) / granularity) * granularity;
  }

  /// -1 means unknown. careful not to return 0 on error, 0 is a real node.
  static int32_t read_numa_file(std::string const& path) {
    std::ifstream f(path);
    if (!f) return -1;
    int32_t node = -1;
    if (!(f >> node)) return -1;
    return node;
  }

  /// is this drm device dir our gpu? read PCI_SLOT_NAME out of its uevent
  static bool dir_pci_matches(std::string const& device_dir, char const* gpu_pci) {
    std::ifstream f(device_dir + "/uevent");
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("PCI_SLOT_NAME=", 0) == 0) {
        return pci_buses_match(gpu_pci, line.substr(14).c_str());
      }
    }
    return false;
  }

  /// same device? compare bus:device, the .Z function suffix can differ
  static bool pci_buses_match(char const* a, char const* b) {
    if (!a || !b) return false;
    std::string sa(a), sb(b);
    auto dot_a = sa.rfind('.');
    auto dot_b = sb.rfind('.');
    if (dot_a != std::string::npos) sa = sa.substr(0, dot_a);
    if (dot_b != std::string::npos) sb = sb.substr(0, dot_b);
    return sa == sb;
  }

  /// charge the counter if it still fits under mem_limit_. 0 = no limit.
  /// same cas loop as fixed_size_host_memory_resource (it's private there,
  /// or i'd call it).
  bool do_reserve(std::size_t bytes) {
    if (bytes == 0) return true;
    if (mem_limit_ != 0) {
      if (bytes > mem_limit_) return false;
      std::size_t cur = total_allocated_.load(std::memory_order_relaxed);
      while (true) {
        if (cur > mem_limit_ - bytes) return false;
        if (total_allocated_.compare_exchange_weak(cur, cur + bytes,
                std::memory_order_relaxed)) {
          break;
        }
      }
    } else {
      total_allocated_.fetch_add(bytes, std::memory_order_relaxed);
    }
    return true;
  }

  /// give the bytes back, clamped at 0
  void do_release(std::size_t bytes) {
    if (bytes == 0) return;
    std::size_t cur = total_allocated_.load(std::memory_order_relaxed);
    while (true) {
      std::size_t next = (cur >= bytes) ? (cur - bytes) : 0;
      if (total_allocated_.compare_exchange_weak(cur, next,
              std::memory_order_relaxed)) {
        break;
      }
    }
  }

  int32_t device_id_{0};
  std::optional<rmm::device_async_resource_ref> upstream_;
  std::size_t mem_limit_{0};
  bool portable_{false};

  std::atomic<std::size_t> total_allocated_{0};

  mutable std::mutex mutex_;
  int32_t numa_node_{-1};
  bool numa_resolved_{false};
  /// user ptr -> (mmap base, len). with big alignments the user ptr is not
  /// the mapping base, so deallocate needs both.
  std::unordered_map<void*, std::pair<void*, std::size_t>> live_allocs_;
};

}  // namespace cucascade::memory
