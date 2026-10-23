/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade disk_data_representation — ROCm real implementation.
//! clone() copies the backing file and wraps the copy, so the clone lives
//! independently of the original file.

#pragma once
#include "cucascade/data/common.hpp"
#include "cucascade/memory/common.hpp"
#include "cucascade/memory/disk_table.hpp"
#ifndef RMM_DEVICE_ASYNC_RESOURCE_REF_STUBBED
#include <rmm/cuda_stream.hpp>
#endif
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace cucascade {

class disk_data_representation : public idata_representation {
 public:
  disk_data_representation(std::unique_ptr<memory::disk_table_allocation> alloc,
                           memory::memory_space& space)
    : idata_representation(space), alloc_(std::move(alloc)) {}

  memory::disk_table_allocation const& get_disk_table() const {
    if (!alloc_) {
      throw std::runtime_error("cuCascade: disk_data_representation has no disk table allocation");
    }
    return *alloc_;
  }

  /// copy the backing file to <path>.clone.<n> and wrap that. the counter
  /// keeps the names unique across clones. plain file io, the stream is
  /// ignored. an empty rep clones to an empty rep.
  std::unique_ptr<idata_representation> clone(rmm::cuda_stream_view) override {
    if (!alloc_) {
      return std::make_unique<disk_data_representation>(
        std::unique_ptr<memory::disk_table_allocation>(), get_memory_space());
    }
    static std::atomic<std::uint64_t> clone_counter{0};
    std::string dst = alloc_->file_path + ".clone." +
      std::to_string(clone_counter.fetch_add(1, std::memory_order_relaxed));
    memory::clone_disk_file(alloc_->file_path, dst, alloc_->size_bytes);
    auto cloned = std::make_unique<memory::disk_table_allocation>();
    cloned->file_path = dst;
    cloned->size_bytes = alloc_->size_bytes;
    return std::make_unique<disk_data_representation>(std::move(cloned), get_memory_space());
  }

  std::size_t get_size_in_bytes() const override { return alloc_ ? alloc_->size_bytes : 0; }
  std::size_t get_uncompressed_data_size_in_bytes() const override { return get_size_in_bytes(); }

 private:
  std::unique_ptr<memory::disk_table_allocation> alloc_;
};

}  // namespace cucascade
