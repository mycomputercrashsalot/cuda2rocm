/*
 * Copyright 2026, Sirius Contributors.
 * Licensed under the Apache License, Version 2.0 (see LICENSE).
 */
//! @file cuCascade disk_table — ROCm real implementation.

#pragma once
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cucascade::memory {

struct disk_table_allocation {
  std::string file_path;
  std::size_t size_bytes{0};
};

inline std::string generate_disk_file_path(std::string const& base, std::size_t idx) {
  return base + "/cucascade_disk_" + std::to_string(idx) + ".bin";
}

/// copy `bytes` from src to dst, a megabyte at a time. throws if either
/// file won't open, or if src runs out before `bytes`.
inline void clone_disk_file(std::string const& src, std::string const& dst,
                            std::size_t bytes) {
  std::ifstream in(src, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cuCascade: clone_disk_file: cannot open source file " + src);
  }
  std::ofstream out(dst, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("cuCascade: clone_disk_file: cannot create destination file " + dst);
  }
  constexpr std::size_t CHUNK = 1 << 20;  // 1 MiB
  std::vector<char> buf(std::min(CHUNK, bytes));
  std::size_t remaining = bytes;
  while (remaining > 0) {
    std::size_t n = std::min(remaining, buf.size());
    in.read(buf.data(), static_cast<std::streamsize>(n));
    if (static_cast<std::size_t>(in.gcount()) != n) {
      throw std::runtime_error(
        "cuCascade: clone_disk_file: source file " + src + " is shorter than expected");
    }
    out.write(buf.data(), static_cast<std::streamsize>(n));
    remaining -= n;
  }
}

}  // namespace cucascade::memory
