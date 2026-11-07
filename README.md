# Kernel Programming in CUDA and Numba

This repository contains high-performance GPU program implementations, optimization strategies, and detailed hardware-level profiling comparison reports between different generations of NVIDIA GPUs (NVIDIA T4 vs. NVIDIA A100).

## Table of Contents
- [Overview](#overview)
- [Programs Included](#programs-included)
  - [1. 2D Max Pooling Kernel (`maxpool2d`)](#1-2d-max-pooling-kernel-maxpool2d)
- [Nsight Compute Profiling & Methodology](#nsight-compute-profiling--methodology)
- [Hardware Performance Comparison Report (T4 vs. A100)](#hardware-performance-comparison-report-t4-vs-a100)
  - [Performance Metrics Summary](#performance-metrics-summary)
  - [Architectural Performance Analysis](#architectural-performance-analysis)
- [How to Run and Profile](#how-to-run-and-profile)

---

## Overview
This project focuses on **Kernel Programming** using **NVIDIA Numba** (Python-based CUDA JIT compiler) and **CuPy** for fast GPU array management. The primary goal is to implement standard parallel patterns (like 2D Max Pooling) from scratch, analyze assembly instruction metrics (SASS), and profile runtime behavior to identify bottlenecks.

---

## Programs Included

### 1. 2D Max Pooling Kernel (`maxpool2d`)
An implementation of the 2D Max Pooling operation commonly used in Convolutional Neural Networks (CNNs). It downsamples input matrices by selecting the maximum value in designated local tiles.

* **Kernel Logic**: Each thread maps to one output element in the output matrix and loops over a local slice of size `pool_size x pool_size` in the input matrix.
* **Implementation**:
```python
@cuda.jit
def maxpool2d(input_mat, output_mat, poolsize):
  out_col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
  out_row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

  if (out_row >= output_mat.shape[0]) or (out_col >= output_mat.shape[1]):
    return

  in_row_mat = out_row * poolsize
  in_col_mat = out_col * poolsize
  in_max_val = input_mat[in_row_mat, in_col_mat]

  for r in range(poolsize):
    for c in range(poolsize):
      in_max_val = max(in_max_val, input_mat[in_row_mat+r, in_col_mat+c])

  output_mat[out_row, out_col] = in_max_val
```

---

## Nsight Compute Profiling & Methodology
We leveraged **NVIDIA Nsight Compute CLI (`ncu`)** to capture low-level hardware metrics of our custom kernel execution. The metrics evaluated are:
* `gpu__time_duration.sum` (Total execution duration in microseconds)
* `smsp__sass_inst_executed.sum` (Total SASS assembly instructions executed across all Streaming Multiprocessors)

---

## Hardware Performance Comparison Report (T4 vs. A100)

### Performance Metrics Summary
We executed the exact same kernel configuration (Matrix size: `4000x4000`, Pooling Window: `4x4`, Thread Block size: `32x32`) across two different GPU architectures:

| Metric | NVIDIA T4 (Turing) | NVIDIA A100-SXM4-40GB (Ampere) | Performance Ratio (A100 vs. T4) |
| :--- | :---: | :---: | :---: |
| **GPU Execution Time** | **357.95 µs** | **63.20 µs** | **~5.66x Faster** |
| **SASS Instructions Executed** | **20,586,752** | **20,778,752** | **1.01x (Equivalent Assembly Code)** |

### Architectural Performance Analysis
1. **Memory Bandwidth Bottleneck Cleared**: Max Pooling is highly **memory bandwidth bound** because it has low arithmetic intensity—it performs 16 global memory reads per output element calculation. While the T4 operates on GDDR6 memory (~320 GB/s), the A100 SXM4 utilizes ultra-high-speed HBM2 memory, delivering up to ~1.5 - 2.0 TB/s bandwidth (~5x to 6x improvement). This matches the **5.66x reduction in execution latency** perfectly.
2. **Instruction Consistency**: The total assembly instructions (SASS) remain almost identical. This confirms that performance variations stem entirely from hardware capabilities (memory bus width, cache sizes, memory subsystem design) rather than compiler optimizations.

---

## How to Run and Profile

To generate and profile the kernel locally under Nsight Compute:
```bash
ncu --metrics gpu__time_duration.sum,smsp__sass_inst_executed.sum --target-processes all python3 maxpool_profile.py
```
