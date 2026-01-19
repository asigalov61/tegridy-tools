#! /usr/bin/env python3

r'''############################################################################
################################################################################
#
#
#	    Numba Haystack Search Python Module
#	    Version 2.0
#
#	    Project Los Angeles
#
#	    Tegridy Code 2026
#
#       https://github.com/asigalov61/tegridy-tools
#
#
################################################################################
#
#       Copyright 2026 Project Los Angeles / Tegridy Code
#
#       Licensed under the Apache License, Version 2.0 (the "License");
#       you may not use this file except in compliance with the License.
#       You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
#       Unless required by applicable law or agreed to in writing, software
#       distributed under the License is distributed on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#       See the License for the specific language governing permissions and
#       limitations under the License.
#
################################################################################
################################################################################
#
#       Critical dependencies
#
#       !pip install numpy
#       !pip install numba
#
################################################################################
'''

################################################################################

print('=' * 70)
print('Loading module...')
print('Please wait...')
print('=' * 70)

################################################################################

import sys
import os
import math
import warnings

###################################################################################

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("NumPy is required.") from e

###################################################################################

# Try to import Numba/CUDA. If not available, we'll provide a CPU fallback.
USE_CUDA = False

try:
    from numba import cuda, int16 as nb_int16, int32 as nb_int32
    USE_CUDA = True
    print('=' * 70)
    print('Numba CUDA is available. GPU path enabled.')
    print('=' * 70)
except Exception as e:
    print('=' * 70)
    print('Numba CUDA not available. Falling back to CPU implementation.')
    print('Install Numba with CUDA support for GPU acceleration.')
    print('=' * 70)
    USE_CUDA = False

###################################################################################

# Tunable constant: maximum needle length to copy into shared memory per block.
# Keep this moderate to avoid exhausting shared memory per block.
MAX_SHARED_NEEDLE = 1024  # adjust if you know your device has more shared memory

###################################################################################

if USE_CUDA:
    # Kernel: each thread checks one candidate start position within a sequence.
    # Grid layout: grid.x = blocks_per_seq (based on max potential), grid.y = batch_size
    # blockDim.x = threads per block (tpb)
    @cuda.jit
    def _search_kernel_shared(
        hay_data, hay_offsets, needle, needle_len,
        potential, res_offsets, counters, matches, seq_start, use_shared
    ):
        """
        Optimized search kernel:
          - Each thread maps to a candidate start position for a particular sequence.
          - Optionally copies the needle into shared memory once per block for faster access.
        Parameters:
          hay_data: int16 flattened haystack values
          hay_offsets: int32 offsets into hay_data for each sequence
          needle: int16 needle array (global)
          needle_len: int32 length of needle
          potential: int32 number of candidate starts per sequence
          res_offsets: int32 result offsets per sequence (length n_seqs)
          counters: int32 per-sequence atomic counters
          matches: int32 flat array to store match start indices
          seq_start: int32 offset for batch start
          use_shared: int32 flag (1 => copy needle to shared memory, 0 => use global)
        """
        # local thread/block indices
        local_seq = cuda.blockIdx.y
        block_id  = cuda.blockIdx.x
        tid       = cuda.threadIdx.x
        tpb       = cuda.blockDim.x

        seq_id = seq_start + local_seq
        # bounds check for sequences in this batch
        if seq_id >= potential.shape[0]:
            return

        n_pos = potential[seq_id]
        start_pos = block_id * tpb + tid
        if start_pos >= n_pos:
            return

        # compute base index in hay_data for this candidate
        base = hay_offsets[seq_id] + start_pos

        # Optionally copy needle into shared memory (only first block thread group does it)
        # Shared buffer declared dynamically by the compiler; we index into it as if it's an array.
        # We use a simple loop to compare; if use_shared==1 we read from shared, else from global.
        if use_shared == 1:
            # allocate shared memory view
            # Note: Numba maps cuda.shared.array to a block-local buffer; we index into it.
            shared_needle = cuda.shared.array(MAX_SHARED_NEEDLE, dtype=nb_int16)
            # copy needle into shared memory (only first tpb threads do the copy in chunks)
            # We must ensure we don't read/write out of bounds
            i = tid
            while i < needle_len:
                shared_needle[i] = needle[i]
                i += tpb
            cuda.syncthreads()

            # compare using shared memory
            for i in range(needle_len):
                if hay_data[base + i] != shared_needle[i]:
                    return
        else:
            # compare directly from global needle
            for i in range(needle_len):
                if hay_data[base + i] != needle[i]:
                    return

        # record match slot atomically
        idx = cuda.atomic.add(counters, seq_id, 1)
        matches[res_offsets[seq_id] + idx] = start_pos

###################################################################################

def _cpu_search(haystack, needle):
    """
    CPU fallback: vectorized sliding-window search using NumPy stride tricks.
    Returns list of lists of match-start-indices per sequence.
    This is very fast for small-to-moderate workloads and avoids GPU overhead.
    """
    needle_arr = np.asanyarray(needle, dtype=np.int16)
    m = needle_arr.size
    results = []
    if m == 0:
        # empty needle matches at every position by convention -> return empty lists
        return [[] for _ in haystack]

    for seq in haystack:
        seq_arr = np.asanyarray(seq, dtype=np.int16)
        n = seq_arr.size
        if n < m:
            results.append([])
            continue
        # Use stride trick to create a (n-m+1, m) view
        shape = (n - m + 1, m)
        strides = (seq_arr.strides[0], seq_arr.strides[0])
        try:
            windows = np.lib.stride_tricks.as_strided(seq_arr, shape=shape, strides=strides)
            # Compare rows to needle
            # For memory safety, copy windows if it's not safe to view (but as_strided is fine here)
            matches_bool = np.all(windows == needle_arr, axis=1)
            matches_idx = np.nonzero(matches_bool)[0].tolist()
            results.append(matches_idx)
        except Exception:
            # Fallback to naive loop if stride trick fails
            idxs = []
            for i in range(n - m + 1):
                if np.array_equal(seq_arr[i:i+m], needle_arr):
                    idxs.append(i)
            results.append(idxs)
    return results

###################################################################################

def NumbaHaystackSearch(haystack, needle, prefer_gpu=True):
    """
    haystack: list of int16 arrays/lists
    needle:   int16 array/list
    prefer_gpu: bool, if True attempt GPU path when available; otherwise use CPU fallback
    returns:  list of lists of match-start-indices per sequence
    """
    # Basic validation and conversion
    if needle is None:
        raise ValueError("needle must be provided (non-None).")
    n_seqs = len(haystack)
    needle_arr = np.asanyarray(needle, dtype=np.int16)
    m = int(needle_arr.size)

    # Quick edge cases
    if n_seqs == 0:
        return []
    if m == 0:
        # define empty needle -> no matches (consistent with original)
        return [[] for _ in range(n_seqs)]

    # Compute lengths and offsets
    lengths = np.array([len(s) for s in haystack], dtype=np.int32)
    offsets = np.empty(n_seqs + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    total_vals = int(offsets[-1])

    # Flatten haystack into contiguous int16 array
    hay_data = np.empty(total_vals, dtype=np.int16)
    for i, seq in enumerate(haystack):
        seq_arr = np.asanyarray(seq, dtype=np.int16)
        start = int(offsets[i])
        hay_data[start:start + lengths[i]] = seq_arr

    # Potential start positions per sequence
    potential = np.maximum(0, lengths - m + 1)
    if potential.max() <= 0:
        return [[] for _ in range(n_seqs)]

    # Result offsets (prefix sum of potentials)
    res_offsets = np.empty(n_seqs + 1, dtype=np.int32)
    res_offsets[0] = 0
    np.cumsum(potential, out=res_offsets[1:])
    total_pot = int(res_offsets[-1])

    # Heuristic: if problem is small, prefer CPU path to avoid GPU overhead
    # Thresholds chosen conservatively; tune for your hardware.
    SMALL_TOTAL_POT = 1_000  # if fewer than this many candidate positions, CPU is often faster
    if (not USE_CUDA) or (not prefer_gpu) or (total_pot <= SMALL_TOTAL_POT):
        return _cpu_search(haystack, needle_arr)

    # GPU path
    try:
        # Use pinned host memory for faster transfers
        matches_host = cuda.pinned_array(total_pot, dtype=np.int32)
        counters_host = cuda.pinned_array(n_seqs, dtype=np.int32)
        # initialize
        counters_host.fill(0)
        # Device copies
        d_data = cuda.to_device(hay_data)
        d_offsets = cuda.to_device(offsets)
        d_needle = cuda.to_device(needle_arr)
        d_potential = cuda.to_device(potential)
        # res_offsets[:-1] maps each sequence to its result base index
        d_res_off = cuda.to_device(res_offsets[:-1])
        d_counters = cuda.to_device(counters_host)
        d_matches = cuda.to_device(matches_host)

        # Determine device properties and tune tpb
        dev = cuda.get_current_device()
        max_threads_per_block = dev.MAX_THREADS_PER_BLOCK
        # choose tpb as power-of-two up to 256 or device limit
        tpb = 256 if max_threads_per_block >= 256 else max_threads_per_block
        # but don't exceed max potential (no point in more threads than max_pot)
        max_pot = int(potential.max())
        if max_pot < tpb:
            # choose nearest power-of-two <= max_pot
            tpb = 1 << (max_pot.bit_length() - 1) if max_pot > 0 else 1
            if tpb == 0:
                tpb = 1

        # blocks per sequence (based on max potential)
        blocks_per_seq = (max_pot + tpb - 1) // tpb
        # grid.y batch size limit
        MAX_GRID_Y = 65_535
        # decide whether to use shared memory for needle
        use_shared = 1 if (m <= MAX_SHARED_NEEDLE) else 0

        # Launch in batches along y dimension
        for seq_start in range(0, n_seqs, MAX_GRID_Y):
            batch_size = min(MAX_GRID_Y, n_seqs - seq_start)
            grid_dims = (blocks_per_seq, batch_size)
            # Launch kernel
            # We pass use_shared as int32 flag
            _search_kernel_shared[grid_dims, tpb](
                d_data, d_offsets, d_needle, np.int32(m),
                d_potential, d_res_off, d_counters, d_matches,
                np.int32(seq_start), np.int32(use_shared)
            )

        # Ensure kernel finished
        cuda.synchronize()

        # Copy results back
        d_counters.copy_to_host(counters_host)
        d_matches.copy_to_host(matches_host)

        # Build per-sequence lists
        results = []
        for i in range(n_seqs):
            cnt = int(counters_host[i])
            start = int(res_offsets[i])
            if cnt == 0:
                results.append([])
            else:
                # slice and convert to Python list
                results.append(matches_host[start:start + cnt].tolist())

        return results

    except cuda.CudaSupportError as e:
        warnings.warn(f"CUDA support error encountered: {e}. Falling back to CPU.")
        return _cpu_search(haystack, needle_arr)
    except Exception as e:
        warnings.warn(f"GPU path failed with exception: {e}. Falling back to CPU.")
        return _cpu_search(haystack, needle_arr)

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Numba Haystack Search Python module
###################################################################################