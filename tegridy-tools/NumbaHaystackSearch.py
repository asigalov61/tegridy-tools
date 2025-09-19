#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	    Numba Haystack Search Python Module
#	    Version 1.0
#
#	    Project Los Angeles
#
#	    Tegridy Code 2025
#
#       https://github.com/asigalov61/tegridy-tools
#
#
################################################################################
#
#       Copyright 2025 Project Los Angeles / Tegridy Code
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

################################################################################

try:
    from numba import cuda
    print('=' * 70)
    print('Numba is found!')
    print('Will use Numba and GPU for processing!')
    print('=' * 70)

except ImportError as e:
    print(f"Error: Could not import Numba. Details: {e}")
    print("Please make sure Numba is installed.")
    print('=' * 70)
    
    raise RuntimeError("Numba could not be loaded!") from e

################################################################################

import numpy as np
from numba import cuda

################################################################################

@cuda.jit
def _search_kernel(
    hay_data, hay_offsets, needle, needle_len,
    potential, res_offsets, counters, matches, seq_start
):
    # local thread/block indices
    local_seq = cuda.blockIdx.y
    block_id  = cuda.blockIdx.x
    tid       = cuda.threadIdx.x
    tpb       = cuda.blockDim.x

    # map to global sequence
    seq_id = seq_start + local_seq
    n_pos  = potential[seq_id]
    start_pos = block_id * tpb + tid
    if start_pos >= n_pos:
        return

    # compare needle
    base = hay_offsets[seq_id] + start_pos
    for i in range(needle_len):
        if hay_data[base + i] != needle[i]:
            return

    # record match slot
    idx = cuda.atomic.add(counters, seq_id, 1)
    matches[res_offsets[seq_id] + idx] = start_pos

def NumbaHaystackSearch(haystack, needle):
    """
    haystack: list of int16 arrays/lists
    needle:   int16 array/list
    returns:  list of lists of match-start-indices per sequence
    """
    n_seqs     = len(haystack)
    needle_arr = np.asanyarray(needle, dtype=np.int16)
    m          = needle_arr.size

    # 1) flatten haystack + compute offsets
    lengths = np.array([len(s) for s in haystack], dtype=np.int32)
    offsets = np.empty(n_seqs+1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    total_vals = int(offsets[-1])

    hay_data = np.empty(total_vals, dtype=np.int16)
    for i, seq in enumerate(haystack):
        seq_arr = np.asanyarray(seq, dtype=np.int16)
        start   = offsets[i]
        hay_data[start:start+lengths[i]] = seq_arr

    # 2) how many possible start positions per sequence
    potential = np.maximum(0, lengths - m + 1)
    if potential.max() <= 0:
        return [[] for _ in range(n_seqs)]

    # 3) prefix‐sum of potentials → result‐offsets
    res_offsets = np.empty(n_seqs+1, dtype=np.int32)
    res_offsets[0] = 0
    np.cumsum(potential, out=res_offsets[1:])
    total_pot = int(res_offsets[-1])

    # 4) host buffers for atomics + matches
    matches_host  = np.empty(total_pot, dtype=np.int32)
    counters_host = np.zeros(n_seqs, dtype=np.int32)

    # 5) copy everything GPU
    d_data      = cuda.to_device(hay_data)
    d_offsets   = cuda.to_device(offsets)
    d_needle    = cuda.to_device(needle_arr)
    d_potential = cuda.to_device(potential)
    d_res_off   = cuda.to_device(res_offsets[:-1])
    d_counters  = cuda.to_device(counters_host)
    d_matches   = cuda.to_device(matches_host)

    # 6) launch parameters
    max_pot        = int(potential.max())
    tpb            = 256 if max_pot >= 256 else max_pot
    blocks_per_seq = (max_pot + tpb - 1) // tpb

    # 7) BATCH on the y‐dimension limit = 65 535
    MAX_GRID_Y = 65_535

    for seq_start in range(0, n_seqs, MAX_GRID_Y):
        batch_size = min(MAX_GRID_Y, n_seqs - seq_start)
        grid_dims  = (blocks_per_seq, batch_size)
        _search_kernel[grid_dims, tpb](
            d_data, d_offsets, d_needle, np.int32(m),
            d_potential, d_res_off, d_counters, d_matches,
            np.int32(seq_start)
        )

    # 8) fetch results back
    d_counters.copy_to_host(counters_host)
    d_matches.copy_to_host(matches_host)

    # 9) split into per‐sequence lists
    results = []
    for i in range(n_seqs):
        cnt   = int(counters_host[i])
        start = int(res_offsets[i])
        results.append(matches_host[start:start+cnt].tolist())

    return results

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Numba Haystack Search Python module
###################################################################################