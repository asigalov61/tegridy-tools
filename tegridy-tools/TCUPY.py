#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	    Tegridy Cupy Python Module (TCUPY)
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
#       Copyright 2024 Project Los Angeles / Tegridy Code
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
#       !pip install cupy-cuda12x
#       !pip install numpy==1.24.4
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
    import cupy as cp
    import cupy as np
    print('=' * 70)
    print('CuPy is found!')
    print('Will use CuPy and GPU for processing!')
    print('=' * 70)

except ImportError as e:
    print(f"Error: Could not import CuPy. Details: {e}")
    # Handle the error, such as providing a fallback or exiting the program
    # For example:
    print("Please make sure CuPy is installed.")
    print('=' * 70)
    
    raise RuntimeError("CuPy could not be loaded!") from e

################################################################################

from collections import defaultdict, deque
from typing import Optional, Tuple, Dict, Any, List

################################################################################

# Constants
MEMORY_LEN = 12       # Autoregressive context length
SEQUENCE_LENGTH = 32  # Each sequence has 24 triplets

# Baseline penalty values:
REPETITION_PENALTY = (1.0, 1.0, 1.0)      # base repetition penalty per element
SPIKE_PENALTY_STRENGTH = (1.0, 1.0, 1.0)    # base spike penalty strength per element
SPIKE_SIGMA = (1.0, 1.0, 1.0)               # baseline sigma value per element (minimum allowed)

###################################################################################

def find_numpy_array(src_array, trg_array):

    """
    Finds 1D numpy array in 2D numpy array
    """

    match_mask = np.all(src_array == trg_array, axis=1)
    
    return np.where(match_mask)[0]

###################################################################################

def vertical_list_search(src_list, trg_list):
    
    """
    For each vertical window of consecutive rows of height len(trg_list) in src_list,
    this function checks whether for every offset j (0 <= j < len(trg_list)) the row
    at index (window_start + j) contains trg_list[j].

    It returns a list of windows (each a list of consecutive row indices) that meet this condition.
    """
    
    if not src_list or not trg_list:
        return []
    
    n = len(src_list)
    k = len(trg_list)
    
    num_windows = n - k + 1
    
    if num_windows <= 0:
        return []
    
    # Determine the maximum row length.
    max_len = max(len(row) for row in src_list)
    
    # Determine a fill value guaranteed to be less than any valid value.
    global_min = min(min(row) for row in src_list if row)
    fill_value = global_min - 1

    # Build a padded 2D array A (shape n x max_len) from src_list.
    A = np.full((n, max_len), fill_value, dtype=np.int64)
    for i, row in enumerate(src_list):
        L = len(row)
        A[i, :L] = row

    # For each unique target in trg_list, compute a Boolean vector of length n.
    # present[t][i] will be True if A[i, :] contains t, else False.
    unique_targets = set(trg_list)
    
    present_dict = {}
    
    for t in unique_targets:
        # Compute along axis=1 so that for each row we see if any element equals t.
        present_dict[t] = np.any(A == t, axis=1)
    
    # Build a Boolean array B of shape (k, num_windows) where for each offset j,
    # B[j, s] = present_dict[ trg_list[j] ][s + j] for each window starting index s.
    B = np.empty((k, num_windows), dtype=bool)
    
    for j in range(k):
        t = trg_list[j]
        # For a vertical window starting at s, row s+j should contain t.
        B[j, :] = present_dict[t][j: j + num_windows]
    
    # A window is valid if all k rows in that window contain the required target.
    valid_windows_mask = np.all(B, axis=0)
    valid_starts = np.nonzero(valid_windows_mask)[0]
    
    # Create output windows (each as a list of consecutive row indices).
    result = [list(range(s, s + k)) for s in valid_starts]
    
    return result


###################################################################################

def pack_sequences(train_data, pad_val=-1):
    """
    Packs a list of variable-length token sequences into a 2D CuPy array.
    
    This version computes lengths and builds the padded array and mask entirely on GPU.
    It converts each sequence into a CuPy array, concatenates them, and assigns tokens in one shot.
    
    Returns:
      batch: a CuPy array of shape (n, max_len)
      lengths: a CuPy array of shape (n,) containing each sequence's length.
    """
    n = len(train_data)
    # Compute lengths of each sequence and convert to a CuPy array.
    lengths = cp.array([len(seq) for seq in train_data], dtype=cp.int64)
    max_len_val = int(cp.max(lengths).get())
    # Allocate the padded 2D array filled with pad_val.
    batch = cp.full((n, max_len_val), pad_val, dtype=cp.int64)
    # Create a boolean mask: for each row, positions less than the sequence length are valid.
    mask = cp.arange(max_len_val).reshape(1, max_len_val) < lengths.reshape(n, 1)
    # Convert each sequence to a CuPy array and concatenate them.
    sequences = [cp.array(seq, dtype=cp.int64) for seq in train_data]
    flat = cp.concatenate(sequences)
    # Fill in the valid positions.
    batch[mask] = flat
    return batch, lengths

###################################################################################

def count_best_pair_gpu(batch, lengths, factor, pad_val=-1):
    """
    Given the entire GPU-resident packed data, compute the most frequent
    adjacent pair (encoded as: pair_val = first * factor + second) on GPU.
    """
    n, L = batch.shape
    cols = cp.arange(L - 1, dtype=cp.int64)
    cols_expanded = cp.broadcast_to(cols, (n, L - 1))
    valid_mask = cols_expanded < cp.reshape(lengths, (n, 1)) - 1

    first_tokens = batch[:, :L - 1]
    second_tokens = batch[:, 1:L]
    valid_first = first_tokens[valid_mask]
    valid_second = second_tokens[valid_mask]

    pairs = valid_first * factor + valid_second
    if pairs.size == 0:
        return None

    sorted_pairs = cp.sort(pairs)
    diff = cp.diff(sorted_pairs)
    boundaries = cp.nonzero(diff)[0] + 1
    group_starts = cp.concatenate([cp.array([0], dtype=cp.int64), boundaries])
    group_ends = cp.concatenate([boundaries, cp.array([sorted_pairs.size], dtype=cp.int64)])
    group_counts = group_ends - group_starts

    max_idx = int(cp.argmax(group_counts))
    best_pair_enc = int(sorted_pairs[group_starts[max_idx]])
    best_freq = int(group_counts[max_idx])
    first = best_pair_enc // factor
    second = best_pair_enc % factor
    return (first, second, best_freq)

###################################################################################

merge_kernel_code = r'''
extern "C" __global__
void merge_pair_kernel(const long* input, long* output, 
                       const long* input_lengths, long* output_lengths,
                       const long num_rows, const long num_cols,
                       const long a, const long b, const long new_token,
                       const long pad_val) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    long in_length = input_lengths[row];
    long out_idx = 0;
    bool skip_next = false;
    for (long i = 0; i < in_length; i++) {
        if (skip_next) {
            skip_next = false;
            continue;
        }
        long token = input[row * num_cols + i];
        if (i < in_length - 1 && token == a && input[row * num_cols + i + 1] == b) {
            output[row * num_cols + out_idx] = new_token;
            out_idx++;
            skip_next = true;
        } else {
            output[row * num_cols + out_idx] = token;
            out_idx++;
        }
    }
    output_lengths[row] = out_idx;
    for (long j = out_idx; j < num_cols; j++) {
        output[row * num_cols + j] = pad_val;
    }
}
'''
merge_kernel = cp.RawKernel(merge_kernel_code, 'merge_pair_kernel')

###################################################################################

def learn_bpe_codes_gpu(train_data, vocab_size=4096, max_merges=None, pad_val=-1):
    """
    Learn BPE merge rules completely on GPU.
    
    The training data is packed once (using the vectorized pack_sequences).
    On each merge iteration, the best adjacent pair is computed on GPU and then merged
    into a new token via a custom merge kernel (with double-buffering).
    
    Returns:
      codes: a list of merge rules as ((first, second), new_token)
      final_data: the merged training data (list of sequences)
    """
    # Pack the entire dataset onto GPU.
    batch, lengths = pack_sequences(train_data, pad_val)
    n, L = batch.shape

    # Initialize vocabulary and the next available token.
    initial_vocab = {token for seq in train_data for token in seq}
    next_token = max(initial_vocab) + 1
    codes = []
    merge_count = 0
    pbar = tqdm.tqdm(total=max_merges if max_merges is not None else None,
                desc="Learning BPE Codes (GPU)", leave=True)

    # Preallocate buffers for double-buffering.
    work_batch = cp.empty_like(batch)
    work_lengths = cp.empty_like(lengths)
    input_batch = batch
    input_lengths = lengths

    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block

    while next_token < vocab_size and (max_merges is None or merge_count < max_merges):
        # Early stop if all sequences have collapsed (checked on GPU).
        if bool(cp.all(input_lengths == 1)):
            pbar.write("All sequences have collapsed; stopping early.")
            break

        factor = next_token  # by construction, every token is < next_token
        best = count_best_pair_gpu(input_batch, input_lengths, factor, pad_val)
        if best is None:
            pbar.write("No mergeable pairs found; stopping early.")
            break
        
        best_pair = (best[0], best[1])
        best_freq = best[2]
        if best_freq < 2:
            pbar.write("Best pair frequency is less than 2; stopping early.")
            break

        codes.append((best_pair, next_token))

        # Launch the merge kernel.
        merge_kernel((blocks,), (threads_per_block,),
                     (input_batch,
                      work_batch,
                      input_lengths,
                      work_lengths,
                      cp.int64(n),
                      cp.int64(L),
                      cp.int64(best_pair[0]),
                      cp.int64(best_pair[1]),
                      cp.int64(next_token),
                      cp.int64(pad_val)))
        # Swap buffers for double-buffering.
        input_batch, work_batch = work_batch, input_batch
        input_lengths, work_lengths = work_lengths, input_lengths

        next_token += 1
        merge_count += 1
        pbar.update(1)
    pbar.close()

    final_batch = cp.asnumpy(input_batch)
    final_lengths = cp.asnumpy(input_lengths)
    final_data = [final_batch[i, :final_lengths[i]].tolist() for i in range(n)]
    return codes, final_data

###################################################################################

fused_merge_kernel_code = r'''
extern "C" __global__
void fused_merge_kernel(long* data_in, long* data_out, long* lengths, const long pad_val,
                          const long num_rows, const long max_len, const long num_merges, const long* merge_rules) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    long base = row * max_len;
    long cur_len = lengths[row];
    long* cur = data_in + base;
    long* other = data_out + base;
    // Process each merge rule sequentially.
    for (int m = 0; m < num_merges; m++) {
        long a = merge_rules[3 * m];
        long b = merge_rules[3 * m + 1];
        long new_token = merge_rules[3 * m + 2];
        long out_idx = 0;
        for (int i = 0; i < cur_len; i++) {
            if (i < cur_len - 1 && cur[i] == a && cur[i+1] == b) {
                other[out_idx] = new_token;
                out_idx++;
                i++;  // Skip the next token.
            } else {
                other[out_idx] = cur[i];
                out_idx++;
            }
        }
        cur_len = out_idx;
        // Swap pointers for the next merge.
        long* temp = cur;
        cur = other;
        other = temp;
    }
    lengths[row] = cur_len;
    // Pad the remaining positions with pad_val.
    for (int i = cur_len; i < max_len; i++) {
        cur[i] = pad_val;
    }
    // If the final result is not in data_in, copy back.
    if (cur != data_in + base) {
        for (int i = 0; i < cur_len; i++) {
            data_in[base + i] = cur[i];
        }
    }
}
'''
fused_kernel = cp.RawKernel(fused_merge_kernel_code, 'fused_merge_kernel')

###################################################################################

def retokenize_train_data_fused_gpu(train_data, codes, pad_val=-1):
    """
    Retokenize training data using the fully fused GPU kernel.
    
    The entire training dataset is first packed into GPU memory (using pack_sequences).
    All learned merge rules (provided in 'codes') are applied via a single kernel launch.
    Each GPU thread processes one sequence by applying all merge rules sequentially.
    
    Returns:
      tokenized_data: list of retokenized sequences.
    """
    # Pack the data.
    batch, lengths = pack_sequences(train_data, pad_val)
    n, max_len = batch.shape
    # Build a flattened merge_rules array using CuPy.
    if len(codes) > 0:
        merge_rules_list = [[rule[0][0], rule[0][1], rule[1]] for rule in codes]
        merge_rules_gpu = cp.array(merge_rules_list, dtype=cp.int64)
        merge_rules_gpu = merge_rules_gpu.reshape(-1)
    else:
        merge_rules_gpu = cp.empty((0,), dtype=cp.int64)
    num_merges = merge_rules_gpu.shape[0] // 3
    # Preallocate a scratch buffer.
    scratch = cp.empty_like(batch)
    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block
    # Launch the fused kernel.
    fused_kernel((blocks,), (threads_per_block,),
                 (batch, scratch, lengths, cp.int64(pad_val),
                  cp.int64(n), cp.int64(max_len), cp.int64(num_merges), merge_rules_gpu))
    final_batch = cp.asnumpy(batch)
    final_lengths = cp.asnumpy(lengths)
    tokenized_data = [final_batch[i, :final_lengths[i]].tolist() for i in range(n)]
    return tokenized_data

###################################################################################

def bpe_encode(seq, codes):
    """
    Iteratively encodes a sequence using BPE merge rules provided in a dictionary.
    
    Args:
        seq (list): A list of tokens (e.g. integers) representing the input sequence.
        codes (dict): A dictionary mapping token pairs (a tuple of two tokens) 
                      to a merged token. For example:
                      { (1, 2): 100, (100, 3): 101 }
    
    Returns:
        list: The encoded sequence after applying all possible merges.
    
    The function repeatedly scans the entire sequence from left to right;
    whenever it finds a contiguous token pair that exists as a key in the
    codes dict, it replaces that pair with the merged token. This pass is
    repeated until no more merges are possible.
    """

    if type(codes) == list:
        codes = dict(codes)
        
    encoded_seq = seq.copy()  # work on a copy so as not to modify the original
    done = False
    while not done:
        new_seq = []
        i = 0
        changed = False
        while i < len(encoded_seq):
            # If a merge is possible, merge the two tokens.
            if i < len(encoded_seq) - 1 and (encoded_seq[i], encoded_seq[i + 1]) in codes:
                new_seq.append(codes[(encoded_seq[i], encoded_seq[i + 1])])
                i += 2  # Skip the next token as it was merged.
                changed = True
            else:
                new_seq.append(encoded_seq[i])
                i += 1
        # If no merges occurred in this pass, exit the loop.
        if not changed:
            done = True
        encoded_seq = new_seq
    return encoded_seq

###################################################################################

def bpe_decode(seq, codes):
    """
    Decodes a sequence encoded with BPE merge rules defined in a codes dictionary.
    
    Args:
        seq (list): The encoded sequence (a list of tokens).
        codes (dict): A dictionary mapping token pairs to the merged token, used during encoding.
    
    Returns:
        list: The fully decoded sequence, with all merged tokens recursively expanded.
    
    The function constructs a reverse mapping that converts a merged token back into 
    its constituent pair. Each token in the sequence is then recursively expanded.
    """

    if type(codes) == list:
        codes = dict(codes)
        
    # Build the reverse mapping: key = merged token, value = tuple (original token pair)
    reverse_mapping = {merged: pair for pair, merged in codes.items()}

    def recursive_expand(token):
        # If the token is a merged token, expand it recursively.
        if token in reverse_mapping:
            a, b = reverse_mapping[token]
            return recursive_expand(a) + recursive_expand(b)
        else:
            return [token]

    decoded_seq = []
    for token in seq:
        decoded_seq.extend(recursive_expand(token))
    return decoded_seq

###################################################################################

def ensure_triplet(val: Any, name: str = "") -> Tuple[float, float, float]:
    """
    Ensure the given parameter is returned as a triplet.
    If provided as a scalar, promote it to a triplet.
    """
    if np.isscalar(val):
        return (float(val), float(val), float(val))
    elif isinstance(val, (list, tuple)) and len(val) == 3:
        return tuple(float(x) for x in val)
    else:
        raise ValueError(f"{name} must be a scalar or a sequence of 3 numbers.")

###################################################################################

REP_PENALTY = ensure_triplet(REPETITION_PENALTY, "REPETITION_PENALTY")
SPIKE_STRENGTH = ensure_triplet(SPIKE_PENALTY_STRENGTH, "SPIKE_PENALTY_STRENGTH")
SPIKE_SIG = ensure_triplet(SPIKE_SIGMA, "SPIKE_SIGMA")

###################################################################################

def sliding_window_view_alternative(a: np.ndarray, window_length: int) -> np.ndarray:
    """
    Create a sliding-window view (without copying) of an array.
    Expected input shape: (n, L, d) and returns: (n, L - window_length + 1, window_length, d)
    """
    n, L, d = a.shape
    new_shape = (n, L - window_length + 1, window_length, d)
    new_strides = (a.strides[0], a.strides[1], a.strides[1], a.strides[2])
    return np.lib.stride_tricks.as_strided(a, shape=new_shape, strides=new_strides)

###################################################################################

def build_ngram_mapping(data: np.ndarray, memory_len: int) -> Dict[Any, Dict[Any, int]]:
    """
    Build an n-gram mapping from a context (a sequence of triplets) to candidate triplets with frequencies.
    """
    n, L, d = data.shape
    window_length = memory_len + 1  # context (memory) + candidate
    windows = sliding_window_view_alternative(data, window_length)
    # windows shape: (n, L - window_length + 1, window_length, d)

    # Split windows into context (first memory_len triplets) and candidates (last triplet)
    contexts = windows[:, :, :memory_len, :]   # shape: (n, num_windows, memory_len, d)
    candidates = windows[:, :, memory_len, :]    # shape: (n, num_windows, d)

    # Flatten the batch and window dimensions.
    contexts_flat = contexts.reshape(-1, memory_len, d)
    candidates_flat = candidates.reshape(-1, d)

    mapping = defaultdict(lambda: defaultdict(int))
    total_windows = contexts_flat.shape[0]
    for context_arr, candidate_arr in tqdm.tqdm(
            zip(contexts_flat, candidates_flat),
            total=total_windows,
            desc="Building n-gram mapping"):
        context_key = tuple(map(tuple, context_arr))  # use a tuple of triplets as the key
        candidate_val = tuple(candidate_arr)
        mapping[context_key][candidate_val] += 1

    return {context: dict(candidates) for context, candidates in mapping.items()}

###################################################################################

def precompute_mapping_lookup(mapping: Dict[Any, Dict[Any, int]]) -> Dict[Any, Tuple[Tuple[Any, ...], np.ndarray]]:
    """
    Converts the mapping into a lookup table: context -> (tuple(candidates), frequencies_array).
    """
    mapping_lookup = {}
    for context, candidate_dict in tqdm.tqdm(mapping.items(), desc="Precomputing lookup"):
        candidates = tuple(candidate_dict.keys())
        frequencies = np.array(list(candidate_dict.values()), dtype=np.float64)
        mapping_lookup[context] = (candidates, frequencies)
    return mapping_lookup

###################################################################################

def build_training_sequences_set(data: np.ndarray) -> set:
    """
    Build a set of training sequences (each as a tuple of triplets) for uniqueness checking.
    """
    return {tuple(map(tuple, seq)) for seq in data}

###################################################################################

def generate_sequence_optimized(mapping_lookup: Dict[Any, Tuple[Tuple[Any, ...], np.ndarray]],
                                training_set: set,
                                memory_len: int,
                                sequence_length: int = 24,
                                max_attempts: int = 1000) -> Optional[Tuple[Tuple[float, float, float], ...]]:
    """
    Autoregressively generate a new, unique sequence using the precomputed mapping lookup.
    The invariant maintained is: the second element of one triplet is never greater than the first element
    of the following triplet.

    Two dynamic adjustments are applied for candidate selection:
    
      1. **Dynamic Repetition Penalty:**  
         For each candidate, count the occurrences of each element in the generated sequence.
         Rather than a fixed penalty, this repetition penalty scales with the ratio
         (current_length / sequence_length). In log-space, it subtracts:
             (current_length / sequence_length) * sum_k(count[k] * log(REP_PENALTY[k])
      2. **Dynamic Spike (Variance) Penalty:**  
         For each candidate, compute the squared difference from the running average for each element.
         Use a dynamic sigma that is the maximum between the running standard deviation and the baseline.
         The penalty term for each element is:
             SPIKE_STRENGTH[k] * ((cand[k] - running_avg[k])^2) / (2 * dynamic_sigma[k]^2)
         The overall spike penalty is the sum of the three terms and is subtracted from the candidate’s log frequency.

    The resulting candidate log score is computed as:
         log(candidate_frequency) - rep_penalty_component - spike_penalty_component
    A numerical stable softmax is then applied over these scores to determine the probability for drawing a candidate.

    If no candidate passing the invariant is found, the attempt is aborted.

    Parameters:
      mapping_lookup: Precomputed lookup mapping (context → (candidates, frequencies)).
      training_set: Set of training sequences to ensure uniqueness.
      memory_len: Number of triplets used as context.
      sequence_length: Desired length of the generated sequence.
      max_attempts: Maximum number of generation attempts.

    Returns:
      A new unique sequence (tuple of triplets) that respects the invariant, or None if not found.
    """
    mapping_keys = list(mapping_lookup.keys())
    num_keys = len(mapping_keys)

    for attempt in range(max_attempts):
        # Select a seed context randomly (from training data so that the invariant holds).
        seed = mapping_keys[np.random.randint(0, num_keys)]
        generated_sequence: List[Tuple[float, float, float]] = list(seed)
        valid_generation = True

        while len(generated_sequence) < sequence_length:
            last_triplet = generated_sequence[-1]
            current_context = tuple(generated_sequence[-memory_len:])  # context as tuple of triplets
            candidate_found = False

            if current_context in mapping_lookup:
                candidates, frequencies = mapping_lookup[current_context]
                # Filter candidates by invariant:
                # Candidate's first element must be >= last triplet's second element.
                valid_indices = [i for i, cand in enumerate(candidates) if cand[0] >= last_triplet[1]]
                if valid_indices:
                    # Filter candidates and their associated frequencies.
                    filtered_freqs = frequencies[valid_indices]
                    filtered_candidates = [candidates[i] for i in valid_indices]

                    # Convert candidates into a NumPy array for vectorized operations.
                    candidate_array = np.array(filtered_candidates, dtype=np.float64)  # shape: (n_candidates, 3)
                    
                    # Prepare generation history as array.
                    generated_array = np.array(generated_sequence, dtype=np.float64)   # shape: (T, 3)
                    current_length = generated_array.shape[0]
                    
                    # Running average and standard deviation for dynamic spike adjustment.
                    running_avg = np.mean(generated_array, axis=0)       # shape: (3,)
                    running_std = np.std(generated_array, axis=0)          # shape: (3,)
                    # Dynamic sigma: ensure a minimum sigma value.
                    dynamic_sigma = np.maximum(running_std, np.array(SPIKE_SIG))
                    
                    # --- Compute Repetition Penalty ---
                    # For each candidate, count the number of occurrences for each element along the corresponding column.
                    rep_counts = np.array([
                        [np.sum(generated_array[:, k] == candidate_array[i, k]) for k in range(3)]
                        for i in range(candidate_array.shape[0])
                    ])  # shape: (n_candidates, 3)
                    # The repetition penalty in log-space.
                    rep_penalty_term = np.sum(rep_counts * np.log(np.array(REP_PENALTY)) *
                                              (current_length / sequence_length), axis=1)  # shape: (n_candidates,)

                    # --- Compute Spike (Variance) Penalty ---
                    # Compute the difference per candidate from the running average.
                    diff = candidate_array - running_avg  # shape: (n_candidates, 3)
                    spike_penalty_term = np.sum(np.array(SPIKE_STRENGTH) * (diff**2) / (2 * (dynamic_sigma**2)),
                                                axis=1)  # shape: (n_candidates,)

                    # --- Compute Candidate Log-Scores ---
                    # Use np.log on frequencies (they are positive by construction).
                    log_freq = np.log(filtered_freqs)
                    log_scores = log_freq - rep_penalty_term - spike_penalty_term

                    # --- Softmax in Log-space (stable computation) ---
                    max_log = np.max(log_scores)
                    exp_scores = np.exp(log_scores - max_log)
                    probabilities = exp_scores / np.sum(exp_scores)
                    
                    # Choose the next candidate using advanced probabilities.
                    chosen_idx = np.random.choice(len(filtered_candidates), p=probabilities)
                    next_triplet = filtered_candidates[chosen_idx]
                    candidate_found = True

            if not candidate_found:
                # Abort this generation attempt if no valid candidate is available.
                valid_generation = False
                break

            generated_sequence.append(next_triplet)

        # Ensure the final sequence meets the invariant and is unique.
        if valid_generation and len(generated_sequence) == sequence_length:
            new_sequence = tuple(generated_sequence)
            invariant_ok = all(a[1] <= b[0] for a, b in zip(new_sequence, new_sequence[1:]))
            if invariant_ok and new_sequence not in training_set:
                return new_sequence

    return None

###################################################################################

def analyze_generated_sequence(sequence: tuple, mapping_lookup: dict, memory_len: int) -> tuple:
    """
    Analyze the generated sequence and return several useful statistics
    as both a dictionary and as a nicely formatted string report.
    
    Statistics Computed:
      - unigram_diversity: Ratio of unique triplets to total triplets.
      - repetition_rate: Fraction of repeated triplets.
      - bigram_diversity: Ratio of unique consecutive pairs to total pairs.
      - max_consecutive_repetitions: Maximum number of identical consecutive triplets.
      - avg_candidate_probability (overfit rate): For the transitions (using a sliding window of size
          MEMORY_LEN as context followed by candidate), the average probability of the chosen candidate
          as per the training mapping.
      
      Additional Analytics:
      - element_stats: For each element (index 0, 1, 2) in a triplet, includes:
            * mean, standard deviation, minimum, maximum, and average consecutive absolute difference.
      - avg_transition_entropy: The average entropy of the candidate distributions (from mapping_lookup)
          for each transition context.
      - context_coverage: The fraction of transitions (based on context of length MEMORY_LEN) that are found 
          in the mapping_lookup.
    
    Parameters:
      sequence: Generated sequence (tuple of triplets).
      mapping_lookup: Precomputed mapping lookup.
      memory_len: The context length used.
    
    Returns:
      A tuple containing:
          (stats_dict, stats_report_string)
    """
    stats = {}
    seq_len = len(sequence)
    
    # --- Basic Statistics ---
    
    # Unigram.
    unique_triplets = len(set(sequence))
    stats["unigram_diversity"] = unique_triplets / seq_len
    stats["repetition_rate"] = 1 - (unique_triplets / seq_len)
    
    # Bigram.
    bigrams = [(sequence[i], sequence[i+1]) for i in range(seq_len - 1)]
    unique_bigrams = len(set(bigrams))
    stats["bigram_diversity"] = unique_bigrams / (seq_len - 1)
    
    # Maximum consecutive repetitions.
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, seq_len):
        if sequence[i] == sequence[i-1]:
            current_consecutive += 1
            if current_consecutive > max_consecutive:
                max_consecutive = current_consecutive
        else:
            current_consecutive = 1
    stats["max_consecutive_repetitions"] = max_consecutive

    # Avg Candidate Probability (Overfit Rate)
    overfit_probs = []
    for i in range(memory_len, seq_len):
        context = tuple(sequence[i - memory_len: i])
        candidate = sequence[i]
        if context in mapping_lookup:
            candidates, frequencies = mapping_lookup[context]
            total_freq = np.sum(frequencies)
            try:
                idx = candidates.index(candidate)
                cand_prob = frequencies[idx] / total_freq
                overfit_probs.append(cand_prob)
            except ValueError:
                pass
    stats["avg_candidate_probability"] = np.mean(overfit_probs) if overfit_probs else None

    # --- Additional Analytics ---

    # 1. Element-Level Statistics.
    seq_arr = np.array(sequence)  # shape: (seq_len, 3)
    element_stats = {}
    for dim in range(seq_arr.shape[1]):
        values = seq_arr[:, dim]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        # Calculate average absolute difference between consecutive values:
        diffs = np.abs(np.diff(values))
        avg_diff = np.mean(diffs) if diffs.size > 0 else 0
        element_stats[f"element_{dim}"] = {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "avg_consecutive_diff": avg_diff,
        }
    stats["element_stats"] = element_stats

    # 2. Transition Entropy:
    entropies = []
    valid_transitions = 0
    for i in range(memory_len, seq_len):
        context = tuple(sequence[i - memory_len: i])
        if context in mapping_lookup:
            candidates, freqs = mapping_lookup[context]
            total_freq = np.sum(freqs)
            if total_freq > 0:
                probs = freqs / total_freq
                # Add a very small constant to avoid log(0)
                epsilon = 1e-10
                entropy = -np.sum(probs * np.log(probs + epsilon))
                entropies.append(entropy)
                valid_transitions += 1
    stats["avg_transition_entropy"] = np.mean(entropies) if entropies else None

    # 3. Context Coverage:
    total_transitions = seq_len - memory_len
    stats["context_coverage"] = (valid_transitions / total_transitions) if total_transitions > 0 else None

    # --- Build a Pretty Report String ---
    sep_line = "-" * 60
    lines = []
    lines.append(sep_line)
    lines.append("Sequence Analytics Report:")
    lines.append(sep_line)
    lines.append("Overall Statistics:")
    lines.append(f"  Unigram Diversity         : {stats['unigram_diversity']:.3f}")
    lines.append(f"  Repetition Rate           : {stats['repetition_rate']:.3f}")
    lines.append(f"  Bigram Diversity          : {stats['bigram_diversity']:.3f}")
    lines.append(f"  Max Consecutive Repetitions: {stats['max_consecutive_repetitions']}")
    cand_prob = stats["avg_candidate_probability"]
    cand_prob_str = f"{cand_prob:.3f}" if cand_prob is not None else "N/A"
    lines.append(f"  Avg Candidate Probability : {cand_prob_str}")
    lines.append("")
    
    lines.append("Element-Level Statistics:")
    for dim in sorted(element_stats.keys()):
        ed = element_stats[dim]
        lines.append(f"  {dim.capitalize()}:")
        lines.append(f"    Mean                 : {ed['mean']:.3f}")
        lines.append(f"    Std Dev              : {ed['std']:.3f}")
        lines.append(f"    Min                  : {ed['min']:.3f}")
        lines.append(f"    Max                  : {ed['max']:.3f}")
        lines.append(f"    Avg Consecutive Diff : {ed['avg_consecutive_diff']:.3f}")
    lines.append("")

    lines.append("Transition Statistics:")
    avg_entropy = stats["avg_transition_entropy"]
    entropy_str = f"{avg_entropy:.3f}" if avg_entropy is not None else "N/A"
    lines.append(f"  Average Transition Entropy: {entropy_str}")
    cc = stats["context_coverage"]
    cc_str = f"{cc:.3f}" if cc is not None else "N/A"
    lines.append(f"  Context Coverage          : {cc_str}")
    lines.append(sep_line)
    
    stats_report = "\n".join(lines)
    
    # Return both the dictionary and the formatted report string.
    return stats, stats_report

###################################################################################

def autoregressive_generate(start_seq, mel_tones, trg_array, trg_matches_array, num_new_tokens, chunk_len=5):
    
    # Convert sequences to NumPy arrays.
    current_seq = np.array(start_seq, dtype=int)  # Shape: (num_tokens, token_dim)
    trg_array = np.array(trg_array, dtype=int)      # Shape: (num_candidates, 2, token_dim)
    start_len = len(start_seq)

    midx = start_len-1
    
    # Deque for sliding memory of candidate pairs (immutable tuples).
    recent_candidates = deque(maxlen=5)

    while (len(current_seq) - start_len) < num_new_tokens:

        midx += 1

        # Get the last two tokens as context.
        context = current_seq[-(chunk_len-1):]  # Shape: (2, token_dim)

        sli = 0
        msize = 0

        ctx = context[:, :-1].reshape(1, -1)
        trg_mat_arr = trg_matches_array

        while msize < 8:

            print('=== Slice', sli)
        
            # Compare context with candidates in trg_array.
            match_mask = np.all(ctx == trg_mat_arr, axis=1)
            match_indices = np.where(match_mask)[0]

            msize = match_indices.size

            if msize < 8:
                sli += 1
                ctx = context[:, :-1].reshape(1, -1)[:, sli:]
                trg_mat_arr = trg_matches_array[:, :-sli]
                
        if match_indices.size == 0:
            if len(current_seq) > start_len:

                #tones_chord = sorted([mel_tones[midx], (mel_tones[midx]+7) % 12])
                tones_chord = sorted([mel_tones[midx]])
                new_tuple = [[mel_tones[midx], TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)]]               
                current_seq = np.concatenate((current_seq, new_tuple), axis=0)
                print('Subbed', midx)
                continue

        # From the matching candidates, filter out those whose candidate pair is in recent memory.
        available_candidates = []
        cseen = []
        for idx in match_indices:

            if idx not in recent_candidates:
                # Convert candidate pair to an immutable tuple
                candidate_pair = tuple(trg_array[idx].tolist())
                if candidate_pair[-1][0] == mel_tones[midx] and candidate_pair[-1][1] not in cseen:
                    available_candidates.append((idx, candidate_pair))
                    cseen.append(candidate_pair[-1][1])

        # If all candidates have recently been used, backtrack.
        if len(available_candidates) < 3:
            if len(current_seq) >= start_len:
                #tones_chord = sorted([mel_tones[midx], (mel_tones[midx]+7) % 12])
                tones_chord = sorted([mel_tones[midx]])
                new_tuple = [[mel_tones[midx], TMIDIX.ALL_CHORDS_SORTED.index(tones_chord)]]               
                current_seq = np.concatenate((current_seq, new_tuple), axis=0)
                #rev_val = random.choice([-1, -2])
                #current_seq = current_seq[:rev_val]
                #print(midx)
                #midx = len(current_seq)
                #print('Reverted', midx, len(current_seq))
                continue

        else:
            print(len(available_candidates))        
            # Choose one available candidate at random.
            chosen_idx, chosen_pair = available_candidates[np.random.choice(len(available_candidates))]
            new_token = trg_array[chosen_idx][-1]  # The second token of the candidate pair.
            
    
            # Append the new token to the sequence.
            current_seq = np.concatenate((current_seq, new_token[None, :]), axis=0)
    
            recent_candidates.append(chosen_idx)

            print('Gen seq len', len(current_seq))

    return current_seq

###################################################################################

def minkowski_distance_vector_to_matrix(x: cp.ndarray, X: cp.ndarray, p: float = 3) -> cp.ndarray:
    
    """
    Computes the Minkowski distance between a 1D CuPy array 'x' and each row of a 2D CuPy array 'X'.
    
    Parameters:
        x (cp.ndarray): A 1D array with shape (n_features,) representing a single vector.
        X (cp.ndarray): A 2D array with shape (n_samples, n_features) where each row is a vector.
        p (float): The order of the Minkowski distance.
                   For instance:
                     - p=1 yields the Manhattan distance,
                     - p=2 yields the Euclidean distance,
                     - p=3 yields the Minkowski distance and will use the cube-root implementation,
                     - p=∞ (or cp.inf) gives the Chebyshev distance.
    
    Returns:
        cp.ndarray: A 1D array of length n_samples containing the Minkowski distance between 'x' 
                    and the corresponding row in 'X'.
    """

    # Compute the element-wise absolute differences between x and every row in X.
    # Broadcasting x over the rows of X results in an array of shape (n_samples, n_features).
    diff = cp.abs(X - x)
    
    if p == float('inf') or p == cp.inf:
        # For the Chebyshev distance, use the maximum absolute difference along the feature axis.
        distances = cp.max(diff, axis=1)
    elif p == 3:
        # Instead of using the generic power operation (sum(diff**3) ** (1/3)),
        # we use cp.cbrt for cube-root calculation when p is exactly 3.
        distances = cp.cbrt(cp.sum(diff ** 3, axis=1))
    else:
        # For general Minkowski distance with finite p,
        # compute the p-th power of differences, sum them, then take the p-th root.
        distances = cp.sum(diff ** p, axis=1) ** (1.0 / p)
        
    return distances

###################################################################################

def pairwise_minkowski_distance(X: cp.ndarray, p: float = 2) -> cp.ndarray:
    
    """
    Computes pairwise Minkowski distances for a 2D CuPy array.
    
    Parameters:
        X (cp.ndarray): A 2D array of shape (n_samples, n_features), where each row represents a vector.
        p (float): The order of the Minkowski distance.
                   For example:
                     - p=1 is the Manhattan distance,
                     - p=2 is the Euclidean distance,
                     - p=∞ (e.g., float('inf') or cp.inf) is the Chebyshev distance.
    
    Returns:
        cp.ndarray: A 2D array of shape (n_samples, n_samples) containing the pairwise Minkowski distances.
    """
    
    # Use broadcasting to compute the absolute difference between every pair of vectors.
    # The result of X[:, None, :] - X[None, :, :] will have shape (n_samples, n_samples, n_features).
    if p == float('inf') or p == cp.inf:
        # For the Chebyshev distance, take the maximum absolute difference along the feature axis.
        return cp.max(cp.abs(X[:, None, :] - X[None, :, :]), axis=-1)
    else:
        # Raise the absolute differences to the power p.
        diff_powered = cp.abs(X[:, None, :] - X[None, :, :]) ** p
        # Sum over the features for each pair (i, j) and then take the p-th root.
        distances = cp.sum(diff_powered, axis=-1) ** (1.0 / p)
        
        return distances
    
###################################################################################

def pairwise_cosine_similarity(X: cp.ndarray, eps: float = 1e-10) -> cp.ndarray:
    
    """
    Computes the pairwise cosine similarity for a 2D CuPy array.
    
    Parameters:
        X (cp.ndarray): A 2D array of shape (n_samples, n_features) where each row represents a vector.
        eps (float): A small constant added to the denominator to prevent division by zero.
    
    Returns:
        cp.ndarray: A 2D array of shape (n_samples, n_samples) containing the pairwise cosine similarities.
    """
    
    # Compute the dot product between every pair of rows.
    # This results in a matrix where element (i, j) is the dot product of X[i] and X[j].
    dot_product = cp.dot(X, X.T)
    
    # Compute the L2 norm (Euclidean norm) for each row vector.
    norms = cp.linalg.norm(X, axis=1)
    
    # Compute the outer product of the norms to form the denominator.
    # The element (i, j) in this matrix is norms[i] * norms[j].
    norm_matrix = cp.outer(norms, norms)
    
    # Compute the cosine similarity matrix.
    # Adding a small epsilon (eps) to the denominator prevents division by zero.
    cosine_similarity = dot_product / (norm_matrix + eps)
    
    return cosine_similarity

###################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the TCUPY Python module
###################################################################################