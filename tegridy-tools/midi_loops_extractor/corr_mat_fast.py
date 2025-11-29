from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Tuple, Dict, List, Optional

import os
import numpy as np
import threading

from note_set_fast import NoteSet  # updated import path if you renamed file

if TYPE_CHECKING:
    from symusic import Note

# Module-level small cache for duration computations to avoid repeated work
# Keyed by (start, end, tb_first, tb_last, tb_len)
_duration_cache: Dict[Tuple[int, int, int, int, int], float] = {}
_duration_cache_lock = threading.Lock()

# Safety caps and windowing parameters (tune to your environment)
_MAX_TICKS_KEEP = 100_000      # hard cap on ticks array length
_TRUNCATE_KEEP = 20_000        # keep this many ticks from head and tail if truncating
_MAX_NOTESETS_FOR_FULL_CORR = 5000  # if n > this, use windowed correlation
_WINDOW_SIZE = 4000           # window size for windowed correlation
_WINDOW_OVERLAP = 500         # overlap between windows to catch cross-boundary runs

# New: sanity threshold for tick arrays used for interpolation
_MAX_TICK = 10 ** 9

# Optional Numba acceleration flags (opt-in via environment)
_USE_NUMBA = os.environ.get("USE_NUMBA", "0") == "1"
_USE_CUDA = os.environ.get("USE_CUDA", "0") == "1"

_NUMBA_AVAILABLE = False
_CUDA_AVAILABLE = False
_njit = None
_cuda = None

if _USE_NUMBA:
    try:
        from numba import njit, prange
        _NUMBA_AVAILABLE = True
        _njit = njit
    except Exception:
        _NUMBA_AVAILABLE = False
        _njit = None

if _USE_CUDA and _NUMBA_AVAILABLE:
    try:
        from numba import cuda
        _CUDA_AVAILABLE = cuda.is_available()
        _cuda = cuda
    except Exception:
        _CUDA_AVAILABLE = False
        _cuda = None

# If CUDA requested but not available, fall back to CPU numba if available
if _USE_CUDA and not _CUDA_AVAILABLE and _NUMBA_AVAILABLE:
    # keep _USE_CUDA False to avoid trying GPU kernels later
    _USE_CUDA = False

# Helper: safe strictly increasing check
def _is_strictly_increasing(arr: np.ndarray) -> bool:
    if arr.size < 2:
        return True
    try:
        return bool((arr[1:] > arr[:-1]).all())
    except Exception:
        prev = arr[0]
        for v in arr[1:]:
            if v <= prev:
                return False
            prev = v
        return True


def _sanitize_ticks_beats_once(ticks_beats: Sequence[int]) -> np.ndarray:
    tb = np.asarray(ticks_beats, dtype=np.int64)
    if tb.size == 0:
        return tb

    # Remove implausible tick values
    try:
        tb = tb[np.isfinite(tb)]
    except Exception:
        pass

    # Clip out-of-range ticks
    if tb.size > 0:
        tb = tb[(tb >= 0) & (tb <= _MAX_TICK)]

    if tb.size == 0:
        return tb

    if _is_strictly_increasing(tb):
        if tb.size > _MAX_TICKS_KEEP:
            head = tb[:_TRUNCATE_KEEP]
            tail = tb[-_TRUNCATE_KEEP:]
            tb = np.concatenate((head, tail))
        return tb

    try:
        tb_unique = np.unique(tb)
    except Exception:
        try:
            tb_list = sorted(set(int(x) for x in tb if np.isfinite(x)))
            tb_unique = np.asarray(tb_list, dtype=np.int64)
        except Exception:
            return np.asarray([], dtype=np.int64)

    if tb_unique.size > _MAX_TICKS_KEEP:
        head = tb_unique[:_TRUNCATE_KEEP]
        tail = tb_unique[-_TRUNCATE_KEEP:]
        tb_unique = np.concatenate((head, tail))

    return tb_unique


# -------------------------
# Numba-accelerated helpers
# -------------------------
# CPU njit implementation of diagonal processing (pure-Python fallback exists)
if _NUMBA_AVAILABLE and not _USE_CUDA:
    # We implement a simplified njit version that mirrors the original logic.
    # Note: numba njit does not support all numpy conveniences used above, so
    # we implement the core loops in a form numba accepts.
    @njit  # type: ignore[name-defined]
    def _process_diagonals_numba_cpu(key_ids: np.ndarray, is_barline: np.ndarray, out_mat: np.ndarray, offset: int) -> None:
        n_local = key_ids.size
        if n_local < 2:
            return

        # Fast path for first element matching later ones when it's a barline
        if is_barline[0]:
            first_id = key_ids[0]
            for idx in range(1, n_local):
                if key_ids[idx] == first_id:
                    out_mat[offset, offset + idx] = 1

        for k in range(1, n_local):
            # left: 0..n_local-k-1, right: k..n_local-1
            prev_val = 0
            for i_local in range(0, n_local - k):
                j_local = i_local + k
                if key_ids[i_local] == key_ids[j_local]:
                    # contiguous run handling: we need to detect run starts/ends
                    # We emulate the "growing" counts by checking previous diagonal value
                    if i_local == 0:
                        if is_barline[i_local]:
                            prev_val = 1
                            out_mat[offset + i_local, offset + j_local] = 1
                        else:
                            prev_val = 0
                    else:
                        if prev_val == 0 and (not is_barline[i_local]):
                            prev_val = 0
                        else:
                            prev_val = prev_val + 1
                            out_mat[offset + i_local, offset + j_local] = prev_val
                else:
                    prev_val = 0

# GPU kernel: compute equality matches for a given diagonal offset k
if _CUDA_AVAILABLE:
    @cuda.jit  # type: ignore[name-defined]
    def _cuda_kernel_match(key_ids, k, n_local, matches):
        i = cuda.grid(1)
        if i < n_local - k:
            matches[i] = 1 if key_ids[i] == key_ids[i + k] else 0


def _process_diagonals_into_matrix(key_ids_slice: np.ndarray, is_barline_slice: np.ndarray, out_mat: np.ndarray, offset: int) -> None:
    """
    Wrapper that selects accelerated implementation if available, otherwise uses
    the original Python/Numpy implementation.
    """
    n_local = key_ids_slice.size
    if n_local < 2:
        return

    # If CUDA is enabled and available, use a GPU kernel to compute per-diagonal matches
    if _CUDA_AVAILABLE:
        try:
            # copy slice to device
            key_ids_dev = _cuda.to_device(key_ids_slice.astype(np.int32))
            # allocate a device array for matches (max length n_local)
            for k in range(1, n_local):
                matches_dev = _cuda.device_array(n_local - k, dtype=np.uint8)
                threadsperblock = 128
                blocks = (n_local - k + threadsperblock - 1) // threadsperblock
                _cuda_kernel_match[blocks, threadsperblock](key_ids_dev, k, n_local, matches_dev)
                matches = matches_dev.copy_to_host()
                if not matches.any():
                    continue

                # find runs in matches (host side)
                true_idx = np.flatnonzero(matches)
                if true_idx.size == 0:
                    continue

                # Identify contiguous runs in true_idx
                if true_idx.size == 1:
                    runs = [(int(true_idx[0]), int(true_idx[0]))]
                else:
                    diffs = np.diff(true_idx)
                    breaks = np.nonzero(diffs > 1)[0]
                    run_starts = np.concatenate(([0], breaks + 1))
                    run_ends = np.concatenate((breaks, [true_idx.size - 1]))
                    runs = [(int(true_idx[s]), int(true_idx[e])) for s, e in zip(run_starts, run_ends)]

                # For each run, walk and set incremental counts
                for run_start, run_end in runs:
                    prev_val = 0
                    for i_local in range(run_start, run_end + 1):
                        j_local = i_local + k
                        global_i = offset + i_local
                        global_j = offset + j_local
                        if i_local == 0:
                            if is_barline_slice[i_local]:
                                prev_val = 1
                                out_mat[global_i, global_j] = 1
                            else:
                                prev_val = 0
                        else:
                            if prev_val == 0 and not is_barline_slice[i_local]:
                                prev_val = 0
                            else:
                                prev_val = prev_val + 1
                                out_mat[global_i, global_j] = prev_val
            # also handle first-element barline fast path (matches against first element)
            if is_barline_slice[0]:
                first_id = int(key_ids_slice[0])
                matches = (key_ids_slice[1:] == first_id)
                if matches.any():
                    global_i = offset
                    global_js = np.nonzero(matches)[0] + offset + 1
                    out_mat[global_i, global_js] = 1
            return
        except Exception:
            # If GPU path fails for any reason, fall back to CPU/NumPy implementation below
            pass

    # If Numba CPU is available and requested, use njit implementation
    if _NUMBA_AVAILABLE and not _USE_CUDA:
        try:
            _process_diagonals_numba_cpu(key_ids_slice.astype(np.int32), is_barline_slice.astype(np.uint8), out_mat, offset)
            return
        except Exception:
            # fall back to Python implementation on error
            pass

    # Original Python/Numpy implementation (fallback)
    # Fast path for first element matching later ones when it's a barline
    if is_barline_slice[0]:
        first_id = key_ids_slice[0]
        matches = (key_ids_slice[1:] == first_id)
        if matches.any():
            global_i = offset
            global_js = np.nonzero(matches)[0] + offset + 1
            out_mat[global_i, global_js] = 1

    # For each diagonal offset k, find matches between left and right slices
    for k in range(1, n_local):
        left = key_ids_slice[: n_local - k]
        right = key_ids_slice[k : n_local]
        matches = (left == right)
        if not matches.any():
            continue

        true_idx = np.flatnonzero(matches)
        if true_idx.size == 0:
            continue

        # Identify contiguous runs in true_idx
        if true_idx.size == 1:
            runs = [(int(true_idx[0]), int(true_idx[0]))]
        else:
            diffs = np.diff(true_idx)
            breaks = np.nonzero(diffs > 1)[0]
            run_starts = np.concatenate(([0], breaks + 1))
            run_ends = np.concatenate((breaks, [true_idx.size - 1]))
            runs = [(int(true_idx[s]), int(true_idx[e])) for s, e in zip(run_starts, run_ends)]

        # For each run, walk and set incremental counts; this inner loop is
        # necessary to produce the "growing" correlation counts along diagonals.
        for run_start, run_end in runs:
            prev_val = 0
            for i_local in range(run_start, run_end + 1):
                j_local = i_local + k
                global_i = offset + i_local
                global_j = offset + j_local
                if i_local == 0:
                    if is_barline_slice[i_local]:
                        prev_val = 1
                        out_mat[global_i, global_j] = 1
                    else:
                        prev_val = 0
                else:
                    if prev_val == 0 and not is_barline_slice[i_local]:
                        prev_val = 0
                    else:
                        prev_val = prev_val + 1
                        out_mat[global_i, global_j] = prev_val


def calc_correlation(note_sets: Sequence[NoteSet]) -> np.ndarray:
    n = len(note_sets)
    if n < 2:
        return np.zeros((n, n), dtype=np.int16)

    key_to_id: Dict[Tuple[int, frozenset], int] = {}
    key_ids = np.empty(n, dtype=np.int32)
    next_id = 1
    for i, ns in enumerate(note_sets):
        # Defensive: ensure ns has expected attributes and reasonable values
        try:
            dur = int(ns.duration)
            pitches = frozenset(ns.pitches)
            if dur < 0 or dur > _MAX_TICK:
                dur = 0
        except Exception:
            dur = 0
            pitches = frozenset()
        key = (dur, pitches)
        kid = key_to_id.get(key)
        if kid is None:
            kid = next_id
            key_to_id[key] = kid
            next_id += 1
        key_ids[i] = kid

    is_barline = np.fromiter((bool(getattr(ns, "is_barline", lambda: False)()) for ns in note_sets), dtype=bool, count=n)
    corr_mat = np.zeros((n, n), dtype=np.int16)

    if n <= _MAX_NOTESETS_FOR_FULL_CORR:
        _process_diagonals_into_matrix(key_ids, is_barline, corr_mat, offset=0)
        return corr_mat

    win = _WINDOW_SIZE
    overlap = min(_WINDOW_OVERLAP, win // 4)
    if overlap >= win:
        overlap = win // 4
    stride = win - overlap
    if stride <= 0:
        stride = max(1, win // 2)

    start = 0
    while start < n:
        end = min(n, start + win)
        key_ids_slice = key_ids[start:end]
        is_barline_slice = is_barline[start:end]
        _process_diagonals_into_matrix(key_ids_slice, is_barline_slice, corr_mat, offset=start)
        if end == n:
            break
        start += stride

    return corr_mat


# The rest of the file (duration helpers and loop detection) remains unchanged
# (kept here for completeness). They are unchanged from the previous version,
# but are included so this module is self-contained.

def get_loop_density(loop: Sequence[NoteSet], num_beats: int | float) -> float:
    if num_beats == 0:
        return 0.0
    active = 0
    for ns in loop:
        if ns.duration != 0:
            active += 1
    return active / float(num_beats)


def is_empty_loop(loop: Sequence[Note]) -> bool:
    for ns in loop:
        if ns.pitches:
            return False
    return True


def compare_loops(p1: Sequence[NoteSet], p2: Sequence[NoteSet], min_rep_beats: int | float) -> int:
    check_len = int(round(min_rep_beats))
    check_len = min(check_len, len(p1), len(p2))
    for i in range(check_len):
        if p1[i] != p2[i]:
            return 0
    return 1 if len(p1) < len(p2) else 2


def test_loop_exists(loop_list: Sequence[Sequence[NoteSet]], loop: Sequence[NoteSet], min_rep_beats: int | float) -> Optional[int]:
    for i, pat in enumerate(loop_list):
        result = compare_loops(loop, pat, min_rep_beats)
        if result == 1:
            return -1
        if result == 2:
            return i
    return None


def filter_sub_loops(candidate_indices: Dict[float, List[Tuple[int, int]]]) -> List[Tuple[int, int, float]]:
    if not candidate_indices:
        return []

    final: List[Tuple[int, int, float]] = []
    for duration in sorted(candidate_indices.keys()):
        intervals = candidate_indices[duration]
        if not intervals:
            continue
        intervals_sorted = sorted(intervals, key=lambda x: x[0])
        merged_start, merged_end = intervals_sorted[0]
        for s, e in intervals_sorted[1:]:
            if s == merged_end:
                merged_end = e
            else:
                final.append((merged_start, merged_end, duration))
                merged_start, merged_end = s, e
        final.append((merged_start, merged_end, duration))

    seen = set()
    unique_final: List[Tuple[int, int, float]] = []
    for s, e, d in sorted(final, key=lambda x: (x[0], x[1], x[2])):
        key = (s, e, d)
        if key not in seen:
            seen.add(key)
            unique_final.append((s, e, d))
    return unique_final


def _compute_frac_positions_vectorized(values: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of fractional positions for an array of tick values.
    Returns a float array where integer positions correspond to exact tick indices,
    and fractional positions are interpolated between indices.
    """
    if values.size == 0:
        return np.array([], dtype=float)

    tb_size = tb.size
    # Use searchsorted to find insertion positions
    pos = tb.searchsorted(values, side='left')  # pos in [0..tb_size]
    res = np.empty(values.shape, dtype=float)

    # Exact matches where pos < tb_size and tb[pos] == value
    exact_mask = (pos < tb_size) & (tb[pos] == values)
    if exact_mask.any():
        res[exact_mask] = pos[exact_mask].astype(float)

    # pos == 0 and not exact
    mask_pos0 = (pos == 0) & (~exact_mask)
    if mask_pos0.any():
        prev_tick = tb[0]
        next_tick = tb[1] if tb_size > 1 else tb[0] + 1
        denom = next_tick - prev_tick if next_tick != prev_tick else 1
        res[mask_pos0] = (values[mask_pos0] - prev_tick) / denom

    # pos >= tb_size (to the right of last tick)
    mask_right = (pos >= tb_size) & (~exact_mask)
    if mask_right.any():
        if tb_size > 1:
            prev_tick = tb[-2]
            last_tick = tb[-1]
            denom = last_tick - prev_tick if last_tick != prev_tick else 1
            res[mask_right] = float(tb_size - 1) + (values[mask_right] - last_tick) / denom
        else:
            # single tick in tb
            res[mask_right] = float(0) + (values[mask_right] - tb[0])  # fallback

    # middle cases: 0 < pos < tb_size and not exact
    mask_mid = (~exact_mask) & (~mask_pos0) & (~mask_right)
    if mask_mid.any():
        pos_mid = pos[mask_mid]
        prev_tick = tb[pos_mid - 1]
        next_tick = tb[pos_mid]
        denom = next_tick - prev_tick
        # avoid division by zero
        denom = np.where(denom == 0, 1, denom)
        res[mask_mid] = (pos_mid - 1).astype(float) + (values[mask_mid] - prev_tick) / denom

    return res


def _compute_durations_batch(starts: np.ndarray, ends: np.ndarray, tb: np.ndarray) -> np.ndarray:
    """
    Compute durations (in beats) for arrays of start and end tick values.
    Vectorized and uses _compute_frac_positions_vectorized.
    """
    if tb is None or tb.size == 0:
        return np.zeros_like(starts, dtype=float)

    # Convert to numpy arrays
    starts_arr = np.asarray(starts, dtype=np.int64)
    ends_arr = np.asarray(ends, dtype=np.int64)

    # Compute fractional positions for starts and ends
    start_pos = _compute_frac_positions_vectorized(starts_arr, tb)
    end_pos = _compute_frac_positions_vectorized(ends_arr, tb)

    durations = end_pos - start_pos
    # Clip negative durations to 0
    durations = np.where(durations < 0.0, 0.0, durations)
    return durations.astype(float)


def get_duration_beats(start: int, end: int, tb: np.ndarray, tick_to_idx: Optional[Dict[int, int]] = None) -> float:
    """
    Backwards-compatible single-pair duration computation.
    For heavy workloads prefer the batch computation used in get_valid_loops.
    """
    if tb is None or tb.size == 0:
        return 0.0

    if end <= start:
        return 0.0

    try:
        tb_first = int(tb[0])
        tb_last = int(tb[-1])
    except Exception:
        return 0.0

    key = (int(start), int(end), tb_first, tb_last, int(tb.size))
    with _duration_cache_lock:
        cached = _duration_cache.get(key)
    if cached is not None:
        return cached

    # Fast path for single-element tb
    if tb.size == 1:
        duration = float(max(0.0, end - start))
        with _duration_cache_lock:
            _duration_cache[key] = duration
        return duration

    # Fallback to vectorized batch helper for a single pair
    dur = _compute_durations_batch(np.array([start], dtype=np.int64), np.array([end], dtype=np.int64), tb)[0]
    with _duration_cache_lock:
        _duration_cache[key] = dur
    return dur


def get_valid_loops(
    note_sets: Sequence[NoteSet],
    corr_mat: np.ndarray,
    ticks_beats: Sequence[int],
    min_rep_notes: int = 4,
    min_rep_beats: float = 2.0,
    min_beats: float = 32.0,
    max_beats: float = 32.0,
    min_loop_note_density: float = 0.5,
) -> Tuple[List[Sequence[NoteSet]], List[Tuple[int, int, float, float]]]:
    """
    Return detected loops and metadata.
    """
    min_rep_notes += 1  # original behavior to not count barlines
    x_idx, y_idx = np.where(corr_mat == min_rep_notes)
    if x_idx.size == 0:
        return [], []

    tb_sanitized = _sanitize_ticks_beats_once(ticks_beats)
    if tb_sanitized.size == 0:
        # no valid beat ticks to compute durations -> no loops
        return [], []

    # Precompute tick->index mapping once (cheap relative to repeated rebuilds)
    try:
        tick_to_idx_global = {int(t): i for i, t in enumerate(tb_sanitized)}
    except Exception:
        tick_to_idx_global = {}

    valid_indices: Dict[float, List[Tuple[int, int]]] = {}
    # Collect unique key pairs to compute durations in batch
    unique_pairs = {}
    pairs_list_starts = []
    pairs_list_ends = []
    pairs_keys = []

    # First pass: collect candidate pairs and unique (start_tick, end_tick) pairs
    for xi, yi in zip(x_idx, y_idx):
        try:
            run_len = int(corr_mat[xi, yi])
        except Exception:
            continue
        start_x = xi - run_len + 1
        start_y = yi - run_len + 1

        if start_x < 0 or start_y < 0 or start_x >= len(note_sets) or start_y >= len(note_sets):
            continue

        try:
            loop_start_time = int(note_sets[start_x].start)
            loop_end_time = int(note_sets[start_y].start)
        except Exception:
            continue

        # sanity check tick magnitudes
        if loop_start_time < 0 or loop_end_time < 0 or loop_start_time > _MAX_TICK or loop_end_time > _MAX_TICK:
            continue

        key_pair = (loop_start_time, loop_end_time)
        if key_pair not in unique_pairs:
            unique_pairs[key_pair] = len(pairs_list_starts)
            pairs_list_starts.append(loop_start_time)
            pairs_list_ends.append(loop_end_time)
            pairs_keys.append(key_pair)

    if not pairs_list_starts:
        return [], []

    # Batch compute durations for all unique pairs
    starts_arr = np.asarray(pairs_list_starts, dtype=np.int64)
    ends_arr = np.asarray(pairs_list_ends, dtype=np.int64)
    durations_arr = _compute_durations_batch(starts_arr, ends_arr, tb_sanitized)
    # Round durations to 2 decimals to match previous behavior
    durations_rounded = np.round(durations_arr.astype(float), 2)

    # Build a mapping from key_pair to rounded duration
    keypair_to_duration: Dict[Tuple[int, int], float] = {
        kp: float(durations_rounded[idx]) for idx, kp in enumerate(pairs_keys)
    }

    # Second pass: populate valid_indices using computed durations
    for xi, yi in zip(x_idx, y_idx):
        try:
            run_len = int(corr_mat[xi, yi])
        except Exception:
            continue
        start_x = xi - run_len + 1
        start_y = yi - run_len + 1

        if start_x < 0 or start_y < 0 or start_x >= len(note_sets) or start_y >= len(note_sets):
            continue

        try:
            loop_start_time = int(note_sets[start_x].start)
            loop_end_time = int(note_sets[start_y].start)
        except Exception:
            continue

        key_pair = (loop_start_time, loop_end_time)
        loop_num_beats = keypair_to_duration.get(key_pair, 0.0)

        if min_beats <= loop_num_beats <= max_beats:
            valid_indices.setdefault(loop_num_beats, []).append((int(xi), int(yi)))

    if not valid_indices:
        return [], []

    filtered = filter_sub_loops(valid_indices)

    loops: List[Sequence[NoteSet]] = []
    loop_bp: List[Tuple[int, int, float, float]] = []
    corr_size = corr_mat.shape[0]

    for start_x, start_y, loop_num_beats in filtered:
        x = start_x
        y = start_y
        while x + 1 < corr_size and y + 1 < corr_size and corr_mat[x + 1, y + 1] > corr_mat[x, y]:
            x += 1
            y += 1

        beginning = x - int(corr_mat[x, y]) + 1
        end = y - int(corr_mat[x, y]) + 1

        if beginning < 0 or end < 0 or beginning >= len(note_sets) or end >= len(note_sets):
            continue

        start_tick = int(note_sets[beginning].start)
        end_tick = int(note_sets[end].start)

        # Try to get duration from keypair_to_duration first
        duration_beats = keypair_to_duration.get((start_tick, end_tick))
        if duration_beats is None:
            # Fallback: compute single pair (rare)
            duration_beats = get_duration_beats(start_tick, end_tick, tb_sanitized, tick_to_idx=tick_to_idx_global)

        if duration_beats >= min_rep_beats and not is_empty_loop(note_sets[beginning:end]):
            loop = note_sets[beginning : (end + 1)]
            loop_density = get_loop_density(loop, loop_num_beats)
            if loop_density < min_loop_note_density:
                continue
            exist_result = test_loop_exists(loops, loop, min_rep_beats)
            if exist_result is None:
                loops.append(loop)
                loop_bp.append((start_tick, end_tick, loop_num_beats, loop_density))
            elif exist_result > 0:
                loops[exist_result] = loop
                loop_bp[exist_result] = (start_tick, end_tick, loop_num_beats, loop_density)

    return loops, loop_bp