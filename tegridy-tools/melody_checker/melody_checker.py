#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	    Melody Checker Python Module
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
#       !pip install numpy==1.24.4
#       !pip install -U scikit-learn
#
################################################################################
'''

################################################################################
# --- Quick demo (uncomment to run) ---
# demo_sequences = [train_data[0], train_data[1], [7, 135, 323]]
# results = predict_batch(pipeline, demo_sequences)
# for r in results:
#     print(r)
################################################################################

################################################################################

print('=' * 70)
print('Loading module...')
print('Please wait...')
print('=' * 70)

################################################################################

import sys
import types
import joblib
import numpy as np
import hashlib
from typing import List, Dict, Any, Tuple

################################################################################

# --- Top-level picklable adapters (must match training definitions) ---
class CalibratedAdapter:
    def __init__(self, scaler, calibrator):
        self.scaler = scaler
        self.calibrator = calibrator

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.calibrator.predict_proba(Xs)

###################################################################################

class RawPipelineAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

###################################################################################

# --- Utilities and feature extraction (must match training implementation) ---
def deterministic_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.astype(np.int64).tobytes())
    return h.hexdigest()

###################################################################################

def pad_or_trim(arr: np.ndarray, length: int, pad_value: int = -1) -> np.ndarray:
    if arr.shape[0] >= length:
        return arr[:length]
    out = np.full(length, pad_value, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out

###################################################################################

def _sequence_entropy(seq: np.ndarray) -> float:
    if seq.size == 0:
        return 0.0
    vals, counts = np.unique(seq, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    ent = -np.sum(probs * np.log2(probs + 1e-12))
    return float(ent)

###################################################################################

def _ngram_counts(seq: np.ndarray, n: int = 2, top_k: int = 8) -> List[int]:
    if seq.size < n:
        return [0] * top_k
    hashes = []
    for i in range(len(seq) - n + 1):
        tok = seq[i:i+n].astype(np.int64)
        h = hashlib.sha256(tok.tobytes()).digest()
        hv = int.from_bytes(h[:4], "little") & 0xFFFFFFFF
        hashes.append(hv)
    take = min(len(hashes), top_k)
    last_hashes = hashes[-take:]
    out = [int(h % 7) for h in last_hashes]
    if take < top_k:
        out = [0] * (top_k - take) + out
    return out

###################################################################################

def extract_features_from_single_sequence(seq: List[int],
                                          suffix_window: int = 12,
                                          prefix_window: int = 12,
                                          checksum_window: int = 10) -> Tuple[np.ndarray, List[str]]:
    arr = np.asarray(seq, dtype=np.int64)
    L = arr.shape[0]
    length = L
    mod3 = L % 3
    unique_count = int(np.unique(arr).size) if L > 0 else 0

    pref = pad_or_trim(arr[:prefix_window], prefix_window, pad_value=-1)
    suff = pad_or_trim(arr[-suffix_window:], suffix_window, pad_value=-1)

    if suff.size > 0:
        diffs = np.diff(suff.astype(np.int64), prepend=(suff[0] if suff.size > 0 else 0))
        last_diffs = pad_or_trim(diffs[-3:], 3, pad_value=-9999)
    else:
        last_diffs = np.array([-9999, -9999, -9999], dtype=np.int64)

    checksum_tokens = arr[-checksum_window:] if L >= checksum_window else arr
    if checksum_tokens.size > 0:
        chash = int(hashlib.sha256(checksum_tokens.astype(np.int64).tobytes()).hexdigest()[:16], 16)
        checksum_comp = int(chash % (2 ** 20))
    else:
        checksum_comp = 0

    smallset = set(arr[(arr >= 0) & (arr <= 31)].tolist())
    bitmask = 0
    for t in sorted(smallset):
        bitmask |= 1 << int(t)

    mean_val = float(np.mean(arr)) if L > 0 else 0.0
    std_val = float(np.std(arr)) if L > 0 else 0.0
    median_val = float(np.median(arr)) if L > 0 else 0.0
    min_val = int(np.min(arr)) if L > 0 else -1
    max_val = int(np.max(arr)) if L > 0 else -1
    entropy = _sequence_entropy(arr)

    p10 = float(np.percentile(arr, 10)) if L > 0 else 0.0
    p25 = float(np.percentile(arr, 25)) if L > 0 else 0.0
    p75 = float(np.percentile(arr, 75)) if L > 0 else 0.0
    p90 = float(np.percentile(arr, 90)) if L > 0 else 0.0

    last_triplet = (-1, -1, -1)
    if L >= 3:
        last_triplet = (int(arr[-3]), int(arr[-2]), int(arr[-1]))
    last_triplet_hash = int(hashlib.sha256(np.array(last_triplet, dtype=np.int64).tobytes()).hexdigest()[:8], 16) % (2 ** 20)

    bigram_summary = _ngram_counts(arr, n=2, top_k=6)
    trigram_summary = _ngram_counts(arr, n=3, top_k=4)

    vec = [
        float(length),
        float(mod3),
        float(unique_count),
        float(checksum_comp),
        float(bitmask),
        float(mean_val),
        float(std_val),
        float(median_val),
        float(min_val),
        float(max_val),
        float(entropy),
        float(p10),
        float(p25),
        float(p75),
        float(p90),
        float(last_triplet_hash),
    ]
    vec.extend([float(x) for x in pref.tolist()])
    vec.extend([float(x) for x in suff.tolist()])
    vec.extend([float(x) for x in last_diffs.tolist()])
    vec.extend([float(x) for x in bigram_summary])
    vec.extend([float(x) for x in trigram_summary])

    feature_names = [
        "length", "mod3", "unique_count", "checksum_comp", "bitmask",
        "mean", "std", "median", "min", "max", "entropy",
        "p10", "p25", "p75", "p90", "last_triplet_hash"
    ]
    feature_names += [f"pref_{i}" for i in range(prefix_window)]
    feature_names += [f"suff_{i}" for i in range(suffix_window)]
    feature_names += [f"last_diff_{i}" for i in range(3)]
    feature_names += [f"bigram_sum_{i}" for i in range(6)]
    feature_names += [f"trigram_sum_{i}" for i in range(4)]

    return np.asarray(vec, dtype=np.float64), feature_names

###################################################################################

# --- Load pipeline (ensure filename matches what was saved) ---
PIPELINE_PATH = "melody_checker.joblib"

###################################################################################

def load_pipeline(path: str = PIPELINE_PATH) -> Dict[str, Any]:
    """
    Lazily load the saved pipeline from disk and handle pickles that
    reference classes defined under __main__ at save time.

    If unpickling raises an AttributeError (e.g. "Can't get attribute
    'CalibratedAdapter' on <module '__main__'>"), this function will
    inject the adapter classes from this module into sys.modules['__main__']
    so pickle can resolve them, then retry the load.
    """
    try:
        return joblib.load(path)
    except AttributeError as exc:
        msg = str(exc)
        # Quick heuristic: only proceed when pickle complains about missing adapter classes
        if "Can't get attribute" in msg and "__main__" in msg:
            main_mod = sys.modules.get("__main__")
            if main_mod is None:
                main_mod = types.ModuleType("__main__")
                sys.modules["__main__"] = main_mod

            # Inject the adapter classes into the __main__ module so unpickler finds them
            # Use the same names that appear in the pickle
            setattr(main_mod, "CalibratedAdapter", CalibratedAdapter)
            setattr(main_mod, "RawPipelineAdapter", RawPipelineAdapter)

            # Retry loading now that classes are visible on __main__
            return joblib.load(path)
        # Re-raise other attribute errors
        raise

###################################################################################

# --- Inference logic mirroring training inference ---
def infer_with_rules_loaded(pipeline: Dict[str, Any], sequence: List[int]) -> Dict[str, Any]:
    seq_arr = np.asarray(sequence, dtype=np.int64)
    seq_hash = deterministic_hash(seq_arr)

    # Triplet alignment hard rule
    if pipeline.get("triplet_rules", {}).get("enforce_triplet_alignment", False):
        if int(seq_arr.shape[0]) % 3 != 0:
            return {
                "decision": False,
                "prob": 0.0,
                "reason": "triplet_alignment_failed",
                "diagnostics": {"sequence_hash": seq_hash, "length": int(seq_arr.shape[0])}
            }

    # Sentinel
    sentinel = pipeline.get("triplet_rules", {}).get("sentinel_triplet", None)
    if sentinel is not None and sentinel != (0, 0, 0):
        if seq_arr.shape[0] >= 3:
            last_triplet = (int(seq_arr[-3]), int(seq_arr[-2]), int(seq_arr[-1]))
            if last_triplet == tuple(map(int, sentinel)):
                return {
                    "decision": True,
                    "prob": 1.0,
                    "reason": "sentinel_triplet_match",
                    "diagnostics": {"sequence_hash": seq_hash, "last_triplet": last_triplet}
                }

    X_feat_vec, feature_names = extract_features_from_single_sequence(sequence)
    X_feat = X_feat_vec.reshape(1, -1)

    clf_adapter = pipeline.get("calibrated_adapter", None)
    if clf_adapter is not None:
        prob = float(clf_adapter.predict_proba(X_feat)[:, 1][0])
    else:
        stored = pipeline.get("scaler_and_clf", None)
        if stored is None:
            raise RuntimeError("No classifier available in pipeline")
        prob = float(stored.predict_proba(X_feat)[:, 1][0])

    chosen_threshold = pipeline.get("operating_point", {}).get("chosen_threshold", 0.5)
    decision = bool(prob >= float(chosen_threshold))

    diagnostics = {
        "sequence_hash": seq_hash,
        "feature_vector": {name: float(val) for name, val in zip(feature_names, X_feat[0].tolist())},
        "threshold": float(chosen_threshold),
    }

    return {
        "decision": decision,
        "prob": prob,
        "reason": "ml_threshold",
        "diagnostics": diagnostics
    }

###################################################################################

def predict_batch(pipeline: Dict[str, Any], sequences: List[List[int]]) -> List[Dict[str, Any]]:
    results = []
    for seq in sequences:
        res = infer_with_rules_loaded(pipeline, seq)
        results.append(res)
    return results

###################################################################################

print('=' * 70)
print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Melody Checker Python module
###################################################################################