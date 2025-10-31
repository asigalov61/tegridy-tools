#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	    Melody Checker Train Python Module
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

print('=' * 70)
print('Loading module...')
print('Please wait...')
print('=' * 70)

################################################################################

# deterministic_pipeline_with_rules_enhanced_picklable.py
from typing import List, Tuple, Dict, Any, Optional
import os
import hashlib
import numpy as np
from collections import Counter
import joblib

# sklearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV

###################################################################################

# Optional newer API: FrozenEstimator (sklearn >= 1.6). Import safely.

try:
    from sklearn.calibration import FrozenEstimator  # type: ignore
    HAS_FROZEN = True
    
except Exception:
    FrozenEstimator = None  # type: ignore
    HAS_FROZEN = False

###################################################################################

# -------------------------
# Global reproducibility
# -------------------------

SEED = 123456789
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

###################################################################################

# -------------------------
# Top-level picklable adapter used for calibrated predict_proba
# -------------------------

class CalibratedAdapter:
    def __init__(self, scaler, calibrator):
        self.scaler = scaler
        self.calibrator = calibrator

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return self.calibrator.predict_proba(Xs)

###################################################################################

# Fallback raw adapter (picklable)
class RawPipelineAdapter:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

###################################################################################

# -------------------------
# Utilities (hashing, seed folding, padding)
# -------------------------

def deterministic_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.astype(np.int64).tobytes())
    return h.hexdigest()

###################################################################################

def _seed32_from_hash_int(h_int: int, global_seed: int) -> int:
    mixed = (h_int ^ (global_seed & 0xFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    seed32 = ((mixed >> 32) ^ (mixed & 0xFFFFFFFF)) & 0xFFFFFFFF
    return int(seed32)

###################################################################################

def pad_or_trim(arr: np.ndarray, length: int, pad_value: int = -1) -> np.ndarray:
    if arr.shape[0] >= length:
        return arr[:length]
    out = np.full(length, pad_value, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out

###################################################################################

# -------------------------
# Negative generation (deterministic truncation + deterministic corruption)
# -------------------------

def create_truncated_and_corrupted_negatives_from_train(train_sequences: List[List[int]],
                                                       truncation_min_frac: float = 0.02,
                                                       truncation_max_frac: float = 0.45,
                                                       preserve_triplet_alignment: bool = True,
                                                       corruption_modes: Tuple[str, ...] = ("truncate", "truncate_mask", "truncate_swap")) -> List[np.ndarray]:
    negatives: List[np.ndarray] = []
    for seq_list in train_sequences:
        seq = np.asarray(seq_list, dtype=np.int64)
        L = seq.shape[0]
        if L == 0:
            negatives.append(np.asarray([-1], dtype=np.int64))
            continue

        h_int = int(deterministic_hash(seq), 16)
        seed32 = _seed32_from_hash_int(h_int, SEED)
        seq_rng = np.random.RandomState(seed32)

        frac = seq_rng.uniform(truncation_min_frac, truncation_max_frac)
        remove = max(1, int(np.floor(L * frac)))
        new_len = L - remove
        if preserve_triplet_alignment:
            new_len = new_len - (new_len % 3)
        if new_len < 1:
            new_len = 1
        truncated = seq[:new_len].copy()

        mode_idx = int(h_int & 0xFFFFFFFF) % len(corruption_modes)
        mode = corruption_modes[mode_idx]

        if mode == "truncate":
            negatives.append(truncated)
        elif mode == "truncate_mask":
            m = min(3, truncated.shape[0])
            mask_positions = [(seed32 + i) % truncated.shape[0] for i in range(m)]
            corrupted = truncated.copy()
            for p in mask_positions:
                corrupted[p] = -1
            negatives.append(corrupted)
        elif mode == "truncate_swap":
            if truncated.shape[0] >= 2:
                i1 = (seed32 >> 3) % truncated.shape[0]
                i2 = (seed32 >> 7) % truncated.shape[0]
                corrupted = truncated.copy()
                corrupted[i1], corrupted[i2] = corrupted[i2], corrupted[i1]
                negatives.append(corrupted)
            else:
                negatives.append(truncated)
        else:
            negatives.append(truncated)
    return negatives

###################################################################################

# -------------------------
# Deterministic feature extraction (enhanced)
# -------------------------

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

def extract_features_from_sequences(sequences: List[np.ndarray],
                                    suffix_window: int = 12,
                                    prefix_window: int = 12,
                                    checksum_window: int = 10) -> Tuple[np.ndarray, List[str]]:
    feature_rows: List[np.ndarray] = []
    for seq in sequences:
        seq = np.asarray(seq, dtype=np.int64)
        L = seq.shape[0]

        length = L
        mod3 = L % 3
        unique_count = int(np.unique(seq).size) if L > 0 else 0

        pref = pad_or_trim(seq[:prefix_window], prefix_window, pad_value=-1)
        suff = pad_or_trim(seq[-suffix_window:], suffix_window, pad_value=-1)

        if suff.size > 0:
            diffs = np.diff(suff.astype(np.int64), prepend=(suff[0] if suff.size > 0 else 0))
            last_diffs = pad_or_trim(diffs[-3:], 3, pad_value=-9999)
        else:
            last_diffs = np.array([-9999, -9999, -9999], dtype=np.int64)

        checksum_tokens = seq[-checksum_window:] if L >= checksum_window else seq
        if checksum_tokens.size > 0:
            chash = int(hashlib.sha256(checksum_tokens.astype(np.int64).tobytes()).hexdigest()[:16], 16)
            checksum_comp = int(chash % (2 ** 20))
        else:
            checksum_comp = 0

        smallset = set(seq[(seq >= 0) & (seq <= 31)].tolist())
        bitmask = 0
        for t in sorted(smallset):
            bitmask |= 1 << int(t)

        mean_val = float(np.mean(seq)) if L > 0 else 0.0
        std_val = float(np.std(seq)) if L > 0 else 0.0
        median_val = float(np.median(seq)) if L > 0 else 0.0
        min_val = int(np.min(seq)) if L > 0 else -1
        max_val = int(np.max(seq)) if L > 0 else -1
        entropy = _sequence_entropy(seq)

        p10 = float(np.percentile(seq, 10)) if L > 0 else 0.0
        p25 = float(np.percentile(seq, 25)) if L > 0 else 0.0
        p75 = float(np.percentile(seq, 75)) if L > 0 else 0.0
        p90 = float(np.percentile(seq, 90)) if L > 0 else 0.0

        last_triplet = (-1, -1, -1)
        if L >= 3:
            last_triplet = (int(seq[-3]), int(seq[-2]), int(seq[-1]))
        last_triplet_hash = int(hashlib.sha256(np.array(last_triplet, dtype=np.int64).tobytes()).hexdigest()[:8], 16) % (2 ** 20)

        bigram_summary = _ngram_counts(seq, n=2, top_k=6)
        trigram_summary = _ngram_counts(seq, n=3, top_k=4)

        vec: List[float] = [
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

        feature_rows.append(np.asarray(vec, dtype=np.float64))

    X = np.vstack(feature_rows) if len(feature_rows) > 0 else np.zeros((0, 0))
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
    return X, feature_names

###################################################################################

# -------------------------
# Deterministic threshold selection helper
# -------------------------

def pick_threshold_for_precision(y_true: np.ndarray, y_prob: np.ndarray, target_precision: float = 0.95) -> float:
    probs = np.unique(np.sort(y_prob)) if y_prob.size > 0 else np.array([0.0, 1.0])
    low, high = probs.min(), probs.max()
    grid = np.concatenate([np.linspace(high, low, num=200, endpoint=True), probs[::-1]])
    grid = np.unique(grid)[::-1]

    best_t: Optional[float] = None
    best_recall = -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        if prec >= target_precision:
            if rec > best_recall:
                best_recall = rec
                best_t = float(t)
    if best_t is None:
        from sklearn.metrics import f1_score
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
    return float(best_t)

###################################################################################

# -------------------------
# Training / model builder (deterministic) with hyperparameter search
# -------------------------

def train_detector_on_train_data(train_data: List[List[int]],
                                 truncation_min_frac: float = 0.02,
                                 truncation_max_frac: float = 0.45,
                                 preserve_triplet_alignment: bool = True,
                                 test_size: float = 0.2) -> Tuple[Dict[str, Any], Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]]:
    complete_sequences = [np.asarray(s, dtype=np.int64) for s in train_data]

    negatives = create_truncated_and_corrupted_negatives_from_train(
        train_data,
        truncation_min_frac=truncation_min_frac,
        truncation_max_frac=truncation_max_frac,
        preserve_triplet_alignment=preserve_triplet_alignment
    )

    X_seqs = complete_sequences + negatives
    y = np.array([1] * len(complete_sequences) + [0] * len(negatives), dtype=np.int8)

    X_train_seqs, X_test_seqs, y_train, y_test = train_test_split(
        X_seqs, y, test_size=test_size, stratify=y, random_state=SEED
    )

    X_train, feature_names = extract_features_from_sequences(X_train_seqs)
    X_test, _ = extract_features_from_sequences(X_test_seqs)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(random_state=SEED))
    ])

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
    param_grid = {
        "clf__max_iter": [200, 500],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [6, 10],
        "clf__l2_regularization": [0.0, 1.0]
    }

    gs = GridSearchCV(pipe, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1, refit=True)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_

    final_clf = None
    try:
        pos_count = int(np.sum(y_test == 1))
        neg_count = int(np.sum(y_test == 0))

        scaler = best.named_steps["scaler"]

        # For newer sklearn prefer FrozenEstimator to avoid FutureWarning about cv='prefit'
        if HAS_FROZEN:
            # Wrap fitted estimator (the inner clf) as FrozenEstimator and calibrate that
            inner_clf = best.named_steps["clf"]
            frozen = FrozenEstimator(inner_clf)
            # We need a CalibratedClassifierCV that accepts an estimator instance (not prefit), so use cv=3 deterministic
            # We will fit calibrator on scaled X_test with a small internal cv to produce stable probability mapping
            calibrator = CalibratedClassifierCV(frozen, method="isotonic" if (pos_count >= 20 and neg_count >= 20) else "sigmoid", cv=3)
            X_test_scaled = scaler.transform(X_test)
            calibrator.fit(X_test_scaled, y_test)
            final_clf = CalibratedAdapter(scaler, calibrator)
        else:
            # Older sklearn: use cv='prefit' pattern but avoid inner local classes.
            X_test_scaled = scaler.transform(X_test)
            method = "isotonic" if (pos_count >= 20 and neg_count >= 20) else "sigmoid"
            calibrator = CalibratedClassifierCV(best.named_steps["clf"], method=method, cv="prefit")  # type: ignore
            calibrator.fit(X_test_scaled, y_test)
            final_clf = CalibratedAdapter(scaler, calibrator)
    except Exception as e:
        # fallback adapter around full pipeline
        final_clf = RawPipelineAdapter(best)

    y_prob = final_clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_test.size > 0 else 0.0
    print(f"Deterministic metrics (threshold=0.5): precision={prec:.4f}, recall={rec:.4f}, f1={f1:.4f}, auc={auc:.4f}")
    print("GridSearch best params:", gs.best_params_)

    pipeline = {
        "feature_names": feature_names,
        "scaler_and_clf": best,
        "calibrated_adapter": final_clf,
        "seed": SEED,
        "truncation_params": {
            "truncation_min_frac": truncation_min_frac,
            "truncation_max_frac": truncation_max_frac,
            "preserve_triplet_alignment": preserve_triplet_alignment,
        },
        "best_params": gs.best_params_
    }
    return pipeline, (X_test_seqs, y_test, X_test, y_prob)

###################################################################################

# -------------------------
# Sentinel discovery and deterministic threshold selection
# -------------------------

def discover_suffix_sentinel_triplet(complete_sequences: List[List[int]],
                                     consider_last_k_triplets: int = 3) -> Tuple[Tuple[int, int, int], float]:
    triplet_counts = Counter()
    eligible = 0
    rng = np.random.RandomState(SEED)
    indices = list(range(len(complete_sequences)))
    rng.shuffle(indices)
    for i in indices:
        seq = np.asarray(complete_sequences[i], dtype=np.int64)
        L = seq.shape[0]
        if L < 3:
            continue
        eligible += 1
        max_triplets = L // 3
        take = min(consider_last_k_triplets, max_triplets)
        for t in range(1, take + 1):
            start = L - 3 * t
            trip = (int(seq[start]), int(seq[start + 1]), int(seq[start + 2]))
            triplet_counts[trip] += 1
    if eligible == 0 or len(triplet_counts) == 0:
        return (0, 0, 0), 0.0
    most_common_triplet, count = triplet_counts.most_common(1)[0]
    rel_freq = count / eligible
    return most_common_triplet, rel_freq

###################################################################################

# -------------------------
# Combined training wrapper that discovers sentinel and chooses threshold
# -------------------------

def train_with_rules_and_threshold(train_data: List[List[int]],
                                   target_precision: float = 0.95,
                                   consider_last_k_triplets: int = 3) -> Dict[str, Any]:
    sentinel_triplet, sentinel_rel = discover_suffix_sentinel_triplet(train_data, consider_last_k_triplets)
    pipeline, test_info = train_detector_on_train_data(
        train_data,
        truncation_min_frac=0.02,
        truncation_max_frac=0.45,
        preserve_triplet_alignment=True,
        test_size=0.2,
    )
    X_test_seqs, y_test, X_test, y_prob = test_info
    threshold = pick_threshold_for_precision(y_test, y_prob, target_precision)

    pipeline["triplet_rules"] = {
        "enforce_triplet_alignment": True,
        "sentinel_triplet": tuple(map(int, sentinel_triplet)),
        "sentinel_relative_freq": float(sentinel_rel),
    }
    pipeline["operating_point"] = {
        "target_precision": float(target_precision),
        "chosen_threshold": float(threshold),
    }

    joblib.dump(pipeline, "melody_checker.joblib", compress=3)
    print("Saved pipeline to melody_checker.joblib")
    print("Sentinel triplet:", pipeline["triplet_rules"]["sentinel_triplet"], "rel_freq:", pipeline["triplet_rules"]["sentinel_relative_freq"])
    print("Chosen threshold for target_precision", target_precision, ":", threshold)
    return pipeline

###################################################################################

# -------------------------
# Inference applying rules then ML (deterministic)
# -------------------------

def infer_with_rules(pipeline: Dict[str, Any], sequence: List[int]) -> Dict[str, Any]:
    seq_arr = np.asarray(sequence, dtype=np.int64)
    seq_hash = deterministic_hash(seq_arr)

    if pipeline.get("triplet_rules", {}).get("enforce_triplet_alignment", False):
        if int(seq_arr.shape[0]) % 3 != 0:
            return {
                "decision": False,
                "prob": 0.0,
                "reason": "triplet_alignment_failed",
                "diagnostics": {"sequence_hash": seq_hash, "length": int(seq_arr.shape[0])}
            }

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

    X_feat, _ = extract_features_from_sequences([seq_arr])
    clf_adapter = pipeline.get("calibrated_adapter", None)
    if clf_adapter is None:
        stored = pipeline.get("scaler_and_clf", None)
        if stored is None:
            raise RuntimeError("No classifier found in pipeline")
        prob = float(stored.predict_proba(X_feat)[:, 1][0])
    else:
        prob = float(clf_adapter.predict_proba(X_feat)[:, 1][0])

    chosen_threshold = pipeline.get("operating_point", {}).get("chosen_threshold", 0.5)
    decision = bool(prob >= float(chosen_threshold))
    return {
        "decision": decision,
        "prob": prob,
        "reason": "ml_threshold",
        "diagnostics": {
            "sequence_hash": seq_hash,
            "feature_vector": {name: float(val) for name, val in zip(pipeline["feature_names"], X_feat[0].tolist())},
            "threshold": float(chosen_threshold),
        }
    }

###################################################################################

# -------------------------
# Example usage (expects `train_data` in scope)
# -------------------------

if __name__ == "__main__":
    try:
        _ = train_data  # type: ignore
    except NameError:
        raise RuntimeError("train_data not found. Provide train_data: List[List[int]] in the environment before running this script.")

    # Recommended start default is 0.95 / 3
    pipeline = train_with_rules_and_threshold(train_data, target_precision=0.97, consider_last_k_triplets=7) 

    loaded = joblib.load("melody_checker.joblib")
    demo_rng = np.random.RandomState(SEED)
    idxs = demo_rng.choice(len(train_data), size=min(5, len(train_data)), replace=False)
    for i in idxs:
        seq = train_data[i]
        result = infer_with_rules(loaded, seq)
        print(f"Demo idx {i}: decision={result['decision']}, prob={result['prob']:.4f}, reason={result['reason']}")
        print("Diagnostics:", result["diagnostics"])

###################################################################################

print('=' * 70)
print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Melody Checker Train Python module
###################################################################################
