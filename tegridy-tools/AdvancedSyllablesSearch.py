#! /usr/bin/python3

r'''############################################################################
################################################################################
#
#
#	    Advanced Syllables Search Python Module
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
#       !pip install nltk pyphen pronouncing
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
    import nltk
    print('=' * 70)
    print('NLTK is found!')
    print('=' * 70)
    print('Downloading CMU dict...')
    status = nltk.download('cmudict')
    print('=' * 70)

except ImportError as e:
    print(f"Error: Could not import NLTK. Details: {e}")
    print("Please make sure NLTK is installed.")
    print('=' * 70)
    
    raise RuntimeError("NLTK could not be loaded!") from e

################################################################################

import re
from typing import List, Sequence, Set, Dict, Tuple
import pronouncing
import pyphen
from functools import lru_cache, wraps

################################################################################

# Build-time imports for cmudict
try:
    from nltk.corpus import cmudict
except Exception as e:
    raise ImportError(
        "NLTK cmudict is required for built-in vowel/onset dictionaries. "
        "Install NLTK and download cmudict: "
        "pip install nltk; then in Python: import nltk; nltk.download('cmudict')"
    ) from e


################################################################################

_pyphen = pyphen.Pyphen(lang="en_US")

_ALPHA_RE = re.compile(r"^[A-Za-z]+$")
_CONSONANT_CLUSTER_RE = re.compile(r"^[^aeiouyAEIOUY]+")

# --- Caching configuration ---
# Toggle caching at runtime using enable_caching() / disable_caching()
CACHING = True

# LRU cache sizes (tune as needed)
_PHONES_CACHE_SIZE = 100_000
_NUCLEI_CACHE_SIZE = 100_000
_PYPHEN_CACHE_SIZE = 50_000
_WORD_SYLL_CACHE_SIZE = 200_000
_LINES_SYLL_CACHE_SIZE = 20_000

def enable_caching() -> None:
    """Enable module-level caching for hot functions."""
    global CACHING
    CACHING = True

def disable_caching() -> None:
    """Disable module-level caching for hot functions."""
    global CACHING
    CACHING = False

# --- Build VOWEL_PHONES and VALID_ONSETS from cmudict ---

def _build_cmudict_derived_sets() -> Tuple[Set[str], Set[str]]:
    """
    Use nltk.corpus.cmudict to derive:
    - vowel phones: ARPAbet symbols that include a stress digit in pronunciations,
      stripped of trailing digits (e.g., 'AH0' -> 'AH').
    - valid onsets: orthographic onset clusters extracted from cmudict word forms:
      everything before the first vowel letter (a,e,i,o,u,y). Include empty onset.
    """
    cmu: Dict[str, List[List[str]]] = cmudict.dict()  # word -> list of phone-lists
    vowel_phones = set()
    onsets = set()
    vowel_letters = set("aeiouy")

    for orth in cmu.keys():
        # orth is lowercase orthographic form (may include punctuation like "can't" as cant)
        # extract onset (letters before first vowel letter)
        cleaned = re.sub(r"[^a-z]", "", orth.lower())
        if cleaned:
            # onset = prefix before first vowel letter
            first_v = next((i for i, ch in enumerate(cleaned) if ch in vowel_letters), None)
            if first_v is None:
                onset = cleaned  # all consonants
            else:
                onset = cleaned[:first_v]
            # limit onset length to a sensible max (e.g., 5)
            onsets.add(onset[:5])
        # inspect pronunciations to collect vowel phones
        pron_lists = cmu.get(orth, [])
        for phones in pron_lists:
            for ph in phones:
                # if phone contains a digit it's a vowel in CMU dict (stress marker)
                if re.search(r"\d", ph):
                    vowel_phones.add(re.sub(r"\d", "", ph))

    # ensure an empty onset is included for vowel-initial syllables
    onsets.add("")
    # normalize to lowercase (onsets are orthographic)
    onsets = set(o.lower() for o in onsets if o is not None)

    return vowel_phones, onsets

VOWEL_PHONES, VALID_ONSETS = _build_cmudict_derived_sets()

# --- Cached low-level helpers ---

def _cache_if_enabled(maxsize):
    """Decorator factory to apply lru_cache when CACHING is True."""
    def decorator(func):
        if CACHING:
            return lru_cache(maxsize=maxsize)(func)
        return func
    return decorator

# For dynamic toggling of caching after module import, we provide both cached and uncached impls,
# and select at call-time via thin wrappers. This allows enable/disable at runtime.

# Actual implementations (uncached)
def _phones_for_word_impl(word: str) -> Sequence[str]:
    """Return ARPAbet phones list for first pron in pronouncing or empty list."""
    p = pronouncing.phones_for_word(word.lower())
    return p[0].split() if p else []

def count_phonetic_nuclei_from_phones(phones: Sequence[str]) -> int:
    """Count vowel phones in ARPAbet phone list (strip digits)."""
    return sum(1 for ph in phones if re.sub(r"\d", "", ph) in VOWEL_PHONES)

def _pyphen_inserted_impl(word: str) -> str:
    """Wrapper around pyphen insertion (may raise); isolated for caching."""
    return _pyphen.inserted(word)

def _get_phonetic_nuclei_for_word_impl(word: str) -> int:
    """Number of phonetic nuclei for a word using pronouncing / CMU dict."""
    if not word or not _ALPHA_RE.match(word):
        return 0
    phones = _phones_for_word_impl(word)
    return count_phonetic_nuclei_from_phones(phones)

def _word_to_syllables_impl(word: str, min_syllable_len: int = 2) -> List[str]:
    """Convert an alphabetic word to syllable-like fragments using CMU-derived sets."""
    if not _ALPHA_RE.match(word):
        return [word]

    nuclei = _get_phonetic_nuclei_for_word_impl(word)

    # Pyphen hyphenation attempt
    try:
        hyph = _pyphen_inserted_impl(word)
    except Exception:
        hyph = word
    if "-" in hyph:
        py_syls = hyph.split("-")
        if nuclei == 0 or abs(len(py_syls) - nuclei) <= 1:
            return _merge_short_fragments(py_syls, min_syllable_len)

    if nuclei > 0:
        parts = _maximal_onset_partition_letters(word, nuclei)
        return _merge_short_fragments(parts, min_syllable_len)

    # fallback: vowel-letter groups
    vowel_groups = []
    buf = ""
    for ch in word:
        if ch.lower() in "aeiouy":
            if buf:
                vowel_groups.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        vowel_groups.append(buf)
    target = max(1, len(vowel_groups))
    parts = _maximal_onset_partition_letters(word, target)
    return _merge_short_fragments(parts, min_syllable_len)

def _lines_to_syllable_lists_impl(text: str, min_syllable_len: int = 2, keep_last_word_unsplit: bool = False) -> List[List[str]]:
    """Convert multiline text to lists of syllable-like tokens per line, preserving punctuation."""
    out: List[List[str]] = []
    lines = text.splitlines() or [text]
    token_re = re.compile(r"[A-Za-z]+|[^A-Za-z\s]")

    for line in lines:
        tokens = token_re.findall(line)
        last_alpha_idx = None
        if keep_last_word_unsplit:
            for idx in range(len(tokens) - 1, -1, -1):
                if _ALPHA_RE.match(tokens[idx]):
                    last_alpha_idx = idx
                    break

        line_sylls: List[str] = []
        for idx, tok in enumerate(tokens):
            if _ALPHA_RE.match(tok):
                if keep_last_word_unsplit and idx == last_alpha_idx:
                    line_sylls.append(tok)
                else:
                    line_sylls.extend(word_to_syllables(tok, min_syllable_len))
            else:
                line_sylls.append(tok)
        out.append(line_sylls)
    return out

# Cached variants created via lru_cache
_phones_for_word_cached = lru_cache(maxsize=_PHONES_CACHE_SIZE)(_phones_for_word_impl)
_get_phonetic_nuclei_for_word_cached = lru_cache(maxsize=_NUCLEI_CACHE_SIZE)(_get_phonetic_nuclei_for_word_impl)
_pyphen_inserted_cached = lru_cache(maxsize=_PYPHEN_CACHE_SIZE)(_pyphen_inserted_impl)
_word_to_syllables_cached = lru_cache(maxsize=_WORD_SYLL_CACHE_SIZE)(_word_to_syllables_impl)
_lines_to_syllable_lists_cached = lru_cache(maxsize=_LINES_SYLL_CACHE_SIZE)(_lines_to_syllable_lists_impl)

# Thin wrappers that honor CACHING flag and call the appropriate implementation
def _phones_for_word(word: str) -> Sequence[str]:
    if CACHING:
        return _phones_for_word_cached(word)
    return _phones_for_word_impl(word)

def get_phonetic_nuclei_for_word(word: str) -> int:
    if CACHING:
        return _get_phonetic_nuclei_for_word_cached(word)
    return _get_phonetic_nuclei_for_word_impl(word)

def _pyphen_inserted(word: str) -> str:
    if CACHING:
        return _pyphen_inserted_cached(word)
    return _pyphen_inserted_impl(word)

def word_to_syllables(word: str, min_syllable_len: int = 2) -> List[str]:
    # lru_cache keyed by both word and min_syllable_len when CACHING enabled
    if CACHING:
        return _word_to_syllables_cached(word, min_syllable_len)
    return _word_to_syllables_impl(word, min_syllable_len)

def lines_to_syllable_lists(text: str, min_syllable_len: int = 2, keep_last_word_unsplit: bool = False) -> List[List[str]]:
    if CACHING:
        return _lines_to_syllable_lists_cached(text, min_syllable_len, keep_last_word_unsplit)
    return _lines_to_syllable_lists_impl(text, min_syllable_len, keep_last_word_unsplit)

# --- Remaining existing helpers (unchanged logic) ---

def _maximal_onset_partition_letters(word: str, target: int) -> List[str]:
    """Partition word into `target` pieces using a letter-based maximal onset heuristic."""
    if target <= 1 or not word:
        return [word]

    letters = list(word)
    vowel_idxs = [i for i, ch in enumerate(letters) if ch.lower() in "aeiouy"]
    if not vowel_idxs:
        return [word]

    boundaries = [(vowel_idxs[i] + vowel_idxs[i + 1]) // 2 for i in range(len(vowel_idxs) - 1)]
    starts = [0] + [b + 1 for b in boundaries]
    ends = boundaries + [len(letters) - 1]
    parts = ["".join(letters[s:e + 1]) for s, e in zip(starts, ends)]

    # Adjust to match target
    if len(parts) > target:
        while len(parts) > target:
            i = len(parts) - 2
            parts[i] = parts[i] + parts.pop(i + 1)
    elif len(parts) < target:
        needed = target - len(parts)
        for _ in range(needed):
            idx = max(range(len(parts)), key=lambda i: len(parts[i]))
            s = parts.pop(idx)
            mid = max(1, len(s) // 2)
            parts.insert(idx, s[:mid])
            parts.insert(idx + 1, s[mid:])

    # Maximal onset assignment using onsets derived from cmudict (orthographic)
    for i in range(len(parts) - 1):
        left, right = parts[i], parts[i + 1]
        if not right:
            continue
        m = _CONSONANT_CLUSTER_RE.match(right)
        right_head = m.group(0) if m else ""
        assigned = ""
        # choose longest valid onset from start of right_head
        for k in range(len(right_head), -1, -1):
            cand = right_head[:k].lower()
            if cand in VALID_ONSETS:
                assigned = right_head[:k]
                break
        move = len(assigned)
        if move > 0:
            non_onset_prefix = right_head[move:]
            if non_onset_prefix:
                parts[i] = left + non_onset_prefix
                parts[i + 1] = right[len(non_onset_prefix):]

    return [p for p in (s.strip() for s in parts) if p]

def _merge_short_fragments(sylls: List[str], min_len: int) -> List[str]:
    """Conservative merging of short fragments; same logic as prior implementation."""
    if min_len <= 1 or not sylls:
        return sylls[:]
    out = sylls[:]
    i = 0
    while i < len(out):
        s = out[i]
        if not _ALPHA_RE.match(s):
            i += 1
            continue
        if len(s) < min_len:
            if len(s) == 1:
                left = i - 1
                right = i + 1
                left_ok = left >= 0 and _ALPHA_RE.match(out[left])
                right_ok = right < len(out) and _ALPHA_RE.match(out[right])
                if left_ok and right_ok:
                    if len(out[left]) >= len(out[right]):
                        out[left] = out[left] + out.pop(i)
                        i = max(left, 0)
                        continue
                    else:
                        out[i] = out[i] + out.pop(right)
                        continue
                elif left_ok:
                    out[left] = out[left] + out.pop(i)
                    i = max(left, 0)
                    continue
                elif right_ok:
                    out[i] = out[i] + out.pop(right)
                    continue
                else:
                    i += 1
                    continue
            else:
                left = i - 1
                right = i + 1
                left_pref = left >= 0 and _ALPHA_RE.match(out[left]) and len(out[left]) >= min_len
                right_pref = right < len(out) and _ALPHA_RE.match(out[right]) and len(out[right]) >= min_len
                if left_pref:
                    out[left] = out[left] + out.pop(i)
                    i = max(left, 0)
                    continue
                if right_pref:
                    out[i] = out[i] + out.pop(right)
                    continue
                if right < len(out) and _ALPHA_RE.match(out[right]):
                    out[i] = out[i] + out.pop(right)
                    continue
                if left >= 0 and _ALPHA_RE.match(out[left]):
                    out[left] = out[left] + out.pop(i)
                    i = max(left, 0)
                    continue
        i += 1
    return out

# --- Module-level cache control utilities ---

def clear_caches() -> None:
    """Clear all internal LRU caches (useful in long-running processes)."""
    try:
        _phones_for_word_cached.cache_clear()
        _get_phonetic_nuclei_for_word_cached.cache_clear()
        _pyphen_inserted_cached.cache_clear()
        _word_to_syllables_cached.cache_clear()
        _lines_to_syllable_lists_cached.cache_clear()
    except Exception:
        # If caching disabled at import time these names still exist (they are lru-wrapped),
        # but guard for safety.
        pass
        
###################################################################################

print('=' * 70)
print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the Numba Haystack Search Python module
###################################################################################