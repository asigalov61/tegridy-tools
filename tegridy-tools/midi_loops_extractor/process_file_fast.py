from __future__ import annotations
from typing import Dict, Tuple, Any, List
import os
import numpy as np
from miditok.utils import get_bars_ticks, get_beats_ticks
from miditok.constants import CLASS_OF_INST, INSTRUMENT_CLASSES
from symusic import Score, Track, TimeSignature

from corr_mat_fast import calc_correlation, get_valid_loops
from note_set_fast import compute_note_sets

MAX_NOTES_PER_TRACK = 50000
MIN_NOTES_PER_TRACK = 5

# New: sanity threshold for tick values (should match other modules)
_MAX_TICK = 10 ** 9

# Optional: allow enabling numba/cuda via environment variables here as well.
# Example:
#   export USE_NUMBA=1
#   export USE_CUDA=1
# The corr_mat_fast module reads these env vars at import time.

def get_instrument_type(track) -> str:
    """
    Determines MIDI instrument class of a track.

    :param track: A pretty_midi.Instrument object
    :return: name of instrument class
    """
    if track.is_drum:
        return "Drums"

    program_number = track.program  # MIDI program number (0–127)
    instrument_class_index = CLASS_OF_INST[program_number]
    instrument_class_name = INSTRUMENT_CLASSES[instrument_class_index]["name"]
    
    return instrument_class_name


def create_loop_dict(endpoint_data: Tuple[int, int, float, float], track_idx: int, instrument_type: str) -> Dict[str,Any]:
    """
    Formats loop metadata into a dictionary for dataset generation
    """
    start, end, beats, density = endpoint_data
    return {
        "track_idx": track_idx,
        "instrument_type": instrument_type,
        "start": start,
        "end": end,
        "duration_beats": beats,
        "note_density": density
    }


def detect_loops_from_path(file_info: Dict) -> Dict[str,List]:
    """
     Given a MIDI file, locate all loops present across all of its tracks
    """
    file_path = file_info['file_path']
    if isinstance(file_path, list):
        file_path = file_path[0]
    try:
        score = Score(file_path)
    except Exception:
        # Unable to parse score (malformed file) -> skip file
        print(f"Unable to parse score for {file_path}, skipping")
        return {
            "track_idx": [],
            "instrument_type": [],
            "start": [],
            "end": [],
            "duration_beats": [],
            "note_density": [],
        }
    return detect_loops(score, file_path=file_path)


def detect_loops(score: Score, file_path: str = None) -> Dict[str,List]:
    """
     Given a MIDI score, locate all loops present across off the tracks
    """
    data = {
        "track_idx": [],
        "instrument_type": [],
        "start": [],
        "end": [],
        "duration_beats": [],
        "note_density": [],
    }
    if file_path is not None:
        data["file_path"] = []
    # Check that there is a time signature. There might be none with abc files
    if len(score.time_signatures) == 0:
        score.time_signatures.append(TimeSignature(0, 4, 4))

    # Extract bars and beats ticks defensively
    try:
        bars_ticks_raw = get_bars_ticks(score)
        beats_ticks_raw = get_beats_ticks(score)
    except Exception:
        print(f"Skipping, couldn't extract bars/beats for {file_path or 'score'} due to malformed events")
        return data

    # sanitize arrays: ensure numeric, finite, reasonable magnitude
    try:
        bars_ticks = np.asarray(bars_ticks_raw, dtype=np.int64)
        bars_ticks = bars_ticks[np.isfinite(bars_ticks)]
        bars_ticks = bars_ticks[(bars_ticks >= 0) & (bars_ticks <= _MAX_TICK)]
    except Exception:
        bars_ticks = np.array([], dtype=np.int64)

    try:
        beats_ticks = np.asarray(beats_ticks_raw, dtype=np.int64)
        beats_ticks = beats_ticks[np.isfinite(beats_ticks)]
        beats_ticks = beats_ticks[(beats_ticks >= 0) & (beats_ticks <= _MAX_TICK)]
    except Exception:
        beats_ticks = np.array([], dtype=np.int64)

    for idx, track in enumerate(score.tracks):
        # Basic track length checks
        try:
            n_notes = len(track.notes)
        except Exception:
            # malformed track object: skip
            continue
        if n_notes > MAX_NOTES_PER_TRACK or n_notes < MIN_NOTES_PER_TRACK:
            continue

        # cut bars_ticks at the end of the track defensively
        try:
            track_end = int(track.end())
        except Exception:
            # malformed end time: skip track
            continue

        if bars_ticks.size and np.any(bars_ticks > track_end):
            try:
                bars_ticks_track = bars_ticks[:np.nonzero(bars_ticks > track_end)[0][0]]
            except Exception:
                bars_ticks_track = bars_ticks
        else:
            bars_ticks_track = bars_ticks

        if beats_ticks.size and np.any(beats_ticks > track_end):
            try:
                beats_ticks_track = beats_ticks[:np.nonzero(beats_ticks > track_end)[0][0]]
            except Exception:
                beats_ticks_track = beats_ticks
        else:
            beats_ticks_track = beats_ticks

        if len(bars_ticks_track) > MAX_NOTES_PER_TRACK:
            # ill-formed bars for this track
            continue   

        # instrument type
        try:
            instrument_type = get_instrument_type(track)
        except Exception:
            instrument_type = "Unknown"

        # Compute note sets with defensive handling of malformed note events
        try:
            note_sets = compute_note_sets(track.notes, bars_ticks_track)
        except Exception:
            # compute_note_sets failed due to malformed notes -> skip track
            continue

        # If compute_note_sets returned nothing and there are no bars, skip
        if not note_sets and (bars_ticks_track.size == 0):
            continue

        # Defensive: ensure note_sets entries look sane
        bad_ns = False
        for ns in note_sets:
            try:
                if ns.start is None or ns.end is None:
                    bad_ns = True
                    break
                if ns.start < 0 or ns.end < 0 or ns.start > _MAX_TICK or ns.end > _MAX_TICK:
                    bad_ns = True
                    break
            except Exception:
                bad_ns = True
                break
        if bad_ns:
            # skip track with malformed note sets
            continue

        # Compute correlation and loops with try/except so a single bad track doesn't crash everything
        try:
            lead_mat = calc_correlation(note_sets)
        except Exception:
            # correlation failed (malformed note_sets) -> skip track
            continue

        try:
            _, loop_endpoints = get_valid_loops(note_sets, lead_mat, beats_ticks_track)
        except Exception:
            # loop detection failed -> skip track
            continue

        for endpoint in loop_endpoints:
            loop_dict = create_loop_dict(endpoint, idx, instrument_type)
            for key in loop_dict.keys():
                data[key].append(loop_dict[key])
            if file_path is not None:
                data["file_path"].append(file_path)

    return data