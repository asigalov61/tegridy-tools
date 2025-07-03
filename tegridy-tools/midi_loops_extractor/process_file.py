
from __future__ import annotations
from typing import Dict, Tuple, Any, List
import numpy as np
from miditok.utils import get_bars_ticks, get_beats_ticks
from miditok.constants import CLASS_OF_INST, INSTRUMENT_CLASSES
from symusic import Score, Track, TimeSignature

from corr_mat import calc_correlation, get_valid_loops
from note_set import compute_note_sets

MAX_NOTES_PER_TRACK = 50000
MIN_NOTES_PER_TRACK = 5


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
    
    :param endpoint_data: tuple of loop start time in ticks, loop end time in
        ticks, loop duration in beats, and density in notes per beat
    :param track_idx: MIDI track index the loop belongs to
    :instrument_type: MIDI instrument the loop represents, as a string
    :return: data entry containing all metadata for a single loop
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

    :param file_info: dictionary containing a file_path key
    :return: dictionary of metadata for each identified loop
    """
    file_path = file_info['file_path']
    if isinstance(file_path, list):
        file_path = file_path[0]
    try:
        score = Score(file_path)
    except:
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

    :param score: score to evaluate for loops
    :return: dictionary of metadata for each identified loop
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

    try:
        bars_ticks = np.array(get_bars_ticks(score))
    except ZeroDivisionError:
        print(f"Skipping, couldn't find any bar lines")
        return data

    beats_ticks = np.array(get_beats_ticks(score))
    for idx, track in enumerate(score.tracks):
        if len(track.notes) > MAX_NOTES_PER_TRACK or len(track.notes) < MIN_NOTES_PER_TRACK:
            #print(f"Skipping track {idx} for length")
            continue
        # cut beats_tick at the end of the track
        if any(track_bars_mask := bars_ticks > track.end()):
            bars_ticks_track = bars_ticks[:np.nonzero(track_bars_mask)[0][0]]
        else:
            bars_ticks_track = bars_ticks

        # cut beats_tick at the end of the track
        if any(track_beats_mask := beats_ticks > track.end()):
            beats_ticks_track = beats_ticks[:np.nonzero(track_beats_mask)[0][0]]
        else:
            beats_ticks_track = beats_ticks

        if len(bars_ticks_track) > MAX_NOTES_PER_TRACK:
            print(f"Skipping track {idx} due to ill-formed bars")
            continue   

        instrument_type = get_instrument_type(track)
        note_sets = compute_note_sets(track.notes, bars_ticks_track)
        lead_mat = calc_correlation(note_sets)
        _, loop_endpoints = get_valid_loops(note_sets, lead_mat, beats_ticks_track)
        for endpoint in loop_endpoints:
            loop_dict = create_loop_dict(endpoint, idx, instrument_type)
            for key in loop_dict.keys():
                data[key].append(loop_dict[key])
            if file_path is not None:
                data["file_path"].append(file_path)

    return data
