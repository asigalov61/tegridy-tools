
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from note_set import NoteSet

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Tuple, Dict

    from symusic import Note


# Implementation of Correlative Matrix approach presented in:
# Jia Lien Hsu, Chih Chin Liu, and Arbee L.P. Chen. Discovering
# nontrivial repeating patterns in music data. IEEE Transactions on
# Multimedia, 3:311–325, 9 2001.
def calc_correlation(note_sets: Sequence[NoteSet]) -> np.ndarray:
    """
    Calculates a correlation matrix of repeated segments with the note_sets. 
    All repetitions are required to start on the downbeat of measure. 

    :param note_sets: list of NoteSets to calculate repetitions for
    :return: 2d square correlation matrix the length of note_sets, each 
        entry is an integer representing the number of continuous matching 
        elements counting backwards from the current row and column
    """
    corr_size = len(note_sets)
    corr_mat = np.zeros((corr_size, corr_size), dtype='int16')

    # Complete the first row
    for j in range(1, corr_size):
        if note_sets[0] == note_sets[j] and note_sets[0].is_barline():
            corr_mat[0, j] = 1
    # Complete rest of the correlation matrix
    for i in range(1, corr_size - 1):
        for j in range(i + 1, corr_size):
            if note_sets[i] == note_sets[j]:
                if corr_mat[i - 1, j - 1] == 0:
                    if not note_sets[i].is_barline():
                        continue  # loops must start on the downbeat (start of bar)
                corr_mat[i, j] = corr_mat[i - 1, j - 1] + 1

    return corr_mat


def get_loop_density(loop: Sequence[NoteSet], num_beats: int | float) -> float:
    """
    Calculates the density of a list of NoteSets in active notes per beat

    :param loop: list of NoteSet groups in the loop
    :param num_beats: duration of the loop in beats
    :return: loop density in active notes per beat
    """
    return len([n_set for n_set in loop if n_set.start != n_set.end]) / num_beats


def is_empty_loop(loop: Sequence[Note]) -> bool:
    """
    Checks if a sequence of notes contains at least one non-rest

    :param loop: sequence of MIDI notes to check
    :return: True if a non-rest note exists, False otherwise
    """
    for note in loop:
        if len(note.pitches) > 0:
            return False
    return True


def compare_loops(p1: Sequence[NoteSet], p2: Sequence[NoteSet], min_rep_beats: int | float) -> int:
    """
    Checks if two lists of NoteSets match up to a certain number of beats.
    Used to track the longest common loop

    :param p1: new loop to compare 
    :param p2: existing loop to compare with
    :return: 0 for a mismatch, 1 if p1 is a subloop of p2, 2 if p2 is a 
        subloop of p1
    """
    min_rep_beats = int(round(min_rep_beats))
    if len(p1) < len(p2):
        for i in range(min_rep_beats):
            if p1[i] != p2[i]:
                return 0  #not a subloop, theres a mismatch
        return 1  #is a subloop
    else:
        for i in range(min_rep_beats):
            if p1[i] != p2[i]:
                return 0  #not a subloop, theres a mismatch
        return 2  #existing loop is subloop of the new one, replace it


def test_loop_exists(loop_list: Sequence[Sequence[NoteSet]], loop: Sequence[NoteSet], min_rep_beats: int | float) -> int:
    """
    Checks if a loop already exists in a loop, and mark it for replacement if 
    it is longer than the existing matching loop

    :param loop_list: list of loops to check
    :param loop: new loop to check for a match
    :param min_rep_beats: number of beats to check for a match
    :return: -1 if loop is a subloop of a current loop in loop_list, idx of
        existing loop to replace if loop is a superstring, or None if loop
        is an entirely new loop 
    """
    for i, pat in enumerate(loop_list):
        result = compare_loops(loop, pat, min_rep_beats)
        if result == 1:
            return -1  #ignore this loop since its a subloop
        if result == 2:
            return i  #replace existing loop with this new longer one
    return None  #we're just appending the new loop


def filter_sub_loops(candidate_indices: Dict[float, Tuple[int, int]]) -> Sequence[Tuple[int, int, float]]:
    """
    Processes endpoints for identified loops, keeping only the largest 
    unique loop when multiple loops intersect, thus eliminating "sub loops."
    For instance, if a 4 bar loop is made up of two 2 bar loops, only a 
    single 2 bar loop will be returned. 

    :param candidate_indices: dictionary of (start_tick, end_tick) for each 
        identified group, keyed by loop length in beats
    :return: filtered list of loops with subloops removed
    """
    candidate_indices = dict(sorted(candidate_indices.items()))

    repeats = {}
    final = []
    for duration in candidate_indices.keys():
        curr_start = 0
        curr_end = 0
        curr_dur = 0
        for start, end in candidate_indices[duration]:
            if start in repeats and repeats[start][0] == end:
                continue

            if start == curr_end:
                curr_end = end
                curr_dur += duration
            else:
                if curr_start != curr_end:
                    repeats[curr_start] = (curr_end, curr_dur)
                curr_start = start
                curr_end = end
                curr_dur = duration

            final.append((start, end, duration))

    return final


def get_duration_beats(start: int, end: int, ticks_beats: Sequence[int]) -> float:
    """
    Given a loop start and end time in ticks and a list of beat tick times, 
    calculate the duration of the loop in beats

    :param start: start time of the loop in ticks
    :param end: end time of the loop in ticks
    :param ticks_beat: list of all the beat times in the track
    :return: duration of  the loop in beats
    """
    idx_beat_previous = None
    idx_beat_first_in = None
    idx_beat_last_in = None
    idx_beat_after = None

    for bi, beat_tick in enumerate(ticks_beats):
        if idx_beat_first_in is None and beat_tick >= start:
            idx_beat_first_in = bi
            idx_beat_previous = max(bi - 1, 0)
        elif idx_beat_last_in is None and beat_tick == end:
            idx_beat_last_in = idx_beat_after = bi
        elif idx_beat_last_in is None and beat_tick > end:
            idx_beat_last_in = max(bi - 1, 0)
            idx_beat_after = bi
    if idx_beat_after is None:
        idx_beat_after = idx_beat_last_in + ticks_beats[-1] - ticks_beats[-2]  # TODO what if length 0?

    beat_length_before = ticks_beats[idx_beat_first_in] - ticks_beats[idx_beat_previous]
    if beat_length_before > 0:
        num_beats_before = (ticks_beats[idx_beat_first_in] - ticks_beats[idx_beat_previous]) / beat_length_before
    else:
        num_beats_before = 0
    beat_length_after = ticks_beats[idx_beat_after] - ticks_beats[idx_beat_last_in]
    if beat_length_after > 0:
        num_beats_after = (ticks_beats[idx_beat_after] - ticks_beats[end]) / beat_length_after
    else:
        num_beats_after = 0
    return float(idx_beat_last_in - idx_beat_first_in + num_beats_before + num_beats_after - 1)


def get_valid_loops(
    note_sets: Sequence[NoteSet],
    corr_mat: np.ndarray,
    ticks_beats: Sequence[int],
    min_rep_notes: int=4,
    min_rep_beats: float=2.0,
    min_beats: float=4.0,
    max_beats: float=32.0,
    min_loop_note_density: float = 0.5,
) -> Tuple[Sequence[NoteSet], Tuple[int, int, float, float]]:
    """
    Returns all of the loops detected in note_sets, filtering based on the 
    specified hyperparameters. Loops that are subloops of larger loops will
    be filtered out
    
    :param min_rep_notes: Minimum number of notes that must be present in 
        the repeated bookend of a loop for it to be considered valid
    :param min_rep_beats: Minimum length in beats of the repeated bookend 
        of a loop for it be considered valid
    :param min_beats: Minimum total length of the loop in beats
    :param max_beats: Maximum total length of the loop in beats
    :param min_loop_note_density: Minimum valid density of a loop in average
        notes per beat across the whole loop
    :return: tuple containing the loop as a sequence of NoteSets, and an
        additional tuple with loop metadata: (start time in ticks, end time 
        in ticks, duration in beats, density)
    """
    min_rep_notes += 1  # don't count bar lines as a repetition
    x_num_elem, y_num_elem = np.where(corr_mat == min_rep_notes)

    # Parse the correlation matrix to retrieve the loops starts/ends ticks
    # keys are loops durations in beats, values tuples of indices TODO ??
    valid_indices = {}
    for i, x in enumerate(x_num_elem):
        y = y_num_elem[i]
        start_x = x - corr_mat[x, y] + 1
        start_y = y - corr_mat[x, y] + 1

        loop_start_time = note_sets[start_x].start
        loop_end_time = note_sets[start_y].start
        loop_num_beats = round(get_duration_beats(loop_start_time, loop_end_time, ticks_beats), 2)
        if max_beats >= loop_num_beats >= min_beats:
            if loop_num_beats not in valid_indices:
                valid_indices[loop_num_beats] = []
            valid_indices[loop_num_beats].append((x_num_elem[i], y_num_elem[i]))

    filtered_indices = filter_sub_loops(valid_indices)

    loops = []
    loop_bp = [] 
    corr_size = corr_mat.shape[0]
    for start_x, start_y, loop_num_beats in filtered_indices:
        x = start_x
        y = start_y
        while x + 1 < corr_size and y + 1 < corr_size and corr_mat[x + 1, y + 1] > corr_mat[x, y]:
            x = x + 1
            y = y + 1
        beginning = x - corr_mat[x, y] + 1
        end = y - corr_mat[x, y] + 1
        start_tick = note_sets[beginning].start
        end_tick = note_sets[end].start
        duration_beats = get_duration_beats(start_tick, end_tick, ticks_beats)

        if duration_beats >= min_rep_beats and not is_empty_loop(note_sets[beginning:end]):
            loop = note_sets[beginning:(end + 1)]
            loop_density = get_loop_density(loop, loop_num_beats)
            if loop_density < min_loop_note_density:
                continue
            exist_result = test_loop_exists(loops, loop, min_rep_beats)
            if exist_result is None:
                loops.append(loop)
                loop_bp.append((start_tick, end_tick, loop_num_beats, loop_density))
            elif exist_result > 0:  # index to replace
                loops[exist_result] = loop
                loop_bp[exist_result] = (start_tick, end_tick, loop_num_beats, loop_density)

    return loops, loop_bp
