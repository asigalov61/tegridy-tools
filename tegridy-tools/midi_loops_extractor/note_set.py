
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from symusic import Note
    from symusic.core import NoteTickList


def compute_note_sets(notes: NoteTickList, bars_ticks: Sequence[int]) -> list[NoteSet]:
    """
    Converts a list of MIDI notes and associated measure start times into 
    a list of NoteSets. Barlines will be represented as empty NoteSets 
    with a duration of 0

    :param notes: list of MIDI notes in a single track
    :param bar ticks: list of measure start times in ticks
    :return: NoteSet representation of the MIDI track
    """
    processed_notes = []
    for note in notes:
        start_new_set = len(processed_notes) == 0 or not processed_notes[-1].fits_in_set(note.start, note.end)
        if start_new_set:
            processed_notes.append(NoteSet(start=note.start, end=note.end))
        processed_notes[-1].add_note(note)

    notes = processed_notes + [NoteSet(start=db, end=db) for db in bars_ticks]
    notes.sort()
    return notes


class NoteSet:
    """
    A set of unique pitches that occur at the same start time and end at 
    the same time (have the same duration).

    If a NoteSet has no pitches are a duration of 0, it represents the 
    start of a measure (ie a barline)

    :param start: start time in MIDI ticks
    :param end: end time in MIDI ticks
    """
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end
        self.duration = self.end - self.start
        self.pitches = set() # MIDI note numbers

    def add_note(self, note: Note) -> None:
        self.pitches.add(note.pitch)

    def fits_in_set(self, start: int, end: int) -> bool:
        return start == self.start and end == self.end

    def is_barline(self) -> bool:
        return self.start == self.end and len(self.pitches) == 0

    def __str__(self) -> str:
        return f"NoteSet({self.start}, {self.duration}, {self.pitches})"

    def __eq__(self, other: object) -> bool:
        """
        Two NoteSets are equal if they match in start time, end time and 
        MIDI pitches present
        """
        if not isinstance(other, NoteSet):
            return False

        if self.duration != other.duration:
            return False
        if len(self.pitches) != len(other.pitches):
            return False

        for m in self.pitches:
            if m not in other.pitches:
                return False

        return True

    def __lt__(self, other: object):
        """
        A NoteSet is sorted based on start time
        """
        return self.start < other.start
