from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from symusic import Note
    from symusic.core import NoteTickList


class NoteSet:
    __slots__ = ("start", "end", "duration", "pitches")

    def __init__(self, start: int, end: int) -> None:
        s = int(start)
        e = int(end)
        self.start = s
        self.end = e
        self.duration = int(e - s)
        self.pitches = set()

    def add_note(self, note: "Note") -> None:
        try:
            self.pitches.add(int(note.pitch))
        except Exception:
            pass

    def fits_in_set(self, start: int, end: int) -> bool:
        return int(start) == self.start and int(end) == self.end

    def is_barline(self) -> bool:
        return self.start == self.end and len(self.pitches) == 0

    def __str__(self) -> str:
        return f"NoteSet({self.start}, {self.duration}, {self.pitches})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NoteSet):
            return False
        if self.duration != other.duration:
            return False
        if len(self.pitches) != len(other.pitches):
            return False
        return self.pitches == other.pitches

    def __lt__(self, other: object):
        return self.start < other.start


_MAX_DURATION_TICKS = 4_000_000
_MAX_GAP_TICKS = 4_000_000
_MIN_NOTE_DURATION = 0

# New: reject implausible tick values (malformed events often have huge tick numbers)
_MAX_TICK = 10 ** 9  # 1 billion ticks; adjust upward if your dataset legitimately uses larger ticks


def compute_note_sets(notes: NoteTickList, bars_ticks: Sequence[int]) -> list[NoteSet]:
    processed_notes = []
    valid_notes = []
    append_valid = valid_notes.append
    for note in notes:
        try:
            s = int(note.start)
            e = int(note.end)
        except Exception:
            # malformed note event: skip
            continue
        # sanity checks: non-negative, reasonable magnitude
        if s < 0 or e < 0:
            continue
        if s > _MAX_TICK or e > _MAX_TICK:
            # suspiciously large tick values -> skip this note
            continue
        dur = e - s
        if dur < _MIN_NOTE_DURATION:
            continue
        if dur > _MAX_DURATION_TICKS:
            continue
        append_valid(note)

    if not valid_notes and not bars_ticks:
        return []

    try:
        valid_notes.sort(key=lambda n: (int(n.start), int(n.end)))
    except Exception:
        valid_notes = sorted(valid_notes, key=lambda n: (getattr(n, "start", 0) or 0, getattr(n, "end", 0) or 0))

    for note in valid_notes:
        try:
            start = int(note.start)
            end = int(note.end)
        except Exception:
            continue
        start_new_set = len(processed_notes) == 0 or not processed_notes[-1].fits_in_set(start, end)
        if start_new_set:
            processed_notes.append(NoteSet(start=start, end=end))
        processed_notes[-1].add_note(note)

    bars_clean = []
    for b in bars_ticks:
        try:
            bi = int(b)
        except Exception:
            continue
        if bi < 0:
            continue
        if bi > _MAX_TICK:
            # skip implausible bar tick
            continue
        bars_clean.append(bi)
    if bars_clean:
        try:
            bars_clean = sorted(set(bars_clean))
        except Exception:
            bars_clean = sorted(list(dict.fromkeys(bars_clean)))
    else:
        bars_clean = []

    bar_note_sets = [NoteSet(start=db, end=db) for db in bars_clean]
    all_sets = processed_notes + bar_note_sets
    try:
        all_sets.sort()
    except Exception:
        all_sets = sorted(all_sets, key=lambda ns: getattr(ns, "start", 0) or 0)

    final_sets = []
    prev_start = None
    for ns in all_sets:
        if ns.start is None or ns.end is None:
            continue
        if ns.start < 0 or ns.duration < 0:
            continue
        if prev_start is not None and (ns.start - prev_start) > _MAX_GAP_TICKS:
            # gap too large; keep but don't crash (original code had a pass)
            pass
        final_sets.append(ns)
        prev_start = ns.start

    return final_sets