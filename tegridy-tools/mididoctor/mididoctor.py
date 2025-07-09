#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#	MIDI Doctor Python module
#
#	Project Los Angeles
#
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
###################################################################################
'''

###################################################################################

__version__ = "25.7.20"

print('=' * 70)
print('MIDI Doctor')
print('Version:', __version__)
print('=' * 70)
print('Loading module...')

###################################################################################

import os

import copy

from collections import OrderedDict, Counter

from itertools import groupby, combinations

import hashlib

from . import MIDI

###################################################################################

ALL_CHORDS_SORTED = [[0], [0, 2], [0, 3], [0, 4], [0, 2, 4], [0, 5], [0, 2, 5], [0, 3, 5], [0, 6],
                    [0, 2, 6], [0, 3, 6], [0, 4, 6], [0, 2, 4, 6], [0, 7], [0, 2, 7], [0, 3, 7],
                    [0, 4, 7], [0, 5, 7], [0, 2, 4, 7], [0, 2, 5, 7], [0, 3, 5, 7], [0, 8],
                    [0, 2, 8], [0, 3, 8], [0, 4, 8], [0, 5, 8], [0, 6, 8], [0, 2, 4, 8],
                    [0, 2, 5, 8], [0, 2, 6, 8], [0, 3, 5, 8], [0, 3, 6, 8], [0, 4, 6, 8],
                    [0, 2, 4, 6, 8], [0, 9], [0, 2, 9], [0, 3, 9], [0, 4, 9], [0, 5, 9], [0, 6, 9],
                    [0, 7, 9], [0, 2, 4, 9], [0, 2, 5, 9], [0, 2, 6, 9], [0, 2, 7, 9],
                    [0, 3, 5, 9], [0, 3, 6, 9], [0, 3, 7, 9], [0, 4, 6, 9], [0, 4, 7, 9],
                    [0, 5, 7, 9], [0, 2, 4, 6, 9], [0, 2, 4, 7, 9], [0, 2, 5, 7, 9],
                    [0, 3, 5, 7, 9], [0, 10], [0, 2, 10], [0, 3, 10], [0, 4, 10], [0, 5, 10],
                    [0, 6, 10], [0, 7, 10], [0, 8, 10], [0, 2, 4, 10], [0, 2, 5, 10],
                    [0, 2, 6, 10], [0, 2, 7, 10], [0, 2, 8, 10], [0, 3, 5, 10], [0, 3, 6, 10],
                    [0, 3, 7, 10], [0, 3, 8, 10], [0, 4, 6, 10], [0, 4, 7, 10], [0, 4, 8, 10],
                    [0, 5, 7, 10], [0, 5, 8, 10], [0, 6, 8, 10], [0, 2, 4, 6, 10],
                    [0, 2, 4, 7, 10], [0, 2, 4, 8, 10], [0, 2, 5, 7, 10], [0, 2, 5, 8, 10],
                    [0, 2, 6, 8, 10], [0, 3, 5, 7, 10], [0, 3, 5, 8, 10], [0, 3, 6, 8, 10],
                    [0, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10], [1], [1, 3], [1, 4], [1, 5], [1, 3, 5],
                    [1, 6], [1, 3, 6], [1, 4, 6], [1, 7], [1, 3, 7], [1, 4, 7], [1, 5, 7],
                    [1, 3, 5, 7], [1, 8], [1, 3, 8], [1, 4, 8], [1, 5, 8], [1, 6, 8], [1, 3, 5, 8],
                    [1, 3, 6, 8], [1, 4, 6, 8], [1, 9], [1, 3, 9], [1, 4, 9], [1, 5, 9], [1, 6, 9],
                    [1, 7, 9], [1, 3, 5, 9], [1, 3, 6, 9], [1, 3, 7, 9], [1, 4, 6, 9],
                    [1, 4, 7, 9], [1, 5, 7, 9], [1, 3, 5, 7, 9], [1, 10], [1, 3, 10], [1, 4, 10],
                    [1, 5, 10], [1, 6, 10], [1, 7, 10], [1, 8, 10], [1, 3, 5, 10], [1, 3, 6, 10],
                    [1, 3, 7, 10], [1, 3, 8, 10], [1, 4, 6, 10], [1, 4, 7, 10], [1, 4, 8, 10],
                    [1, 5, 7, 10], [1, 5, 8, 10], [1, 6, 8, 10], [1, 3, 5, 7, 10],
                    [1, 3, 5, 8, 10], [1, 3, 6, 8, 10], [1, 4, 6, 8, 10], [1, 11], [1, 3, 11],
                    [1, 4, 11], [1, 5, 11], [1, 6, 11], [1, 7, 11], [1, 8, 11], [1, 9, 11],
                    [1, 3, 5, 11], [1, 3, 6, 11], [1, 3, 7, 11], [1, 3, 8, 11], [1, 3, 9, 11],
                    [1, 4, 6, 11], [1, 4, 7, 11], [1, 4, 8, 11], [1, 4, 9, 11], [1, 5, 7, 11],
                    [1, 5, 8, 11], [1, 5, 9, 11], [1, 6, 8, 11], [1, 6, 9, 11], [1, 7, 9, 11],
                    [1, 3, 5, 7, 11], [1, 3, 5, 8, 11], [1, 3, 5, 9, 11], [1, 3, 6, 8, 11],
                    [1, 3, 6, 9, 11], [1, 3, 7, 9, 11], [1, 4, 6, 8, 11], [1, 4, 6, 9, 11],
                    [1, 4, 7, 9, 11], [1, 5, 7, 9, 11], [1, 3, 5, 7, 9, 11], [2], [2, 4], [2, 5],
                    [2, 6], [2, 4, 6], [2, 7], [2, 4, 7], [2, 5, 7], [2, 8], [2, 4, 8], [2, 5, 8],
                    [2, 6, 8], [2, 4, 6, 8], [2, 9], [2, 4, 9], [2, 5, 9], [2, 6, 9], [2, 7, 9],
                    [2, 4, 6, 9], [2, 4, 7, 9], [2, 5, 7, 9], [2, 10], [2, 4, 10], [2, 5, 10],
                    [2, 6, 10], [2, 7, 10], [2, 8, 10], [2, 4, 6, 10], [2, 4, 7, 10],
                    [2, 4, 8, 10], [2, 5, 7, 10], [2, 5, 8, 10], [2, 6, 8, 10], [2, 4, 6, 8, 10],
                    [2, 11], [2, 4, 11], [2, 5, 11], [2, 6, 11], [2, 7, 11], [2, 8, 11],
                    [2, 9, 11], [2, 4, 6, 11], [2, 4, 7, 11], [2, 4, 8, 11], [2, 4, 9, 11],
                    [2, 5, 7, 11], [2, 5, 8, 11], [2, 5, 9, 11], [2, 6, 8, 11], [2, 6, 9, 11],
                    [2, 7, 9, 11], [2, 4, 6, 8, 11], [2, 4, 6, 9, 11], [2, 4, 7, 9, 11],
                    [2, 5, 7, 9, 11], [3], [3, 5], [3, 6], [3, 7], [3, 5, 7], [3, 8], [3, 5, 8],
                    [3, 6, 8], [3, 9], [3, 5, 9], [3, 6, 9], [3, 7, 9], [3, 5, 7, 9], [3, 10],
                    [3, 5, 10], [3, 6, 10], [3, 7, 10], [3, 8, 10], [3, 5, 7, 10], [3, 5, 8, 10],
                    [3, 6, 8, 10], [3, 11], [3, 5, 11], [3, 6, 11], [3, 7, 11], [3, 8, 11],
                    [3, 9, 11], [3, 5, 7, 11], [3, 5, 8, 11], [3, 5, 9, 11], [3, 6, 8, 11],
                    [3, 6, 9, 11], [3, 7, 9, 11], [3, 5, 7, 9, 11], [4], [4, 6], [4, 7], [4, 8],
                    [4, 6, 8], [4, 9], [4, 6, 9], [4, 7, 9], [4, 10], [4, 6, 10], [4, 7, 10],
                    [4, 8, 10], [4, 6, 8, 10], [4, 11], [4, 6, 11], [4, 7, 11], [4, 8, 11],
                    [4, 9, 11], [4, 6, 8, 11], [4, 6, 9, 11], [4, 7, 9, 11], [5], [5, 7], [5, 8],
                    [5, 9], [5, 7, 9], [5, 10], [5, 7, 10], [5, 8, 10], [5, 11], [5, 7, 11],
                    [5, 8, 11], [5, 9, 11], [5, 7, 9, 11], [6], [6, 8], [6, 9], [6, 10],
                    [6, 8, 10], [6, 11], [6, 8, 11], [6, 9, 11], [7], [7, 9], [7, 10], [7, 11],
                    [7, 9, 11], [8], [8, 10], [8, 11], [9], [9, 11], [10], [11]]

###################################################################################

text_events = ['text_event',
               'copyright_text_event', 
               'track_name',
               'instrument_name',
               'lyric',
               'marker',
               'cue_point',
               'text_event_08',
               'text_event_09',
               'text_event_0a',
               'text_event_0b',
               'text_event_0c',
               'text_event_0d',
               'text_event_0e',
               'text_event_0f'
              ]

###################################################################################

_TEXT_ENCODINGS = [
    'iso-8859-1',   # Western Europe
    'windows-1251', # Cyrillic (Russian)
    'koi8-r',       # Cyrillic (Russian)
    'iso-8859-5',   # Cyrillic
    'cp866',        # DOS Cyrillic

    'shift_jis',    # Japanese
    'euc_jp',       # Japanese
    'gb18030',      # Simplified Chinese
    'big5',         # Traditional Chinese
    'euc_kr',       # Korean

    'utf-8',        # Universal fallback
    'utf-16',       # Wide-char fallback
]

###################################################################################

def read_midi(midi_file,
              ignore_bad_signature=True,
              ignore_bad_header=False
             ):

    score = MIDI.midi2score(open(midi_file, 'rb').read(),
                            ignore_bad_signature=True,
                            ignore_bad_header=False
                           )

    ticks = score[0]
    
    tracks = []
    
    itrack = 1
    while itrack < len(score):
        
        notes = []
        other = []
        
        for event in score[itrack]:
            if event[0] == 'note':
                
                event[3] %= 16
                event[4] %= 128
                event[5] %= 128
                
                notes.append(event)
    
            else:
                if event[0] == 'patch_change':
                    event[2] %= 16
                    event[3] %= 128
                    
                other.append(event)

        if notes or other:
            tracks.append([notes, other])
    
        itrack += 1

    return [ticks, tracks]

###################################################################################

def chordify_notes(notes, 
                   timings_divider=1
                  ):

    pe = notes[0]

    chords = []
    cho = []
    
    for e in notes:
        if int(e[1] / timings_divider)-int(pe[1] / timings_divider) == 0:
            cho.append(e)

        else:
            if cho:
                chords.append(cho)

            cho = [e]

        pe = e
            
    if cho:
        chords.append(cho)

    return chords

###################################################################################

def check_and_fix_tones_chord(tones_chord):

  tones_chord_combs = [list(comb) for i in range(len(tones_chord), 0, -1) for comb in combinations(tones_chord, i)]

  for c in tones_chord_combs:
    if c in ALL_CHORDS_SORTED:
      checked_tones_chord = c
      break

  return sorted(checked_tones_chord)

###################################################################################

def remove_duplicate_notes(notes, 
                           timings_divider=1, 
                           return_dupes_count=False
                          ):

    bp_count = 0

    if notes:

        cnotes = chordify_notes(notes, timings_divider=timings_divider)
    
        new_notes = []

        for c in cnotes:
            
            cho = []
            seen = []
    
            for cc in c:
                if [cc[3], cc[4]] not in seen:
                    cho.append(cc)
                    seen.append([cc[3], cc[4]])
    
                else:
                    bp_count += 1
    
            new_notes.extend(cho)
            
        if return_dupes_count:
            return new_notes, bp_count
            
        else:
            return new_notes

    else:
        if return_dupes_count:
            return notes, bp_count

        else:
            return notes

###################################################################################
    
def repair_chords(notes, 
                  timings_divider=1, 
                  return_bad_chords_count=False
                 ):

    bcount = 0

    if notes:

        chords = chordify_notes(notes, timings_divider=timings_divider)

        fixed_chords = []
    
        for c in chords:
            c.sort(key=lambda x: x[3])
    
            if len(c) > 1:
    
                groups = groupby(c, key=lambda x: x[3])
        
                for cha, gr in groups:
    
                    gr = list(gr)
                    
                    tones_chord = sorted(set([p[4] % 12 for p in gr]))
        
                    if tones_chord not in ALL_CHORDS_SORTED:
                        tones_chord = check_and_fix_tones_chord(tones_chord)
                        bcount += 1
        
                    ngr = []
                    
                    for n in gr:
                        if n[4] % 12 in tones_chord:
                            ngr.append(n)
        
                    fixed_chords.extend(ngr)
                        
    
            else:
                fixed_chords.extend(c)
                
        fixed_chords.sort(key=lambda x: (x[1], -x[4]))

        if return_bad_chords_count:
            return fixed_chords, bcount

        else:
            return fixed_chords
            
    else:
        if return_bad_chords_count:
            return notes, bcount

        else:
            return notes
    
###################################################################################
    
def ordered_groups(data):
    
    groups = OrderedDict()
    
    for sublist in data:
        key = tuple([sublist[3], sublist[4]])
        
        if key not in groups:
            groups[key] = []
            
        groups[key].append(sublist)
    
    return list(groups.items())

###################################################################################

def fix_monophonic_notes_durations(mono_notes,
                                   min_notes_gap=0,
                                   min_notes_dur=20,
                                   max_notes_dur=500,
                                   return_bad_durs_count=False
                                   ):

    monophonic_score = copy.deepcopy(mono_notes)

    for n in monophonic_score:
        if n[2] > max_notes_dur:
            n[2] = max_notes_dur

        elif n[2] < min_notes_dur:
            n[2] = min_notes_dur
  
    fixed_score = []

    bd_count = 0

    for i in range(len(monophonic_score)-1):
        note = monophonic_score[i]
    
        nmt = monophonic_score[i+1][1]
    
        if note[1]+note[2] >= nmt:
          note_dur = max(1, nmt-note[1]-min_notes_gap)
          bd_count += 1
            
        else:
          note_dur = note[2]
    
        new_note = [note[0], note[1], note_dur] + note[3:]
        
        if new_note[2] >= min_notes_dur:
            fixed_score.append(new_note)

        else:
            bd_count += 1
      

    fixed_score.append(monophonic_score[-1])

    if return_bad_durs_count:
        return fixed_score, bd_count

    else:
        return fixed_score

###################################################################################

def fix_notes_durations(notes,
                        min_notes_gap=0,
                        min_notes_dur=1,
                        max_notes_dur=500,
                        return_bad_durs_count=False
                       ):
    
    bd_count = 0
    
    if notes:
        
        non_drums = [e for e in notes if e[3] != 9]
        drums = [e for e in notes if e[3] == 9]
        
        notes_groups = ordered_groups(non_drums)
    
        merged_score = []
    
        for k, g in notes_groups:
            if len(g) > 2:
                fg, bc = fix_monophonic_notes_durations(g, 
                                                        min_notes_gap=min_notes_gap, 
                                                        min_notes_dur=min_notes_dur,
                                                        max_notes_dur=max_notes_dur,
                                                        return_bad_durs_count=True
                                                       )
                merged_score.extend(fg)
                bd_count += bc
    
            elif len(g) == 2:
                bd1 = False
                bd2 = False
                
                if g[0][2] > max_notes_dur:
                    g[0][2] = max_notes_dur
                    bd1 = True

                if g[1][2] > max_notes_dur:
                    g[1][2] = max_notes_dur
                    bd2 = True

                if g[0][2] < min_notes_dur:
                    g[0][2] = min_notes_dur
                    bd1 = True

                if g[1][2] < min_notes_dur:
                    g[1][2] = min_notes_dur
                    bd2 = True
    
                if g[0][1]+g[0][2] >= g[1][1]:
                    g[0][2] = max(1, g[1][1] - g[0][1] - min_notes_gap)
                    bd1 = True
                    
                merged_score.extend(g)
                bd_count += sum([bd1, bd2])
    
            elif len(g) == 1:
                bd1 = 0
                
                if g[0][2] > max_notes_dur:
                    g[0][2] = max_notes_dur
                    bd1 = 1

                if g[0][2] < min_notes_dur:
                    g[0][2] = min_notes_dur
                    bd1 = 1
                    
                merged_score.extend(g)
                bd_count += bd1
    
        for d in drums:
            bd1 = 0
            
            if d[2] < min_notes_dur:
                d[2] = min_notes_dur
                bd1 = 1
    
            if d[2] > max_notes_dur:
                d[2] = max_notes_dur
                bd1 = 1

            bd_count += bd1

        if return_bad_durs_count:
            return sorted(merged_score + drums, key=lambda x: (x[1], -x[4])), bd_count

        else:
            return sorted(merged_score + drums, key=lambda x: (x[1], -x[4]))    
    
    else:
        if return_bad_durs_count:
            return notes, bd_count

        else:
            return notes
        
###################################################################################
    
def adjust_notes_velocities(notes, 
                            max_velocity=124, 
                            adjustment_threshold_velocity=56, 
                            return_adj_vels_count=False
                           ):

    av_count = 0
    
    if notes:

        vels = [e[5] for e in notes]
        avg_vel = sum(vels) / len(vels)
    
        if avg_vel < adjustment_threshold_velocity:

            score = copy.deepcopy(notes)
    
            min_velocity = min([c[5] for c in score])
            max_velocity_all_channels = max([c[5] for c in score])
            min_velocity_ratio = min_velocity / max_velocity_all_channels
        
            max_channel_velocity = max([c[5] for c in score])
            
            if max_channel_velocity < min_velocity:
                factor = max_velocity / min_velocity
            else:
                factor = max_velocity / max_channel_velocity
                
            for i in range(len(score)):
                score[i][5] = int(score[i][5] * factor)
                av_count += 1

            if return_adj_vels_count:
                return score, av_count

            else:
                return score

        if return_adj_vels_count:
            return notes, av_count

        else:
            return notes

    else:
        if return_adj_vels_count:
            return notes, av_count

        else:
            return notes

###################################################################################
        
def repair_flat_dynamics(notes, 
                         adjustment_threshold=0.95, 
                         return_adj_vels_count=False
                        ):

    notes.sort(key=lambda x: x[3])
    chan_groups = groupby(notes, key=lambda x: x[3])

    new_notes = []
    av_count = 0

    for chan, chan_group in chan_groups:

        chan_group = copy.deepcopy(list(chan_group))

        chan_ptcs = [e[4] for e in chan_group]

        chan_vels = [e[5] for e in chan_group]

        chan_vels_counts = Counter(chan_vels).most_common()

        if chan_vels_counts[0][1] / len(chan_vels) > adjustment_threshold:
            avg_chan_ptc = round(sum(chan_ptcs) / len(chan_ptcs))

            vel_shift = avg_chan_ptc // 4

            if chan == 9:
                vel_shift = 70

            else:
                if avg_chan_ptc < 48:
                    vel_shift = 48

            for e in chan_group:
                e[5] = max(36, min(124, e[4] + vel_shift))
                new_notes.append(e)
                av_count += 1

        else:
            new_notes.extend(chan_group)

    new_notes = sorted(new_notes, key=lambda x: (x[1], -x[4]))
    
    if return_adj_vels_count:
        return new_notes, av_count

    else:
        return new_notes
    
###################################################################################

def convert_bytes_in_nested_list(lst,
                                 encoding=None,
                                 errors='ignore',
                                 return_changed_events_count=False
                                ):

    new_list = []
    ce_count = 0

    for item in lst:
        if isinstance(item, list):
            sub, sub_count = convert_bytes_in_nested_list(item, 
                                                          encoding, 
                                                          errors, 
                                                          True
                                                         )
            new_list.append(sub)
            ce_count += sub_count

        elif isinstance(item, bytes):
            b = item

            if encoding is None:
                chosen = None
                fallback = None

                for enc in _TEXT_ENCODINGS:
                    try:
                        s = b.decode(enc)
                        
                    except (UnicodeDecodeError, LookupError):
                        continue

                    if s.encode(enc, errors=errors) == b:
                        chosen = s
                        break

                    if fallback is None:
                        fallback = s

                if chosen is not None:
                    decoded = chosen
                    
                elif fallback is not None:
                    decoded = fallback
                    
                else:
                    decoded = b.decode('iso-8859-1', errors)

            else:
                decoded = b.decode(encoding, errors)

            new_list.append(decoded)
            ce_count += 1

        else:
            new_list.append(item)

    if return_changed_events_count:
        return new_list, ce_count
        
    return new_list

###################################################################################

def unbyte_text_events(events,
                       encoding=None,
                       return_unbyte_count=False
                      ):

    fixed_events = []

    uc_count = 0
    
    for e in events:
        if e[0] in text_events:
            ne, cc = convert_bytes_in_nested_list(e, 
                                                  encoding=encoding,
                                                  return_changed_events_count=True
                                                 )
            fixed_events.append(ne)
            uc_count += cc

        else:
            fixed_events.append(e)
    
    if return_unbyte_count:
        return fixed_events, uc_count

    else:
        return fixed_events

###################################################################################

def write_midi(ticks, 
               tracks, 
               file_name='healed.mid',
               sort_notes=True,
               write_midi=True,
               return_midi_data=False,
               return_score=False,
               return_stats=False,
               return_md5_hash=False,
               force_utf8=False
              ):

    score = [ticks]
    
    for n, o in tracks:
        if n and sort_notes:
            n.sort(key=lambda x: (x[3] == 9, -x[4]))
        score.append(sorted(o+n, key=lambda x: x[1]))

    if write_midi:
        midi_data = MIDI.score2midi(score, 
                                    force_utf8=force_utf8
                                   )
        
        with open(file_name, 'wb') as fi:
            fi.write(midi_data)
            fi.close

    out_data = {}

    if return_midi_data or return_md5_hash:
        midi_data = MIDI.score2midi(score, force_utf8=force_utf8)

    if return_midi_data:
        out_data.update({'midi_data': midi_data})
        
    if return_stats:
        out_data.update({'midi_stats': MIDI.score2stats(score)})

    if return_score:
        out_data.update({'midi_score': score})

    if return_md5_hash:
        out_data.update({'midi_md5_hash': hashlib.md5(mdata).hexdigest()})

    return out_data

###################################################################################
    
def heal_midi(midi_file, 
              output_dir='./healed_midis/',
              timings_divider=1,
              max_notes_dur=-1,
              text_events_encoding=None,
              force_utf8=False,
              quiet_vels_adj_threshold=56,
              flat_vels_adj_threshold=0.95,
              write_midi_to_file=True,
              return_midi_data=False,
              return_midi_score=False,
              return_midi_hashes=False,
              return_original_midi_stats=False,
              return_healed_midi_stats=False,
              return_repair_stats=False
             ):

    if os.path.exists(midi_file):

        #===================================================================================

        with open(midi_file, 'rb') as fi:
            fdata = fi.read()
            fi.close()

        ticks, tracks = read_midi(midi_file)

        if return_original_midi_stats:
            oscore = [ticks]

            for n, o in tracks:
                oscore.extend([sorted(o+n, key=lambda x: x[1])])
            ostats = MIDI.score2stats(oscore)

        #===================================================================================

        notes_and_other_events_counts = {}

        for i, (n, o) in enumerate(tracks):
            notes_and_other_events_counts[i] = {'num_notes': len(n),
                                                'num_other': len(o)
                                               }
            

        #===================================================================================
    
        dd_tracks = []
        dd_counts = {}
    
        for i, (n, o) in enumerate(tracks):
            out, dd_count = remove_duplicate_notes(n, 
                                                   timings_divider=timings_divider, 
                                                   return_dupes_count=True
                                                  )
            dd_tracks.append([out, o])
            dd_counts[i] = {'duplicate': dd_count, 
                            'total':len(n)
                           }

        #===================================================================================

        cd_tracks = []
        cd_counts = {}

        for i, (n, o) in enumerate(dd_tracks):
            out, cd_count = repair_chords(n, 
                                          timings_divider=timings_divider, 
                                          return_bad_chords_count=True
                                         ) 
            cd_tracks.append([out, o])
            cd_counts[i] = {'bad': cd_count, 
                            'total': len(n)
                           }

        #===================================================================================
    
        fd_tracks = []
        fd_counts = {}

        if max_notes_dur < 1:
            mndur = ticks*8

        else:
            mndur = max_notes_dur

        for i, (n, o) in enumerate(cd_tracks):
            out, fd_count = fix_notes_durations(n, 
                                                max_notes_dur=mndur,
                                                return_bad_durs_count=True
                                               )
            fd_tracks.append([out, o])
            fd_counts[i] = {'adjusted': fd_count, 
                            'total': len(n)
                           }

        #===================================================================================

        va_tracks = []
        va_counts = {}

        for i, (n, o) in enumerate(fd_tracks):
            out, va_count = adjust_notes_velocities(n,
                                                    adjustment_threshold_velocity=quiet_vels_adj_threshold,
                                                    return_adj_vels_count=True
                                                   )
            va_tracks.append([out, o])
            va_counts[i] = {'adjusted': va_count, 
                            'total': len(n)
                           }

        #===================================================================================

        fv_tracks = []
        fv_counts = {}

        for i, (n, o) in enumerate(va_tracks):
            out, fv_count = repair_flat_dynamics(n,
                                                 adjustment_threshold=flat_vels_adj_threshold,
                                                 return_adj_vels_count=True
                                                )
            fv_tracks.append([out, o])
            fv_counts[i] = {'adjusted': fv_count, 
                            'total': len(n)
                           }
        
        #===================================================================================
    
        ut_tracks = []
        ut_counts = {}
    
        for i, (n, o) in enumerate(fv_tracks):
            out, ut_count = unbyte_text_events(o, 
                                               return_unbyte_count=True,
                                               encoding=text_events_encoding
                                              )
            ut_tracks.append([n, out])
            ut_counts[i] = {'adjusted': ut_count, 
                            'total': len(o)
                           }
            
        #===================================================================================

        repair_counts_dict = {}

        repair_counts_dict['duplicate_pitches_counts_per_track'] = dd_counts
        repair_counts_dict['bad_chords_counts_per_track'] = cd_counts
        repair_counts_dict['adjusted_durations_counts_per_track'] = fd_counts
        repair_counts_dict['adjusted_quiet_velocities_counts_per_track'] = va_counts
        repair_counts_dict['adjusted_flat_velocities_counts_per_track'] = fv_counts
        repair_counts_dict['adjusted_text_events_counts_per_track'] = ut_counts

        #===================================================================================

        if write_midi or return_midi_data or return_midi_hashes or return_midi_stats:
            midi_dict = write_midi(ticks, 
                                   ut_tracks, 
                                   write_midi=False, 
                                   return_midi_data=True, 
                                   return_stats=True, 
                                   return_score=True,
                                   force_utf8=force_utf8
                                  )

            mdata = midi_dict['midi_data']
            mscore = midi_dict['midi_score']
            hstats = midi_dict['midi_stats']

        #===================================================================================

        out_data = {'original_midi_file_name': os.path.basename(midi_file)}

        #===================================================================================

        if write_midi:
    
            fn = os.path.basename(midi_file)
            os.makedirs(output_dir, exist_ok=True)
            fpath = os.path.join(output_dir, fn)

            with open(fpath, 'wb') as fi:
                fi.write(mdata)
                fi.close()

        #===================================================================================

        if return_midi_data:
            out_data['raw_midi_data'] = mdata

        #===================================================================================

        if return_midi_score:
           out_data['healed_midi_score'] = mscore

        #===================================================================================

        if return_midi_hashes:
            orig_md5 = hashlib.md5(fdata).hexdigest()
            new_md5 = hashlib.md5(mdata).hexdigest()
            out_data.update({'original_md5_hash': orig_md5, 'healed_md5_hash': new_md5})

        #===================================================================================

        if return_original_midi_stats:
            out_data.update({'original_midi_stats': ostats})

        #===================================================================================

        if return_healed_midi_stats:
            out_data.update({'healed_midi_stats': hstats})

        #===================================================================================

        if return_repair_stats:
            repair_stats_dict = {}

            repair_stats_dict['file_name'] = os.path.basename(midi_file)
            repair_stats_dict['num_ticks'] = ticks
            repair_stats_dict['num_tracks'] = len(tracks)
            repair_stats_dict['num_notes_and_other_events_per_track'] = notes_and_other_events_counts
            repair_stats_dict['repair_stats'] = repair_counts_dict

            out_data.update(repair_stats_dict)

        return out_data

    #=======================================================================================

    else:
        return {}
    
###################################################################################

print('Module loaded!')
print('=' * 70)
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the TMIDI X Python module
###################################################################################