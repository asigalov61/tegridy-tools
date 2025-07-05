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

__version__ = "25.7.5"

print('=' * 70)
print('MIDI Doctor')
print('Version:', __version__)
print('=' * 70)
print('Loading module...')

###################################################################################

import os

import MIDI

import copy

from collections import OrderedDict

from itertools import groupby, combinations

import hashlib

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

def chordify_notes(notes, timings_divider=1):

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

    if notes:

        cnotes = chordify_notes(notes, timings_divider=timings_divider)
    
        new_notes = []
    
        bp_count = 0
    
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
            return bp_count
            
        else:
            return new_notes

    else:
        return notes

###################################################################################
    
def repair_chords(notes,
                  timings_divider=1,
                  return_bad_chords_count=False
                  ):

    if notes:

        chords = chordify_notes(notes, timings_divider=timings_divider)
    
        bcount = 0
    
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
                                   min_notes_dur=25,
                                   max_notes_dur=500
                                   ):
    if mono_notes:
        
        monophonic_score = copy.deepcopy(mono_notes)

        for n in monophonic_score:
            if n[2] > max_notes_dur:
                n[2] = max_notes_dur

            elif n[2] < min_notes_dur:
                n[2] = min_notes_dur
      
        fixed_score = []

        for i in range(len(monophonic_score)-1):
            note = monophonic_score[i]
        
            nmt = monophonic_score[i+1][1]
        
            if note[1]+note[2] >= nmt:
              note_dur = max(1, nmt-note[1]-min_notes_gap)
            else:
              note_dur = note[2]
        
            new_note = [note[0], note[1], note_dur] + note[3:]
            
            if new_note[2] >= min_notes_dur:
                fixed_score.append(new_note)
          

        fixed_score.append(monophonic_score[-1])

        return fixed_score

    else:
        return mono_notes

###################################################################################

def fix_notes_durations(notes,
                        min_notes_gap=0,
                        min_notes_dur=1,
                        max_notes_dur=500,
                       ):

    if notes:
        non_drums = [e for e in notes if e[3] != 9]
        drums = [e for e in notes if e[3] == 9]
        
        notes_groups = ordered_groups(non_drums)
    
        merged_score = []
    
        for k, g in notes_groups:
            if len(g) > 2:
                fg = fix_monophonic_notes_durations(g, 
                                                    min_notes_gap=min_notes_gap, 
                                                    min_notes_dur=min_notes_dur,
                                                    max_notes_dur=max_notes_dur
                                                   )
                merged_score.extend(fg)
    
            elif len(g) == 2:
                if g[0][2] > max_notes_dur:
                    g[0][2] = max_notes_dur

                if g[1][2] > max_notes_dur:
                    g[1][2] = max_notes_dur

                if g[0][2] > min_notes_dur:
                    g[0][2] = min_notes_dur

                if g[1][2] < min_notes_dur:
                    g[1][2] = min_notes_dur
    
                if g[0][1]+g[0][2] >= g[1][1]:
                    g[0][2] = max(1, g[1][1] - g[0][1] - 1)
                    
                merged_score.extend(g)
    
            else:
                if g[0][2] > max_notes_dur:
                    g[0][2] = max_notes_dur

                if g[0][2] > min_notes_dur:
                    g[0][2] = min_notes_dur
                    
                merged_score.extend(g)
    
        for d in drums:
            if d[2] < min_notes_dur:
                d[2] = min_notes_dur
    
            if d[2] > max_notes_dur:
                d[2] = max_notes_dur           
    
        return sorted(merged_score + drums, key=lambda x: x[1])
    
    else:
        return notes

###################################################################################
    
def adjust_notes_velocities(notes, max_velocity=120, adjustment_threshold_velocity=60):

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
    
            return score

        else:
            return notes
    
    else:
        return notes

###################################################################################

def convert_bytes_in_nested_list(lst, 
                                 encoding='utf-8', 
                                 errors='ignore'
                                ):
    
    new_list = []
    
    for item in lst:
        if isinstance(item, list):
            new_list.append(convert_bytes_in_nested_list(item))
            
        elif isinstance(item, bytes):
            new_list.append(item.decode(encoding, errors=errors))
            
        else:
            new_list.append(item)
            
    return new_list

###################################################################################

def unbyte_text_events(events, encoding='utf-8'):

    fixed_events = []
    
    for e in events:
        if e[0] in text_events:
            ne = convert_bytes_in_nested_list(e, encoding=encoding)
            fixed_events.append(ne)

        else:
            fixed_events.append(e)

    return fixed_events

###################################################################################

def write_midi(ticks, 
               tracks, 
               file_name='healed.mid',
               sort_notes=True,
               return_stats=False,
               return_midi_data=False
              ):

    score = [ticks]
    
    for n, o in tracks:
        if n and sort_notes:
            n.sort(key=lambda x: (x[3] == 9, -x[4]))
        score.append(sorted(o+n, key=lambda x: x[1]))

    midi_data = MIDI.score2midi(score)

    if return_midi_data:
        return midi_data
        
    with open(file_name, 'wb') as fi:
        fi.write(midi_data)
        fi.close

    if return_stats:
        return MIDI.score2stats(score)

###################################################################################
    
def heal_midi(midi_file, 
              output_dir='./healed_midis/',
              return_hashes=False
             ):

    if os.path.exists(midi_file):

        with open(midi_file, 'rb') as fi:
            fdata = fi.read()
            fi.close()

        orig_md5 = hashlib.md5(fdata).hexdigest()
    
        ticks, tracks = read_midi(midi_file)
    
        dd_tracks = []
    
        for n, o in tracks:
            dd_tracks.append([remove_duplicate_notes(n), o])
    
        fd_tracks = []
    
        for n, o in dd_tracks:
            fd_tracks.append([fix_notes_durations(n, max_notes_dur=ticks*8), o])

        va_tracks = []

        for n, o in fd_tracks:
            va_tracks.append([adjust_notes_velocities(n), o])
    
        ut_tracks = []
    
        for n, o in va_tracks:
            ut_tracks.append([n, unbyte_text_events(o)])
    
        fn = os.path.basename(midi_file)
        os.makedirs(output_dir, exist_ok=True)
        fpath = os.path.join(output_dir, fn)
    
        mdata = write_midi(ticks, ut_tracks, return_midi_data=True)

        new_md5 = hashlib.md5(mdata).hexdigest()

        with open(fpath, 'wb') as fi:
            fi.write(mdata)
            fi.close()

        if return_hashes:
            return [orig_md5, new_md5]

    else:
        return []

###################################################################################

print('Module loaded!')
print('=' * 70)
print('Enjoy! :)')
print('=' * 70)

###################################################################################
# This is the end of the TMIDI X Python module
###################################################################################