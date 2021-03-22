################################################################################
# 
# TinyMIDI Processor
#
# Simple, yet fully-featured, Tegridy MIDI<>INT Processor for Music AI purposes
#
# Multi-instrumental, with augmentation functions
#
################################################################################
# 
# Requirements: pip install pretty_midi
#
################################################################################
#
# TinyMIDI MIDI to INT Processor:
#
# Input: MIDI file
# Output: Flat list of byte integers (0-128)
# MIDI Events INTs/Events separator = 255
#
################################################################################
#
# TinyMIDI INT to MIDI Processor:
#
# Input: List of lists of 7 INTs per list, representing a single MIDI event
# Output: pretty_midi.Pretty_MIDI object that can be written to a file like so:
#
# midi.write('MIDI_file.mid')
#
################################################################################
# 
# Source code is courtesy of Rick McCoy of GitHub:
# https://github.com/Rick-McCoy/Reformer-pytorch/tree/master/datasets
#
################################################################################
# 
# Project Los Angele
# Tegridy Code 2021
# License: Apache 2.0
#
################################################################################

import warnings
import pretty_midi as pm
import numpy as np

def TinyMIDI_MIDI_to_INTs_Processor(path, augment=False):
   
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        song = pm.PrettyMIDI(str(path))
    event_list = []
    for inst in song.instruments:
        for note in inst.notes:
            event_list.append((
                int(note.start * 2048),
                (128 if inst.is_drum else inst.program),
                note.pitch,
                note.velocity,
                int(note.end * 2048)
            ))
    event_list.sort()
    input_list = [[129, 128, 128, 128, 128, 128, 128]]
    current_time = 0
    pitch_augment = np.random.randint(-6, 6) if augment else 0
    velocity_augment = np.random.randint(-10, 11) if augment else 0
    time_augment = np.random.rand() + 0.5 if augment else 1
    for event in event_list:
        delta = min(int((event[0] - current_time) * time_augment), 16383)
        dur = min(int((event[4] - event[0]) * time_augment), 16383)
        instrument = event[1]
        pitch = min(max(event[2] + pitch_augment, 0), 127)
        velocity = min(max(event[3] + velocity_augment, 0), 127)
        input_list.append([
            instrument, pitch, velocity,
            dur // 128, dur % 128, delta // 128, delta % 128, 255
        ])
        current_time = event[0]
    
    input_list.append([130, 129, 129, 129, 129, 129, 129])

    return input_list

################################################################################

def TinyMIDI_INTs_to_MIDI_Processor(INTs_list):
    
    midi = pm.PrettyMIDI(resolution=960)
    
    instruments = [pm.Instrument(i) for i in range(128)] \
                + [pm.Instrument(0, is_drum=True)]
    current_time = 0
    
    for event in INTs_list:
        if event[0] == 130 or 129 in event[1:]:
            break
        if event[0] == 129 or 128 in event[1:]:
            continue
        if event[0] == 131 or 130 in event[1:]:
            continue
        instrument = event[0]
        pitch = event[1]
        velocity = event[2]
        dur = event[3] * 128 + event[4]
        delta = event[5] * 128 + event[6]
        instruments[instrument].notes.append(
            pm.Note(
                velocity=velocity,
                pitch=pitch,
                start=(current_time + delta) / 2048,
                end=(current_time + delta + dur) / 2048
            )
        )
        current_time += delta
    for inst in instruments:
        if inst.notes:
            midi.instruments.append(inst)
    
    return midi

################################################################################