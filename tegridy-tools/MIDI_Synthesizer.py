#===================================================================================================================
#
# MIDI Synthesizer Python Module
#
# Converts MIDI.py score or opus to audio compatible with Google Colab or HUgging Face Gradio
#
# Version 1.0
#
# Original source code courtesy of SkyTNT
# https://github.com/SkyTNT/midi-model/
#
# Original source code retrieved on 10/20/2023
#
# Project Los Angeles
# Tegridy Code 2023
#
#===================================================================================================================
#
# Critical dependencies
#
# pip install numpy
# sudo apt install fluidsynth
# pip install pyfluidsynth
#
#===================================================================================================================
#
# Example Google colab code:
#
# audio = TMIDIX.MIDI_opus_to_audio(opus, '/usr/share/sounds/sf2/FluidR3_GM.sf2')
# display(Audio(audio, rate=44100, normalize=False))   
#
#===================================================================================================================

def MIDI_opus_to_audio(midi_opus, soundfont_path, sample_rate=44100, volume_scale=20):

    # Such imports are due to TMIDIX legacy support
    fluidsynth = __import__('fluidsynth')
    np = __import__('numpy')

    def normalize_volume(matrix, factor=20):
        norm = np.linalg.norm(matrix)
        matrix = matrix/norm  # normalized matrix
        return matrix * factor

    ticks_per_beat = midi_opus[0]
    event_list = []
    for track_idx, track in enumerate(midi_opus[1:]):
        abs_t = 0
        for event in track:
            abs_t += event[1]
            event_new = [*event]
            event_new[1] = abs_t
            event_list.append(event_new)
    event_list = sorted(event_list, key=lambda e: e[1])

    tempo = int((60 / 120) * 10 ** 6)  # default 120 bpm
    ss = np.empty((0, 2), dtype=np.int16)
    fl = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = fl.sfload(soundfont_path)
    last_t = 0
    for c in range(16):
        fl.program_select(c, sfid, 128 if c == 9 else 0, 0)
    for event in event_list:
        name = event[0]
        sample_len = int(((event[1] / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
        sample_len -= int(((last_t / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
        last_t = event[1]
        if sample_len > 0:
            sample = fl.get_samples(sample_len).reshape(sample_len, 2)
            ss = np.concatenate([ss, sample])
        if name == "set_tempo":
            tempo = event[2]
        elif name == "patch_change":
            c, p = event[2:4]
            fl.program_select(c, sfid, 128 if c == 9 else 0, p)
        elif name == "control_change":
            c, cc, v = event[2:5]
            fl.cc(c, cc, v)
        elif name == "note_on" and event[3] > 0:
            c, p, v = event[2:5]
            fl.noteon(c, p, v)
        elif name == "note_off" or (name == "note_on" and event[3] == 0):
            c, p = event[2:4]
            fl.noteoff(c, p)

    fl.delete()
    if ss.shape[0] > 0:
        max_val = np.abs(ss).max()
        if max_val != 0:
            ss = (ss / max_val) * np.iinfo(np.int16).max
    ss = ss.astype(np.int16)

    ss = ss.swapaxes(1, 0)

    ss = normalize_volume(ss, volume_scale)
    
    return ss
    
#===================================================================================================================

def Tegridy_ms_SONG_to_MIDI_Converter(SONG,
                                      output_signature = 'Tegridy TMIDI Module', 
                                      track_name = 'Composition Track',
                                      list_of_MIDI_patches = [0, 24, 32, 40, 42, 46, 56, 71, 73, 0, 0, 0, 0, 0, 0, 0],
                                      output_file_name = 'TMIDI-Composition',
                                      text_encoding='ISO-8859-1',
                                      verbose=True):

    '''Tegridy milisecond SONG to MIDI Converter
     
    Input: Input ms SONG in TMIDI ms SONG/MIDI.py ms Score format
           Output MIDI Track 0 name / MIDI Signature
           Output MIDI Track 1 name / Composition track name
           List of 16 MIDI patch numbers for output MIDI. Def. is MuseNet compatible patches.
           Output file name w/o .mid extension.
           Optional text encoding if you are working with text_events/lyrics. This is especially useful for Karaoke. Please note that anything but ISO-8859-1 is a non-standard way of encoding text_events according to MIDI specs.

    Output: MIDI File
            Detailed MIDI stats
            MIDI.py score
            MIDI.py opus

    Project Los Angeles
    Tegridy Code 2020'''                                  
    
    if verbose:
        print('Converting to MIDI. Please stand-by...')

    output_header = [1000,
                    [['set_tempo', 0, 1000000],
                     ['time_signature', 0, 4, 2, 24, 8],
                     ['track_name', 0, bytes(output_signature, text_encoding)]]]

    patch_list = [['patch_change', 0, 0, list_of_MIDI_patches[0]], 
                    ['patch_change', 0, 1, list_of_MIDI_patches[1]],
                    ['patch_change', 0, 2, list_of_MIDI_patches[2]],
                    ['patch_change', 0, 3, list_of_MIDI_patches[3]],
                    ['patch_change', 0, 4, list_of_MIDI_patches[4]],
                    ['patch_change', 0, 5, list_of_MIDI_patches[5]],
                    ['patch_change', 0, 6, list_of_MIDI_patches[6]],
                    ['patch_change', 0, 7, list_of_MIDI_patches[7]],
                    ['patch_change', 0, 8, list_of_MIDI_patches[8]],
                    ['patch_change', 0, 9, list_of_MIDI_patches[9]],
                    ['patch_change', 0, 10, list_of_MIDI_patches[10]],
                    ['patch_change', 0, 11, list_of_MIDI_patches[11]],
                    ['patch_change', 0, 12, list_of_MIDI_patches[12]],
                    ['patch_change', 0, 13, list_of_MIDI_patches[13]],
                    ['patch_change', 0, 14, list_of_MIDI_patches[14]],
                    ['patch_change', 0, 15, list_of_MIDI_patches[15]],
                    ['track_name', 0, bytes(track_name, text_encoding)]]

    output = output_header + [patch_list + SONG]

    opus = score2opus(output)

    midi_data = score2midi(output, text_encoding)
    detailed_MIDI_stats = score2stats(output)

    with open(output_file_name + '.mid', 'wb') as midi_file:
        midi_file.write(midi_data)
        midi_file.close()
    
    if verbose:    
        print('Done! Enjoy! :)')
    
    return detailed_MIDI_stats, output, opus

#===================================================================================================================
# Original synthesizer function code is below just in case...
#===================================================================================================================

import fluidsynth
import numpy as np

def synthesis(midi_opus, soundfont_path, sample_rate=44100):
    ticks_per_beat = midi_opus[0]
    event_list = []
    for track_idx, track in enumerate(midi_opus[1:]):
        abs_t = 0
        for event in track:
            abs_t += event[1]
            event_new = [*event]
            event_new[1] = abs_t
            event_list.append(event_new)
    event_list = sorted(event_list, key=lambda e: e[1])

    tempo = int((60 / 120) * 10 ** 6)  # default 120 bpm
    ss = np.empty((0, 2), dtype=np.int16)
    fl = fluidsynth.Synth(samplerate=float(sample_rate))
    sfid = fl.sfload(soundfont_path)
    last_t = 0

    for c in range(16):
        fl.program_select(c, sfid, 128 if c == 9 else 0, 0)

    for event in event_list:
        name = event[0]
        sample_len = int(((event[1] / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
        sample_len -= int(((last_t / ticks_per_beat) * tempo / (10 ** 6)) * sample_rate)
        last_t = event[1]
        if sample_len > 0:
            sample = fl.get_samples(sample_len).reshape(sample_len, 2)
            ss = np.concatenate([ss, sample])
        if name == "set_tempo":
            tempo = event[2]
        elif name == "patch_change":
            c, p = event[2:4]
            fl.program_select(c, sfid, 128 if c == 9 else 0, p)
        elif name == "control_change":
            c, cc, v = event[2:5]
            fl.cc(c, cc, v)
        elif name == "note_on" and event[3] > 0:
            c, p, v = event[2:5]
            fl.noteon(c, p, v)
        elif name == "note_off" or (name == "note_on" and event[3] == 0):
            c, p = event[2:4]
            fl.noteoff(c, p)

    fl.delete()

    if ss.shape[0] > 0:
        max_val = np.abs(ss).max()
        if max_val != 0:
            ss = (ss / max_val) * np.iinfo(np.int16).max
    ss = ss.astype(np.int16)

    return ss

#===================================================================================================================