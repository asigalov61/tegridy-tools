# tegridy-tools usage examples

***

## Here are the example code snippets for all main TMIDIX module MIDI processing functions

### MIDI input

```
import TMIDIX

#===============================================================================
# Input MIDI file (as filepath or bytes)

input_midi = './tegridy-tools/tegridy-tools/seed2.mid'

#===============================================================================
# Raw single-track ms score

raw_score = TMIDIX.midi2single_track_ms_score(input_midi)

#===============================================================================
# Enhanced score notes

escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]

#===============================================================================
# Augmented enhanced score notes

escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes)

#===============================================================================
# Chordified augmented enhanced score

cscore = TMIDIX.chordify_score([1000, escore_notes])

#===============================================================================
# Monophonic melody score with adjusted ruations

melody = TMIDIX.fix_monophonic_score_durations(TMIDIX.extract_melody(cscore))
```

### MIDI output

```
import copy
import TMIDIX
import midi_to_colab_audio
from IPython.display import Audio, display

#===============================================================================
# Input variables

output_file_name = './melody' # output MIDI file path without extension
output_score = copy.deepcopy(melody) # or escore_notes (any notes list in TMIDIX/MIDI score format)
melody_patch = 40 # Violin

print('=' * 70)
#===============================================================================
# Restoring timings back after using augment_enhanced_score_notes function

for e in output_score:
  e[1] = e[1] * 16 # start times
  e[2] = e[2] * 16 # durations
  e[6] = melody_patch # patches

#===============================================================================
# Creating output score patches list from the enhanced score notes

# This is optional and you can skip this if you do not care about patches
# You can use provided code to detect all composition patches
# Alternatively you can remove the option from SONG converter below for defaults

patches = [-1] * 16 

for e in output_score:
    if e[3] != 9:
        if patches[e[3]] == -1:
            patches[e[3]] = e[6]
        else:
            if patches[e[3]] != e[6]:
                if -1 in patches:
                    patches[patches.index(-1)] = e[6]
                else:
                    patches[-1] = e[6]

patches = [p if p != -1 else 0 for p in patches]

#===============================================================================
# Converting to MIDI
            
detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                          output_signature = 'TMIDIX MIDI Composition',
                                                          output_file_name = output_file_name,
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=patches
                                                          )

#===============================================================================
# Printing resulting MIDI stats

print('=' * 70)
print(detailed_stats)

#===============================================================================
# Rendering MIDI to (raw) audio for listening to and for further processing

print('=' * 70)
print('Converting MIDI to audio...Please wait...')
midi_audio = midi_to_colab_audio.midi_to_colab_audio('./melody.mid')
display(Audio(midi_audio, rate=16000, normalize=False))

#===============================================================================
# Resulting MIDI plot

TMIDIX.plot_ms_SONG(output_score, plot_title=output_file_name+'.mid')
```

***

### Make sure to check out [Jupyter/Google Colab Notebooks](https://github.com/asigalov61/tegridy-tools/tree/main/tegridy-tools/notebooks) dir for many useful and practical examples and applications

***

### Project Los Angeles
### Tegridy Code 2024
