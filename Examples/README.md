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
# Monophonic melody score with adjusted durations and custom patch (40 / Violin)

melody = TMIDIX.fix_monophonic_score_durations(TMIDIX.extract_melody(cscore, melody_patch=40))
```

### MIDI output

```
import copy
import TMIDIX
import midi_to_colab_audio
from IPython.display import display, Audio

#===============================================================================
# Input variables

output_file_name = './melody' # output MIDI file path without extension
output_score = copy.deepcopy(melody) # or escore_notes (any notes list in TMIDIX/MIDI score format)

print('=' * 70)

#===============================================================================
# Creating output score patches list from the enhanced score notes

# This is optional and you can skip this if you do not care about patches
# Alternatively You can use TMIDIX functions to detect all composition patches

# This function will create a single track patch list (16 patches)
# patches = TMIDIX.patch_list_from_enhanced_score_notes(output_score)

# And this functions will create a full patch list (any number of patches)
# It will also patch the score if there are more than 16 patches
# This function is preferred and recommended

output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(output_score)

#===============================================================================
# Converting to MIDI

detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                          output_signature = 'TMIDIX MIDI Composition',
                                                          output_file_name = output_file_name,
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=patches,
                                                          timings_multiplier=16 # Restoring augmented timings
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

### Advanced MIDI processing

```
import TMIDIX

#===============================================================================
# Input MIDI file (as filepath or bytes)

input_midi = './tegridy-tools/tegridy-tools/seed-lyrics.mid'

#===============================================================================
# Output MIDI file

output_file_name = './song' # output MIDI file path without extension

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
# Let's say we have a melody with patch 1 (Bright Acoustic Piano)

mel_patch = 1

#===============================================================================
# Splitting enhanced score into melody and accompaniment scores...

mel_score = [e for e in escore_notes if e[6] == mel_patch]
acc_score = [e for e in escore_notes if e[6] != mel_patch]

#===============================================================================
# If both scores present...

if mel_score and acc_score:

  #=============================================================================
  # Making sure that melody score is monophonic...

  mel_score = [c[0] for c in TMIDIX.chordify_score([1000, mel_score])]

  #=============================================================================
  # Making sure that notes do not overlap...

  fixed_mel_score = TMIDIX.fix_monophonic_score_durations(mel_score)

  #=============================================================================
  # Making sure that accompaniment score consists of only good chords...

  fixed_acc_score = TMIDIX.flatten(TMIDIX.advanced_check_and_fix_chords_in_chordified_score(TMIDIX.chordify_score([1000, acc_score]))[0])

  #=============================================================================
  # Combining melody and accompaniment scores back into one score...

  fixed_escore_notes = sorted(fixed_mel_score+fixed_acc_score, key=lambda x: (x[1], x[4]))

  #=============================================================================
  # Resetting score timings to start from zero...

  recalculated_escore_notes = TMIDIX.recalculate_score_timings(fixed_escore_notes)

  #=============================================================================
  # Converting absolute score timings into relative timings...

  delta_escore_notes = TMIDIX.enhanced_delta_score_notes(recalculated_escore_notes)

  #=============================================================================
  # Processing delta enhanced score into integer tokens seq...

  tokenized_data = TMIDIX.basic_enhanced_delta_score_notes_tokenizer(delta_escore_notes)

  final_score_tokens_ints_seq = tokenized_data[1]
  tokens_shifts = tokenized_data[2]

  #=============================================================================
  # Double-checking that tokens seq is good and within range...
  
  assert min(final_score_tokens_ints_seq) >= tokens_shifts[0], print('Bad seq!!! Check seq!!!')
  assert max(final_score_tokens_ints_seq) < tokens_shifts[-1], print('Bad seq!!! Check seq!!!')

  #===============================================================================
  # Converting tokenized score seq back to MIDI...

  output_score = TMIDIX.basic_enhanced_delta_score_notes_detokenizer(final_score_tokens_ints_seq, tokens_shifts)

  output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(output_score)

  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                            output_signature = 'TMIDIX MIDI Composition',
                                                            output_file_name = output_file_name,
                                                            track_name='Project Los Angeles',
                                                            list_of_MIDI_patches=patches
                                                           )
```

***

### Make sure to check out [Jupyter/Google Colab Notebooks](https://github.com/asigalov61/tegridy-tools/tree/main/tegridy-tools/notebooks) dir for many useful and practical examples and applications

***

### Project Los Angeles
### Tegridy Code 2024
