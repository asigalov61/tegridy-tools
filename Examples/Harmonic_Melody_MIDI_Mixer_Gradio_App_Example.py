# https://huggingface.co/spaces/asigalov61/Harmonic-Melody-MIDI-Mixer

import os.path

import time as reqtime
import datetime
from pytz import timezone

from itertools import groupby
import copy

import gradio as gr

import random

from midi_to_colab_audio import midi_to_colab_audio
import TMIDIX

import matplotlib.pyplot as plt

in_space = os.getenv("SYSTEM") == "spaces"
         
# =================================================================================================

def pitches_counts(melody_score):

  pitches = [p[4] for p in melody_score]

  pcounts = []

  count = 0
  pp = -1
    
  for p in pitches:
    if p == pp:
      count += 1
      pcounts.append(count)
    else:
      count = 0
      pcounts.append(count)
    pp = p

  return pcounts

# =================================================================================================

def find_similar_song(songs, src_melody):

  src_pcount = pitches_counts(src_melody)

  ratios = []

  for s in songs:
    patch = s[1]

    trg_melody = [e for e in s[3] if e[6] == patch]
    trg_pcount = pitches_counts(trg_melody)

    pcount = 0

    for i, c in enumerate(src_pcount):
      if c == trg_pcount[i]:
        pcount += 1

    ratios.append(pcount / len(src_pcount))

  max_ratio = max(ratios)

  return songs[ratios.index(max_ratio)], max_ratio, ratios.count(max_ratio)

# =================================================================================================

def mix_chord(chord, tones_chord, mel_patch, mel_pitch, next_note_dtime):

  cho = []

  for k, g in groupby(sorted(chord, key=lambda x: x[6]), lambda x: x[6]):

    if k != 128:
      if k == mel_patch:
          
        cg = list(g)
          
        c = copy.deepcopy(cg[0])
          
        if cg[0][2] > next_note_dtime:
            c[2] = next_note_dtime
            
        c[4] = mel_pitch
        c[5] = 105 + (mel_pitch % 12)
          
        cho.append(c)

      else:
        cg = list(g)

        tclen = len(tones_chord)

        tchord = tones_chord

        if len(cg) > tclen:
          tchord = tones_chord + [random.choice(tones_chord) for _ in range(len(cg)-tclen)]

        for i, cc in enumerate(cg):
            
          c = copy.deepcopy(cc)
            
          if cc[2] > next_note_dtime:
              c[2] = next_note_dtime
              
          c[4] = ((c[4] // 12) * 12) + tchord[i]
          c[5] += c[4] % 12
            
          cho.append(c)

    else:
      cho.extend(list(g))

  return cho

# =================================================================================================

def MixMelody(input_midi, input_find_best_match, input_adjust_melody_notes_durations, input_adjust_accompaniment_notes_durations):
    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    print('=' * 70)

    fn = os.path.basename(input_midi.name)
    fn1 = fn.split('.')[0]

    print('-' * 70)
    print('Input file name:', fn)
    print('Find best matches', input_find_best_match)
    print('Adjust melody notes durations:', input_adjust_melody_notes_durations)
    print('Adjust accompaniment notes durations:', input_adjust_accompaniment_notes_durations)
    print('-' * 70)

    #===============================================================================
    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)
    
    #===============================================================================
    # Enhanced score notes
    
    raw_escore = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
    if len(raw_escore) > 0:
    
        #===============================================================================
        # Augmented enhanced score notes
        
        src_escore = TMIDIX.recalculate_score_timings(TMIDIX.augment_enhanced_score_notes([e for e in raw_escore if e[6] < 80]))
        
        src_cscore = TMIDIX.chordify_score([1000, src_escore])
        
        src_melody = [c[0] for c in src_cscore][:256]
        
        src_melody_pitches = [p[4] for p in src_melody]
        
        src_harm_tones_chords = TMIDIX.harmonize_enhanced_melody_score_notes(src_melody)
        
        #===============================================================================
        
        matched_songs = [a for a in all_songs if a[2] == max(32, len(src_melody))]
        
        random.shuffle(matched_songs)

        max_match_ratio = -1
        max_match_ratios_count = len(matched_songs)

        if input_find_best_match:
            new_song, max_match_ratio, max_match_ratios_count = find_similar_song(matched_songs, src_melody)
        else:
            new_song = random.choice(matched_songs)
        
        print('Selected Monster Mono Melodies MIDI:', new_song[0])
        print('Selected melody match ratio:', max_match_ratio)
        print('Selected melody instrument:', TMIDIX.Number2patch[new_song[1]], '(', new_song[1], ')')
        print('Melody notes count:', new_song[2])
        print('Matched melodies pool count', max_match_ratios_count)
        
        MIDI_Summary = 'Selected Monster Mono Melodies MIDI: ' + str(new_song[0]) + '\n'
        MIDI_Summary += 'Selected melody match ratio: ' + str(max_match_ratio) + '\n'
        MIDI_Summary += 'Selected melody instrument: ' + str(TMIDIX.Number2patch[new_song[1]]) + ' (' + str(new_song[1]) + ')' + '\n'
        MIDI_Summary += 'Melody notes count: ' + str(new_song[2]) + '\n'
        MIDI_Summary += 'Matched melodies pool count: ' + str(max_match_ratios_count)

        fn1 += '_' + str(new_song[0]) + '_' + str(TMIDIX.Number2patch[new_song[1]]) + '_' + str(new_song[1]) + '_' + str(new_song[2])
        
        trg_patch = new_song[1]
        
        trg_song = copy.deepcopy(new_song[3])
        TMIDIX.adjust_score_velocities(trg_song, 95)
        
        cscore = TMIDIX.chordify_score([1000, trg_song])
        
        print('=' * 70)
        print('Done loading source and target MIDIs...!')
        print('=' * 70)
        print('Mixing...')

        mixed_song = []
        
        midx = 0
        next_note_dtime = 255
        
        for i, c in enumerate(cscore):
            cho = copy.deepcopy(c)
            
            patches = sorted(set([e[6] for e in c]))
            
            if trg_patch in patches:

                if input_adjust_melody_notes_durations:
                    if midx < len(src_melody)-1:
                        next_note_dtime = src_melody[midx+1][1] - src_melody[midx][1]
                    else:
                        next_note_dtime = 255
                    
                mixed_song.extend(mix_chord(c, src_harm_tones_chords[midx], trg_patch, src_melody_pitches[midx], next_note_dtime))
                
                midx += 1
            
            else:
                if input_adjust_accompaniment_notes_durations:
                    if i < len(cscore)-1:
                        next_note_dtime = cscore[i+1][0][1] - cscore[i][0][1]
                    else:
                        next_note_dtime = 255
                
                mixed_song.extend(mix_chord(cho, src_harm_tones_chords[midx], trg_patch, src_melody_pitches[midx], next_note_dtime))

            if midx == len(src_melody):
                break      

        print('=' * 70)
        print('Done!')
        print('=' * 70)

        #===============================================================================
        print('Rendering results...')
        
        print('=' * 70)
        print('Sample INTs', mixed_song[:5])
        print('=' * 70)
        
        output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(mixed_song)

        detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                                  output_signature = 'Harmonic Melody MIDI Mixer',
                                                                  output_file_name = fn1,
                                                                  track_name='Project Los Angeles',
                                                                  list_of_MIDI_patches=patches,
                                                                  timings_multiplier=16
                                                                  )
        
        new_fn = fn1+'.mid'
                
        
        audio = midi_to_colab_audio(new_fn, 
                            soundfont_path=soundfont,
                            sample_rate=16000,
                            volume_scale=10,
                            output_for_gradio=True
                            )
        
        print('Done!')
        print('=' * 70)
    
        #========================================================
    
        output_midi_title = str(fn1)
        output_midi_summary = str(MIDI_Summary)
        output_midi = str(new_fn)
        output_audio = (16000, audio)

        for o in output_score:
            o[1] *= 16
            o[2] *= 16
        
        output_plot = TMIDIX.plot_ms_SONG(output_score, plot_title=output_midi_title, return_plt=True)
    
        print('Output MIDI file name:', output_midi)
        print('Output MIDI title:', output_midi_title)
        print('Output MIDI summary:', MIDI_Summary)
        print('=' * 70) 
        
    
        #========================================================
        
        print('-' * 70)
        print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
        print('-' * 70)
        print('Req execution time:', (reqtime.time() - start_time), 'sec')
    
        return output_midi_title, output_midi_summary, output_midi, output_audio, output_plot

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    soundfont = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"
    
    all_songs = TMIDIX.Tegridy_Any_Pickle_File_Reader('Monster_Mono_Melodies_MIDI_Dataset_65536_32_256')
    print('=' * 70)
    
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Harmonic Melody MIDI Mixer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Harmonize and mix any MIDI melody</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Harmonic-Melody-MIDI-Mixer&style=flat)\n\n"
            "This is a demo for TMIDIX Python module from tegridy-tools and Monster Mono Melodies MIDI Dataset\n\n"
            "Check out [tegridy-tools](https://github.com/asigalov61/tegridy-tools) on GitHub!\n\n"
            "Check out [Monster-MIDI-Dataset](https://github.com/asigalov61/Monster-MIDI-Dataset) on GitHub!\n\n"
        )
        gr.Markdown("## Upload your MIDI or select a sample example MIDI below")
        
        input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
        input_find_best_match = gr.Checkbox(label="Find best match", value=True)
        input_adjust_melody_notes_durations = gr.Checkbox(label="Adjust melody notes durations", value=True)
        input_adjust_accompaniment_notes_durations = gr.Checkbox(label="Adjust accompaniment notes durations", value=True)
        
        run_btn = gr.Button("mix melody", variant="primary")

        gr.Markdown("## Output results")

        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])


        run_event = run_btn.click(MixMelody, [input_midi, input_find_best_match, input_adjust_melody_notes_durations, input_adjust_accompaniment_notes_durations],
                                  [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot])

        gr.Examples(
            [["Abracadabra-Sample-Melody.mid", True, True, True], 
             ["Sparks-Fly-Sample-Melody.mid", True, True, True],
            ],
            [input_midi, input_find_best_match, input_adjust_melody_notes_durations, input_adjust_accompaniment_notes_durations],
            [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot],
            MixMelody,
            cache_examples=True,
        )
        
        app.queue().launch()