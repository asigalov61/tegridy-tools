# =================================================================================================
# https://huggingface.co/spaces/asigalov61/Monophonic-MIDI-Melody-Harmonizer
# =================================================================================================

import os
import time as reqtime
import datetime
from pytz import timezone

import gradio as gr

import os
import random
from tqdm import tqdm

import TMIDIX
import HaystackSearch

from midi_to_colab_audio import midi_to_colab_audio

# =================================================================================================

def Harmonize_Melody(input_src_midi,
                    source_melody_transpose_value,
                    harmonizer_melody_chunk_size,
                    harmonizer_max_matches_count,
                    melody_MIDI_patch_number,
                    harmonized_accompaniment_MIDI_patch_number,
                    base_MIDI_patch_number
                    ):

    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)
    
    start_time = reqtime.time()

    sfn = os.path.basename(input_src_midi.name)
    sfn1 = sfn.split('.')[0]
    print('Input src MIDI name:', sfn)

    print('=' * 70)
    print('Requested settings:')
    print('Source melody transpose value:', source_melody_transpose_value)
    print('Harmonizer melody chunk size:', harmonizer_melody_chunk_size)
    print('Harmonizer max matrches count:', harmonizer_max_matches_count)
    print('Melody MIDI patch number:', melody_MIDI_patch_number)
    print('Harmonized accompaniment MIDI patch number:', harmonized_accompaniment_MIDI_patch_number)
    print('Base MIDI patch number:', base_MIDI_patch_number)
    print('=' * 70)
    
    #==================================================================

    print('=' * 70)
    print('Loading seed melody...')
    
    #===============================================================================
    # Raw single-track ms score
    
    raw_score = TMIDIX.midi2single_track_ms_score(input_src_midi.name)
    
    #===============================================================================
    # Enhanced score notes
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
    #===============================================================================
    # Augmented enhanced score notes
    
    escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes, timings_divider=16)
    
    cscore = [c[0] for c in TMIDIX.chordify_score([1000, escore_notes])]
    
    mel_score = TMIDIX.fix_monophonic_score_durations(TMIDIX.recalculate_score_timings(cscore))
    
    mel_score = TMIDIX.transpose_escore_notes(mel_score, source_melody_transpose_value)
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    mel_pitches = [p[4] % 12 for p in mel_score]
    
    print('Melody has', len(mel_pitches), 'notes')
    print('=' * 70)

    #==================================================================

    print('=' * 70)
    print('Melody Harmonizer')
    print('=' * 70)

    print('=' * 70)
    print('Harmonizing...')
    print('=' * 70)

    #===============================================================================
    
    song = []
    
    csize = harmonizer_melody_chunk_size
    matches_mem_size = harmonizer_max_matches_count
    
    i = 0
    dev = 0
    dchunk = []
    
    #===============================================================================

    def find_best_match(matches):
    
      mlens = []
    
      for sidx in matches:
        mlen = len(TMIDIX.flatten(long_chords_chunks_mult[sidx[0]][sidx[1]:sidx[1]+(csize // 2)]))
        mlens.append(mlen)
    
      max_len = max(mlens)
      max_len_idx = mlens.index(max_len)
    
      return matches[max_len_idx]

    #===============================================================================
    
    while i < len(mel_pitches):
    
      matches = []
    
      for midx, mel in enumerate(long_mels_chunks_mult):
        if len(mel) >= csize:
          schunk = mel_pitches[i:i+csize]
          idx = HaystackSearch.HaystackSearch(schunk, mel)
    
          if idx != -1:
            matches.append([midx, idx])
            if matches_mem_size > -1:
              if len(matches) > matches_mem_size:
                break
    
      if matches:
    
        sidx = find_best_match(matches)
    
        fchunk = long_chords_chunks_mult[sidx[0]][sidx[1]:sidx[1]+csize]
    
        song.extend(fchunk[:(csize // 2)])
        i += (csize // 2)
        dchunk = fchunk
        dev = 0
        print('step', i)
        
      else:
    
        if dchunk:
    
          song.append(dchunk[(csize // 2)+dev])
          dev += 1
          i += 1
          print('dead chord', i, dev)
        else:
          print('DEAD END!!!')
          song.append([mel_pitches[0]+48])
          break
    
    
        if dev == csize // 2:
          print('DEAD END!!!')
          break
    
    song = song[:len(mel_pitches)]
    
    print('Harmonized', len(song), 'out of', len(mel_pitches), 'notes')

    print('Done!')
    print('=' * 70)
    
    #===============================================================================
    
    print('Rendering results...')
    print('=' * 70)

    output_score = []
    
    time = 0
    
    patches = [0] * 16
    patches[0] = harmonized_accompaniment_MIDI_patch_number
    
    if base_MIDI_patch_number > -1:
      patches[2] = base_MIDI_patch_number
    
    patches[3] = melody_MIDI_patch_number
    
    for i, s in enumerate(song):
    
      time = mel_score[i][1] * 16
      dur = mel_score[i][2] * 16

      output_score.append(['note', time, dur, 3,  mel_score[i][4], 115+(mel_score[i][4] % 12), 40])
    
      for p in s:
        output_score.append(['note', time, dur, 0, p, max(40, p), harmonized_accompaniment_MIDI_patch_number])
                    
      if base_MIDI_patch_number > -1:
        output_score.append(['note', time, dur, 2, (s[-1] %  12)+24, 120-(s[-1] %  12), base_MIDI_patch_number])

    fn1 = "Monophonic-MIDI-Melody-Harmonizer-Composition"
    
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                              output_signature = 'Monophonic MIDI Melody Harmonizer',
                                                              output_file_name = fn1,
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches
                                                              )
    
    new_fn = fn1+'.mid'
            
    
    audio = midi_to_colab_audio(new_fn, 
                        soundfont_path=soundfont,
                        sample_rate=16000,
                        volume_scale=10,
                        output_for_gradio=True
                        )
    
    #========================================================

    output_midi_title = str(fn1)
    output_midi = str(new_fn)
    output_audio = (16000, audio)
    
    output_plot = TMIDIX.plot_ms_SONG(output_score, plot_title=output_midi, return_plt=True)
    
    print('Done!')
    
    #========================================================

    harmonization_summary_string = '=' * 70
    harmonization_summary_string += '\n'

    harmonization_summary_string += 'Source melody has ' + str(len(mel_pitches)) + ' monophonic pitches' + '\n'
    harmonization_summary_string += '=' * 70
    harmonization_summary_string += '\n'
    
    harmonization_summary_string += 'Harmonized ' + str(len(song)) + ' out of ' + str(len(mel_pitches)) + ' source melody pitches' + '\n'
    harmonization_summary_string += '=' * 70
    harmonization_summary_string += '\n'
    
    #========================================================
    
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')

    return output_audio, output_plot, output_midi, harmonization_summary_string

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    #===============================================================================

    soundfont = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"

    print('Loading Monster Harmonized Melodies MIDI Dataset...')
    print('=' * 70)
    all_chords_chunks = TMIDIX.Tegridy_Any_Pickle_File_Reader('Monster_Harmonized_Melodies_MIDI_Dataset')
    
    print('=' * 70)
    print('Total number of harmonized melodies:', len(all_chords_chunks))
    print('=' * 70)
    print('Loading melodies...')

    long_mels_chunks_mult = []
    long_chords_chunks_mult = []
    
    for c in tqdm(all_chords_chunks):
      long_mels_chunks_mult.append([p % 12 for p in c[0]])
      long_chords_chunks_mult.append(c[1])
    
    print('Done!')
    print('=' * 70)
    print('Total loaded melodies count:', len(long_mels_chunks_mult))
    print('=' * 70)
    
    #===============================================================================
    
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Monophonic MIDI Melody Harmonizer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Retrieval augmented harmonization of any MIDI melody</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Monophonic-MIDI-Melody-Harmonizer&style=flat)\n\n"
            "This is a demo for Monster MIDI Dataset\n\n"
            "Check out [Monster MIDI Dataset](https://github.com/asigalov61/Monster-MIDI-Dataset) on GitHub!\n\n"
        )
        
        gr.Markdown("## Upload your MIDI or select a sample example below")
        gr.Markdown("### For best results upload only monophonic melody MIDIs")
        
        input_src_midi = gr.File(label="Source MIDI", file_types=[".midi", ".mid", ".kar"])
        
        gr.Markdown("## Select harmonization options")

        source_melody_transpose_value = gr.Slider(-6, 6, value=0, step=1, label="Source melody transpose value", info="You can transpose source melody by specified number of semitones if the original melody key does not harmonize well")
        harmonizer_melody_chunk_size = gr.Slider(4, 16, value=8, step=2, label="Hamonizer melody chunk size", info="Larger chunk sizes result in better harmonization at the cost of speed and harminzation length")
        harmonizer_max_matches_count = gr.Slider(-1, 20, value=0, step=1, label="Harmonizer max matches count", info="Maximum number of harmonized chords per melody note to collect and to select from")
       
        melody_MIDI_patch_number = gr.Slider(0, 127, value=40, step=1, label="Source melody MIDI patch number")
        harmonized_accompaniment_MIDI_patch_number = gr.Slider(0, 127, value=0, step=1, label="Harmonized accompaniment MIDI patch number")
        base_MIDI_patch_number = gr.Slider(-1, 127, value=35, step=1, label="Base MIDI patch number")

        gr.Markdown("## PLEASE NOTE: Harmonization may take a long time and it is dependent on the selected harmonization settings")

        run_btn = gr.Button("harmonize melody", variant="primary")

        gr.Markdown("## Harmonization results")

        output_summary = gr.Textbox(label="Melody harmonization summary")
        
        output_audio = gr.Audio(label="Output MIDI audio", format="mp3", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])

        run_event = run_btn.click(Harmonize_Melody, 
                                                  [input_src_midi,
                                                    source_melody_transpose_value,
                                                    harmonizer_melody_chunk_size,
                                                    harmonizer_max_matches_count,
                                                    melody_MIDI_patch_number,
                                                    harmonized_accompaniment_MIDI_patch_number,
                                                    base_MIDI_patch_number],                                                                                                                       
                                                   [output_audio, output_plot, output_midi, output_summary]
                                 )

        gr.Examples(
            [
            ["USSR Anthem Seed Melody.mid", 0, 12, -1, 40, 0, 35],
            ],
            [input_src_midi,
            source_melody_transpose_value,
            harmonizer_melody_chunk_size,
            harmonizer_max_matches_count,
            melody_MIDI_patch_number,
            harmonized_accompaniment_MIDI_patch_number,
            base_MIDI_patch_number],                                                                                                                       
            [output_audio, output_plot, output_midi, output_summary],
            Harmonize_Melody,
            cache_examples=True,
        )
   
        app.queue().launch()