#================================================================
# https://huggingface.co/spaces/asigalov61/Advanced-MIDI-Renderer
#================================================================
# Packages:
#
#   sudo apt install fluidsynth
#
#================================================================
# Requirements:
#   
#   pip install gradio
#   pip install numpy
#   pip install scipy
#   pip install matplotlib
#   pip install networkx
#   pip install scikit-learn
#
#================================================================
# Core modules:
#
# git clone --depth 1 https://github.com/asigalov61/tegridy-tools
#
# import TMIDIX
# import TPLOTS
# import midi_to_colab_audio
#
#================================================================

import os
import hashlib

import time
import datetime
from pytz import timezone

import copy
from collections import Counter
import random
import statistics

import gradio as gr

import TMIDIX
import TPLOTS

from midi_to_colab_audio import midi_to_colab_audio

#==========================================================================================================

def Render_MIDI(input_midi, 
                render_type, 
                soundfont_bank, 
                render_sample_rate,
                render_with_sustains,
                merge_misaligned_notes,
                custom_render_patch,
                render_align,
                render_transpose_value,
                render_transpose_to_C4,
                render_output_as_solo_piano,
                render_remove_drums 
                ):
    
    print('*' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = time.time()
    
    print('=' * 70)
    print('Loading MIDI...')

    fn = os.path.basename(input_midi)
    fn1 = fn.split('.')[0]

    fdata = open(input_midi, 'rb').read()

    input_midi_md5hash = hashlib.md5(fdata).hexdigest()
    
    print('=' * 70)
    print('Requested settings:')
    print('=' * 70)
    print('Input MIDI file name:', fn)
    print('Input MIDI md5 hash', input_midi_md5hash)
    print('-' * 70)
    print('Render type:', render_type)
    print('Soudnfont bank', soundfont_bank)
    print('Audio render sample rate', render_sample_rate)

    if render_type != 'Render as-is':
        print('Render with sustains:', render_with_sustains)
        print('Merge misaligned notes:', merge_misaligned_notes)
        print('Custom MIDI render patch', custom_render_patch)
        print('Align to bars:', render_align)
        print('Transpose value:', render_transpose_value)
        print('Transpose to C4', render_transpose_to_C4)
        print('Output as Solo Piano', render_output_as_solo_piano)
        print('Remove drums:', render_remove_drums)

    print('=' * 70)
    print('Processing MIDI...Please wait...')
    
    #=======================================================
    # START PROCESSING

    raw_score = TMIDIX.midi2single_track_ms_score(fdata)
    
    escore = TMIDIX.advanced_score_processor(raw_score, 
                                             return_enhanced_score_notes=True, 
                                             apply_sustain=render_with_sustains
                                            )[0]

    if merge_misaligned_notes > 0:
        escore = TMIDIX.merge_escore_notes(escore, merge_threshold=merge_misaligned_notes)

    escore = TMIDIX.augment_enhanced_score_notes(escore, timings_divider=1)

    first_note_index = [e[0] for e in raw_score[1]].index('note')
    
    cscore = TMIDIX.chordify_score([1000, escore])

    meta_data = raw_score[1][:first_note_index] + [escore[0]] + [escore[-1]] + [raw_score[1][-1]]

    aux_escore_notes = TMIDIX.augment_enhanced_score_notes(escore, sort_drums_last=True)
    song_description = TMIDIX.escore_notes_to_text_description(aux_escore_notes)
    
    print('Done!')
    print('=' * 70)
    print('Input MIDI metadata:', meta_data[:5])
    print('=' * 70)
    print('Input MIDI song description:', song_description)
    print('=' * 70)
    print('Processing...Please wait...')

    output_score = copy.deepcopy(escore)

    if render_type == "Extract melody":
        output_score = TMIDIX.add_expressive_melody_to_enhanced_score_notes(escore, return_melody=True)
        output_score = TMIDIX.recalculate_score_timings(output_score)

    elif render_type == "Flip":
        output_score = TMIDIX.flip_enhanced_score_notes(escore)
        
    elif render_type == "Reverse":
        output_score = TMIDIX.reverse_enhanced_score_notes(escore)

    elif render_type == 'Repair Durations':
        output_score = TMIDIX.even_out_durations_in_escore_notes(escore)
        output_score = TMIDIX.fix_escore_notes_durations(output_score, min_notes_gap=0)
    
    elif render_type == 'Repair Chords':
        fixed_cscore = TMIDIX.advanced_check_and_fix_chords_in_chordified_score(cscore)[0]
        output_score = TMIDIX.flatten(fixed_cscore)

    elif render_type == 'Remove Duplicate Pitches':
        output_score = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore)

    elif render_type == 'Quantize':
        output_score = TMIDIX.quantize_escore_notes(escore)

    elif render_type == 'Humanize Velocities':
        output_score = TMIDIX.humanize_velocities_in_escore_notes(escore)
    
    elif render_type == "Add Drum Track":
        nd_escore = [e for e in escore if e[3] != 9]
        nd_escore = TMIDIX.augment_enhanced_score_notes(nd_escore)
        output_score = TMIDIX.advanced_add_drums_to_escore_notes(nd_escore)

        for e in output_score:
            e[1] *= 16
            e[2] *= 16

    print('Done processing!')
    print('=' * 70)
    
    print('Repatching if needed...')
    print('=' * 70)

    if -1 < custom_render_patch < 128:
        for e in output_score:
            if e[3] != 9:
                e[6] = custom_render_patch
        
    print('Done repatching!')
    print('=' * 70)
    
    print('Sample output events', output_score[:5])
    print('=' * 70)
    print('Final processing...')
    
    new_fn = fn1+'.mid'

    if render_type != "Render as-is":

        if render_transpose_value != 0:
            output_score = TMIDIX.transpose_escore_notes(output_score, render_transpose_value)

        if render_transpose_to_C4:
            output_score = TMIDIX.transpose_escore_notes_to_pitch(output_score)

        if render_align == "Start Times":
            output_score = TMIDIX.recalculate_score_timings(output_score)
            output_score = TMIDIX.align_escore_notes_to_bars(output_score)
    
        elif render_align == "Start Times and Durations":
            output_score = TMIDIX.recalculate_score_timings(output_score)
            output_score = TMIDIX.align_escore_notes_to_bars(output_score, trim_durations=True)
    
        elif render_align == "Start Times and Split Durations":
            output_score = TMIDIX.recalculate_score_timings(output_score)
            output_score = TMIDIX.align_escore_notes_to_bars(output_score, split_durations=True)

        if render_type == "Longest Repeating Phrase":
            zscore = TMIDIX.recalculate_score_timings(output_score)
            lrno_score = TMIDIX.escore_notes_lrno_pattern_fast(zscore)

            if lrno_score is not None:
                output_score = lrno_score

            else:
                output_score = TMIDIX.recalculate_score_timings(TMIDIX.escore_notes_middle(output_score, 50))

        if render_type == "Multi-Instrumental Summary":
            zscore = TMIDIX.recalculate_score_timings(output_score)
            c_escore_notes = TMIDIX.compress_patches_in_escore_notes_chords(zscore)
            
            if len(c_escore_notes) > 128:
                cmatrix = TMIDIX.escore_notes_to_image_matrix(c_escore_notes, filter_out_zero_rows=True, filter_out_duplicate_rows=True)
                smatrix = TPLOTS.square_image_matrix(cmatrix, num_pca_components=max(1, min(5, len(c_escore_notes) // 128)))
                output_score = TMIDIX.image_matrix_to_original_escore_notes(smatrix)
                
                for o in output_score:
                    o[1] *= 250
                    o[2] *= 250            

        if render_output_as_solo_piano:
            output_score = TMIDIX.solo_piano_escore_notes(output_score, keep_drums=True)
            
        if render_remove_drums:
            output_score = TMIDIX.strip_drums_from_escore_notes(output_score)
            
        if render_type == "Solo Piano Summary":
            sp_escore_notes = TMIDIX.solo_piano_escore_notes(output_score, keep_drums=False)
            zscore = TMIDIX.recalculate_score_timings(sp_escore_notes)

            if len(zscore) > 128:
                
                bmatrix = TMIDIX.escore_notes_to_binary_matrix(zscore)
                cmatrix = TMIDIX.compress_binary_matrix(bmatrix, only_compress_zeros=True)
                smatrix = TPLOTS.square_binary_matrix(cmatrix, interpolation_order=max(1, min(5, len(zscore) // 128)))
                output_score = TMIDIX.binary_matrix_to_original_escore_notes(smatrix)
    
                for o in output_score:
                    o[1] *= 200
                    o[2] *= 200
            
        SONG, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(output_score)
                    
        detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(SONG,
                                                                  output_signature = 'Advanced MIDI Renderer',
                                                                  output_file_name = fn1,
                                                                  track_name='Project Los Angeles',
                                                                  list_of_MIDI_patches=patches
                                                                  )

    else:
        with open(new_fn, 'wb') as f:
            f.write(fdata)
            f.close()
            
    if soundfont_bank in ["Super GM",
                            "Orpheus GM",
                            "Live HQ GM",
                            "Nice Strings + Orchestra", 
                            "Real Choir", 
                            "Super Game Boy", 
                            "Proto Square"
                            ]:
        
        sf2bank = ["Super GM",
                    "Orpheus GM",
                    "Live HQ GM",
                    "Nice Strings + Orchestra", 
                    "Real Choir", 
                    "Super Game Boy", 
                    "Proto Square"
                    ].index(soundfont_bank)
    
    else:
        sf2bank = 0

    if render_sample_rate in ["16000", "32000", "44100"]:
        srate = int(render_sample_rate)
    
    else:
        srate = 16000

    print('-' * 70)
    print('Generating audio with SF2 bank', sf2bank, 'and', srate, 'Hz sample rate')
    
    audio = midi_to_colab_audio(new_fn, 
                        soundfont_path=soundfonts[sf2bank],
                        sample_rate=srate,
                        output_for_gradio=True
                        )

    print('-' * 70)

    new_md5_hash = hashlib.md5(open(new_fn,'rb').read()).hexdigest()
    
    print('Done!')
    print('=' * 70)

    #========================================================

    output_midi_md5 = str(new_md5_hash)
    output_midi_title = str(fn1)
    output_midi_summary = str(meta_data)
    output_midi = str(new_fn)
    output_audio = (srate, audio)
    
    output_plot = TMIDIX.plot_ms_SONG(output_score, plot_title=output_midi, return_plt=True)

    print('Output MIDI file name:', output_midi)
    print('Output MIDI title:', output_midi_title)
    print('Output MIDI hash:', output_midi_md5)
    print('Output MIDI summary:', output_midi_summary[:5])
    print('=' * 70) 
    
    #========================================================

    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (time.time() - start_time), 'sec')
    print('*' * 70)
    
    #========================================================
    
    return output_midi_md5, output_midi_title, output_midi_summary, output_midi, output_audio, output_plot, song_description
    
#==========================================================================================================

if __name__ == "__main__":

    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    soundfonts = ["SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2",
                  "Orpheus_18.06.2020.sf2",
                  "Live HQ Natural SoundFont GM.sf2",
                  "Nice-Strings-PlusOrchestra-v1.6.sf2", 
                  "KBH-Real-Choir-V2.5.sf2", 
                  "SuperGameBoy.sf2", 
                  "ProtoSquare.sf2"
                 ]

    app = gr.Blocks()
    
    with app:
        
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Advanced MIDI Renderer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Transform and render any MIDI</h1>")
        
        gr.Markdown("This is a demo for tegridy-tools\n\n"
                    "Please see [tegridy-tools](https://github.com/asigalov61/tegridy-tools) GitHub repo for more information\n\n"
                   )
        
        gr.Markdown("## Upload your MIDI")
        
        input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"], type="filepath")

        gr.Markdown("## Select desired Sound Font bank and render sample rate")

        soundfont_bank = gr.Radio(["Super GM",
                                   "Orpheus GM",
                                   "Live HQ GM",
                                   "Nice Strings + Orchestra", 
                                   "Real Choir", 
                                   "Super Game Boy", 
                                   "Proto Square"
                                  ], 
                                  label="SoundFont bank", 
                                  value="Super GM"
                                 )

        render_sample_rate = gr.Radio(["16000", 
                                       "32000", 
                                       "44100"
                                      ], 
                                      label="MIDI audio render sample rate", 
                                      value="16000"
                                     )

        gr.Markdown("## Select desired render type")

        render_type = gr.Radio(["Render as-is", 
                                "Custom render", 
                                "Extract melody", 
                                "Flip", 
                                "Reverse",
                                "Repair Durations",
                                "Repair Chords",
                                "Remove Duplicate Pitches",
                                "Longest Repeating Phrase",
                                "Multi-Instrumental Summary",
                                "Solo Piano Summary",
                                "Quantize",
                                "Humanize Velocities",
                                "Add Drum Track"
                               ], 
                               label="Render type", 
                               value="Render as-is"
                              )

        gr.Markdown("## Select custom render options")

        render_with_sustains = gr.Checkbox(label="Render with sustains (if present)", value=True)
        merge_misaligned_notes = gr.Slider(-1, 127, value=-1, label="Merge misaligned notes")
        custom_render_patch = gr.Slider(-1, 127, value=-1, label="Custom render MIDI patch")
        
        render_align = gr.Radio(["Do not align", 
                                 "Start Times", 
                                 "Start Times and Durations", 
                                 "Start Times and Split Durations"
                                ], 
                                label="Align output to bars", 
                                value="Do not align"
                               )        
        
        render_transpose_value = gr.Slider(-12, 12, value=0, step=1, label="Transpose value")
        render_transpose_to_C4 = gr.Checkbox(label="Transpose to C4", value=False)


        render_output_as_solo_piano = gr.Checkbox(label="Output as Solo Piano", value=False)
        render_remove_drums = gr.Checkbox(label="Remove drums", value=False)
        
        submit = gr.Button("Render MIDI", variant="primary")

        gr.Markdown("## Render results")
        
        output_midi_md5 = gr.Textbox(label="Output MIDI md5 hash")
        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_song_description = gr.Textbox(label="Output MIDI description")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])
        
        run_event = submit.click(Render_MIDI, [input_midi, 
                                               render_type, 
                                               soundfont_bank, 
                                               render_sample_rate,
                                               render_with_sustains,
                                               merge_misaligned_notes,
                                               custom_render_patch,
                                               render_align,
                                               render_transpose_value,
                                               render_transpose_to_C4,
                                               render_output_as_solo_piano,
                                               render_remove_drums                                              
                                              ],
                                                [output_midi_md5, 
                                                 output_midi_title, 
                                                 output_midi_summary, 
                                                 output_midi, 
                                                 output_audio, 
                                                 output_plot,
                                                 output_song_description
                                                ])
        
    app.queue().launch()