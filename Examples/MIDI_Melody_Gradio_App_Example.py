#=====================================================
# https://huggingface.co/spaces/asigalov61/MIDI-Melody
#=====================================================

import os

import time as reqtime
import datetime
from pytz import timezone

import gradio as gr

import random
import tqdm

from midi_to_colab_audio import midi_to_colab_audio

import TMIDIX

import matplotlib.pyplot as plt

# =================================================================================================
                       
def AddMelody(input_midi, 
              input_mel_type, 
              input_channel, 
              input_patch, 
              input_start_chord, 
              input_apply_sustains
             ):
    
    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    print('=' * 70)

    fn = os.path.basename(input_midi.name)
    fn1 = fn.split('.')[0]

    print('-' * 70)
    print('Input file name:', fn)
    print('Req mel type:', input_mel_type)
    print('Req channel:', input_channel)
    print('Req patch:', input_patch)
    print('Req start chord:', input_start_chord)
    print('Req apply sustains:', input_apply_sustains)
    print('-' * 70)

    #===============================================================================
    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)
    
    #===============================================================================
    # Enhanced score notes
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, 
                                                   return_enhanced_score_notes=True,
                                                   apply_sustain=input_apply_sustains
                                                  )[0]
    
    if len(escore_notes) > 0:
    
        #=======================================================
        # PRE-PROCESSING
        
        #===============================================================================
        # Augmented enhanced score notes
        
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes)
        
        #===============================================================================
        # Recalculate timings
        
        escore_notes = TMIDIX.recalculate_score_timings(escore_notes)
        
        #===============================================================================
        # Add melody

        if input_mel_type == "Original":
            output = TMIDIX.add_melody_to_enhanced_score_notes(escore_notes, 
                                                               melody_channel=input_channel, 
                                                               melody_patch=input_patch, 
                                                               melody_start_chord=input_start_chord
                                                              )

        elif input_mel_type == "Expressive":
            output = TMIDIX.add_expressive_melody_to_enhanced_score_notes(escore_notes, 
                                                                          melody_channel=input_channel, 
                                                                          melody_patch=input_patch, 
                                                                          melody_start_chord=input_start_chord
                                                                         )

        elif input_mel_type == "Smooth Expressive":
            output = TMIDIX.add_smooth_expressive_melody_to_enhanced_score_notes(escore_notes, 
                                                                                 melody_channel=input_channel, 
                                                                                 melody_patch=input_patch, 
                                                                                 melody_start_chord=input_start_chord
                                                                                )
        
        elif input_mel_type == "Smooth":
            output = TMIDIX.add_smooth_melody_to_enhanced_score_notes(escore_notes,
                                                                      melody_channel=input_channel, 
                                                                      melody_patch=input_patch, 
                                                                      melody_start_chord=input_start_chord
                                                                     )
            
        print('=' * 70)
        print('Done!')
        print('=' * 70)
        
        #===============================================================================
        print('Rendering results...')
        
        print('=' * 70)
        print('Sample INTs', output[:12])
        print('=' * 70)
        
        output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(output)

        detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                                  output_signature = 'MIDI Melody',
                                                                  output_file_name = fn1,
                                                                  track_name='Project Los Angeles',
                                                                  list_of_MIDI_patches=patches,
                                                                  timings_multiplier=16
                                                                  )
        
        new_fn = fn1+'.mid'
                
        
        audio = midi_to_colab_audio(new_fn, 
                            soundfont_path=soundfont,
                            sample_rate=16000,
                            output_for_gradio=True
                            )
        
        print('Done!')
        print('=' * 70)
    
        #========================================================
    
        output_midi_title = str(fn1)
        output_midi_summary = str(output_score[:3])
        output_midi = str(new_fn)
        output_audio = (16000, audio)
        
        output_plot = TMIDIX.plot_ms_SONG(output_score,
                                          plot_title=output_midi_title,
                                          timings_multiplier=16,
                                          return_plt=True
                                         )
    
        print('Output MIDI file name:', output_midi)
        print('Output MIDI title:', output_midi_title)
        print('Output MIDI summary:', '')
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
   
    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>MIDI Melody</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Add a unique melody to any MIDI</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.MIDI-Melody&style=flat)\n\n"
            "This is a demo for TMIDIX Python module from tegridy-tools\n\n"
            "Check out [tegridy-tools](https://github.com/asigalov61/tegridy-tools) on GitHub!\n\n"
        )
        gr.Markdown("## Upload your MIDI or select a sample example MIDI")

        input_midi = gr.File(label="Input MIDI")
        
        input_mel_type = gr.Dropdown(['Expressive', 'Smooth Expressive', 'Smooth', 'Original'], value="Expressive", label="Melody type")
        input_channel = gr.Slider(0, 15, value=3, step=1, label="Melody MIDI channel")
        input_patch = gr.Slider(0, 127, value=40, step=1, label="Melody MIDI patch")
        input_start_chord = gr.Slider(0, 128, value=0, step=1, label="Melody start chord")
        input_apply_sustains = gr.Checkbox(value=True, label="Apply sustains (if present)")
        
        run_btn = gr.Button("add melody", variant="primary")

        gr.Markdown("## Output results")

        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])


        run_event = run_btn.click(AddMelody, [input_midi, 
                                              input_mel_type, 
                                              input_channel, 
                                              input_patch, 
                                              input_start_chord,
                                              input_apply_sustains
                                             ],
                                              [output_midi_title, 
                                               output_midi_summary, 
                                               output_midi, 
                                               output_audio, 
                                               output_plot
                                              ])

        gr.Examples([["Sharing The Night Together.kar", "Expressive", 3, 40, 0, False], 
                     ["Deep Relaxation Melody #6.mid", "Expressive", 3, 40, 0, False],
                    ],
                    [input_midi, 
                     input_mel_type, 
                     input_channel, 
                     input_patch, 
                     input_start_chord, 
                     input_apply_sustains
                    ],
                    [output_midi_title, 
                     output_midi_summary, 
                     output_midi, 
                     output_audio, 
                     output_plot
                    ],
                    AddMelody,
                    cache_examples=True,
                )
        
        app.queue().launch()