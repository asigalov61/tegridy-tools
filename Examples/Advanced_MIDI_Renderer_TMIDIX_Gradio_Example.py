import argparse
import glob
import os.path
import hashlib
import time
import datetime
from pytz import timezone

import gradio as gr

import pickle
import tqdm
import json

import TMIDIX
from midi_to_colab_audio import midi_to_colab_audio

import copy
from collections import Counter
import random
import statistics

import matplotlib.pyplot as plt

#==========================================================================================================

in_space = os.getenv("SYSTEM") == "spaces"

#==========================================================================================================

def render_midi(input_midi, render_type, soundfont_bank, render_sample_rate, custom_render_patch):
    
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
    print('Input MIDI file name:', fn)
    print('Input MIDI md5 hash', input_midi_md5hash)
    print('Render type:', render_type)
    print('Soudnfont bank', soundfont_bank)
    print('Audio render sample rate', render_sample_rate)
    print('Custom MIDI render patch', custom_render_patch)
    
    print('=' * 70)
    print('Processing MIDI...Please wait...')
    
    #=======================================================
    # START PROCESSING

    raw_score = TMIDIX.midi2single_track_ms_score(fdata)
    
    escore = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
    
    escore= TMIDIX.augment_enhanced_score_notes(escore)

    first_note_index = [e[0] for e in raw_score[1]].index('note')
    
    cscore = TMIDIX.chordify_score([1000, escore])

    meta_data = raw_score[1][:first_note_index] + [escore[0]] + [escore[-1]] + [raw_score[1][-1]]
    
    print('Done!')
    print('=' * 70)
    print('Input MIDI metadata:', meta_data[:5])
    print('=' * 70)
    print('Processing...Please wait...')

    if render_type == 'Render as-is':
        output_score = copy.deepcopy(escore)

    elif render_type == "Custom render" or not render_type:
        output_score = copy.deepcopy(escore)

    elif render_type == "Extract melody":
        output_score = TMIDIX.extract_melody(cscore, melody_range=[48, 84])

    elif render_type == "Flip":
        output_score = TMIDIX.flip_enhanced_score_notes(escore)
        
    elif render_type == "Reverse":
        output_score = TMIDIX.reverse_enhanced_score_notes(escore)
        
    elif render_type == 'Repair':
        fixed_cscore = TMIDIX.advanced_check_and_fix_chords_in_chordified_score(cscore)[0]
        output_score = TMIDIX.flatten(fixed_cscore)

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

        SONG, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(output_score)
                    
        detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(SONG,
                                                                  output_signature = 'Advanced MIDI Renderer',
                                                                  output_file_name = fn1,
                                                                  track_name='Project Los Angeles',
                                                                  list_of_MIDI_patches=patches,
                                                                  timings_multiplier=16
                                                                  )

    else:
        with open(new_fn, 'wb') as f:
            f.write(fdata)
            f.close()
            
    if soundfont_bank in ["General MIDI", "Nice strings plus orchestra", "Real choir"]:
        sf2bank = ["General MIDI", "Nice strings plus orchestra", "Real choir"].index(soundfont_bank)
    
    else:
        sf2bank = 0

    if render_sample_rate in ["16000", "32000", "44100"]:
        srate = int(render_sample_rate)
    
    else:
        srate = 16000
    
    audio = midi_to_colab_audio(new_fn, 
                        soundfont_path=soundfonts[sf2bank],
                        sample_rate=srate,
                        volume_scale=10,
                        output_for_gradio=True
                        )

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
    
    yield output_midi_md5, output_midi_title, output_midi_summary, output_midi, output_audio, output_plot
    
#==========================================================================================================

if __name__ == "__main__":

    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--port", type=int, default=7860, help="gradio server port")
    
    opt = parser.parse_args()
    
    soundfonts = ["SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2", "Nice-Strings-PlusOrchestra-v1.6.sf2", "KBH-Real-Choir-V2.5.sf2"]

    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Advanced MIDI Renderer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Transform and render any MIDI</h1>")
        
        gr.Markdown("![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Advanced-MIDI-Renderer&style=flat)\n\n"
                    "This is a demo for tegridy-tools\n\n"
                    "Please see [tegridy-tools](https://github.com/asigalov61/tegridy-tools) GitHub repo for more information\n\n"
                   )
        
        gr.Markdown("## Upload your MIDI")
        
        input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"], type="filepath")

        gr.Markdown("## Select desired render type")

        render_type = gr.Radio(["Render as-is", "Custom render", "Extract melody", "Flip", "Reverse", "Repair"], label="Render type", value="Render as-is")

        gr.Markdown("## Select desired render options")

        soundfont_bank = gr.Radio(["General MIDI", "Nice strings plus orchestra", "Real choir"], label="SoundFont bank", value="General MIDI")

        render_sample_rate = gr.Radio(["16000", "32000", "44100"], label="MIDI audio render sample rate", value="16000")

        custom_render_patch = gr.Slider(-1, 127, value=-1, label="Custom MIDI render patch")

        submit = gr.Button()

        gr.Markdown("## Render results")
        
        output_midi_md5 = gr.Textbox(label="Output MIDI md5 hash")
        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])
        
        run_event = submit.click(render_midi, [input_midi, render_type, soundfont_bank, render_sample_rate, custom_render_patch],
                                                [output_midi_md5, output_midi_title, output_midi_summary, output_midi, output_audio, output_plot])
        
    app.queue(1).launch(server_port=opt.port, share=opt.share, inbrowser=True) 