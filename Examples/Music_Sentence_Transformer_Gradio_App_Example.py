# https://huggingface.co/spaces/asigalov61/Music-Sentence-Transformer

import os.path

import time as reqtime
import datetime
from pytz import timezone

import gradio as gr

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import numpy as np

import random
import tqdm

from midi_to_colab_audio import midi_to_colab_audio
import TMIDIX

import matplotlib.pyplot as plt

in_space = os.getenv("SYSTEM") == "spaces"
         
# =================================================================================================
                       
def SearchMIDIs(input_text):
    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    print('-' * 70)
    print('Req search query:', input_text)
    print('-' * 70)

    print('Searching...')

    query_embedding = model.encode([input_text])
    
    # Compute cosine similarity between query and each sentence in the corpus
    similarities = util.cos_sim(query_embedding, corpus_embeddings)
    
    # Find the index of the most similar sentence
    closest_index = np.argmax(similarities)
    closest_index_match_ratio = max(similarities[0]).tolist()

    best_corpus_match = clean_midi_artist_song_description_summaries_lyrics_score[closest_index]

    print('Done!')
    print('=' * 70)
    
    print('Match corpus index', closest_index)
    print('Match corpus ratio', closest_index_match_ratio)
    
    print('=' * 70)
    print('Done!')
    print('=' * 70)
    
    #===============================================================================
    print('Rendering results...')

    artist = best_corpus_match[0]
    song = best_corpus_match[1]
    descr = ' '.join(best_corpus_match[2:4])
    lyr = best_corpus_match[-2]
    score = best_corpus_match[-1]
    
    print('=' * 70)
    print('Sample INTs', score[:3])
    print('=' * 70)

    title = song + ' by ' + artist
    fn1 = title.replace(' ', "_")
    
    output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(score)

    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(output_score,
                                                              output_signature = '"' + song + '" by ' + artist,
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

    output_midi_title = str(title)
    output_midi_summary = str(descr)
    output_midi = str(new_fn)
    output_audio = (16000, audio)
    output_midi_lyrics = str(lyr)

    for o in output_score:
        o[1] *= 16
        o[2] *= 16
    
    output_plot = TMIDIX.plot_ms_SONG(output_score, plot_title=output_midi_title, return_plt=True)

    print('Output MIDI file name:', output_midi)
    print('Output MIDI title:', output_midi_title)
    print('Output MIDI summary:', output_midi_summary)
    print('=' * 70) 
    

    #========================================================
    
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')

    return output_midi_title, output_midi_summary, output_midi, output_audio, output_plot, output_midi_lyrics

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)

    soundfont = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"

    print('Loading clean_midi corpus...')

    clean_midi_artist_song_description_summaries_lyrics_score = TMIDIX.Tegridy_Any_Pickle_File_Reader('clean_midi_artist_song_description_summaries_lyrics_scores')

    print('Done!')
    print('=' * 70)

    print('Loading clean_midi corpus embeddings...')
    
    corpus_embeddings = np.load('clean_midi_corpus_embeddings_all_mpnet_base_v2.npz')['data']
    
    print('Done!')
    print('=' * 70)

    print('Loading Sentence Transformer model...')
    
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print('Done!')
    print('=' * 70)
    
    app = gr.Blocks()
    
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Music Sentence Transformer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Discover and search MIDI music with sentence transformer</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Music-Sentence-Transformer&style=flat)\n\n"
            "This is a demo for TMIDIX Python module from tegridy-tools\n\n"
            "Check out [tegridy-tools](https://github.com/asigalov61/tegridy-tools) on GitHub!\n\n"
        )
        gr.Markdown("## Enter any desired search query\n"
                    "### You can enter song description, music description, or both")
        
        input_text = gr.Textbox(value="The song about yesterday with Grand Piano lead and drums", interactive=True, label='Search query')

        run_btn = gr.Button("search", variant="primary")

        gr.Markdown("## Output results")

        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_midi_lyrics = gr.Textbox(label="Output MIDI lyrics")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])

        run_event = run_btn.click(SearchMIDIs, [input_text],
                                  [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot, output_midi_lyrics])
        
        app.queue().launch()