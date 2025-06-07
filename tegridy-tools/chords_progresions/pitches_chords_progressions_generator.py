r'''###############################################################################
###################################################################################
#
#
#	Pitches Chords Progressions Generator Python module
#
#	Version 1.0
#
#	Project Los Angeles
#	Tegridy Code 2025
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
###################################################################################
# 
#   Critical requirements:
#
#   !git clone --depth 1 https://github.com/asigalov61/tegridy-tools
#
#   !pip install numpy
#   !pip install huggingface_hub
#   !pip install tqdm
#
###################################################################################
# 
#   Simple use example:
#
#   from pitches_chords_progressions_generator import *
#
#   dataset_file_path = download_dataset()
#   chords_chunks_np_array, chords_chunks_data = load_dataset(dataset_file_path)
#
#   stats = Generate_Chords_Progression(chords_chunks_np_array, chords_chunks_data)
#
###################################################################################
#
#   Project links
#
#   https://huggingface.co/spaces/asigalov61/Chords-Progressions-Generator
#   https://soundcloud.com/aleksandr-sigalov-61/sets/pitches-chords-progressions
#   https://github.com/asigalov61/Tegridy-MIDI-Dataset/tree/master/Chords-Progressions
#   https://github.com/asigalov61/tegridy-tools/tree/main/tegridy-tools/chords_progresions
#
###################################################################################
#
#   Copyright 2025 Project Los Angeles / Tegridy Code
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
###################################################################################
'''

###################################################################################

__version__ = "1.0.0"

###################################################################################

print('=' * 70)
print('Pitches Chords Progressions Generator Python module')
print('Version:', __version__)
print('=' * 70)

###################################################################################

print('=' * 70)
print('Loading needed modules...')
print('=' * 70)

import os
import sys
import random
from collections import Counter
import math

import time as reqtime
import datetime
from pytz import timezone

import numpy as np

sys.path.append('./tegridy-tools/tegridy-tools/')

import TMIDIX

from huggingface_hub import hf_hub_download

print('=' * 70)
print('Done!')
print('=' * 70)

###################################################################################

def download_dataset(repo_id='asigalov61/Chords-Progressions-Generator',
                     repo_type='space',
                     filename='processed_chords_progressions_chunks_data.pickle',
                     local_dir='./',
                     verbose=True
                    ):

    '''
    Pitches Chords Progressions dataset downloader

    Function options:
        repo_id: Hugging Face repo ID
        repo_type: Hugging Face repo type
        filename: Pitches Chords Progressions dataset file name in the specified repo
        local_dir: Local dir to store the dataset at
        verbose: Verbosity option

    Function returns:
    
        Full path to the processed Pitches Chords Progressions dataset 
    '''
    
    if verbose:
        print('=' * 70)
        print('Downloading processed Pitches Chords Progressions dataset...')
        print('=' * 70)
        
    dataset_file_path = hf_hub_download(repo_id=repo_id,
                                        repo_type=repo_type,
                                        filename=filename,
                                        local_dir=local_dir                                        
                                        )

    if verbose:
        print('=' * 70)
        print('Done!')
        print('=' * 70)
    
    return dataset_file_path

###################################################################################

def load_dataset(dataset_file_path, chords_chunks_size=6, verbose=True):

    '''
    Pitches Chords Progressions dataset loader

    Function options:

        dataset_file_path: Full path to the processed Pitches Chords Progressions dataset
        chords_chunks_size: Number of chords in each chords chunk (min 3 and max 8 chords per chunk)
        verbose: Verbosity option

    Function returns:
    
        Chords chunks NumPy array
        Chords chunks data
    '''

    if verbose:
        print('=' * 70)
        print('Loading processed Pitches Chords Progressions dataset...')
        print('=' * 70)

    chords_chunks_data = TMIDIX.Tegridy_Any_Pickle_File_Reader(dataset_file_path, verbose=verbose)
    long_tones_chords_dict, all_long_chords_tokens_chunks, all_long_good_chords_chunks = chords_chunks_data

    if verbose:
        print('=' * 70)
        print('Resulting chords dictionary size:', len(long_tones_chords_dict))
        print('=' * 70)
        print('Loading chords chunks...')
    
    chords_chunks_np_array = np.array([a[:max(3, min(8, chords_chunks_size))] for a in all_long_chords_tokens_chunks])

    if verbose:
        print('Done!')
        print('=' * 70)
        print('Total chords chunks count:', len(all_long_chords_tokens_chunks))
        print('=' * 70)

    return chords_chunks_np_array, chords_chunks_data
    
###################################################################################

def Generate_Chords_Progression(chords_chunks_np_array,
                                chords_chunks_data,
                                output_MIDI_file_name="./Pitches-Chords-Progression-Composition",
                                minimum_song_length_in_chords_chunks=30,
                                chords_chunks_memory_ratio=1,
                                chord_time_step=250,
                                merge_chords_notes=2000,
                                melody_MIDI_patch_number=40,
                                chords_progression_MIDI_patch_number=0,
                                base_MIDI_patch_number=35,
                                add_drums=True,
                                return_MIDI_score=False,
                                return_MIDI_stats=False,
                                return_chords_stats=False,
                                verbose=True
                               ):

    '''
    Chords progression generator

    Function options:
    
        chords_chunks_np_array: Chords chunks Numpy Array
        chords_chunks_data: Chords chunks data
        output_MIDI_file_name: Output MIDI file name without extension
        minimum_song_length_in_chords_chunks: Minimum song length in chords chunks
        chords_chunks_memory_ratio: Chords chunks memory ratio
        chord_time_step: Chord time step in ms
        merge_chords_notes: Merged chords notes max time
        melody_MIDI_patch_number: Melody MIDI patch number
        chords_progression_MIDI_patch_number: Chords progression MIDI patch number
        base_MIDI_patch_number: Base MIDI patch number
        add_drums: Add drum track
        return_MIDI_score: Return final chords progression MIDI score
        return_MIDI_stats: Return MIDI file stats
        return_chords_stats: Return chords progression stats string
        verbose: Verbosity option

    Function returns:

        List of specified return options
    '''

    #==================================================================

    PDT = timezone('US/Pacific')

    if verbose:
        print('=' * 70)
        print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    #==================================================================

    chunk_size = chords_chunks_np_array.shape[1]

    src_long_chunks = chords_chunks_np_array

    long_tones_chords_dict, all_long_chords_tokens_chunks, all_long_good_chords_chunks = chords_chunks_data 

    #==================================================================

    if verbose:
        print('=' * 70)
        print('Requested settings:')
        print('-' * 70)
        print('Chords chunks length:', chunk_size)
        print('Output MIDI file name:', output_MIDI_file_name)
        print('Minimum song length in chords chunks:', minimum_song_length_in_chords_chunks)
        print('Chords chunks memory ratio:', chords_chunks_memory_ratio)
        print('Chord time step:', chord_time_step)
        print('Merge chords notes max time:', merge_chords_notes)
        print('Melody MIDI patch number:', melody_MIDI_patch_number)
        print('Chords progression MIDI patch number:', chords_progression_MIDI_patch_number)
        print('Base MIDI patch number:', base_MIDI_patch_number)
        print('Add drum track:', add_drums)
        print('Return MIDI score:', return_MIDI_score)
        print('Return MIDI stats:', return_MIDI_stats)
        print('Return chords stats:', return_chords_stats)
        print('-' * 70)
        
    #==================================================================
    
        print('=' * 70)
        print('Pitches Chords Progressions Generator')
        print('=' * 70)

    #==================================================================

        print('=' * 70)
        print('Chunk-by-chunk generation')
        print('=' * 70)
        print('Generating...')
        print('=' * 70)

    #==================================================================
    
    matching_long_chords_chunks = []
    
    ridx = random.randint(0, chords_chunks_np_array.shape[0]-1)
    
    matching_long_chords_chunks.append(ridx)
    
    max_song_len = 0
    
    tries = 0
    
    while len(matching_long_chords_chunks) < minimum_song_length_in_chords_chunks:

        matching_long_chords_chunks = []
    
        ridx = random.randint(0, chords_chunks_np_array.shape[0]-1)
    
        matching_long_chords_chunks.append(ridx)
        seen = [ridx]
        gseen = [ridx]
    
        for a in range(minimum_song_length_in_chords_chunks * 10):
    
          if not matching_long_chords_chunks:
            break
    
          if len(matching_long_chords_chunks) > minimum_song_length_in_chords_chunks:
            break
    
          schunk = all_long_chords_tokens_chunks[matching_long_chords_chunks[-1]]
          trg_long_chunk = np.array(schunk[-chunk_size:])
          idxs = np.where((src_long_chunks == trg_long_chunk).all(axis=1))[0].tolist()
    
          if len(idxs) > 1:
    
            random.shuffle(idxs)
    
            eidxs = [i for i in idxs if i not in seen]
    
            if eidxs:
              eidx = eidxs[0]
              matching_long_chords_chunks.append(eidx)
              seen.append(eidx)
              gseen.append(eidx)
    
              if 0 < chords_chunks_memory_ratio < 1:
                seen = random.choices(gseen, k=math.ceil(len(gseen) * chords_chunks_memory_ratio))
              elif chords_chunks_memory_ratio == 0:
                seen = []
    
            else:
                gseen.pop()
                matching_long_chords_chunks.pop()
    
          else:
            gseen.pop()
            matching_long_chords_chunks.pop()
    
    
        if len(matching_long_chords_chunks) > max_song_len:

            if verbose:
                print('Current song length:', len(matching_long_chords_chunks), 'chords chunks')
                print('=' * 70)
            final_song = matching_long_chords_chunks
    
        max_song_len = max(max_song_len, len(matching_long_chords_chunks))
    
        tries += 1
    
        if tries % 500 == 0:
            if verbose:
                print('Number of passed tries:', tries)
                print('=' * 70)

    if len(matching_long_chords_chunks) > max_song_len:
        if verbose:
            print('Current song length:', len(matching_long_chords_chunks), 'chords chunks')
            print('=' * 70)
        final_song = matching_long_chords_chunks
    
    f_song = []
    
    for mat in final_song:
        f_song.extend(all_long_good_chords_chunks[mat][:-chunk_size])
    f_song.extend(all_long_good_chords_chunks[mat][-chunk_size:])

    if verbose:
        print('Generated final song after', tries, 'tries with', len(final_song), 'chords chunks and', len(f_song), 'chords')
        print('=' * 70)
    
        print('Done!')
        print('=' * 70)
    
    #===============================================================================

    if verbose:
        print('Rendering results...')
        print('=' * 70)

    output_score = []
    
    time = 0
    
    patches = [0] * 16
    patches[0] = chords_progression_MIDI_patch_number
    
    if base_MIDI_patch_number > -1:
      patches[2] = base_MIDI_patch_number
    
    if melody_MIDI_patch_number > -1:
      patches[3] = melody_MIDI_patch_number
    
    chords_labels = []
    
    for i, s in enumerate(f_song):
    
      time += chord_time_step
    
      dur = chord_time_step
    
      chord_str = str(i+1)
    
      for t in sorted(set([t % 12 for t in s])):
        chord_str += '-' + str(t)
    
      chords_labels.append(['text_event', time, chord_str])
    
      for p in s:
        output_score.append(['note', time, dur, 0, p, max(40, p), chords_progression_MIDI_patch_number])
                    
      if base_MIDI_patch_number > -1:
        output_score.append(['note', time, dur, 2, (s[-1] %  12)+24, 120-(s[-1] %  12), base_MIDI_patch_number])
    
    if melody_MIDI_patch_number > -1:
      output_score = TMIDIX.add_melody_to_enhanced_score_notes(output_score, 
                                                               melody_patch=melody_MIDI_patch_number,
                                                               melody_notes_max_duration=max(merge_chords_notes, chord_time_step)
                                                              )

    if merge_chords_notes > 0:
        escore_matrix = TMIDIX.escore_notes_to_escore_matrix(output_score)
        output_score = TMIDIX.escore_matrix_to_merged_escore_notes(escore_matrix, max_note_duration=merge_chords_notes)

    if add_drums:
        output_score = TMIDIX.augment_enhanced_score_notes(output_score)
        output_score = TMIDIX.advanced_add_drums_to_escore_notes(output_score)

        for e in output_score:
            e[1] *= 16
            e[2] *= 16

    midi_score = sorted(chords_labels + output_score, key=lambda x: x[1])

    return_list = []

    if return_MIDI_score:
        return_list.append(midi_score)
    
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(midi_score,
                                                              output_signature = 'Pitches Chords Progression',
                                                              output_file_name = output_MIDI_file_name,
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches,
                                                              verbose=verbose
                                                              )

    if return_MIDI_stats:
        return_list.append(detailed_stats)

    #===============================================================================
    if verbose:
        print('=' * 70)
        print('Generated chords progression info and stats:')
        print('=' * 70)

    chords_progression_summary_string = '=' * 70
    chords_progression_summary_string += '\n'

    all_song_chords = []
    
    for pc in f_song:
      tones_chord = tuple(sorted(set([p % 12 for p in pc])))
      all_song_chords.append([pc, tones_chord])
        
    if verbose:
        print('=' * 70)
        print('Total number of chords:', len(all_song_chords))
    chords_progression_summary_string += 'Total number of chords: ' + str(len(all_song_chords)) + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'

    if verbose:
        print('=' * 70)
        print('Most common pitches chord:', list(Counter(tuple([a[0] for a in all_song_chords])).most_common(1)[0][0]), '===', Counter(tuple([a[0] for a in all_song_chords])).most_common(1)[0][1], 'count')
    chords_progression_summary_string += 'Most common pitches chord: ' + str(list(Counter(tuple([a[0] for a in all_song_chords])).most_common(1)[0][0])) + ' === ' + str(Counter(tuple([a[0] for a in all_song_chords])).most_common(1)[0][1]) + ' count' + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'

    if verbose:
        print('=' * 70)
        print('Most common tones chord:', list(Counter(tuple([a[1] for a in all_song_chords])).most_common(1)[0][0]), '===', Counter(tuple([a[1] for a in all_song_chords])).most_common(1)[0][1], 'count')
    chords_progression_summary_string += 'Most common tones chord: ' + str(list(Counter(tuple([a[1] for a in all_song_chords])).most_common(1)[0][0])) + ' === ' + str(Counter(tuple([a[1] for a in all_song_chords])).most_common(1)[0][1]) + ' count' + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'

    if verbose:
        print('=' * 70)
        print('Sorted unique songs chords set:', len(sorted(set(tuple([a[1] for a in all_song_chords])))), 'count')
    chords_progression_summary_string += 'Sorted unique songs chords set: ' + str(len(sorted(set(tuple([a[1] for a in all_song_chords]))))) +  ' count' + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'
    for c in sorted(set(tuple([a[1] for a in all_song_chords]))):
        chords_progression_summary_string += str(list(c)) + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'

    if verbose:
        print('=' * 70)
        print('Grouped songs chords set:', len(TMIDIX.grouped_set(tuple([a[1] for a in all_song_chords]))), 'count')
    chords_progression_summary_string += 'Grouped songs chords set: ' + str(len(TMIDIX.grouped_set(tuple([a[1] for a in all_song_chords])))) + ' count' + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'

    if verbose:
        print('=' * 70)
    for c in TMIDIX.grouped_set(tuple([a[1] for a in all_song_chords])):
        chords_progression_summary_string += str(list(c)) + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'
    chords_progression_summary_string += 'All songs chords' + '\n'
    chords_progression_summary_string += '=' * 70
    chords_progression_summary_string += '\n'

    for i, pc_tc in enumerate(all_song_chords):
        chords_progression_summary_string += 'Song chord # ' + str(i) + '\n'
        chords_progression_summary_string += str(list(pc_tc[0])) + ' === ' + str(list(pc_tc[1])) + '\n'
        chords_progression_summary_string += '=' * 70
        chords_progression_summary_string += '\n'

    if return_chords_stats:
        return_list.append(chords_progression_summary_string)

    #===============================================================================

    if verbose:
        print('-' * 70)
        print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
        print('-' * 70)
        print('Req execution time:', (reqtime.time() - start_time), 'sec')
        print('=' * 70)

    #===============================================================================

    return return_list
    
###################################################################################
# This is the end of Pitches Chords Progressions Generator Python module
###################################################################################