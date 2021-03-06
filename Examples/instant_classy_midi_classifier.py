# -*- coding: utf-8 -*-
"""Instant_Classy_MIDI_Classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1opiN1uC06IWGlja9WI_iVkvYIkOKJSjQ

# Instant Classy (ver 1.0)

***

## Very fast and precise MIDI classifier

***

### Powered by FuzzyWuzzy: https://github.com/seatgeek/fuzzywuzzy

***

#### Classification is done according to the LAKH dataset: https://colinraffel.com/projects/lmd/

#### Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.

***

#### Project Los Angeles
#### Tegridy Code 2021

***

# Setup environment
"""

#@title Install dependencies
!git clone https://github.com/asigalov61/tegridy-tools
!pip install fuzzywuzzy[speedup]

# Commented out IPython magic to ensure Python compatibility.
#@title Import needed modules

# %cd /content/tegridy-tools/tegridy-tools
import TMIDI
# %cd /content

import pickle
import gzip

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from tqdm import auto

"""# Classify"""

#@title Prep classifier data and load your MIDI

full_path_to_MIDI_to_classify = "/content/tegridy-tools/tegridy-tools/seed2.mid" #@param {type:"string"}

print('=' * 70)
print('Loading data...')
print('=' * 70)

# Loading signatures and target MIDI
data = pickle.load(gzip.open('./tegridy-tools/tegridy-data/Instant_Classy_LAKH_MIDI_Signatures_Pack.pickle.gz'))

print('Total number of loaded signatures:', data[5])
text, melody, chords = TMIDI.Optimus_MIDI_TXT_Processor(full_path_to_MIDI_to_classify, MIDI_channel = 16, MIDI_patch=range(0,127))

# prepping data
melody_list_f = []
chords_list_f = []

# melody
m_st_sum = sum([y[1] for y in melody])
m_st_avg = int(m_st_sum / len(melody))
m_du_sum = sum([y[2] for y in melody])
m_du_avg = int(m_du_sum / len(melody))
m_ch_sum = sum([y[3] for y in melody])
m_ch_avg = int(m_ch_sum / len(melody))
m_pt_sum = sum([y[4] for y in melody])
m_pt_avg = int(m_pt_sum / len(melody))
m_vl_sum = sum([y[5] for y in melody])
m_vl_avg = int(m_vl_sum / len(melody))
melody_list_f.append([m_st_sum, 
                      m_st_avg,
                      m_du_sum,
                      m_du_avg,
                      m_ch_sum,
                      m_ch_avg,
                      m_pt_sum,
                      m_pt_avg,
                      m_vl_sum,
                      m_vl_avg])

# chords
c_st_sum = sum([y[1] for y in chords])
c_st_avg = int(c_st_sum / len(chords))
c_du_sum = sum([y[2] for y in chords])
c_du_avg = int(c_du_sum / len(chords))
c_ch_sum = sum([y[3] for y in chords])
c_ch_avg = int(c_ch_sum / len(chords))
c_pt_sum = sum([y[4] for y in chords])
c_pt_avg = int(c_pt_sum / len(chords))
c_vl_sum = sum([y[5] for y in chords])
c_vl_avg = int(c_vl_sum / len(chords))

chords_list_f.append([c_st_sum, 
                      c_st_avg,
                      c_du_sum,
                      c_du_avg,
                      c_ch_sum,
                      c_ch_avg,
                      c_pt_sum,
                      c_pt_avg,
                      c_vl_sum,
                      c_vl_avg])

print('=' * 70)
print('Input MIDI file name:', full_path_to_MIDI_to_classify)
print('=' * 70)
print('Melody:',melody_list_f[0])
print('Chords:',chords_list_f[0])
print('=' * 70)

#@title Classify your MIDI

#@markdown Match Points: 

#@markdown 1-2 = Start Time (sum/avg)

#@markdown 3-4 = Duration (sum/avg)

#@markdown 5-6 = Channel (sum/avg)

#@markdown 7-8 = Pitch (sum/avg)

#@markdown 9-10 = Velocity (sum/avg)


number_of_match_points = 10 #@param {type:"slider", min:1, max:10, step:1}

print('=' * 70)
print('Instant Classy MIDI Classifier')
print('=' * 70)
print('Classifying...')
print('Please wait...')
print('=' * 70)
print('Number of requested match points', number_of_match_points)

# melody

ratings = []
for m in auto.tqdm(data[1]):
  ratings.append(fuzz.ratio(melody_list_f[0][:number_of_match_points], m[:number_of_match_points]))

print('=' * 70)
print('Melody match:')
print('Rating:', max(ratings))
print('Match:', data[3][ratings.index(max(ratings))][2])
print('MIDI:', data[3][ratings.index(max(ratings))][1])
print('Hash:', data[3][ratings.index(max(ratings))][0])
print('Data', data[1][ratings.index(max(ratings))])

# chords

ratings = []
for c in auto.tqdm(data[0]):
  ratings.append(fuzz.ratio(chords_list_f[0][:number_of_match_points], c[:number_of_match_points]))

print('=' * 70)
print('Melody and Chords match:')
print('Rating:', max(ratings))
print('Match:', data[3][ratings.index(max(ratings))][2])
print('MIDI:', data[3][ratings.index(max(ratings))][1])
print('Hash:', data[3][ratings.index(max(ratings))][0])
print('Data', data[0][ratings.index(max(ratings))])

print('=' * 70)

"""# Congrats! You did it! :)"""