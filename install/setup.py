print('=' * 70)
print('Setting up tegridy-tools...Please wait...')

import sys
import os

sys.path.insert(0,os.getcwd()+'/tegridy-tools')
sys.path.insert(0,os.getcwd()+'/tegridy-tools/tegridy-tools')
sys.path.insert(0,os.getcwd()+'/tegridy-tools/tegridy-data')
sys.path.insert(0,os.getcwd()+'/tegridy-tools/tegridy-tools/X-Transformer')

import pickle
from collections import Counter
from itertools import groupby
import secrets
import tqdm
import math
import statistics
import copy
import shutil
import pprint
import wave

from joblib import Parallel, delayed, parallel_config

import matplotlib.pyplot as plt
import numpy as np

import networkx as nx

from sklearn.manifold import TSNE
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from scipy.ndimage import zoom
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import zscore

import TMIDIX
import TPLOTS
import TMELODIES
import HaystackSearch
import midi_to_colab_audio

import random

from IPython.display import display, Audio

import locale
locale.getpreferredencoding = lambda: "UTF-8"

print('=' * 70)
print('Done!')
print('Enjoy!')
print('=' * 70)
