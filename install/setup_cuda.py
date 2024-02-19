print('=' * 70)
print('Setting up tegridy-tools...Please wait...')

import sys
import os

sys.path.insert(0,'~/tegridy-tools')
sys.path.insert(0,'~/tegridy-tools/tegridy-tools')
sys.path.insert(0,'~/tegridy-tools/tegridy-data')
sys.path.insert(0,'~/tegridy-tools/tegridy-tools/X-Transformer')

import pickle
from collections import Counter
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

import TMIDIX
import midi_to_colab_audio

import torch
import cupy as cp

from x_transformer_1_23_2 import *

import random

import locale
locale.getpreferredencoding = lambda: "UTF-8"

print('=' * 70)
print('Done!')
print('Enjoy!')
print('=' * 70)