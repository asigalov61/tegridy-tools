# tegridy-tools

***

## Symbolic music artificial Intelligence toolkit for rapid prototyping, design and evaluation of the symbolic music AI architectures, systems and models

![Tegridy-Tools-Logo](https://github.com/user-attachments/assets/25539aeb-c4fa-4a84-ae43-f73c1db3fe9b)

***

# Install

## Recommended and required for the latest version of the modules/files: 

```sh
!git clone --depth 1 https://github.com/asigalov61/tegridy-tools
```

***

## Hassle-free automatic install , setup, and all core modules imports

```sh
!curl -O https://raw.githubusercontent.com/asigalov61/tegridy-tools/main/install/setup.sh
# !wget https://raw.githubusercontent.com/asigalov61/tegridy-tools/main/install/setup.sh
!chmod +x setup.sh
!bash setup.sh
!rm setup.sh
```

***

## After install you can import/re-import modules like so...

### Standard Python auto-imports/re-imports

```sh
# CPU setup and imports
!python ./tegridy-tools/install/setup.py
```

```sh
# GPU setup and imports
!python ./tegridy-tools/install/setup_cuda.py
```

### Google Colab auto-imports/re-imports

```sh
# CPU setup and imports
%run ./tegridy-tools/install/setup.py
```

```sh
# GPU setup and imports
%run ./tegridy-tools/install/setup_cuda.py
```

### Manual imports/re-imports

```python
import os
import copy
import math
import statistics
import pickle
import shutil
from itertools import groupby
from collections import Counter
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed, parallel_config
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import TMIDIX
import TPLOTS
import TMELODIES
import HaystackSearch
import midi_to_colab_audio
from x_transformer_1_23_2 import *

import random
```

***

# Quick Start Guide

## Start by checking out [example code snippets and scripts](https://github.com/asigalov61/tegridy-tools/tree/main/Examples)

## Or you can check out many practical, specific and detailed [Jupyter/Google Colab Notebooks](https://github.com/asigalov61/tegridy-tools/tree/main/tegridy-tools/notebooks)

***

# Detailed core modules documentation

## [tegridy-tools docs](https://github.com/asigalov61/tegridy-tools/tree/main/docs)

***

```bibtex
@inproceedings{lev2026tegridytools,
    title       = {tegridy-tools: Symbolic Music NLP Artificial Intelligence Toolkit},
    author      = {Aleksandr Lev},
    booktitle   = {GitHub},
    year        = {2026},
}
```
***

### Project Los Angeles
### Tegridy Code 2026
