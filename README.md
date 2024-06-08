# tegridy-tools

***

## Symbolic music artificial Intelligence toolkit for rapid prototyping, design and evaluation of the symbolic music AI architectures, systems and models

***

# Install

## Recommended and required for the latest version of the modules/files: 

```
!git clone --depth 1 https://github.com/asigalov61/tegridy-tools
```

***

## Hassle-free automatic install , setup, and all core modules imports

```
!curl -O https://raw.githubusercontent.com/asigalov61/tegridy-tools/main/install/setup.sh
# !wget https://raw.githubusercontent.com/asigalov61/tegridy-tools/main/install/setup.sh
!chmod +x setup.sh
!bash setup.sh
!rm setup.sh
```

***

## After install you can import/re-import modules like so...

### Standard Python auto-imports/re-imports

```
# CPU setup and imports
!python ./tegridy-tools/install/setup.py
```

```
# GPU setup and imports
!python ./tegridy-tools/install/setup_cuda.py
```

### Google Colab auto-imports/re-imports

```
# CPU setup and imports
%run ./tegridy-tools/install/setup.py
```

```
# GPU setup and imports
%run ./tegridy-tools/install/setup_cuda.py
```

### Manual imports/re-imports

```
import TMIDIX
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
@inproceedings{lev2024tegridytools,
    title       = {tegridy-tools: Symbolic Music NLP Artificial Intelligence Toolkit},
    author      = {Aleksandr Lev},
    booktitle   = {GitHub},
    year        = {2024},
}
```
***

### Project Los Angeles
### Tegridy Code 2024
