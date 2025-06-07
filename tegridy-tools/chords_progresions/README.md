# Pitches Chords Progressions Generator Python Module

***

## Requirements

```sh
!git clone --depth 1 https://github.com/asigalov61/tegridy-tools

!pip install numpy
!pip install huggingface_hub
!pip install tqdm
```

***

## Simple use example

```python
from pitches_chords_progressions_generator import *

dataset_file_path = download_dataset()
chords_chunks_np_array, chords_chunks_data = load_dataset(dataset_file_path)

stats = Generate_Chords_Progression(chords_chunks_np_array, chords_chunks_data)
```

***

## Project links

### [Pitches Chords Progressions Generator LIVE demo on Hugging Face](https://huggingface.co/spaces/asigalov61/Chords-Progressions-Generator)
### [Pitches Chords Progressions Generator Samples on SoundCloud](https://soundcloud.com/aleksandr-sigalov-61/sets/pitches-chords-progressions)
### [Pitches Chords Progressions in Tegridy MIDI Dataset](https://github.com/asigalov61/Tegridy-MIDI-Dataset/tree/master/Chords-Progressions)

***

### Project Los Angeles
### Tegridy Code 2025
