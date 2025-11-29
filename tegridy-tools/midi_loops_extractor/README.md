# MIDI Loops Extractor
## Detect and extract loops from MIDIs

***

### Source code is courtesy of [GIGA MIDI](https://github.com/GigaMidiDataset/The-GigaMIDI-dataset-with-loops-and-expressive-music-performance-detection)

### Source code was retrieved on 11/22/2024

***

## Requirements

```sh
pip install pretty-midi
pip install symusic
pip install miditok
pip install numba
pip install numpy==1.24.4
```

***

## Basic use

### Extract loops info

```python
import os

# Set desired environment variables
os.environ["USE_NUMBA"] = "1"
os.environ["USE_CUDA"] = "1"

# Check the variables
print(os.environ["USE_NUMBA"])
print(os.environ["USE_CUDA"])

from process_file_fast import detect_loops_from_path

midi_file = './tegridy-tools/tegridy-tools/seed-lyrics.mid'

result = detect_loops_from_path({'file_path': [midi_file]})
```

### Save loop as MIDI

```python
from symusic import Score

output_loop_MIDI = './output_loop_MIDI.mid'

select_loop_idx = 0

score = Score({'file_path': [midi_file]})

raw_loop = score.clip(result['start'][select_loop_idx], result['end'][select_loop_idx])

zero_start_time_loop = raw_loop.shift_time(-result['start'][select_loop_idx])

zero_start_time_loop.dump_midi(output_loop_MIDI)
```

***

### Project Los Angeles
### Tegridy Code 2025
