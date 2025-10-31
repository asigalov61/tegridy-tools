# Melody Checker
## Check any MIDI melody for completeness

***

## Dependencies

```sh
!pip install -U joblib
!pip install numpy==1.24.4
!pip install -U scikit-learn
```

***

## Basic use example

```python
import melody_checker

# Load default pre-trained model
# PLEASE NOTE: Default pre-trained model was trained on melody sequences 24-64 notes
# So make sure that your sequences are 24-64 notes (72-192 tokens) long for optimal results
# Also, make sure that you feed sequnces that are multiples of 3 (i.e 72 tokens/24 notes long)
pipeline = melody_checker.load_pipeline("melody_checker.joblib")

# Melody sequence as a list of flattened triplets
# Triplet format: [delta_start_time, duration+128, midi_pitch+256]
# dtime and duration range is 0-127 (4096ms / 32)
melody_seq = [0, 198, 328, 82, 139, 323, 11, 151, 328, 24, 145, 323, 17, 133, 325, 6, 151,
              327, 24, 139, 320, 11, 139, 320, 12, 151, 325, 23, 145, 323, 18, 133, 321, 6,
              151, 323, 23, 139, 316, 12, 139, 316, 12, 151, 318, 23, 139, 318, 12, 139, 320,
              12, 151, 321, 23, 139, 321, 12, 139, 323, 12, 151, 325, 23, 139, 327, 12, 139,
              328, 11, 163, 330, 36, 139, 323, 11, 151, 332, 24, 145, 330, 17, 133, 328, 6,
              151, 330, 24, 139, 327, 11, 139, 323, 12, 151, 328, 23, 145, 327, 18, 133, 325,
              6, 151, 327, 23, 139, 320, 12, 139, 320, 12, 151, 325, 23, 139, 323, 12, 139,
              321, 12, 151, 323, 23, 139, 316, 12, 139, 316, 12, 151, 328, 23, 145, 327, 18,
              133, 325, 5, 174, 323
             ]

# This example melody sequence should return decision == True
res = melody_checker.infer_with_rules_loaded(pipeline, melody_seq)

print(res)
```

***

### Project Los Angeles
### Tegridy Code 2025
