# Melody Checker
## Check any MIDI melody for completeness

***

## Dependencies

```sh
!pip install numpy==1.24.4
!pip install -U scikit-learn
```

***

## Basic use example

```python
import melody_checker

pipeline = melody_checker.load_pipeline("melody_checker.joblib")

# Melody sequence as a list of flattened triplets
# Triplet format: [delta_start_time, duration+128, midi_pitch+256]
# dtime and duration range is 0-127 (4000ms / 32)

melody_seq = [0, 128, 256, 2, 135, 260, ...]

res = melody_checker.infer_with_rules_loaded(pipeline, melody_seq)

print(res)
```

***

### Project Los Angeles
### Tegridy Code 2025
