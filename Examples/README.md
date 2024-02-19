# tegridy-tools usage examples

***

### Here is a full code snippet for TMIDIX modules main functions

```
input_midi = './tegridy-tools/tegridy-tools/seed2.mid'

raw_score = TMIDIX.midi2single_track_ms_score(input_midi)

escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]

escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes)

cscore = TMIDIX.chordify_score([1000, escore_notes])

melody = TMIDIX.fix_monophonic_score_durations(TMIDIX.extract_melody(cscore))
```

***

### Project Los Angeles
### Tegridy Code 2024
