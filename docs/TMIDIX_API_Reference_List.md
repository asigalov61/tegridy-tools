# TMIDIX API Reference List
## All TMIDIX functions listed in alphabetical order

---

## 📚 Table of Contents

- [#](##)
- [A](#a)
- [B](#b)
- [C](#c)
- [D](#d)
- [E](#e)
- [F](#f)
- [G](#g)
- [H](#h)
- [I](#i)
- [J](#j)
- [L](#l)
- [M](#m)
- [N](#n)
- [O](#o)
- [P](#p)
- [Q](#q)
- [R](#r)
- [S](#s)
- [T](#t)
- [V](#v)
- [W](#w)

---

## #
<a name="#"></a>

**27 entries**

<details>
<summary>Show entries</summary>

* `_auto_clusters`
* `_ber_compressed_int`
* `_candidates_in_span`
* `_choose_fixed_type`
* `_choose_rep_from_cluster`
* `_clean_up_warnings`
* `_consistentise_ticks`
* `_decode`
* `_encode`
* `_fits_in_signed`
* `_fits_in_unsigned`
* `_int2twobytes`
* `_mad`
* `_median`
* `_read_14_bit`
* `_read_varint_from_bytearray`
* `_rep`
* `_round_int`
* `_some_text_event`
* `_twobytes2int`
* `_unshift_ber_int`
* `_viterbi_with_jump_penalty`
* `_warn`
* `_write_14_bit`
* `_write_varint_to_bytearray`
* `_zigzag_decode`
* `_zigzag_encode`

</details>

---

## A
<a name="a"></a>

**29 entries**

<details>
<summary>Show entries</summary>

* `add_arrays`
* `add_base_to_escore_notes`
* `add_drums_to_escore_notes`
* `add_expressive_melody_to_enhanced_score_notes`
* `add_melody_to_enhanced_score_notes`
* `add_smooth_expressive_melody_to_enhanced_score_notes`
* `add_smooth_melody_to_enhanced_score_notes`
* `adjust_escore_notes_timings`
* `adjust_escore_notes_to_average`
* `adjust_list_of_values_to_target_average`
* `adjust_numbers_to_sum`
* `adjust_score_velocities`
* `advanced_add_drums_to_escore_notes`
* `advanced_align_escore_notes_to_bars`
* `advanced_check_and_fix_chords_in_chordified_score`
* `advanced_check_and_fix_tones_chord`
* `advanced_score_processor`
* `advanced_validate_chord_pitches`
* `align_escore_notes_to_bars`
* `align_escore_notes_to_escore_notes`
* `align_integer_lists`
* `all_consequtive`
* `alpha_str`
* `alpha_str_to_toks`
* `analyze_score_pitches`
* `apply_sustain_to_ms_score`
* `ascii_text_words_counter`
* `ascii_texts_search`
* `augment_enhanced_score_notes`

</details>

---

## B
<a name="b"></a>

**14 entries**

<details>
<summary>Show entries</summary>

* `bad_chord`
* `basic_enhanced_delta_score_notes_detokenizer`
* `basic_enhanced_delta_score_notes_tokenizer`
* `binary_matrix_to_original_escore_notes`
* `binary_matrix_to_rle_toks`
* `binary_rle_decoder`
* `binary_rle_encoder`
* `bits_to_int`
* `bits_to_tones_chord`
* `both_chords`
* `bpe_decode`
* `bpe_encode`
* `build_lcp_array`
* `build_suffix_array`

</details>

---

## C
<a name="c"></a>

**40 entries**

<details>
<summary>Show entries</summary>

* `calculate_combined_distances`
* `ceil_with_precision`
* `check_and_fix_chord`
* `check_and_fix_chords_in_chordified_score`
* `check_and_fix_pitches_chord`
* `check_and_fix_tones_chord`
* `check_monophonic_melody`
* `chord_cost`
* `chord_to_pchord`
* `chordified_score_pitches`
* `chordify_score`
* `chords_common_tones_chain`
* `chords_to_escore_notes`
* `chunk_by_threshold_mode`
* `chunk_list`
* `chunks_shuffle`
* `clean_string`
* `common_subpatterns`
* `compress_binary_matrix`
* `compress_patches_in_escore_notes`
* `compress_patches_in_escore_notes_chords`
* `compress_tokens_sequence`
* `computeLPSArray`
* `compute_base`
* `compute_sustain_intervals`
* `concat_cols`
* `concat_rows`
* `convert_bytes_in_nested_list`
* `convert_escore_notes_pitches_chords_signature`
* `copy_file`
* `cosine_similarity`
* `count_bad_chords_in_chordified_score`
* `count_escore_notes_patches`
* `count_patterns`
* `covariance`
* `create_enhanced_monophonic_melody`
* `create_files_list`
* `create_nested_chords_tree`
* `create_similarity_matrix`
* `cubic_kernel`

</details>

---

## D
<a name="d"></a>

**17 entries**

<details>
<summary>Show entries</summary>

* `decode_bpe_corpus`
* `decode_delta_chord_tok`
* `decode_delta_chord_tok_raw`
* `decode_from_ord`
* `decode_int_auto`
* `decode_matrix_marker_prefixed`
* `decode_row_zero_counts`
* `decode_sparse_list`
* `delta_pitches`
* `delta_score_notes`
* `delta_score_to_abs_score`
* `delta_tones`
* `destack_list`
* `detect_list_values_type`
* `detect_plateaus`
* `distribute_k_values`
* `dot_product`

</details>

---

## E
<a name="e"></a>

**57 entries**

<details>
<summary>Show entries</summary>

* `encode_bpe_corpus`
* `encode_delta_chord_tok`
* `encode_delta_chord_tok_raw`
* `encode_int_auto`
* `encode_int_manual`
* `encode_matrix_marker_prefixed`
* `encode_row_zero_counts`
* `encode_sparse_list`
* `encode_to_ord`
* `enhanced_chord_to_chord_token`
* `enhanced_chord_to_tones_chord`
* `enhanced_delta_score_notes`
* `equalize_closest_elements_dynamic`
* `escore_chord_to_chord_token`
* `escore_matrix_to_merged_escore_notes`
* `escore_matrix_to_original_escore_notes`
* `escore_notes_averages`
* `escore_notes_core`
* `escore_notes_delta_times`
* `escore_notes_durations`
* `escore_notes_durations_counter`
* `escore_notes_even_timings`
* `escore_notes_grouped_patches`
* `escore_notes_lrno_pattern`
* `escore_notes_lrno_pattern_fast`
* `escore_notes_medley`
* `escore_notes_middle`
* `escore_notes_monoponic_melodies`
* `escore_notes_patch_lrno_patterns`
* `escore_notes_patches`
* `escore_notes_pitches_chords_signature`
* `escore_notes_pitches_range`
* `escore_notes_primary_features`
* `escore_notes_scale`
* `escore_notes_times_tones`
* `escore_notes_to_binary_matrix`
* `escore_notes_to_chords`
* `escore_notes_to_escore_matrix`
* `escore_notes_to_expanded_binary_matrix`
* `escore_notes_to_image_matrix`
* `escore_notes_to_parsons_code`
* `escore_notes_to_rle_tokens`
* `escore_notes_to_text_description`
* `escore_notes_velocities`
* `even_out_durations_in_escore_notes`
* `even_out_values_in_list_of_lists`
* `even_out_velocities_in_escore_notes`
* `even_timings`
* `event2alsaseq`
* `exists`
* `exists_noncontig`
* `exists_ratio`
* `expert_check_and_fix_chords_in_escore_notes`
* `expert_check_and_fix_pitches_chord`
* `expert_check_and_fix_tones_chord`
* `extract_melody`
* `extract_non_overlapping_chords`

</details>

---

## F
<a name="f"></a>

**35 entries**

<details>
<summary>Show entries</summary>

* `find_best_tones_chord`
* `find_chords_chunk_in_escore_notes`
* `find_chunk_indexes`
* `find_closest_tone`
* `find_closest_value`
* `find_common_divisors`
* `find_deepest_midi_dirs`
* `find_divisors`
* `find_exact_match_variable_length`
* `find_fuzzy_lrno_pattern_fast`
* `find_highest_density_escore_notes_chunk`
* `find_indexes`
* `find_lrno_pattern_fast`
* `find_lrno_patterns`
* `find_matching_tones_chords`
* `find_most_similar_matrix`
* `find_next`
* `find_next_bar`
* `find_paths`
* `find_pattern_idxs`
* `find_pattern_start_indexes`
* `find_similar_tones_chord`
* `find_value_power`
* `fix_bad_chords_in_escore_notes`
* `fix_escore_notes_durations`
* `fix_monophonic_score_durations`
* `fixed_escore_notes_timings`
* `flatten`
* `flatten_spikes`
* `flatten_spikes_advanced`
* `flip_enhanced_score_notes`
* `flip_list_columns`
* `flip_list_rows`
* `frame_monophonic_melody`
* `full_chords_to_sorted_chords`

</details>

---

## G
<a name="g"></a>

**10 entries**

<details>
<summary>Show entries</summary>

* `generate_colors`
* `generate_tones_chords_progression`
* `get_chords_by_semitones`
* `get_chords_with_prefix`
* `get_md5_hash`
* `get_weighted_score`
* `grep`
* `group_by_threshold`
* `group_sublists_by_length`
* `grouped_set`

</details>

---

## H
<a name="h"></a>

**7 entries**

<details>
<summary>Show entries</summary>

* `hamming_distance`
* `harmonize_enhanced_melody_score_notes`
* `harmonize_enhanced_melody_score_notes_to_ms_SONG`
* `has_consecutive_trend`
* `horizontal_ordered_list_search`
* `hsv_to_rgb`
* `humanize_velocities_in_escore_notes`

</details>

---

## I
<a name="i"></a>

**8 entries**

<details>
<summary>Show entries</summary>

* `image_matrix_to_original_escore_notes`
* `insert_caps_newlines`
* `insert_newlines`
* `int_to_bits`
* `int_to_pitches_chord`
* `int_to_tones_chord`
* `is_mostly_wide_peaks_and_valleys`
* `is_valid_md5_hash`

</details>

---

## J
<a name="j"></a>

**1 entry**

<details>
<summary>Show entries</summary>

* `jaccard_similarity`

</details>

---

## L
<a name="l"></a>

**6 entries**

<details>
<summary>Show entries</summary>

* `list_md5_hash`
* `lists_differences`
* `lists_intersections`
* `lists_similarity`
* `lists_sym_differences`
* `longest_common_chunk`

</details>

---

## M
<a name="m"></a>

**23 entries**

<details>
<summary>Show entries</summary>

* `max_sum_chunk_idxs`
* `md5_hash`
* `mean`
* `merge_adjacent_pairs`
* `merge_chords`
* `merge_counts`
* `merge_escore_notes`
* `merge_escore_notes_start_times`
* `merge_melody_notes`
* `merge_text_files`
* `midi2ms_score`
* `midi2opus`
* `midi2score`
* `midi2single_track_ms_score`
* `min_max_cum_low_perc_value`
* `minkowski_distance`
* `monophonic_check`
* `morph_tones_chord`
* `most_common_delta_time`
* `most_common_ordered_set`
* `mult_pitches`
* `multi_instrumental_escore_notes_tokenized`
* `multiprocessing_wrapper`

</details>

---

## N
<a name="n"></a>

**4 entries**

<details>
<summary>Show entries</summary>

* `needleman_wunsch_aligner`
* `norm`
* `normalize_chord_durations`
* `normalize_chordified_score_durations`

</details>

---

## O
<a name="o"></a>

**11 entries**

<details>
<summary>Show entries</summary>

* `Optimus_Data2TXT_Converter`
* `Optimus_MIDI_TXT_Processor`
* `Optimus_Signature`
* `Optimus_Squash`
* `Optimus_TXT_to_Notes_Converter`
* `opus2midi`
* `opus2score`
* `ordered_groups`
* `ordered_groups_unsorted`
* `ordered_lists_match_ratio`
* `ordered_set`

</details>

---

## P
<a name="p"></a>

**11 entries**

<details>
<summary>Show entries</summary>

* `patch_enhanced_score_notes`
* `patch_list_from_enhanced_score_notes`
* `patch_to_instrument_family`
* `patches_onset_times`
* `pearson_correlation`
* `pitches_chord_to_int`
* `pitches_to_tones`
* `pitches_to_tones_chord`
* `plot_ms_SONG`
* `proportional_adjust`
* `proportions_counter`

</details>

---

## Q
<a name="q"></a>

**2 entries**

<details>
<summary>Show entries</summary>

* `quantize_escore_notes`
* `quantize_median_existing`

</details>

---

## R
<a name="r"></a>

**13 entries**

<details>
<summary>Show entries</summary>

* `read_jsonl`
* `read_jsonl_lines`
* `recalculate_score_timings`
* `remove_duplicate_pitches_from_escore_notes`
* `remove_events_from_escore_notes`
* `replace_bad_tones_chord`
* `replace_chords_in_escore_notes`
* `resize_matrix`
* `reverse_enhanced_score_notes`
* `rle_decode_ones`
* `rle_encode_ones`
* `rle_tokens_to_escore_notes`
* `rle_toks_to_binary_matrix`

</details>

---

## S
<a name="s"></a>

**29 entries**

<details>
<summary>Show entries</summary>

* `score2midi`
* `score2opus`
* `score2stats`
* `score_chord_to_tones_chord`
* `shift_bits`
* `shift_to_smallest_integer_type`
* `smooth_escore_notes`
* `smooth_values`
* `solo_piano_escore_notes`
* `solo_piano_escore_notes_tokenized`
* `sort_list_by_other`
* `sorted_chords_to_full_chords`
* `sparse_random_int_list`
* `split_escore_notes_by_channel`
* `split_escore_notes_by_patch`
* `split_escore_notes_by_time`
* `split_list`
* `split_melody`
* `square_binary_matrix`
* `squash_monophonic_escore_notes_pitches`
* `squash_pitches_to_octaves`
* `ssim_index`
* `stack_list`
* `strings_dict`
* `strip_drums_from_escore_notes`
* `summarize_escore_notes`
* `symmetric_match_ratio`
* `system_cpus_utilization`
* `system_memory_utilization`

</details>

---

## T
<a name="t"></a>

**49 entries**

<details>
<summary>Show entries</summary>

* `Tegridy_Any_Pickle_File_Reader`
* `Tegridy_Any_Pickle_File_Writer`
* `Tegridy_Chord_Match`
* `Tegridy_Chords_Generator`
* `Tegridy_Chords_List_Music_Features`
* `Tegridy_FastSearch`
* `Tegridy_File_Time_Stamp`
* `Tegridy_INT_String_to_TXT_Converter`
* `Tegridy_INT_to_TXT_Converter`
* `Tegridy_Last_Chord_Finder`
* `Tegridy_List_Slicer`
* `Tegridy_MIDI_Zip_Notes_Summarizer`
* `Tegridy_SONG_to_Full_MIDI_Converter`
* `Tegridy_SONG_to_MIDI_Converter`
* `Tegridy_Score_Chords_Pairs_Generator`
* `Tegridy_Score_Slicer`
* `Tegridy_Sliced_Score_Pairs_Generator`
* `Tegridy_Split_List`
* `Tegridy_TXT_DeTokenizer`
* `Tegridy_TXT_Tokenizer`
* `Tegridy_TXT_to_INT_Converter`
* `Tegridy_Timings_Converter`
* `Tegridy_Transform`
* `Tegridy_ms_SONG_to_MIDI_Converter`
* `t_to_n`
* `to_millisecs`
* `tokenize_features_to_ints_winsorized`
* `toks_to_alpha_str`
* `tone_type`
* `tones_chord_to_bits`
* `tones_chord_to_int`
* `tones_chord_to_pitches`
* `tones_chord_type`
* `tones_chords_to_bits`
* `tones_chords_to_ints`
* `tones_chords_to_types`
* `tones_to_pitches`
* `top_k_list_value`
* `top_k_list_values`
* `train_bpe`
* `transpose_chord_token`
* `transpose_escore_notes`
* `transpose_escore_notes_to_pitch`
* `transpose_list`
* `transpose_pitches`
* `transpose_pitches_chord`
* `transpose_tones`
* `transpose_tones_chord`
* `trim_list_trail_range`

</details>

---

## V
<a name="v"></a>

**5 entries**

<details>
<summary>Show entries</summary>

* `validate_pitches`
* `validate_pitches_chord`
* `values_percentile`
* `variance`
* `vertical_list_search`

</details>

---

## W
<a name="w"></a>

**2 entries**

<details>
<summary>Show entries</summary>

* `winsorized_normalize`
* `write_jsonl`

</details>

---

