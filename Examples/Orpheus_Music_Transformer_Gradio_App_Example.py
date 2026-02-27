#===================================================================
# https://huggingface.co/spaces/asigalov61/Orpheus-Music-Transformer
#===================================================================

"""
Orpheus Music Transformer Gradio App - Single Model, Simplified Version
SOTA 8k multi-instrumental music transformer trained on 2.31M+ high-quality MIDIs
Using one large optimized model which was trained for 4 full epochs"
"""

#===================================================================
# pip requirements (fully cross platform compatible and minimal)
#===================================================================
# !pip install tqdm
# !pip install numpy
# !pip install soundfile
# !pip install midirenderer
# !pip install matplotlib
# !pip install gradio
# !pip install huggingface_hub
# !pip install torch
# !pip install einops
# !pip install einx
#===================================================================
# Required modules (fully cross platform compatible and minimal)
#-------------------------------------------------------------------
# Download modules from https://github.com/asigalov61/tegridy-tools
#-------------------------------------------------------------------
# TMIDIX.py
# x_transformer_2_3_1.py
#===================================================================

# -----------------------------
# START-UP INFO FUNCTIONS
# -----------------------------
SEP = '=' * 70

def print_sep():
    print(SEP)

print_sep()
print("Orpheus Music Transformer Gradio App")
print_sep()
print("Loading modules...")

# -----------------------------
# MODULES IMPORTS
# -----------------------------

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

RUNNING_IN_SPACE = (
    os.environ.get("SYSTEM", "").lower() == "spaces"
    or "SPACE_ID" in os.environ
    or "HF_SPACE_ID" in os.environ
)

import argparse

from pathlib import Path
from io import BytesIO

import time as reqtime
import datetime
from pytz import timezone
import random

if RUNNING_IN_SPACE:
    import spaces
    GPU = spaces.GPU
else:
    def GPU(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper
        
import gradio as gr

import TMIDIX

import matplotlib.pyplot as plt

import numpy as np
import soundfile as sf
import midirenderer

from huggingface_hub import hf_hub_download

# -----------------------------
# ENVIRONMENT & PyTorch
# -----------------------------

import torch

os.environ['USE_FLASH_ATTENTION'] = '1'

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

# -----------------------------
# X-Transformer
# -----------------------------

from x_transformer_2_3_1 import TransformerWrapper, AutoregressiveWrapper, Decoder, top_p

print_sep()
print("PyTorch version:", torch.__version__)
print("Done loading modules!")
print_sep()

# -----------------------------
# SPACES AND LOCAL ARGS
# -----------------------------

def parse_local_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str, default="large")
    parser.add_argument("--soundfont-name", type=str, default="SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2")
    return parser.parse_args()

args = parse_local_args() if not RUNNING_IN_SPACE else None

# -----------------------------
# CONFIGURATION & GLOBALS
# -----------------------------
PDT = timezone('US/Pacific')

SMALL_MODEL_CHECKPOINT = 'Orpheus_Music_Transformer_Trained_Model_128497_steps_0.6934_loss_0.7927_acc.pth'
LARGE_MODEL_CHECKPOINT = 'Orpheus_Music_Transformer_Large_Trained_Model_31087_steps_0.6878_loss_0.7889_acc.pth'

MODEL_SIZE = args.model_size if args else "large"
MODEL_DEVICE = 'cuda'
MODEL_DTYPE = torch.bfloat16

SOUNDFONT_BANK = args.soundfont_name if args else'SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'

NUM_OUT_BATCHES = 10
PREVIEW_LENGTH = 120  # in tokens

# -----------------------------
# MODEL INIT FUNCTIONS
# -----------------------------
print_sep()

SEQ_LEN = 8192
PAD_IDX = 18819

if "large" in MODEL_SIZE.lower().strip():
    depth = 16
    heads = 16
    MODEL_CHECKPOINT = LARGE_MODEL_CHECKPOINT
    print("Instantiating large model...")

else:
    depth = 8
    heads = 32
    MODEL_CHECKPOINT = SMALL_MODEL_CHECKPOINT
    print(f"Instantiating small model...")

model = TransformerWrapper(
    num_tokens=PAD_IDX + 1,
    max_seq_len=SEQ_LEN,
    attn_layers=Decoder(
        dim=2048,
        depth=depth,
        heads=heads,
        rotary_pos_emb=True,
        attn_flash=True
    )
)
model = AutoregressiveWrapper(model,
                              ignore_index=PAD_IDX,
                              pad_value=PAD_IDX
                             )

print('Done!')
print_sep()
print("Model will use", MODEL_DTYPE.__repr__().split('.')[-1], "precision...")
print("Model will use", MODEL_DEVICE, "device...")
print_sep()
print("Loading model checkpoint...")
print_sep()

checkpoint = hf_hub_download(
    repo_id='asigalov61/Orpheus-Music-Transformer',
    filename=MODEL_CHECKPOINT
)

model.load_state_dict(torch.load(checkpoint, map_location=MODEL_DEVICE))

model.eval()

model.to(MODEL_DEVICE)

model = torch.compile(model, mode='max-autotune')

ctx = torch.amp.autocast(device_type=MODEL_DEVICE,
                         dtype=MODEL_DTYPE
                        )

print_sep()
print("Done!")
print_sep()

# -----------------------------
# SOUNDFONT LOADING FUNCTIONS
# -----------------------------
print('Loading SoundFont...')
print_sep()

SOUNDFONT_PATH = hf_hub_download(repo_id='projectlosangeles/soundfonts4u',
                                 repo_type='dataset',
                                 filename=SOUNDFONT_BANK
                                )

print_sep()
print('Done!')
print('=' * 70)

# -----------------------------
# MIDI PROCESSING FUNCTIONS
# -----------------------------
def load_midi(input_midi, 
              apply_sustains=True, 
              remove_duplicate_pitches=True, 
              remove_overlapping_durations=True
             ):
    
    """Process the input MIDI file and create a token sequence."""

    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)
    
    escore_notes = TMIDIX.advanced_score_processor(raw_score, 
                                                   return_enhanced_score_notes=True, 
                                                   apply_sustain=apply_sustains
                                                  )

    if escore_notes and escore_notes[0]:
    
        escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes[0], 
                                                           sort_drums_last=True
                                                          )

        if remove_duplicate_pitches:
            escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)

        if remove_overlapping_durations:
            escore_notes = TMIDIX.fix_escore_notes_durations(escore_notes, 
                                                             min_notes_gap=0
                                                            )
        
        dscore = TMIDIX.delta_score_notes(escore_notes)
        
        dcscore = TMIDIX.chordify_score([d[1:] for d in dscore])
        
        melody_chords = [18816]
        
        #=======================================================
        # MAIN PROCESSING CYCLE
        #=======================================================
        
        for i, c in enumerate(dcscore):
        
            delta_time = c[0][0]
        
            melody_chords.append(delta_time)
        
            for e in c:
            
                #=======================================================
                
                # Durations
                dur = max(1, min(255, e[1]))
        
                # Patches
                pat = max(0, min(128, e[5]))
                
                # Pitches
                ptc = max(1, min(127, e[3]))
                
                # Velocities
                # Calculating octo-velocity
                vel = max(8, min(127, e[4]))
                velocity = round(vel / 15)-1
                
                #=======================================================
                # FINAL NOTE SEQ
                #=======================================================
                
                # Writing final note
                pat_ptc = (128 * pat) + ptc 
                dur_vel = (8 * dur) + velocity
        
                melody_chords.extend([pat_ptc+256, dur_vel+16768])
            
        return melody_chords

    else:
        return [18816]

def save_midi(tokens):
    
    """Convert token sequence back to a MIDI score and write it using TMIDIX.
    """

    time = 0
    dur = 1
    vel = 90
    pitch = 60
    channel = 0
    patch = 0

    patches = [-1] * 16

    channels = [0] * 16
    channels[9] = 1

    song_f = []

    for ss in tokens:

        if 0 <= ss < 256:

            time += ss * 16

        if 256 <= ss < 16768:

            patch = (ss-256) // 128

            if patch < 128:

                if patch not in patches:
                  if 0 in channels:
                      cha = channels.index(0)
                      channels[cha] = 1
                  else:
                      cha = 15

                  patches[cha] = patch
                  channel = patches.index(patch)
                else:
                  channel = patches.index(patch)

            if patch == 128:
                channel = 9

            pitch = (ss-256) % 128


        if 16768 <= ss < 18816:

            dur = ((ss-16768) // 8) * 16
            vel = (((ss-16768) % 8)+1) * 15

            song_f.append(['note', time, dur, channel, pitch, vel, patch])

    song_f = TMIDIX.remove_duplicate_pitches_from_escore_notes(song_f)

    song_f = TMIDIX.fix_escore_notes_durations(song_f, 
                                               min_notes_gap=0
                                              )

    output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(song_f)

    fname = f"Orpheus-Music-Transformer-Composition"
    fname += "-"+datetime.datetime.now(PDT).strftime("%Y-%m-%d-%H-%M-%S")
    
    TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
        output_score,
        output_signature='Orpheus Music Transformer',
        output_file_name=fname,
        track_name='Project Los Angeles',
        list_of_MIDI_patches=patches,
        verbose=False
    )
    return fname, output_score

# -----------------------------
# MIDI RENDERING FUNCTIONS
# -----------------------------

def render_midi_for_gradio(midi_path: str | Path,
                           soundfont_path: str | Path
                          ) -> tuple[int, np.ndarray]:

    midi_bytes = Path(midi_path).read_bytes()
    sf_bytes = Path(soundfont_path).read_bytes()

    wav_bytes = midirenderer.render_wave_from(sf_bytes, midi_bytes)

    with BytesIO(wav_bytes) as bio:
        audio, sr = sf.read(bio, dtype="float32")

    audio = np.asarray(audio, dtype=np.float32)
    np.clip(audio, -1.0, 1.0, out=audio)

    audio_int16 = (audio * 32767.0).round().astype(np.int16)

    return sr, audio_int16

# -----------------------------
# TOKENS SANITIZER FUNCTIONS
# -----------------------------

def extract_pairs_and_prefix(lst):
    
    RANGE1 = (0, 255)
    RANGE2 = (256, 16767)
    RANGE3 = (16768, 18815)
    RANGE4 = (18816, 18819)

    def in_range(x, r):
        return r[0] <= x <= r[1]

    prefix = []
    started = False
    
    for x in lst:
        if in_range(x, RANGE2):
            started = True
            break
            
        prefix.append(x)

    pairs = []
    pending = None

    for x in lst:
        if in_range(x, RANGE2):
            pending = x
            
        elif in_range(x, RANGE3):
            if pending is not None:
                pairs.append((pending, x))
                pending = None

        elif in_range(x, RANGE4):
            pairs.append((x, x))

    return prefix, pairs

def sanitize_tokens(tokens):

    chords = []
    cho = []

    for t in tokens:
        if t < 256:
            if cho:
                chords.append(cho)

            cho = [t]

        else:
            cho.append(t)
            
    if cho:
        chords.append(cho)

    san_tokens = []

    for cho in chords:
        pfx, ptcs_durs = extract_pairs_and_prefix(cho)

        san_tokens.extend(pfx)

        san_ptcs_durs = []
        seen = []

        for ptc, dur in ptcs_durs:
            if 256 <= ptc < 16768:
                if ptc not in seen:
                    san_tokens.append(ptc)
                    san_tokens.append(dur)
                    seen.append(ptc)

            else:
                san_tokens.append(ptc)
        
    return san_tokens       
        
# -----------------------------
# MUSIC GENERATION FUNCTIONS
# -----------------------------
@GPU
def generate_music(prime, num_gen_tokens, num_gen_batches, model_temperature, model_top_p):
    
    """Generate music tokens given prime tokens and parameters."""

    if len(prime) >= 6656:
        prime = [18816] + prime[-6656:]
    
    inputs = prime
    
    print("Generating...")
    inp = torch.LongTensor([inputs] * num_gen_batches).to(MODEL_DEVICE)

    if model_top_p < 1:
        with ctx:
            out = model.generate(
                inp,
                num_gen_tokens,
                filter_logits_fn=top_p,
                filter_kwargs={'thres': model_top_p},
                temperature=model_temperature,
                eos_token=18818,
                return_prime=False,
                verbose=False
            )

    else:
        with ctx:
            out = model.generate(
                inp,
                num_gen_tokens,
                temperature=model_temperature,
                eos_token=18818,
                return_prime=False,
                verbose=False
            )
            
    print("Done!")
    print_sep()
    return out.tolist()

def generate_music_and_state(input_midi, 
                             apply_sustains,
                             remove_duplicate_pitches,
                             remove_overlapping_durations,
                             prime_instruments, 
                             num_prime_tokens, 
                             num_gen_tokens, 
                             model_temperature, 
                             model_top_p, 
                             add_drums, 
                             add_outro,
                             final_composition, 
                             generated_batches, 
                             block_lines
                            ):
    
    """
    Generate tokens using the model, update the composition state, and prepare outputs.
    This function combines seed loading, token generation, and UI output packaging.
    """
    
    print_sep()
    print("Request start time:", datetime.datetime.now(PDT).strftime("%Y-%m-%d %H:%M:%S"))
    start_time = reqtime.time()

    print_sep()
    if input_midi is not None:
        fn = os.path.basename(input_midi.name)
        fn1 = fn.split('.')[0]
        print('Input file name:', fn)
        print('Apply sustains:', apply_sustains)
        print('Remove duplicate pitches:', remove_duplicate_pitches)
        print('Remove overlapping duriations', remove_overlapping_durations)

    print('Prime instruments:', prime_instruments)
    print('Num prime tokens:', num_prime_tokens)
    print('Num gen tokens:', num_gen_tokens)

    print('Model temp:', model_temperature)
    print('Model top p:', model_top_p)
    
    print('Add drums:', add_drums)
    print('Add outro:', add_outro)
    print_sep()
    
    # Load seed from MIDI if there is no existing composition.
    if not final_composition and input_midi is not None:
        final_composition = load_midi(input_midi, 
                                      apply_sustains=apply_sustains, 
                                      remove_duplicate_pitches=remove_duplicate_pitches, 
                                      remove_overlapping_durations=remove_overlapping_durations
                                     )

        if num_prime_tokens < 6656:
            final_composition = final_composition[:num_prime_tokens]
        
        midi_fname, midi_score = save_midi(final_composition)
        # Use the last note's time as a marker.
        last_nd_note = [e for e in midi_score if e[3] != 9]
        block_lines.append((last_nd_note[-1][1]+last_nd_note[-1][2]) // 1000 if final_composition else 0)

    if not final_composition and input_midi is None and prime_instruments:
        final_composition = [18816, 0]

        if "Drums" in prime_instruments:
            ci_num = random.choice([37, 42])
    
            for _ in range(4):
                final_composition.append((128*128)+ci_num+256)
                final_composition.append((8*16)+7+16768)
                final_composition.append(32)

        nd_instruments = [i for i in prime_instruments[:4] if i != 'Drums']

        if nd_instruments:
            prime_chord = random.choice([c for c in TMIDIX.ALL_CHORDS_FULL if len(c) == len(nd_instruments)])
            
            for i, instr in enumerate(nd_instruments):
                instr_num = Patch2number[instr]
                instr_oct = TMIDIX.Patch2octave[instr]
                
                final_composition.append((128*instr_num)+(instr_oct+prime_chord[i])+256)
                dur = random.randint(16, 32)
                vel = random.randint(5, 7)
                final_composition.append((8*dur)+vel+16768)

        if 'Drums' in prime_instruments:
            drum_pitch = random.choice([35, 36, 41, 43, 45, 47, 48, 50])
            final_composition.append((128*128)+(drum_pitch)+256)
            final_composition.append((8*16)+7+16768)

    drum_seq = []
    outro_seq = []
 
    if final_composition:
        if add_drums:
            drum_pitches = random.sample([35, 36, 41, 43, 45], k=1)
            for dp in drum_pitches:
                drum_seq.append((128*128)+dp+256) # Drum patch/pitch token
                drum_seq.append((8*16)+7+16768) # Dur/vel
                
        if add_outro:
            outro_seq.append(18817) # Outro token

    if not final_composition and input_midi is None and not prime_instruments:
        final_composition = [18816, 0]
            
    print_sep()
    print('Composition has', len(final_composition+drum_seq+outro_seq), 'tokens')
    print_sep()
    
    batched_gen_tokens = generate_music(final_composition+drum_seq+outro_seq, num_gen_tokens,
                                        NUM_OUT_BATCHES, model_temperature, model_top_p)

    batched_gen_tokens_san = []

    for tokens in batched_gen_tokens:
        san_tokens = sanitize_tokens(tokens)
        batched_gen_tokens_san.append(san_tokens)

    batched_gen_tokens = batched_gen_tokens_san

    batched_gen_tokens_ext = []
    
    if drum_seq or outro_seq:
        for tokens in batched_gen_tokens:
            batched_gen_tokens_ext.append(drum_seq+outro_seq+tokens)

        batched_gen_tokens = batched_gen_tokens_ext
    
    output_batches = []
    for i, tokens in enumerate(batched_gen_tokens):
        preview_composition = final_composition+drum_seq+outro_seq
        preview_tokens = preview_composition[-PREVIEW_LENGTH:]

        plot_kwargs = {'plot_title': f'Batch # {i}', 'return_plt': True}
        
        if len(preview_composition) > PREVIEW_LENGTH:
            preview_score = save_midi(preview_tokens[:PREVIEW_LENGTH])[1]
            plot_kwargs['block_lines_times_list'] = [(preview_score[-1][1]+preview_score[-1][2]) // 1000]

        midi_fname, midi_score = save_midi(preview_tokens + tokens)
        midi_plot = TMIDIX.plot_ms_SONG(midi_score,
                                        **plot_kwargs
                                       )

        gradio_audio = render_midi_for_gradio(midi_fname + '.mid',
                                              SOUNDFONT_PATH
                                             )
        
        output_batches.append([gradio_audio, midi_plot, tokens, midi_fname + '.mid'])

    # Update generated_batches (for use by add/remove functions)
    generated_batches = batched_gen_tokens
    
    # Flatten outputs: states then audio and plots for each batch.
    outputs_flat = []
    for batch in output_batches:
        outputs_flat.extend([batch[0], batch[1], batch[3]])

    print("Request end time:", datetime.datetime.now(PDT).strftime("%Y-%m-%d %H:%M:%S"))
    print_sep()
    
    end_time = reqtime.time()
    execution_time = end_time - start_time
    
    print(f"Request execution time: {execution_time} seconds")
    print_sep()
        
    return [final_composition, generated_batches, block_lines] + outputs_flat

# -----------------------------
# BATCH HANDLING FUNCTIONS
# -----------------------------
def add_batch(batch_number, final_composition, generated_batches, block_lines):
    """Add tokens from the specified batch to the final composition and update outputs."""
    if generated_batches:
        final_composition.extend(generated_batches[batch_number])
        midi_fname, midi_score = save_midi(final_composition)
        last_nd_note = [e for e in midi_score if e[3] != 9]
        block_lines.append((last_nd_note[-1][1]+last_nd_note[-1][2]) // 1000 if final_composition else 0)
        midi_plot = TMIDIX.plot_ms_SONG(
            midi_score,
            plot_title='Orpheus Music Transformer Composition',
            block_lines_times_list=block_lines[:-1],
            return_plt=True
        )
        gradio_audio = render_midi_for_gradio(midi_fname + '.mid',
                                              SOUNDFONT_PATH
                                             )
        print("Added batch #", batch_number)
        print_sep()
        return gradio_audio, midi_plot, midi_fname + '.mid', final_composition, generated_batches, block_lines
    else:
        return None, None, None, [], [], []

def remove_batch(batch_number, num_tokens, final_composition, generated_batches, block_lines):
    """Remove tokens from the final composition and update outputs."""
    if final_composition and len(final_composition) > num_tokens:
        final_composition = final_composition[:-num_tokens]
        if block_lines:
            block_lines.pop()
        midi_fname, midi_score = save_midi(final_composition)
        midi_plot = TMIDIX.plot_ms_SONG(
            midi_score,
            plot_title='Orpheus Music Transformer Composition',
            block_lines_times_list=block_lines[:-1],
            return_plt=True
        )
        gradio_audio = render_midi_for_gradio(midi_fname + '.mid',
                                              SOUNDFONT_PATH
                                             )
        print("Removed batch #", batch_number)
        print_sep()
        return gradio_audio, midi_plot, midi_fname + '.mid', final_composition, generated_batches, block_lines
    else:
        return None, None, None, [], [], []

# -----------------------------
# MISC FUNCTIONS
# -----------------------------

def clear():
    """Clear outputs and reset state."""
    print_sep()
    print('Clear batch...')
    print_sep()
    return None, None, None, [], []

def reset(final_composition=[], generated_batches=[], block_lines=[]):
    """Reset composition state."""
    print_sep()
    print('Reset composition...')
    print_sep()
    return [], [], []

Patch2number = TMIDIX.reverse_dict(TMIDIX.Number2patch)
Patch2number['Drums'] = 128

# -----------------------------
# GRADIO INTERFACE SETUP
# -----------------------------
with gr.Blocks() as orpheus_app:

    gr.Markdown("<h1 style='text-align: left; margin-bottom: 1rem'>Orpheus Music Transformer</h1>")
    gr.Markdown("<h1 style='text-align: left; margin-bottom: 1rem'>SOTA 8k multi-instrumental music transformer trained on 2.31M+ high-quality MIDIs</h1>")
    gr.Markdown("<h1 style='text-align: left; margin-bottom: 1rem'>🔥[2026]🔥 Now featuring large optimized model!</h1>")
    
    with gr.Row(elem_classes="duplicate-row"):
        gr.DuplicateButton(
            value="🤗 Duplicate 🤗",
            variant="huggingface",
            size="md",
            link="https://huggingface.co/spaces/asigalov61/Orpheus-Music-Transformer?duplicate=true",
            link_target="_blank"
        )
        
        gr.Button(
            value="❤️ Models ❤️",
            variant="huggingface",
            size="md",
            link="https://huggingface.co/asigalov61/Orpheus-Music-Transformer",
            link_target="_blank"
        )

        gr.Button(
            value="🦖 Dataset 🦖",
            variant="huggingface",
            size="md",
            link="https://huggingface.co/datasets/projectlosangeles/Godzilla-MIDI-Dataset",
            link_target="_blank"
        )

    gr.HTML("""
    <iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/playlists/2042253855&color=%23ff5500&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe><div style="font-size: 10px; color: #cccccc;line-break: anywhere;word-break: normal;overflow: hidden;white-space: nowrap;text-overflow: ellipsis; font-family: Interstate,Lucida Grande,Lucida Sans Unicode,Lucida Sans,Garuda,Verdana,Tahoma,sans-serif;font-weight: 100;"><a href="https://soundcloud.com/aleksandr-sigalov-61" title="Project Los Angeles" target="_blank" style="color: #cccccc; text-decoration: none;">Project Los Angeles</a> · <a href="https://soundcloud.com/aleksandr-sigalov-61/sets/orpheus-music-transformer" title="Orpheus Music Transformer" target="_blank" style="color: #cccccc; text-decoration: none;">Orpheus Music Transformer</a></div>
    """)

    gr.Markdown("## Key Features")
    gr.Markdown("""
    - **Efficient Architecture with RoPE**: Large optimized 748M full attention autoregressive transformer with RoPE.
    - **Extended Sequence Length**: 8k tokens that comfortably fit most music compositions and facilitate long-term music structure generation.
    - **Premium Training Data**: Trained solely on the highest-quality MIDIs from the Godzilla MIDI dataset.
    - **Optimized MIDI Encoding**: Extremely efficient MIDI representation using only 3 tokens per note and 7 tokens per tri-chord.
    - **Distinct Encoding Order**: Features a unique duration/velocity last MIDI encoding order for refined musical expression.
    - **Full-Range Instrumental Learning**: True full-range MIDI instruments encoding enabling the model to learn each instrument separately.
    - **Natural Composition Endings**: Outro tokens that help generate smooth and natural musical conclusions.
    """)

    # Global state variables for composition
    final_composition = gr.State([])
    generated_batches = gr.State([])
    block_lines = gr.State([])

    gr.Markdown("## Upload seed MIDI or click 'Generate' for random output")
    
    gr.Markdown("### PLEASE NOTE:")
    gr.Markdown("* Orpheus Music Transformer is a primarily music continuation/co-composition model!")
    gr.Markdown("* The model works best if given some music context to work with")
    gr.Markdown("* Random generation from SOS token/embeddings may not always produce good results")
    
    input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
    input_midi.upload(reset, [final_composition, generated_batches, block_lines],
                      [final_composition, generated_batches, block_lines])
    apply_sustains = gr.Checkbox(value=True, label="Apply sustains (if present)")
    remove_duplicate_pitches = gr.Checkbox(value=True, label="Remove duplicate pitches (if present)")
    remove_overlapping_durations = gr.Checkbox(value=True, label="Trim overlapping durations (if present)")

    gr.Markdown("## Generation options")
    prime_instruments = gr.Dropdown(label="Prime instruments (select up to 5)", choices=list(Patch2number.keys()),
                                    multiselect=True, max_choices=5, type="value",
                                    info="NOTE: Custom MIDI overrides prime instruments"
                                   )

    prime_instruments.input(reset, [final_composition, generated_batches, block_lines],
                          [final_composition, generated_batches, block_lines])
    
    num_prime_tokens = gr.Slider(16, 6656, value=6656, step=1, label="Number of prime tokens")
    num_gen_tokens = gr.Slider(16, 1024, value=512, step=1, label="Number of tokens to generate")
    model_temperature = gr.Slider(0.1, 1, value=0.9, step=0.01, label="Model temperature")
    model_top_p = gr.Slider(0.1, 1.0, value=0.96, step=0.01, label="Model sampling top p value", info="1 == Disabled")
    add_drums = gr.Checkbox(value=False, label="Add drums")
    add_outro = gr.Checkbox(value=False, label="Add an outro")
    generate_btn = gr.Button("Generate", variant="primary")

    gr.Markdown("## Batch Previews")
    outputs = [final_composition, generated_batches, block_lines]
    # Two outputs (audio and plot) for each batch
    for i in range(NUM_OUT_BATCHES):
        with gr.Tab(f"Batch # {i}"):
            audio_output = gr.Audio(label=f"Batch # {i} MIDI Audio", format="mp3")
            plot_output = gr.Plot(label=f"Batch # {i} MIDI Plot")
            midi_file = gr.File(label=f"Batch # {i} MIDI File")
            outputs.extend([audio_output, plot_output, midi_file])
            
    generate_btn.click(
        generate_music_and_state,
        [input_midi, 
         apply_sustains,
         remove_duplicate_pitches,
         remove_overlapping_durations,
         prime_instruments, 
         num_prime_tokens, 
         num_gen_tokens, 
         model_temperature, 
         model_top_p, 
         add_drums, 
         add_outro,
         final_composition, 
         generated_batches, 
         block_lines
        ],
        outputs
    )

    gr.Markdown("## Add/Remove Batch")
    batch_number = gr.Slider(0, NUM_OUT_BATCHES - 1, value=0, step=1, label="Batch number to add/remove")
    add_btn = gr.Button("Add batch", variant="primary")
    remove_btn = gr.Button("Remove batch", variant="stop")
    clear_btn = gr.ClearButton()

    final_audio_output = gr.Audio(label="Final MIDI audio", format="mp3")
    final_plot_output = gr.Plot(label="Final MIDI plot")
    final_file_output = gr.File(label="Final MIDI file")

    add_btn.click(
        add_batch,
        [batch_number, final_composition, generated_batches, block_lines],
        [final_audio_output, final_plot_output, final_file_output, final_composition, generated_batches, block_lines]
    )
    remove_btn.click(
        remove_batch,
        [batch_number, num_gen_tokens, final_composition, generated_batches, block_lines],
        [final_audio_output, final_plot_output, final_file_output, final_composition, generated_batches, block_lines]
    )
    clear_btn.click(clear, inputs=None,
                    outputs=[final_audio_output, final_plot_output, final_file_output, final_composition, block_lines])

# -----------------------------
# APP LAUNCHER
# -----------------------------

if __name__ == "__main__":
    orpheus_app.launch(
        mcp_server=RUNNING_IN_SPACE,   # MCP only on HF
        share=not RUNNING_IN_SPACE,    # Share only locally
        server_name="0.0.0.0",
        server_port=7860,
    )
