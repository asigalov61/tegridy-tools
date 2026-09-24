#==================================================================
# https://huggingface.co/spaces/projectlosangeles/Drum-Track-Master
#==================================================================

"""
----------------------------
Drum Track Master Gradio App
----------------------------

-------------
Requirements:
-------------

tqdm
numpy
scikit-learn
matplotlib
gradio
huggingface_hub
torch
einops
einx

---------
Packages:
---------

fluidsynth

--------
Modules:
--------

https://github.com/asigalov61/tegridy-tools

TMIDIX
x_transformer_2_3_1
midi_to_colab_audio
print_collector

-------
Models:
-------

https://huggingface.co/projectlosangeles/midisimx

-----------------------
License and Attribution
-----------------------

Apache 2.0 / Version 1.0.0

Project Los Angeles
Tegridy Code 2026
"""

# =================================================================================================

print('=' * 70)
print('Drum Track Master Gradio App')
print('=' * 70)

# =================================================================================================

import os
import copy

os.environ['USE_FLASH_ATTENTION'] = '1'

import time as reqtime
from pytz import timezone

import torch

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

from torch.utils.data import Dataset, DataLoader

import spaces
import gradio as gr

from x_transformer_2_3_1 import *

import datetime
import random
import statistics
import tqdm

from midi_to_colab_audio import midi_to_colab_audio

import TMIDIX

import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download

from print_collector import PrintCollector

# =================================================================================================

print = PrintCollector(pretty=True, width=70) # Debug log
         
# =================================================================================================

OUTPUT_MIDIS_DIR = 'output_midis'

# =================================================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================================================================

dtype = torch.bfloat16

ctx = torch.amp.autocast(device_type=DEVICE, dtype=dtype)

# =================================================================================================

# -------------------------------------------------------------------------------------------------
# Multi-label classifier model
# -------------------------------------------------------------------------------------------------

def build_model(device, num_classes=30, vocab_size=769, max_seq_len=1024, pad_idx=768):
    """
    Transformer encoder -> average pooled -> multi-logits (multi-class).
    """
    model = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        logits_dim=num_classes,
        use_cls_token=False,
        average_pool_embed=True,
        emb_dropout = 0.2,
        attn_layers=Encoder(
            dim=384,
            depth=8,
            heads=8,
            rotary_pos_emb=True,
            attn_flash=True,
            layer_dropout = 0.2,   # stochastic depth - dropout entire layer
            attn_dropout = 0.2,    # dropout post-attention
            ff_dropout = 0.2       # feedforward dropout
        ),
    )
    return model.to(device)
    
# -------------------------------------------------------------------------------------------------

def load_model(checkpoint_path, num_classes, device='cuda'):
    """
    Rebuilds the architecture, loads weights.
    """
    model = build_model(device, num_classes)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

# -------------------------------------------------------------------------------------------------

class InferenceDataset(Dataset):
    """
    Dataset for pairs (src_seq, label).
    src_seq: list of token IDs (ints).
    label: single int or float (0 or 1).
    """
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_seq = self.data_pairs[idx]
        x = torch.tensor(src_seq, dtype=torch.long)
        return x

# -------------------------------------------------------------------------------------------------

@torch.no_grad()
def predict(model, seqs, device=torch.device('cuda'), batch_size=512, pad_idx=768):
    """
    Returns two lists:
      - preds: int class predictions (0, 1, 2, 3)
      - probs: float probabilities for the predicted class
    """
    model.eval()
    
    # --- 1. Pad sequences and build masks ---
    lengths = [len(s) for s in seqs]
    max_len = max(lengths) if lengths else 1
    
    x = torch.full((len(seqs), max_len), pad_idx, dtype=torch.long)
    mask = torch.zeros((len(seqs), max_len), dtype=torch.bool)
    
    for i, (seq, l) in enumerate(zip(seqs, lengths)):
        x[i, :l] = torch.tensor(seq, dtype=torch.long)
        mask[i, :l] = True

    all_preds = []
    all_probs = []

    # --- 2. Batched Inference ---
    for i in range(0, len(seqs), batch_size):
        batch_x = x[i:i+batch_size].to(device, non_blocking=True)
        batch_mask = mask[i:i+batch_size].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(batch_x, mask=batch_mask)  # [B, 4]

        # Softmax to get probabilities, then pick the max
        probs = torch.softmax(logits.float(), dim=-1)  # [B, 4]
        confidences, preds = probs.max(dim=-1)         # [B], [B]

        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(confidences.cpu().tolist())

    return all_preds, all_probs

# =================================================================================================

print('=' * 70)
print('Loading models...')
print('=' * 70)
print('Loading Drum Track Master masked encoder model...')
print('=' * 70)

ENC_SEQ_LEN      = 1280
ENC_MASK_PROB    = 0.15
ENC_MASK_IDX     = 821
ENC_PAD_IDX      = ENC_MASK_IDX+1
ENC_VOCAB_SIZE   = ENC_PAD_IDX+1

print('Encoder Vocab size:', ENC_VOCAB_SIZE)

enc_model = TransformerWrapper(
    num_tokens = ENC_VOCAB_SIZE,
    max_seq_len = ENC_SEQ_LEN,
    attn_layers = Encoder(
        dim   = 768,
        depth = 16,
        heads = 12,
        rotary_pos_emb = True,
        attn_flash = True,
    ),
)

print('=' * 70)
print('Loading model checkpoint...')

checkpoint = hf_hub_download(
    repo_id='projectlosangeles/midisimx',
    filename='midisimx_drums_encoder_trained_model_6417_steps_0.6751_loss_0.7739_acc.pth'
)

DEVICE = 'cuda'

enc_model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))

enc_model.cuda()

enc_model.eval()

print('=' * 70)
print('Done!')
print('=' * 70)

# =================================================================================================

print('=' * 70)
print('Loading models...')
print('=' * 70)
print('Loading Drum Track Master drum track style classifier model...')
print('=' * 70)

STYLE_CLS_NUM_CLASSES = 18
STYLE_CLS_SEQ_LEN     = 1024
STYLE_CLS_PAD_IDX     = 768
STYLE_CLS_VOCAB_SIZE  = 769

print('=' * 70)
print('Downloading model checkpoint...')

style_checkpoint = hf_hub_download(
    repo_id='projectlosangeles/midisimx',
    filename='midisimx_drums_style_cls_trained_model_13914_steps_0.2076_loss_0.9347_acc.pth'
)

print('=' * 70)
print('Loading model checkpoint...')

style_model = load_model(style_checkpoint, STYLE_CLS_NUM_CLASSES)

style_model.cuda()

style_model.eval()

print('=' * 70)
print('Done!')
print('=' * 70)

# =================================================================================================

print('=' * 70)
print('Loading models...')
print('=' * 70)
print('Loading Drum Track Master drum track BPM classifier model...')
print('=' * 70)

BPM_NUM_CLASSES = 30
BPM_SEQ_LEN     = 1024
BPM_PAD_IDX     = 768
BPM_VOCAB_SIZE  = 769

print('=' * 70)
print('Downloading model checkpoint...')

bpm_checkpoint = hf_hub_download(
    repo_id='projectlosangeles/midisimx',
    filename='midisimx_drums_bpm_cls_trained_model_6956_steps_0.267_loss_0.9271_acc.pth'
)

print('=' * 70)
print('Loading model checkpoint...')

bpm_model = load_model(bpm_checkpoint, BPM_NUM_CLASSES)

bpm_model.cuda()

bpm_model.eval()

print('=' * 70)
print('Done!')
print('=' * 70)

# =================================================================================================

print('Loading SoundFont...')

SOUNDFONT_PATH = hf_hub_download(repo_id='projectlosangeles/soundfonts4u',
                                 repo_type='dataset',
                                 filename='SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2'
                                )

print('Done!')
print('=' * 70)

# =================================================================================================

style_to_id = {
    'afrobeat': 0,
    'afrocuban': 1,
    'blues': 2,
    'country': 3,
    'dance': 4,
    'funk': 5,
    'gospel': 6,
    'highlife': 7,
    'hiphop': 8,
    'jazz': 9,
    'latin': 10,
    'middleeastern': 11,
    'neworleans': 12,
    'pop': 13,
    'punk': 14,
    'reggae': 15,
    'rock': 16,
    'soul': 17,
    'none': 18,
    'unknown': 19
}

# -------------------------------------------------------------------------------------------------

bpm_to_id = {
    50: 0,
    60: 1,
    65: 2,
    70: 3,
    75: 4,
    80: 5,
    85: 6,
    90: 7,
    95: 8,
    100: 9,
    105: 10,
    110: 11,
    115: 12,
    120: 13,
    125: 14,
    130: 15,
    135: 16,
    140: 17,
    145: 18,
    150: 19,
    155: 20,
    160: 21,
    170: 22,
    175: 23,
    180: 24,
    185: 25,
    190: 26,
    200: 27,
    215: 28,
    290: 29,
    'none': 30,
    'unknown': 31
}

# -------------------------------------------------------------------------------------------------

id_to_style = TMIDIX.reverse_dict(style_to_id)

id_to_bpm = TMIDIX.reverse_dict(bpm_to_id)

# =================================================================================================

mastering_modes = {'Start Times': list(range(0, 256)),
                   'Durations': list(range(384, 640)),
                   'Pitches': list(range(256, 384)),
                   'Velocities': list(range(640, 768)),
                   'Style': list(range(768, 788)),
                   'BPM': list(range(788, 820))
                  }

# =================================================================================================

def process_midi(midi_file):

    try:

        #========================================
        
        raw_score = TMIDIX.midi2single_track_ms_score(midi_file)
    
        pre_score = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]
        
        drum_score = [e for e in pre_score if e[3] == 9]

        #========================================
    
        if drum_score:
    
            escore_notes = TMIDIX.augment_enhanced_score_notes(pre_score, sort_drums_last=True)
        
            escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)
        
            inst_score = TMIDIX.strip_drums_from_escore_notes(TMIDIX.fix_escore_notes_durations(escore_notes, min_notes_gap=0))
    
            #========================================
            
            escore_notes = TMIDIX.augment_enhanced_score_notes(drum_score, sort_drums_last=True)
        
            escore_notes = TMIDIX.remove_duplicate_pitches_from_escore_notes(escore_notes)
        
            fixed_score = TMIDIX.fix_escore_notes_durations(escore_notes, min_notes_gap=0)

            #========================================
        
            vels = [e[5] for e in fixed_score]
            avg_vel = sum(vels) / len(vels)
        
            if len(set(vels)) < 4:
                fixed_score = TMIDIX.humanize_velocities_in_escore_notes(fixed_score)
        
            if avg_vel < 80:
                TMIDIX.adjust_score_velocities(fixed_score, 100)

            #========================================
        
            dscore = TMIDIX.delta_score_notes(fixed_score)
        
            score = [0]
        
            for e in dscore:
                if e[1] != 0:
                    score.append(e[1])
        
                score.extend([e[4]+256, e[2]+384, e[5]+640]) # 768
            
            return score, inst_score, fixed_score[0][1]

        return None, None, -1

    except:
        return None, None, -2

# =================================================================================================

def save_midi(tokens, inst_score, drum_track_start_time):
    
    """Convert token sequence back to a MIDI score and write it using TMIDIX.
    """

    song_f = []
    
    time = drum_track_start_time
    dur = 1
    vel = 90
    pitch = 60
    channel = 0
    patch = 0
    
    patches = [0] * 16
    
    for m in tokens:
    
        if 0 <= m < 256:
            time += m
    
        elif 256 < m < 384:
            pitch = (m-256)
    
        elif 384 < m < 640:
            dur = (m-384)
            
        elif 640 < m < 768:
            vel = (m-640)
            
            song_f.append(['note', time, dur, 9, pitch, vel, 128])
    
    song_f = sorted(copy.deepcopy(inst_score)+copy.deepcopy(song_f), key=lambda x: x[1])

    if song_f is not None and song_f:

        song_f = TMIDIX.remove_duplicate_pitches_from_escore_notes(song_f)
    
        song_f = TMIDIX.fix_escore_notes_durations(song_f)
        
        output_score, patches, overflow_patches = TMIDIX.patch_enhanced_score_notes(song_f)
        
        now = datetime.datetime.now(PDT)
        ms4 = now.strftime("%f")[:4]
        
        fname = (
            "Drum-Track-Master-Composition-"
            + now.strftime(f"%Y-%m-%d-%H-%M-%S-{ms4}")
        )

        os.makedirs(OUTPUT_MIDIS_DIR, exist_ok=True)

        output_fname = os.path.join(OUTPUT_MIDIS_DIR, fname)
        
        TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(
            output_score,
            output_signature='Drum Track Master',
            output_file_name=output_fname,
            track_name='Project Los Angeles',
            list_of_MIDI_patches=patches,
            timings_multiplier=16,
            verbose=False
        )
        return output_fname, output_score

    else:
        return None, None

# =================================================================================================

@spaces.GPU
def encoder_mastering(selected_model,
                      drum_score,
                      num_master_iter,
                      style,
                      bpm,
                      mastering_mode,
                      debug=False
                     ):

    chunk_len = (ENC_SEQ_LEN // 2)-2


    drum_score_chunks = TMIDIX.chunk_by_size(drum_score,
                                             chunk_len
                                            )

    drum_score_chunks_lengths = [len(c) for c in drum_score_chunks]

    style_tok = style_to_id[style.lower()]+768
    bpm_tok = bpm_to_id[bpm]+788

    mas_toks = []
    
    for mode in mastering_mode:
        mas_toks.extend(mastering_modes[mode])

    print('-' * 70)
    print('Encoder seq len:', ENC_SEQ_LEN)
    print('Drum score was split into', len(drum_score_chunks), 'chunks')
    print('Chunks lengths:', drum_score_chunks_lengths)
    print('-' * 70)

    output_score = []
    output_styles_bpms_probs = []
    style_bpm_mask = []

    if 'Style' in mastering_mode and 'BPM' in mastering_mode:
        style_bpm_mask = [1, 2]
        
    elif 'Style' in mastering_mode:
        style_bpm_mask = [1]

    elif 'BPM' in mastering_mode:
        style_bpm_mask [2]

    else:
        style_bpm_mask = []

    if debug:
        print('Style and BPM mask:', style_bpm_mask)
        
    for cidx, chunk in enumerate(tqdm.tqdm(drum_score_chunks, disable=not debug)):

        if cidx == 0:
            inp_seq = [820, style_tok, bpm_tok] + chunk
            masked_pos = [i for i in style_bpm_mask + list(range(3, len(inp_seq))) if inp_seq[i] in mas_toks]
            if debug:
                print('Chunk #', cidx, 'inp_seq len:', len(inp_seq), 'ctx len:', 0)
                print('inp_seq:', inp_seq[:10])
                print('masked_pos:', masked_pos[:10])

        else:
            ctx = output_score[2:][-chunk_len:]
            inp_seq = [820, style_tok, bpm_tok] + ctx + chunk
            masked_pos = [i for i in range(3 + len(ctx), len(inp_seq)) if inp_seq[i] in mas_toks]
            if debug:
                print('Chunk #', cidx, 'inp_seq len:', len(inp_seq), 'ctx len:', len(ctx))
        
        results = predict_masked_tokens_iter(selected_model,
                                             inp_seq,
                                             mask_positions=masked_pos,
                                             iterations=num_master_iter,
                                             topk=1,
                                             seq_len=ENC_SEQ_LEN,
                                             mask_idx=ENC_MASK_IDX,
                                             pad_idx=ENC_PAD_IDX,
                                             vocab_size=ENC_VOCAB_SIZE
                               )
        
        enc_output = results['predicted_ids']

        output_probs = [r['topk'][0][1] for r in results['predictions']]
        avg_output_prob = sum(output_probs) / len(output_probs)

        if debug:
            print('enc_output:', enc_output[:10])
            print(output_probs)
            print(avg_output_prob)

        style_tok, bpm_tok = enc_output[1], enc_output[2]

        if not 768 <= style_tok < 788:
            style_tok = enc_output[1] = 787

        if not 788 <= bpm_tok < 820:
            bpm_tok = enc_output[2] = 819

        if cidx == 0:
            if style_bpm_mask:
                style_bpm_mask = []

            output_score.extend(enc_output)

        else:
            output_score.extend(enc_output[-len(chunk):])

        if debug:
            print('enc_output len:', len(enc_output), 'output_score len:', len(output_score))
            print('SOS:', enc_output[0])
            print('Style:', id_to_style[style_tok-768].title())
            print('BPM:', id_to_bpm[bpm_tok-788])
            
        output_styles_bpms_probs.append(tuple([style_tok, bpm_tok, avg_output_prob]))

        '''
        style_min_avg_max_prob = min(style_probs), sum(style_probs) / len(style_probs), max(style_probs)
        bpm_min_avg_max_prob = min(bpm_probs), sum(bpm_probs) / len(bpm_probs), max(bpm_probs)

        if debug:
            print('Style min/avg/max:', style_min_avg_max_prob)
            print('Mode style:', mode_style)
            print('BPM', bpm_min_avg_max_prob)
            print('Mode BPM:', mode_bpm)

        mode_style = statistics.mode(style_preds)
        mode_bpm = statistics.mode(bpm_preds)
        '''
        
    return output_score, output_styles_bpms_probs
    
# =================================================================================================

@spaces.GPU
def cls_style_bpm(drum_score,
                  style,
                  bpm,
                  debug=False,
                 ):

    style_tok = style_to_id[style.lower()]+768
    bpm_tok = bpm_to_id[bpm]+788

    if not 768 <= style_tok < 788:
        style_tok = enc_output[1] = 787

    if not 788 <= bpm_tok < 820:
        bpm_tok = enc_output[2] = 819

    if debug:
        print('-' * 70)
        print('Style / BPM CLS seq lens:', STYLE_CLS_SEQ_LEN, '/', BPM_SEQ_LEN)
        print('Input style:', style)
        print('Input BPM:', bpm)
        print('-' * 70)

    inp_seq = drum_score[:ENC_SEQ_LEN]

    style_preds, style_probs = predict(style_model, [inp_seq])
    bpm_preds, bpm_probs = predict(bpm_model, [inp_seq])

    if debug:
        print('Predicted style:', style_preds[0], '/', id_to_style[style_preds[0]])
        print('Style prob:', style_probs[0])
        print('Predicted BPM', bpm_preds[0], '/', id_to_bpm[bpm_preds[0]])
        print('BPM prob:', bpm_probs[0])

    return drum_score, (id_to_style[style_preds[0]].title(), id_to_bpm[bpm_preds[0]], style_probs[0], bpm_probs[0])

# =================================================================================================

@spaces.GPU
def Master_Drum_Track(input_midi,
                      requested_model,
                      drum_track_style,
                      drum_track_bpm,
                      mastering_mode,
                      num_master_iter,
                      debug
                     ):

    debug_log = None
    print.clear()

    if input_midi is not None:
    
        print('=' * 70)
        print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
        start_time = reqtime.time()

        print('=' * 70)
        fn = os.path.basename(input_midi.name)
        fn1 = fn.split('.')[0]
        print('Input file name:', fn)
        print('Req model:', requested_model)
        print('Drum track style:', drum_track_style)
        print('Drum track BPM:', drum_track_bpm)
        print('Mastering mode:', mastering_mode)
        print('Num main iter:', num_master_iter)
        print('Debug:', debug)
        
        print('=' * 70)
        print('Loading MIDI...')
        drum_score, inst_score, drum_track_start_time = process_midi(input_midi.name)

        if drum_track_start_time == -1:
            run_status = 'MIDI does not have drum track! Please upload a MIDI with a drum track.'
            print('Run status:', run_status)

            if debug:
                debug_log = print.get()
            
            return run_status, None, None, None, None, None, debug_log

        if drum_track_start_time == -2:
            run_status = 'Bad MIDI! Please upload a good MIDI!'
            print('Run status:', run_status)

            if debug:
                debug_log = print.get()
            
            run_status, None, None, None, None, None, debug_log
        
        print('Instrumental score has', len(inst_score), 'tokens')
        print('Sample instrumental score events:', inst_score[:3])    
        print('Drum track has', len(drum_score), 'tokens')
        print('Sample drum track tokens:', drum_score[:10])
        print('=' * 70)
    
        #===============================================================================

        if requested_model == 'Mastering MIRE':
            print('Will use MIRE Encoder...')
            print('Predicting...')
            
            score_output, styles_bpms_probs = encoder_mastering(enc_model,
                                                                drum_score,
                                                                num_master_iter,
                                                                drum_track_style,
                                                                drum_track_bpm,
                                                                mastering_mode,
                                                                debug=debug
                                                               )

            styles = []
            bpms = []
            probs = []
            
            for style, bpm, prob in styles_bpms_probs:
                styles.append(id_to_style[style-768].title())
                bpms.append(id_to_bpm[bpm-788])
                probs.append(prob)
                
            mode_style = statistics.mode(styles)
            mode_bpm = statistics.mode(bpms)
            avg_prob = statistics.mean(probs)
    
            print('Styles:', styles)
            print('Mode style:', mode_style)
            print('BPMs:', bpms)
            print('Mode BPM:', mode_bpm)
            print('Probs:', probs)
            print('Avg prob:', avg_prob)

            output_style, output_bpm = f'{mode_style} / {avg_prob}', f'{mode_bpm} / {avg_prob}'
            
        else:
            print('Will use Style and BPM MLCLS...')
            print('Predicting...')
            
            score_output, cls_output = cls_style_bpm(drum_score,
                                                     drum_track_style,
                                                     drum_track_bpm,
                                                     debug=debug
                                                    )

            print(cls_output)

            mode_style = cls_output[0]
            mode_bpm = cls_output[1]
            style_prob = cls_output[2]
            bpm_prob = cls_output[3]

            output_style, output_bpm = f'{mode_style} / {style_prob}', f'{mode_bpm} / {bpm_prob}'

        print('Done!')
        print('=' * 70)
        
        #===============================================================================
        
        print('=' * 70)        
        print('Saving MIDI...')
        print('=' * 70)

        output_fname, output_score = save_midi(score_output, inst_score, drum_track_start_time)

        if output_fname is None or output_score is None:
            run_status = 'There was a problem saving MIDI! Please try again!'
            print('Run status:', run_status)

            if debug:
                debug_log = print.get()
            
            return run_status, None, None, None, None, None, debug_log
            
        print('Done!')
        print('=' * 70)
        
        #===============================================================================        
        print('Rendering results...')
        print('=' * 70)            
        
        audio = midi_to_colab_audio(output_fname+'.mid', 
                                    soundfont_path=SOUNDFONT_PATH,
                                    sample_rate=16000,
                                    output_for_gradio=True
                                    )
    
        #========================================================
    
        output_audio = (16000, audio)
        
        output_plot = TMIDIX.plot_ms_SONG(output_score,
                                          timings_multiplier=16,
                                          plot_title=os.path.basename(output_fname)+'.mid',
                                          return_plt=True
                                         )
    
        print('Done!')
        print('=' * 70) 
        
        #========================================================
        
        print('-' * 70)
        print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
        print('-' * 70)
        print('Req execution time:', (reqtime.time() - start_time), 'sec')

        run_status = 'Done!'
        print('Run status:', run_status)

        if debug:
            debug_log = print.get()
    
        return run_status, output_style, output_bpm, output_audio, output_plot, output_fname+'.mid', debug_log
        
    run_status = 'No MIDI was uploaded! Please upload a MIDI!'
    print('Run status:', run_status)

    if debug:
        debug_log = print.get()
    
    return run_status, None, None, None, None, None, debug_log

# =================================================================================================
  
PDT = timezone('US/Pacific')

print('=' * 70)
print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
print('=' * 70)
 
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: left; margin-bottom: 1rem'>Drum Track Master</h1>")
    gr.Markdown("<h1 style='text-align: left; margin-bottom: 1rem'>Master, manipulate, and edit any MIDI drum track with midisimx MIRE and MLCLS transformers</h1>")
    with gr.Row(elem_classes="duplicate-row"):
        
        gr.DuplicateButton(
            value="🤗 Duplicate 🤗",
            variant="huggingface",
            size="md",
            link="https://huggingface.co/spaces/projectlosangeles/Drum-Track-Master?duplicate=true",
            link_target="_blank"
        )
        
        gr.Button(
            value="❤️ Models ❤️",
            variant="huggingface",
            size="md",
            link="https://huggingface.co/projectlosangeles/midisimx",
            link_target="_blank"
        )
      
        gr.Button(
            value="🦖 Dataset 🦖",
            variant="huggingface",
            size="md",
            link="https://huggingface.co/datasets/projectlosangeles/Discover-MIDI-Dataset",
            link_target="_blank"
        )
   
    gr.Markdown("## Upload your MIDI")

    input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])

    gr.Markdown("### Select task")
    requested_model = gr.Dropdown(label="Model to use",
                                  choices=['Mastering MIRE', 'Style and BPM MLCLS'],
                                  value='Mastering MIRE',
                                 )
    
    gr.Markdown("### MIRE settings")
    drum_track_style = gr.Dropdown(label="Drum track style", choices=[k.title() for k in list(style_to_id.keys())],
                                value="Pop",
                               )
    drum_track_bpm = gr.Dropdown(label="Drum track BPM", choices=list(bpm_to_id.keys()),
                                value=120,
                               )
    mastering_mode = gr.Dropdown(label="Which drum track components to master",
                                 choices=['Style',
                                          'BPM',
                                          'Velocities',
                                          'Pitches',
                                          'Durations',
                                          'Start Times'
                                         ],
                                          multiselect=True,
                                          type="value",
                                          value=['Velocities'],
                                         )
    num_master_iter = gr.Slider(1, 500, value=100, step=1, label="Number of mastering iterations")
    
    gr.Markdown("### Debug")
    inp_debug = gr.Checkbox(label="Debug", value=False)
    
    run_btn = gr.Button("Master", variant="primary")

    gr.Markdown("## Generation results")

    run_status = gr.Textbox(label="Run status")
    output_style = gr.Textbox(label="Drum track style and its probability")
    output_bpm = gr.Textbox(label="Drum track BPM and its probability")
    output_audio = gr.Audio(label="Output MIDI audio", format="mp3", elem_id="midi_audio")
    output_plot = gr.Plot(label="Output MIDI score plot")
    output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])
    out_debug = gr.Textbox(label="Debug")

    run_event = run_btn.click(Master_Drum_Track,
                              [input_midi,
                               requested_model,
                               drum_track_style,
                               drum_track_bpm,
                               mastering_mode,
                               num_master_iter,
                               inp_debug
                              ],
                              [run_status,
                               output_style,
                               output_bpm,
                               output_audio,
                               output_plot,
                               output_midi,
                               out_debug
                              ])
    
    demo.launch()