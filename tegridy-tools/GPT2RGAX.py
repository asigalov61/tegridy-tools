#! /usr/bin/python3

r'''###############################################################################
###################################################################################
#
#   GPT2RGAX.py
#
#	GPT-2 with Relative Global Attention Python Module
#   Experimental Version
#
#	Version 1.0
#
#   PLEASE NOTE THAT THIS IS A WORK IN PROGRESS
#   CHECK BACK FOR UPDATES SOON
#
#	Based upon a source-code of Sashmark97:
#   https://github.com/Sashmark97/midigen
#
#	Project Los Angeles
#	Tegridy Code 2021
#
#   https://github.com/Tegridy-Code/Project-Los-Angeles
#
#
###################################################################################
###################################################################################
#       Copyright 2021 Project Los Angeles / Tegridy Code
#
#       Licensed under the Apache License, Version 2.0 (the "License");
#       you may not use this file except in compliance with the License.
#       You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
#       Unless required by applicable law or agreed to in writing, software
#       distributed under the License is distributed on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#       See the License for the specific language governing permissions and
#       limitations under the License.
###################################################################################
###################################################################################'''

########################################################
#
# Critical dependencies/requirements:
#
# pip install torch
# pip install tqdm
# pip install matplotlib
#
########################################################

print('Loading GPT2-RGA Experimental Module...')

########################################################

import glob
import os
import sys
import math
import time
import random
import pickle
import joblib

from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.modules.normalization import LayerNorm
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import *

from torch.nn.functional import linear, softmax, dropout

########################################################

# Constants

SEQUENCE_START = 0
RANGE_NOTE_ON = 128
RANGE_NOTE_OFF = 128
RANGE_VEL = 32
RANGE_TIME_SHIFT = 100

# Taken from the paper
ADAM_BETA_1             = 0.9
ADAM_BETA_2             = 0.98
ADAM_EPSILON            = 10e-9

LR_DEFAULT_START        = 1.0
SCHEDULER_WARMUP_STEPS  = 4000
# LABEL_SMOOTHING_E       = 0.1

# DROPOUT_P               = 0.1

TOKEN_END               = 256+512 # RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_VEL + RANGE_TIME_SHIFT
TOKEN_PAD               = TOKEN_END + 1

VOCAB_SIZE              = TOKEN_PAD + 1

TORCH_FLOAT             = torch.float32
TORCH_INT               = torch.int32

TORCH_LABEL_TYPE        = torch.long

PREPEND_ZEROS_WIDTH     = 4

TORCH_CPU_DEVICE = torch.device("cpu")
USE_CUDA = 1
TORCH_CUDA_DEVICE = torch.device("cuda")

#====

weight_modulus = 1
print_modulus = 1
n_workers = 1

lr = None
ce_smoothing = None
batch_size = 4
random_seq = True
epochs = 5

rpr = False #'store_true'

enable_rpr = True
max_seq = 1024
n_layers = 6
num_heads = 8
d_model = 512
dim_feedforward = 512
dropout_prob = 0.1

########################################################

def cpu_device():

    return TORCH_CPU_DEVICE

def get_device():

    if((not USE_CUDA) or (TORCH_CUDA_DEVICE is None)):
        return TORCH_CPU_DEVICE
    else:
        return TORCH_CUDA_DEVICE

def train(cur_epoch, model, dataloader, loss, opt, lr_scheduler=None, num_iters=-1, save_checkpoint_steps=1000):
    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1
    loss_hist = []
    save_steps = 0
    out = -1
    model.train()
    with tqdm(total=len(dataloader)) as bar_train:
        for batch_num, batch in enumerate(dataloader):
            time_before = time.time()

            x   = batch[0].to(get_device())
            tgt = batch[1].to(get_device())

            y, _ = model(x)

            y   = y.reshape(y.shape[0] * y.shape[1], -1)
            tgt = tgt.flatten()

            out = loss.forward(y, tgt)

            out.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()

            if(lr_scheduler is not None):
                lr_scheduler.step()

            time_after = time.time()
            time_took = time_after - time_before
            lr = opt.param_groups[0]['lr']
            bar_train.set_description(f'Epoch: {cur_epoch} Loss: {float(out):.4} LR: {float(lr):.8}')
            bar_train.update(1)
            loss_hist.append(out.item())
            
            
            
            if save_steps % save_checkpoint_steps == 0:
                print('Saving model progress. Please wait...')
                print('gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')
                torch.save(model.state_dict(), 'gpt2_rpr_checkpoint_' + str(cur_epoch) + '_epoch_' + str(save_steps) + '_steps_' + str(round(float(out), 4)) + '_loss.pth')
                print('Done!')
                print('Saving training loss graph...')
                tr_loss_list = [sublist for sublist in loss_hist]
                plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
                plt.savefig('gpt2_rpr_checkpoint_training_loss_graph.png')
                print('Done! Continuing training...')
            save_steps +=1
            
            if batch_num == num_iters:
                break

    return loss_hist

def compute_epiano_accuracy(out, tgt):
    softmax = nn.Softmax(dim=-1)
    out = torch.argmax(softmax(out), dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != TOKEN_PAD)

    out = out[mask]
    tgt = tgt[mask]

    if(len(tgt) == 0):
        return 1.0

    num_right = (out == tgt)
    num_right = torch.sum(num_right).type(TORCH_FLOAT)

    acc = num_right / len(tgt)

    return acc

def eval_model(model, dataloader, loss, num_iters=-1):
    model.eval()

    avg_acc     = -1
    avg_loss    = -1
    with torch.set_grad_enabled(False):
        n_test      = len(dataloader)
        sum_loss   = 0.0
        sum_acc    = 0.0
        with tqdm(total=len(dataloader)) as bar_eval:
            for batch in dataloader:
                x   = batch[0].to(get_device())
                tgt = batch[1].to(get_device())

                y, _ = model(x)

                sum_acc += float(compute_epiano_accuracy(y, tgt))

                y   = y.reshape(y.shape[0] * y.shape[1], -1)
                tgt = tgt.flatten()

                out = loss.forward(y, tgt)

                sum_loss += float(out)
                bar_eval.set_description(f'Loss val: {float(out):.4}  Acc: {float(sum_acc / (bar_eval.n + 1)):.4}')
                bar_eval.update(1)
                if bar_eval.n == num_iters:
                    break

        avg_loss    = sum_loss / n_test
        avg_acc     = sum_acc / n_test

    return avg_loss, avg_acc

class LrStepTracker:

    def __init__(self, model_dim=512, warmup_steps=4000, init_steps=0):
        # Store Values
        self.warmup_steps = warmup_steps
        self.model_dim = model_dim
        self.init_steps = init_steps

        # Begin Calculations
        self.invsqrt_dim = (1 / math.sqrt(model_dim))
        self.invsqrt_warmup = (1 / (warmup_steps * math.sqrt(warmup_steps)))

    # step
    def step(self, step):

        step += self.init_steps
        if(step <= self.warmup_steps):
            return self.invsqrt_dim * self.invsqrt_warmup * step
        else:
            invsqrt_step = (1 / math.sqrt(step))
            return self.invsqrt_dim * invsqrt_step

# get_lr
def get_lr(optimizer):

    for param_group in optimizer.param_groups:
        return param_group['lr']

########################################################

#@title Functions
class EPianoDataset(Dataset):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Pytorch Dataset for the Maestro e-piano dataset (https://magenta.tensorflow.org/datasets/maestro).
    Recommended to use with Dataloader (https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)
    Uses all files found in the given root directory of pre-processed (preprocess_midi.py)
    Maestro midi files.
    ----------
    """

    def __init__(self, midi_list, max_seq=2048, random_seq=True):
        self.max_seq    = max_seq
        self.random_seq = random_seq
        self.data_files = midi_list

    def __len__(self):
        """
        ----------
        Author: Damon Gwinn
        ----------
        How many data files exist in the given directory
        ----------
        """

        return len(self.data_files)

    def __getitem__(self, idx):
        """
        ----------
        Author: Damon Gwinn
        ----------
        Gets the indexed midi batch. Gets random sequence or from start depending on random_seq.
        Returns the input and the target.
        ----------
        """

        raw_mid = torch.tensor(self.data_files, dtype=TORCH_LABEL_TYPE, device=cpu_device())
        x, tgt = process_midi(raw_mid, self.max_seq, self.random_seq)

        return x, tgt
    
def process_midi(raw_mid, max_seq, random_seq):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Takes in pre-processed raw midi and returns the input and target. Can use a random sequence or
    go from the start based on random_seq.
    ----------
    """

    x   = torch.full((max_seq, ), TOKEN_PAD,  dtype=TORCH_LABEL_TYPE, device=cpu_device())
    tgt = torch.full((max_seq, ), TOKEN_PAD,  dtype=TORCH_LABEL_TYPE, device=cpu_device())

    raw_len     = len(raw_mid)
    full_seq    = max_seq + 1 # Performing seq2seq

    if(raw_len == 0):
        return x, tgt

    start = 0
    end = 0

    # Randomly selecting a range
    if (random_seq):
        end_range = raw_len - full_seq
        start = random.randint(abs(SEQUENCE_START), abs(end_range))

    # Always taking from the start to as far as we can
    else:
      start = SEQUENCE_START

    end = start + full_seq

    data = raw_mid[start:end]

    x = data[:max_seq]
    tgt = data[1:full_seq]

    return x, tgt

########################################################

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.enable_rpr = config.enable_rpr
        if config.enable_rpr:
            self.attn = MultiheadAttentionRPR(config.n_embd, config.n_head, config.attn_pdrop, er_len=config.er_len)
        else:
            self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.dim_feedforward),
            nn.GELU(),
            nn.Linear(config.dim_feedforward, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask=None):
        if self.enable_rpr:
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)[0]
        else:
            x = x + self.attn(self.ln1(x)) 
        x = x + self.mlp(self.ln2(x))
        return x

class MultiheadAttentionRPR(nn.Module):
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/modules/activation.html#MultiheadAttention
    Modification to add RPR embedding Er and call custom multi_head_attention_forward_rpr
    ----------
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False,
                 add_zero_attn=False, kdim=None, vdim=None, er_len=None):
        super(MultiheadAttentionRPR, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        # Adding RPR embedding matrix
        if(er_len is not None):
            self.Er = Parameter(torch.rand((er_len, self.head_dim), dtype=torch.float32))
        else:
            self.Er = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):

        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            # return F.multi_head_attention_forward(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask, use_separate_proj_weight=True,
            #     q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
            #     v_proj_weight=self.v_proj_weight)

            return multi_head_attention_forward_rpr(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, rpr_mat=self.Er)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            # return F.multi_head_attention_forward(
            #     query, key, value, self.embed_dim, self.num_heads,
            #     self.in_proj_weight, self.in_proj_bias,
            #     self.bias_k, self.bias_v, self.add_zero_attn,
            #     self.dropout, self.out_proj.weight, self.out_proj.bias,
            #     training=self.training,
            #     key_padding_mask=key_padding_mask, need_weights=need_weights,
            #     attn_mask=attn_mask)

            return multi_head_attention_forward_rpr(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, rpr_mat=self.Er)

# multi_head_attention_forward_rpr
def multi_head_attention_forward_rpr(query,                       # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: Tensor
                                 in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: Tensor
                                 out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None,                   # type: Optional[Tensor]
                                 rpr_mat=None
                                 ):
    '''
    print('Query: ', query.shape, 'Key: ', key.shape, 'Value: ', value.shape)
    print('Equal: ', torch.equal(query, key) and torch.equal(key, value))
    print('embed_dim_to_check: ', embed_dim_to_check)
    print('num_heads:', num_heads)
    print('in_proj_weight: ', in_proj_weight.shape)
    print('in_proj_bias: ', in_proj_bias.shape)
    print('bias_k:', bias_k, 'bias_v', bias_v)
    print('add_zero_attn:', add_zero_attn)
    print('dropout_p: ', dropout_p)
    print('out_proj_weight: ', out_proj_weight.shape)
    print('out_proj_bias:', out_proj_bias.shape)
    print('training:', training)
    print('need_weights:', need_weights)
    print('use_separate_proj_weight:', use_separate_proj_weight)

    print('key_padding_mask:', key_padding_mask)
    print('attn_mask:', attn_mask.shape)
    print('q_proj_weight:', q_proj_weight)
    print('k_proj_weight:', k_proj_weight)
    print('v_proj_weight:', v_proj_weight)
    print('static_k:', static_k)
    print('static_v:', static_v)
    print('rpr_mat:', rpr_mat.shape)
    '''
    """
    ----------
    Author: Pytorch
    Modified: Damon Gwinn
    ----------
    For Relative Position Representation support (https://arxiv.org/abs/1803.02155)
    https://pytorch.org/docs/1.2.0/_modules/torch/nn/functional.html
    Modification to take RPR embedding matrix and perform skew optimized RPR (https://arxiv.org/abs/1809.04281)
    ----------
    """

    # type: (...) -> Tuple[Tensor, Optional[Tensor]]

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    ######### ADDITION OF RPR ###########
    if(rpr_mat is not None):
        rpr_mat = _get_valid_embedding(rpr_mat, q.shape[1], k.shape[1])
        qe = torch.einsum("hld,md->hlm", q, rpr_mat)
        srel = _skew(qe)

        attn_output_weights += srel

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)

    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

def _get_valid_embedding(Er, len_q, len_k):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Gets valid embeddings based on max length of RPR attention
    ----------
    """

    len_e = Er.shape[0]
    start = max(0, len_e - len_q)
    return Er[start:, :]

def _skew(qe):
    """
    ----------
    Author: Damon Gwinn
    ----------
    Performs the skew optimized RPR computation (https://arxiv.org/abs/1809.04281)
    ----------
    """

    sz = qe.shape[1]
    mask = (torch.triu(torch.ones(sz, sz).to(qe.device)) == 1).float().flip(0)

    qe = mask * qe
    qe = F.pad(qe, (1,0, 0,0, 0,0))
    qe = torch.reshape(qe, (qe.shape[0], qe.shape[2], qe.shape[1]))

    srel = qe[:, 1:, :]
    return srel

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.enable_rpr = config.enable_rpr

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        if self.enable_rpr:
            mask = generate_square_subsequent_mask(t).to(get_device())
        else:
            mask = None
            
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        if self.enable_rpr:
            x = x.permute(1,0,2)
            for module in self.blocks:
                x = module(x, mask=mask)
            x = x.permute(1,0,2)
        else:
            x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if self.enable_rpr:
            del mask
        return logits, loss
    
    def generate(self, primer=None, target_seq_length=1024, beam=0, beam_chance=1.0, temperature=1, 
                 stop_token=TOKEN_END, verbose=True):

        assert (not self.training), "Cannot generate while in training mode"

        if verbose: print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            logits, _ = self.forward(gen_seq[..., :cur_i])
            y = self.softmax(logits)[..., :stop_token+1]
            token_probs = y[:, cur_i-1, :] / (temperature if temperature > 0 else 1.)

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token


                # Let the transformer decide to end if it wants to
                if(next_token == stop_token):
                    if verbose: print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 50 == 0):
                if verbose: print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]

    def generate_batches(self, primer=None, target_seq_length=1024, temperature=1, num_batches=1, verbose=True):

        assert (not self.training), "Cannot generate while in training mode"

        if verbose: print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((num_batches,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())

        cur_i = num_primer
        while(cur_i < target_seq_length):
            logits, _ = self.forward(gen_seq[..., :cur_i])
            y = self.softmax(logits)[..., :]
            token_probs = y[:, cur_i-1, :] / (temperature if temperature > 0 else 1.)


            distrib = torch.distributions.categorical.Categorical(probs=token_probs)
            next_token = distrib.sample()
            gen_seq[:, cur_i] = next_token

            cur_i += 1
            if(cur_i % 50 == 0):
                if verbose: print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]
    
def generate_square_subsequent_mask(sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, dim_feedforward, enable_rpr=False, er_len=None, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.dim_feedforward = dim_feedforward
        self.enable_rpr = enable_rpr
        self.er_len = er_len
        for k,v in kwargs.items():
            setattr(self, k, v)
import logging
logger = logging.getLogger(__name__)

########################################################

def TrainDataLoader(train_data, train_data_ratio=0.333, 
                    train_data_shuffle=True, val_data_ratio=0.03, 
                    test_data_ratio=0.03, number_of_batches = 16, 
                    n_workers = 6):
    
    print('=' * 50)
    print('Loading train data. Please wait...')
    
    train_d = train_data[:int(len(train_data) * train_data_ratio)]

    val_dataset = train_data[:int(len(train_data) * val_data_ratio)]
    test_dataset = train_data[:int(len(train_data) * test_data_ratio)]

    train_list = train_d
    val_list = val_dataset
    test_list = [] # test_dataset
    print('=' * 50)

    print('Processing INTs datasets...')
    train_dataset = EPianoDataset(train_list, max_seq, random_seq)
    val_dataset = EPianoDataset(val_list, max_seq)
    test_dataset = EPianoDataset(test_list, max_seq)
    print('=' * 50)

    print('Loading INTs datasets...')
    batch_size = number_of_batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=train_data_shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=n_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers)
    print('=' * 50)

    print('Total INTs in the dataset', len(train_d))
    print('Total unique INTs in the dataset', len(set(train_d)))
    print('Max INT in the dataset', max(train_d))
    print('Min INT in the dataset', min(train_d))
    print('=' * 50)

    print('Checking datasets shapes...')
    print('=' * 50)

    print('Train loader')
    for x, tgt in train_loader:
        print(f'X shape: {x.shape}')
        print(f'Target shape: {tgt.shape}')
        break
    print('=' * 50)

    print('Validation loader')
    for x, tgt in val_loader:
        print(f'X shape: {x.shape}')
        print(f'Target shape: {tgt.shape}')
        break
    print('=' * 50)

    print('Test loader')
    for x, tgt in test_loader:
        print(f'X shape: {x.shape}')
        print(f'Target shape: {tgt.shape}')
        break
    print('=' * 50)

    print('Done! Enjoy! :)')
    print('=' * 50)

    return train_loader, val_loader, test_loader

########################################################

def TrainNewModel(train_loader, val_loader, test_loader):

    config = GPTConfig(VOCAB_SIZE, 
                       max_seq,
                       dim_feedforward=dim_feedforward,
                       n_layer=n_layers, 
                       n_head=num_heads, 
                       n_embd=d_model,
                       enable_rpr=enable_rpr,
                       er_len=max_seq)

    model = GPT(config).to(get_device())

    #=====

    init_step = 0
    lr = LR_DEFAULT_START
    lr_stepper = LrStepTracker(d_model, SCHEDULER_WARMUP_STEPS, init_step)
    eval_loss_func = nn.CrossEntropyLoss(ignore_index=TOKEN_PAD)
    train_loss_func = eval_loss_func

    opt = Adam(model.parameters(), lr=lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
    lr_scheduler = LambdaLR(opt, lr_stepper.step)


    #===

    best_eval_acc        = 0.0
    best_eval_acc_epoch  = -1
    best_eval_loss       = float("inf")
    best_eval_loss_epoch = -1
    best_acc_file = 'gpt2_rpr_acc.pth'
    best_loss_file = 'gpt2_rpr_loss.pth'
    loss_train, loss_val, acc_val = [], [], []

    for epoch in range(0, epochs):
        new_best = False

        loss = train(epoch+1, model, train_loader, train_loss_func, opt, lr_scheduler, num_iters=-1)
        loss_train.append(loss)

        eval_loss, eval_acc = eval_model(model, val_loader, eval_loss_func, num_iters=-1)
        loss_val.append(eval_loss)
        acc_val.append(eval_acc)

        if(eval_acc > best_eval_acc):
            best_eval_acc = eval_acc
            best_eval_acc_epoch  = epoch+1
            torch.save(model.state_dict(), best_acc_file)
            new_best = True

        if(eval_loss < best_eval_loss):
            best_eval_loss       = eval_loss
            best_eval_loss_epoch = epoch+1
            torch.save(model.state_dict(), best_loss_file)
            new_best = True

        if(new_best):
            print("Best eval acc epoch:", best_eval_acc_epoch)
            print("Best eval acc:", best_eval_acc)
            print("")
            print("Best eval loss epoch:", best_eval_loss_epoch)
            print("Best eval loss:", best_eval_loss)

########################################################        
        

def plot_losses(losses, path_to_output_image_to):
    print('Plotting Loss Graph...')
    tr_loss_list = [item for sublist in losses for item in sublist]
    plt.plot([i for i in range(len(tr_loss_list))] ,tr_loss_list, 'b')
    plt.savefig(path_to_output_image_to)
    print('Done!')

########################################################

print('GPT2-RGA-X loading complete!')
print('Enjoy!')

########################################################
########################################################
