r'''############################################################################
#===============================================================================
#
# X Trasformer Module
#
# x-transformers code With useful modifications
#
# Version 1.27.16
#
# Original source code courtesy of lucidrains
# https://github.com/lucidrains/x-transformers
#
# Source code retrieved on 02/20/2024
#
# Project Los Angeles
# Tegridy Code 2024
#
#===============================================================================
#
# Critical dependencies
#
# !pip install torch
# !pip install einops
#
#===============================================================================
#
# Basic use example
#
# from x_transformer_1_27_16 import *
#
# model = instantiate_x_transformer_model()
# save_x_transformer_model(model)
# load_x_transformer_model('model_checkpoint_1_epochs_1_steps_0_loss_1_acc.pth')
#
#===============================================================================
'''

################################################################################

################################################################################
# Code for x-transformers Python module attend.py
################################################################################

from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat

# constants

@dataclass
class Intermediates:
    qk_similarities: Optional[Tensor] = None
    pre_softmax_attn: Optional[Tensor] = None
    post_softmax_attn: Optional[Tensor] = None
    cached_kv: Optional[Tuple[Tensor, Tensor]] = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def compact(arr):
    return [*filter(exists, arr)]

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# functions for creating causal mask
# need a special one for onnx cpu (no support for .triu)

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def onnx_create_causal_mask(i, j, device):
    r = torch.arange(i, device = device)
    causal_mask = rearrange(r, 'i -> i 1') < rearrange(r, 'j -> 1 j')
    causal_mask = F.pad(causal_mask, (j - i, 0), value = False)
    return causal_mask

# main class

class Attend(nn.Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        talking_heads = False,
        sparse_topk = None,
        scale = None,
        qk_norm = False,
        flash = False,
        add_zero_kv = False,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.scale = scale
        self.qk_norm = qk_norm

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        self.attn_fn = partial(F.softmax, dtype = torch.float32) if not qk_norm else F.softmax

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and talking_heads), 'talking heads not compatible with flash attention'

        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)

        # sparse topk

        assert not (flash and sparse_topk), 'sparse topk not compatible with flash attention'
        self.sparse_topk = sparse_topk

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # flash attention

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        self.sdp_kwargs = sdp_kwargs

    def flash_attn(
        self,
        q, k, v,
        mask = None,
        attn_bias = None
    ):
        batch, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if self.qk_norm:
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        causal = self.causal

        # in the case of kv caching with one token (q_len == 1), just turn off causal masking
        # in speculative decoding, this may go up to 5-6, so right aligned causal mask will be needed there

        if q_len == 1 and causal:
            causal = False

        # expand key padding mask

        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, 'h i j -> 1 h i j').expand(batch, heads, -1, -1)

            # if mask given, the mask would already contain the causal mask from above logic
            # otherwise, if no mask given but still causal, mask out alibi positional bias to a large negative number

            mask_value = -torch.finfo(q.dtype).max

            if exists(mask):
                attn_bias = attn_bias.masked_fill(~mask, mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device = device)
                attn_bias = attn_bias.masked_fill(causal_mask, mask_value // 2)
                causal = False

            # scaled_dot_product_attention handles attn_mask either as bool or additive bias
            # make it an additive bias here

            mask = attn_bias

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked):
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, Intermediates()

    def forward(
        self,
        q, k, v,
        mask = None,
        attn_bias = None,
        prev_attn = None
    ):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, heads, kv_heads, device = q.shape[-2], q.shape[1], k.shape[1], q.device

        scale = default(self.scale, q.shape[-1] ** -0.5)

        causal = self.causal

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = map(lambda t: rearrange(t, 'b 1 n d -> b n d'), (k, v))
        elif kv_heads < heads:
            k, v = map(lambda t: repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads), (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = map(lambda t: F.pad(t, (0, 0, 1, 0), value = 0.), (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        if exists(prev_attn):
            dots = dots + prev_attn

        qk_similarities = dots.clone()

        if self.talking_heads:
            dots = self.pre_softmax_talking_heads(dots)

        if exists(attn_bias):
            dots = dots + attn_bias

        i, j, dtype = *dots.shape[-2:], dots.dtype

        mask_value = -torch.finfo(dots.dtype).max

        if exists(self.sparse_topk) and self.sparse_topk < j:
            top_values, _ = dots.topk(self.sparse_topk, dim = -1)
            sparse_topk_mask = dots < top_values[..., -1:]
            mask = (mask & sparse_topk_mask) if exists(mask) else sparse_topk_mask

        if exists(mask):
            dots = dots.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            dots = dots.masked_fill(causal_mask, mask_value)

        pre_softmax_attn = dots.clone()

        attn = self.attn_fn(dots, dim = -1)
        attn = attn.type(dtype)

        post_softmax_attn = attn.clone()

        attn = self.attn_dropout(attn)

        if self.talking_heads:
            attn = self.post_softmax_talking_heads(attn)

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        return out, intermediates

################################################################################

################################################################################
# Code for x-transformers Python module x_transformers.py
################################################################################

import math
from random import random
from typing import Dict
from packaging import version

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast

from functools import partial, wraps
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Callable, Optional, Union

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

# constants

DEFAULT_DIM_HEAD = 64

@dataclass
class LayerIntermediates:
    hiddens:            Optional[List[Tensor]] = None   # all hiddens, before the final norm (in pre-norm architecture)
    last_hidden:        Optional[Tensor] = None         # very last hidden after all attention layers, after the final norm
    attn_intermediates: Optional[List[Intermediates]] = None
    layer_hiddens:      Optional[List[Tensor]] = None
    attn_z_loss:        Optional[Tensor] = None
    mems:               Optional[Tensor] = None
    memory_tokens:      Optional[Tensor] = None

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def at_most_one_of(*bools):
    return sum(map(int, bools)) <= 1

class always():
    def __init__(self, val):
        self.val = val
    def __call__(self, *args, **kwargs):
        return self.val

class not_equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x != self.val

class equals():
    def __init__(self, val):
        self.val = val
    def __call__(self, x, *args, **kwargs):
        return x == self.val

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

# tensor helpers

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: List[Tensor],
    mask = None,
    weight = 1.
):
    # the same loss applied to the mixture of experts router logits in https://arxiv.org/abs/2202.08906
    # in the paper, in a tiny footnote, they mention using it on attention logits with stabilizing effects
    # also used in PaLM as one of the measures

    lse = 0.

    for attn in pre_softmax_attns:
        lse = lse + attn.logsumexp(dim = -1)

    loss = torch.square(lse)
    loss = reduce(loss, 'b h n -> b n', 'sum')

    if not exists(mask):
        return loss.mean() * weight

    loss = loss[mask].sum() / mask.sum().clamp(min = 1e-5)
    return loss * weight

# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# structured dropout, more effective than traditional attention dropouts

def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device = device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1. - dropout
    num_keep = max(1,  int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim = 1).indices

    batch_indices = torch.arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2

# embedding

class TokenEmbedding(nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal = False, num_buckets = 32, max_distance = 128, heads = 8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        device = self.device
        q_pos = torch.arange(j - i, j, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class DynamicPositionBias(nn.Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = nn.ModuleList([])

        self.mlp.append(Sequential(
            nn.Linear(1, dim),
            LayerNorm(dim) if norm else None,
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else None,
                nn.SiLU()
            ))

        self.mlp.append(nn.Linear(dim, heads))

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, i, j):
        assert i == j
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = torch.arange(n, device = device)
        context_arange = torch.arange(n, device = device)
        indices = rearrange(seq_arange, 'i -> i 1') - rearrange(context_arange, 'j -> 1 j')
        indices += (n - 1)

        # input to continuous positions MLP
        pos = torch.arange(-n + 1, n, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, total_heads, **kwargs):
        super().__init__()
        self.heads = heads
        self.total_heads = total_heads

        slopes = Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    def get_bias(self, i, j, device):
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = 0)
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast(enabled = False)
    def forward(self, t):
        max_pos = t.max()+1

        freqs = torch.einsum('i , j -> i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[-seq_len:, :]
    scale = scale[-seq_len:, :] if isinstance(scale, torch.Tensor) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t, t_unrotated), dim = -1)

# norms

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1) * (dim ** -0.5))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True)
        return x / norm.clamp(min = self.eps) * self.g

class LayerNorm(nn.Module):
    def __init__(self, dim):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

if version.parse(torch.__version__) >= version.parse('2.1.0'):
    LayerNorm = partial(nn.LayerNorm, bias = False)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.g

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

# residual and residual gates

class Residual(nn.Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1.):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t
    else:
        amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        mult_bias = False
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.mult_bias = nn.Parameter(torch.ones(dim_out)) if mult_bias else 1.

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate) * self.mult_bias

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            project_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            project_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        self.ff = Sequential(
            project_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

# attention. it is all we need

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        dim_context = None,
        heads = 8,
        causal = False,
        flash = False,
        talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_value_heads = False,
        swiglu_values = False,
        gate_values = False,
        zero_init_output = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        qk_norm_dim_scale = False,
        one_kv_head = False,
        kv_heads = None,
        shared_kv = False,
        value_dim_head = None,
        tensor_product = False,      # https://arxiv.org/abs/2208.06061
        add_zero_kv = False,         # same as add_zero_attn in pytorch
        rotary_embed_values = False,
        onnxable = False
    ):
        super().__init__()
        dim_kv = default(dim_context, dim)

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        assert not (exists(kv_heads) and one_kv_head), 'either attn_one_kv_head is set to True (in which case kv_heads is set to 1), or attn_kv_heads is set, but not both'

        value_dim_head = default(value_dim_head, dim_head)
        kv_heads = default(kv_heads, heads)

        kv_heads = 1 if one_kv_head else kv_heads
        assert divisible_by(heads, kv_heads)

        self.kv_heads = kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim_kv, k_dim, bias = False)

        # shared key / values, for further memory savings during inference
        assert not (shared_kv and value_dim_head != dim_head), 'key and value head dimensions must be equal for shared key / values'
        self.to_v = nn.Linear(dim_kv, v_dim, bias = False) if not shared_kv else None

        # relations projection from tp-attention
        self.to_r = nn.Linear(dim, v_dim, bias = False) if tensor_product else None

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            self.to_v_gate_activation = F.silu if swiglu_values else F.sigmoid
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 10)

        # add per head gating of the output values, from 'Attend to nothing' paper
        self.to_v_head_gate = None
        if gate_value_heads:
            self.to_v_head_gate = nn.Linear(dim, heads)
            nn.init.constant_(self.to_v_head_gate.weight, 0)
            nn.init.constant_(self.to_v_head_gate.bias, 10)

        # cosine sim attention
        self.qk_norm = qk_norm
        self.qk_norm_groups = qk_norm_groups
        self.qk_norm_scale = qk_norm_scale

        # whether to use the rmsnorm (equivalent to cosine sim attention when scale is equal to 1) - https://arxiv.org/abs/2302.05442
        self.qk_norm_dim_scale = qk_norm_dim_scale

        self.qk_norm_q_scale = self.qk_norm_k_scale = 1
        if qk_norm and qk_norm_dim_scale:
            self.qk_norm_q_scale = nn.Parameter(torch.ones(heads, 1, dim_head))
            self.qk_norm_k_scale = nn.Parameter(torch.ones(heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads = heads,
            causal = causal,
            talking_heads = talking_heads,
            dropout = dropout,
            sparse_topk = sparse_topk,
            qk_norm = qk_norm,
            scale = qk_norm_scale if qk_norm else self.scale,
            add_zero_kv = add_zero_kv,
            flash = flash,
            onnxable = onnxable
        )

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(kv_heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(out_dim, dim * 2, bias = False), nn.GLU()) if on_attn else nn.Linear(out_dim, dim, bias = False)

        # whether to rotate positions into values, for absolute positions in addition to relative
        self.rotary_embed_values = rotary_embed_values

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None,
        mem_mask = None,
        return_intermediates = False,
        cache: Optional[Intermediates] = None,
    ):
        b, n, h, kv_h, head_scale, device, has_context = x.shape[0], x.shape[1], self.heads, self.kv_heads, self.head_scale, x.device, exists(context)

        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input
        r_input = x

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input) if exists(self.to_v) else k
        r = self.to_r(r_input) if exists(self.to_r) else None

        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k, v, r = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = kv_h), (k, v, r))

        if exists(cache) and not has_context:
            ck, cv = cache.cached_kv

            if exists(mem):
                mk, k = unpack(k, mem_packed_shape, 'b h * d')
                mv, v = unpack(v, mem_packed_shape, 'b h * d')

            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

            if exists(mem):
                k = torch.cat((mk, k), dim = -2)
                v = torch.cat((mv, v), dim = -2)

        if return_intermediates:
            mem_len = mem.shape[-2] if exists(mem) else 0
            cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        if exists(rotary_pos_emb) and not has_context:
            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)
            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if self.rotary_embed_values:
                v = apply_rotary_pos_emb(v, freqs, k_xpos_scale)

        input_mask = context_mask

        if not exists(input_mask) and not has_context:
            input_mask = mask

            if (exists(input_mask) or exists(mem_mask)) and exists(mem):
                seq_len, mem_len = n, mem.shape[-2]

                if not exists(mem_mask):
                    input_mask = pad_at_dim(input_mask, (mem_len, 0), dim = -1, value = True)
                elif not exists(input_mask):
                    input_mask = pad_at_dim(mem_mask, (0, seq_len), dim = -1, value = True)
                else:
                    input_mask = torch.cat((mem_mask, input_mask), dim = -1)

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = torch.cat((mem_k, k), dim = -2)
            v = torch.cat((mem_v, v), dim = -2)

            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)

        i, j = map(lambda t: t.shape[-2], (q, k))

        # determine masking

        mask_value = max_neg_value(q)
        masks = []
        final_attn_mask = None

        if exists(input_mask):
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            masks.append(~attn_mask)

        if exists(self.max_attend_past):
            range_q = torch.arange(j - i, j, device = device)
            range_k = torch.arange(j, device = device)
            dist = rearrange(range_q, 'i -> 1 1 i 1') - rearrange(range_k, 'j -> 1 1 1 j')
            max_attend_past_mask = dist > self.max_attend_past
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        attn_bias = None
        if exists(rel_pos):
            attn_bias = rel_pos(i, j)

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask = final_attn_mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )

        # https://arxiv.org/abs/2208.06061 proposes to add a residual for better gradients

        if exists(r):
            out = out * r + out

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # per head gating, from https://arxiv.org/abs/2306.12929

        if exists(self.to_v_head_gate):
            head_gate = self.to_v_head_gate(x)
            out = out * rearrange(head_gate, 'b n h -> b h n 1').sigmoid()

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * self.to_v_gate_activation(gates)

        # combine the heads

        out = self.to_out(out)

        if exists(mask):
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_simple_rmsnorm = False,
        alibi_pos_bias = False,
        alibi_num_heads = None,
        rel_pos_bias = False,
        rel_pos_num_buckets = 32,
        rel_pos_max_distance = 128,
        dynamic_pos_bias = False,
        dynamic_pos_bias_log_distance = False,
        dynamic_pos_bias_mlp_depth = 2,
        dynamic_pos_bias_norm = False,
        rotary_pos_emb = False,
        rotary_emb_dim = None,
        rotary_xpos = False,
        rotary_interpolation_factor = 1.,
        rotary_xpos_scale_base = 512,
        rotary_base_rescale_factor = 1.,
        custom_layers = None,
        sandwich_coef = None,
        par_ratio = None,
        weight_tie_layers = False,   # Albert - https://arxiv.org/abs/1909.11942
        layers_execute_order = None, # generalizes weight tying, can do arbitrary layer execution orders
        residual_attn = False,
        cross_residual_attn = False,
        macaron = False,
        pre_norm = True,
        pre_norm_has_final_norm = True,
        gate_residual = False,
        scale_residual = False,
        scale_residual_constant = 1.,
        shift_tokens = 0,
        sandwich_norm = False,
        resi_dual = False,
        resi_dual_scale = 1.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        disable_abs_pos_emb = None,
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)
        cross_attn_kwargs, kwargs = groupby_prefix_and_trim('cross_attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.disable_abs_pos_emb = default(disable_abs_pos_emb, (rel_pos_bias or rotary_pos_emb))

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)

        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        assert not (alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert (int(rel_pos_bias) + int(dynamic_pos_bias) + int(alibi_pos_bias)) <= 1, 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None
        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads)

        assert (int(sandwich_norm) + int(resi_dual)) <= 1, 'either sandwich norm or resiDual is selected, but not both'
        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        if resi_dual:
            pre_norm = False

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.resi_dual = resi_dual
        assert 0 < resi_dual_scale <= 1., 'resiDual prenorm residual must be scaled by a factor greater than 0 and less than or equal to 1.'
        self.resi_dual_scale = resi_dual_scale

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        assert (int(use_scalenorm) + int(use_rmsnorm) + int(use_simple_rmsnorm)) <= 1, 'you can only use either scalenorm, rmsnorm, or simple rmsnorm'

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        else:
            norm_class = LayerNorm

        norm_fn = partial(norm_class, dim)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn  = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm or resi_dual else nn.Identity()

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads = heads, causal = causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **{**attn_kwargs, **cross_attn_kwargs})
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant)

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        self_attn_kv_mask = None,
        mems = None,
        mem_masks = None,
        seq_start_pos: Optional[Tensor] = None,
        cache: Optional[LayerIntermediates] = None,
        cache_age = 1,
        return_hiddens = False,
        rotary_pos_emb = None
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        mem_masks = mem_masks.copy() if exists(mem_masks) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = torch.arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        if not exists(rotary_pos_emb) and exists(self.rotary_pos_emb):
            maybe_mem = mems[0] # todo - handle edge case where different layers get different memory lengths. don't think this will ever come up but who knows
            mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

            pos = torch.arange(x.shape[1] + mem_len, device = x.device) - mem_len
            rotary_pos_emb = self.rotary_pos_emb(pos)

        # assume cached key / values

        attn_cache = []

        if exists(cache):
            assert not self.training and self.causal and not any([*map(exists, (mask, attn_mask))])

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

            attn_cache = cache.attn_intermediates

        iter_attn_cache = iter(attn_cache)

        # outer residual - for resiDual paper

        outer_residual = x * self.resi_dual_scale

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.layers,
            self.layer_dropouts
        )

        layer_variables = tuple(tuple(layer_variable[i] for i in self.layers_execute_order) for layer_variable in layer_variables)

        # go through the attention and feedforward layers

        for ind, (layer_type, (norm, block, residual_fn), layer_dropout) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            if self.training and layer_dropout > 0. and random() < layer_dropout:
                continue

            if layer_type == 'a':
                if return_hiddens:
                    hiddens.append(x)

                layer_mem = mems.pop(0) if mems else None
                layer_mem_mask = mem_masks.pop(0) if mem_masks else None

            if layer_type == 'c':
                if self.training and self.cross_attn_tokens_dropout > 0.:
                    context, context_mask = dropout_seq(context, context_mask, self.cross_attn_tokens_dropout)

            inner_residual = x

            if return_hiddens:
                layer_hiddens.append(x)

            pre_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == 'a' and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, rotary_pos_emb = rotary_pos_emb, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, mem_mask = layer_mem_mask, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), return_intermediates = True)
            elif layer_type == 'f':
                out = block(x)

            if self.resi_dual:
                outer_residual = outer_residual + out * self.resi_dual_scale

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual)

            if layer_type in ('a', 'c') and return_hiddens:
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.resi_dual:
            x = x + self.final_norm(outer_residual)
        else:
            x = self.final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            last_hidden = x,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens,
        )

        return x, intermediates

class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal = False, **kwargs)

class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = True, **kwargs)

class PrefixDecoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal = False, **kwargs)

    def forward(
        self,
        x,
        *args,
        attn_mask = None,
        prefix_attn_len = None,
        **kwargs
    ):
        b, n, device = x.shape[0], x.shape[1], x.device
        causal_mask = torch.ones((n, n), device = device, dtype = torch.bool).triu(1)

        forwarded_mask = ~causal_mask

        if exists(prefix_attn_len):
            if isinstance(prefix_attn_len, int):
                prefix_attn_len = torch.full((b,), prefix_attn_len, device = device)

            prefix_mask = torch.arange(n, device = device) < rearrange(prefix_attn_len, 'b -> b 1 1 1')
            forwarded_mask = forwarded_mask | prefix_mask

        if exists(attn_mask):
            forwarded_mask = forwarded_mask & attn_mask

        return super().forward(x, *args, attn_mask = forwarded_mask, **kwargs)

class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend = True, only_cross = True, **kwargs)

class ViTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        attn_layers: Encoder,
        channels = 3,
        num_classes = None,
        post_emb_norm = False,
        num_register_tokens = 0,
        emb_dropout = 0.
    ):
        super().__init__()
        assert divisible_by(image_size, patch_size), 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        has_register_tokens = num_register_tokens > 0
        self.has_register_tokens = has_register_tokens

        if has_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))

        self.patch_to_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim)
        )

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers

        self.mlp_head = nn.Linear(dim, num_classes) if exists(num_classes) else nn.Identity()

    def forward(
        self,
        img,
        return_embeddings = False,
        return_logits_and_embeddings = False
    ):
        b, p = img.shape[0], self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        n = x.shape[1]

        x = x + self.pos_embedding[:, :n]

        x = self.post_emb_norm(x)
        x = self.dropout(x)

        if self.has_register_tokens:
            r = repeat(self.register_tokens, 'n d -> b n d', b = b)
            x, ps = pack((x, r), 'b * d')

        embed = self.attn_layers(x)

        if self.has_register_tokens:
            embed, _ = unpack(embed, ps, 'b * d')

        assert at_most_one_of(return_embeddings, return_logits_and_embeddings)

        if not exists(self.mlp_head) or return_embeddings:
            return embed

        pooled = embed.mean(dim = -2)
        logits = self.mlp_head(pooled)

        if not return_logits_and_embeddings:
            return logits

        return logits, embed

class TransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: Dict[str, int] = dict(),
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        tie_embedding = False,
        logits_dim = None,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        emb_frac_gradient = 1., # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed = l2norm_embed)

        # additional embeddings - say type embedding from BERT

        self.embeds = None

        if len(embed_num_tokens) > 0:
            self.embeds = nn.ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim, bias = False) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std = 1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)

    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: Dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: Optional[LayerIntermediates] = None,
        **kwargs
    ):
        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient = x.shape[0], x.shape[1], x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient
        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb

        # add additional embeddings

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if has_memory_tokens:
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim = -2, value = 0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b = x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, seq_start_pos = seq_start_pos, **kwargs)

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            x = x[:, :n]

        if return_logits_and_embeddings:
            out = (self.to_logits(x), x)
        elif return_embeddings:
            out = x
        else:
            out = self.to_logits(x)

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim = -2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

class XTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        tie_token_emb = False,
        ignore_index = -100,
        pad_value = 0,
        cross_attn_tokens_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec_kwargs, kwargs = groupby_prefix_and_trim('dec_', kwargs)

        assert 'dim' not in enc_kwargs and 'dim' not in dec_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)
        enc_transformer_kwargs['scaled_sinu_pos_emb'] = enc_kwargs.pop('scaled_sinu_pos_emb', False)
        enc_transformer_kwargs['use_abs_pos_emb'] = enc_kwargs.pop('use_abs_pos_emb', True)

        dec_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec_kwargs)
        dec_transformer_kwargs['emb_dropout'] = dec_kwargs.pop('emb_dropout', 0)
        dec_transformer_kwargs['scaled_sinu_pos_emb'] = dec_kwargs.pop('scaled_sinu_pos_emb', False)
        dec_transformer_kwargs['use_abs_pos_emb'] = dec_kwargs.pop('use_abs_pos_emb', True)

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout  # how many tokens from the encoder to dropout when cross attending from decoder - seen in a couple papers, including Perceiver AR - this will also be very effective regularization when cross attending to very long memories

        self.encoder = TransformerWrapper(
            **enc_transformer_kwargs,
            attn_layers = Encoder(dim = dim, **enc_kwargs)
        )

        self.decoder = TransformerWrapper(
            **dec_transformer_kwargs,
            attn_layers = Decoder(dim = dim, cross_attend = True, **dec_kwargs)
        )

        if tie_token_emb:
            self.decoder.token_emb = self.encoder.token_emb

        self.decoder = AutoregressiveWrapper(self.decoder, ignore_index=ignore_index, pad_value=pad_value)

    @torch.no_grad()
    def generate(self, seq_in, seq_out_start, seq_len, mask = None, attn_mask = None, **kwargs):
        encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)
        return self.decoder.generate(seq_out_start, seq_len, context = encodings, context_mask = mask, **kwargs)

    def forward(self, src, tgt, mask = None, attn_mask = None, src_prepend_embeds = None):

        enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)

        if exists(src_prepend_embeds) and exists(mask):
            mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.training and self.cross_attn_tokens_dropout > 0:
            enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)

        out = self.decoder(tgt, context = enc, context_mask = mask)
        return out

################################################################################

################################################################################
# Code for x-transformers Python module continuous.py
################################################################################

import torch
from torch import nn
import torch.nn.functional as F

from einops import pack, repeat, unpack

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# main classes

class ContinuousTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers: AttentionLayers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.post_emb_norm = LayerNorm(dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # project in and out

        self.project_in = nn.Linear(dim_in, dim, bias = False) if exists(dim_in) else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias = False) if exists(dim_out) else nn.Identity()

    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        mem_masks = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        x = self.project_in(x)
        x = x + self.pos_emb(x, pos = pos)

        x = self.post_emb_norm(x)

        # memory tokens

        if self.has_memory_tokens:
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((batch, seq), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((batch, prepend_seq), device = device, dtype = torch.bool))

                mask = torch.cat((prepend_mask, mask), dim = -1)

        x = self.emb_dropout(x)

        # attention layers

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, return_hiddens = True, **kwargs)

        # splice out memory tokens

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        out = self.project_out(x) if not return_embeddings else x

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

class ContinuousAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net: ContinuousTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        loss_fn = nn.MSELoss(reduction = 'none')
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.loss_fn = loss_fn

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'

        if num_dims == 2:
            start_tokens = start_tokens[None, :]        

        b, t, _, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            last = self.net(x, **kwargs)[:, -1:]
            out = torch.cat((out, last), dim = -2)

        out = out[:, t:]

        if num_dims == 2:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        inp, target = x[:, :-1], x[:, 1:]

        assert 'prepend_embeds' not in kwargs

        mask = kwargs.get('mask', None)
        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(inp, **kwargs)

        loss = self.loss_fn(out, target)

        if exists(mask):
            assert loss.ndim > 1, 'loss should not be reduced if mask is passed in'
            loss = loss[mask]

        return loss.mean()

################################################################################

################################################################################
# Code for x-transformers Python module dpo.py
################################################################################

from copy import deepcopy

import torch
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange

# helper functions

def exists(v):
    return v is not None

def freeze_all_layers_(module):
    for param in module.parameters():
        param.requires_grad = False

def log_prob_from_model_and_seq(model, seq):
    logits = model(seq)
    log_prob = logits.log_softmax(dim = -1)
    indices = rearrange(seq, '... -> ... 1')
    log_probs = log_prob.gather(-1, indices)
    return rearrange(log_probs, '... 1 -> ...')

def masked_mean(log_probs, mask = None):
    if not exists(mask):
        return log_probs.mean(dim = -1)

    log_probs = log_probs.masked_fill(~mask, 0.)
    num = log_probs.sum(dim = -1)
    den = mask.sum(dim = -1)
    return num / den.clamp(min = 1e-5)

def maybe_and_mask(*masks):
    masks = [*filter(exists, masks)]
    if len(masks) == 0:
        return None

    mask, *rest_masks = masks
    for rest_mask in rest_masks:
        mask = mask & rest_mask

    return mask

# main class

class DPO(Module):
    def __init__(
        self,
        model: TransformerWrapper,
        *,
        beta = 0.1,
        pad_id = None
    ):
        super().__init__()
        self.policy_model = model

        self.ref_model = deepcopy(model)
        freeze_all_layers_(self.ref_model)

        self.beta = beta
        self.pad_id = pad_id

    def parameters(self):
        return self.policy_model.parameters()

    def forward(
        self,
        preferred_seq,
        unpreferred_seq,
        *,
        prompt_mask,
        preferred_seq_mask = None,
        unpreferred_seq_mask = None,
    ):
        assert preferred_seq.ndim == 2
        assert preferred_seq.shape == unpreferred_seq.shape

        if exists(self.pad_id):
            if not exists(preferred_seq_mask):
                preferred_seq_mask = preferred_seq != self.pad_id

            if not exists(unpreferred_seq_mask):
                unpreferred_seq_mask = unpreferred_seq != self.pad_id

        """
        Following Appendix B in https://arxiv.org/abs/2305.18290
        """

        with torch.no_grad():
            self.ref_model.eval()
            ref_preferred_logprob = log_prob_from_model_and_seq(self.ref_model, preferred_seq)
            ref_unpreferred_logprob = log_prob_from_model_and_seq(self.ref_model, unpreferred_seq)

        policy_preferred_logprob = log_prob_from_model_and_seq(self.policy_model, preferred_seq)
        policy_unpreferred_logprob = log_prob_from_model_and_seq(self.policy_model, unpreferred_seq)

        # masked mean of log probs

        preferred_seq_mask = maybe_and_mask(~prompt_mask, preferred_seq_mask)
        unpreferred_seq_mask = maybe_and_mask(~prompt_mask, unpreferred_seq_mask)

        ref_preferred_logprob, policy_preferred_logprob = map(lambda t: masked_mean(t, preferred_seq_mask), (ref_preferred_logprob, policy_preferred_logprob))
        ref_unpreferred_logprob, policy_unpreferred_logprob = map(lambda t: masked_mean(t, unpreferred_seq_mask), (ref_unpreferred_logprob, policy_unpreferred_logprob))

        # main dpo formula

        policy_logratios = policy_preferred_logprob - policy_unpreferred_logprob
        ref_logratios = ref_preferred_logprob - ref_unpreferred_logprob

        losses = -F.logsigmoid(self.beta * (policy_logratios - ref_logratios))

        return losses.mean()

################################################################################

################################################################################
# Code for x-transformers Python module nonautoregressive_wrapper.py
################################################################################

import math
from random import random
from contextlib import nullcontext
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat, pack, unpack

from typing import Optional

# constants

Losses = namedtuple('Losses', ['loss', 'generator_loss', 'critic_loss'])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# sampling helpers

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

# prob helpers

def sample_prob(prob):
    return random() < prob

def coin_flip():
    return sample_prob(0.5)

# tensor helpers

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

# schedules

def linear_schedule(t):
    return 1 - t

def cosine_schedule(t):
    """ https://arxiv.org/abs/2202.04200 """
    return torch.cos(t * math.pi / 2)

# self token critic
# inspired by Nijkamp et al. - https://aclanthology.org/2021.naacl-main.409/

class SelfCritic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

        dim = net.attn_layers.dim
        self.to_logits = nn.Linear(dim, 1)

    def forward(self, x):
        embed = self.net(x, return_embeddings = True)
        return self.to_logits(embed)

class NonAutoregressiveWrapper(nn.Module):
    """
    https://arxiv.org/abs/1904.09324
    https://arxiv.org/abs/2202.04200
    """

    def __init__(
        self,
        net,
        *,
        mask_id,
        steps = 18,
        self_cond = False,
        self_cond_train_prob = 0.75,
        no_replace_prob = 0.15,          # which percentage of the tokens masked will stay the same, done in original MLM paper
        random_token_prob = 0.1,         # which percentage of tokens to be replaced with random token, done in original MLM paper
        schedule = 'linear',
        can_mask_prev_unmasked = False,  # when unmasking, whether it can remask previously unmasked
        token_critic: Optional[TransformerWrapper] = None,
        self_token_critic = False,
        critic_loss_weight = 1.
    ):
        super().__init__()
        assert not (self_token_critic and exists(token_critic))

        self.net = net

        dim = net.emb_dim
        self.dim = dim
        self.num_tokens = net.num_tokens

        self.mask_id = mask_id

        # afaict, maskgit paper did not do this
        # but may help for self conditioning, as used successfully in original BERT

        self.no_replace_prob = no_replace_prob
        self.random_token_prob = random_token_prob

        self.max_seq_len = net.max_seq_len
        self.steps = steps

        if callable(schedule):
            self.schedule_fn = schedule
        if schedule == 'linear':
            self.schedule_fn = linear_schedule
        elif schedule == 'cosine':
            self.schedule_fn = cosine_schedule
        else:
            raise ValueError(f'invalid schedule {schedule}')

        self.can_mask_prev_unmasked = can_mask_prev_unmasked

        # self conditioning

        self.self_cond = self_cond

        if self_cond:
            self.null_embed = nn.Parameter(torch.randn(dim))
            self.to_self_cond = nn.Linear(dim, dim, bias = False) if self_cond else None
            self.self_cond_train_prob = self_cond_train_prob

        # token critic

        self.token_critic = token_critic

        if self_token_critic:
            self.token_critic = SelfCritic(net)

        self.critic_loss_weight = critic_loss_weight

    @torch.no_grad()
    def generate(
        self,
        batch_size = None,
        start_temperature = 1.,
        filter_thres = 0.7,
        noise_level_scale = 1.,
        **kwargs
    ):
        sample_one = not exists(batch_size)
        batch_size = default(batch_size, 1)

        device = next(self.net.parameters()).device

        was_training = self.training
        self.eval()

        times = torch.linspace(0., 1., self.steps + 1)

        # sequence starts off as all masked

        shape = (batch_size, self.max_seq_len)

        seq = torch.full(shape, self.mask_id, device = device)
        mask = torch.full(shape, True, device = device)

        # slowly demask

        all_mask_num_tokens = (self.schedule_fn(times[1:]) * self.max_seq_len).long()

        # self conditioning

        has_self_cond = self.self_cond
        last_embed = self.null_embed if has_self_cond else None

        for mask_num_tokens, steps_until_x0 in zip(all_mask_num_tokens.tolist(), reversed(range(self.steps))):

            self_cond = self.to_self_cond(last_embed) if has_self_cond else None

            logits, embeds = self.net(
                seq,
                sum_embeds = self_cond,
                return_logits_and_embeddings = True,
                **kwargs
            )

            if has_self_cond:
                last_embed = embeds

            if exists(filter_thres):
                logits = top_k(logits, filter_thres)

            annealing_scale = steps_until_x0 / self.steps
            temperature = start_temperature * annealing_scale

            probs = (logits / max(temperature, 1e-3)).softmax(dim = -1)

            sampled_ids = gumbel_sample(logits, temperature = max(temperature, 1e-3))

            seq = torch.where(mask, sampled_ids, seq)

            if exists(self.token_critic):
                scores = self.token_critic(seq)
                scores = rearrange(scores, 'b n 1 -> b n')
                scores = scores + noise_level_scale * gumbel_noise(scores) * annealing_scale
            else:
                scores = 1 - logits.softmax(dim = -1)
                scores = scores.gather(2, rearrange(sampled_ids, 'b n -> b n 1'))
                scores = rearrange(scores, 'b n 1 -> b n')

            if mask_num_tokens == 0:
                pass

            if not self.can_mask_prev_unmasked:
                scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

            mask_indices = scores.topk(mask_num_tokens, dim = -1).indices
            mask = torch.zeros_like(scores, dtype = torch.bool).scatter(1, mask_indices, True)
            seq = seq.masked_fill(mask, self.mask_id)

        self.train(was_training)

        if sample_one:
            seq = rearrange(seq, '1 n -> n')

        return seq

    def forward(
        self,
        x,
        only_train_generator = False,
        only_train_critic = False,
        generator_sample_temperature = None,
        **kwargs
    ):
        b, n, device = *x.shape, x.device
        assert n == self.max_seq_len

        orig_seq = x.clone()

        rand_times = torch.empty(b, device = device).uniform_(0, 1)
        batched_randperm = torch.rand((b, n), device = device).argsort(dim = -1).float()

        rand_probs = self.schedule_fn(rand_times)
        num_tokens_mask = (rand_probs * n).clamp(min = 1.)
        mask = batched_randperm < rearrange(num_tokens_mask, 'b -> b 1')

        # to ensure all tokens produce embeddings, instead of just the ones with [mask] input, as done in seminal BERT MLM paper
        # potentially needed for self-conditioning (on embedding) to work well

        replace_mask_id_mask = mask.clone()
        frac_seq_left = 1.

        if self.no_replace_prob > 0. and coin_flip():
            frac_seq_left -= self.no_replace_prob

            no_replace_prob_mask = get_mask_subset_prob(mask, self.no_replace_prob)
            replace_mask_id_mask &= ~no_replace_prob_mask

        if self.random_token_prob > 0. and coin_flip():
            random_token_prob_mask = get_mask_subset_prob(replace_mask_id_mask, self.random_token_prob * frac_seq_left)
            random_tokens = torch.randint(0, self.num_tokens, (b, n), device = device)

            x = torch.where(random_token_prob_mask, random_tokens, x)
            replace_mask_id_mask &= ~random_token_prob_mask

        masked = torch.where(replace_mask_id_mask, self.mask_id, x)

        # self conditioning

        if self.self_cond:
            self_cond = self.null_embed

            if sample_prob(self.self_cond_train_prob):
                with torch.no_grad():
                    self_cond = self.net(masked, return_embeddings = True, **kwargs).detach()

            kwargs.update(sum_embeds = self.to_self_cond(self_cond))

        # logits

        context = torch.no_grad if only_train_critic else nullcontext

        with context():
            logits = self.net(masked, **kwargs)

        # cross entropy loss

        loss = F.cross_entropy(
            logits[mask],
            orig_seq[mask]
        )

        if not exists(self.token_critic) or only_train_generator:
            return Losses(loss, loss, None)

        sampled_ids = gumbel_sample(logits, temperature = default(generator_sample_temperature, random()))
        generated = torch.where(mask, sampled_ids, orig_seq)

        critic_logits = self.token_critic(generated)
        critic_labels = (sampled_ids != orig_seq).float()

        critic_loss = F.binary_cross_entropy_with_logits(
            rearrange(critic_logits, '... 1 -> ...'),
            critic_labels
        )

        # determine losses to be returned based on what researcher wants to train

        if only_train_critic:
            total_loss = critic_loss
            loss = None
        else:
            total_loss = loss + critic_loss * self.critic_loss_weight

        return Losses(total_loss, loss,  critic_loss)

################################################################################

################################################################################
# Code for x-transformers Python module autoregressive_wrapper.py
################################################################################

from math import ceil, log
from typing import Optional, Union, Tuple, Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F

from einops import rearrange, pack, unpack

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else (t,) * length

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# for variable lengthed prefixes

def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape, t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = F.pad(t, (max_pad_len, 0), value = 0)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None]]
    return aligned

# nucleus

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending = True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, frac_num_tokens = 0.1, k = None):
    num_tokens = logits.shape[-1]

    k = default(k, ceil(frac_num_tokens * num_tokens))
    k = min(k, num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# top_a

def top_a(logits, min_p_pow = 2.0, min_p_ratio = 0.02):
    probs = F.softmax(logits, dim = -1)
    max_probs = torch.amax(probs, dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# contrastive decoding function

def contrastive_decode_fn(
    expert_logits,
    amateur_logits,
    alpha = 0.1,
    beta = 0.5
):
    """
    Appendix A Algorithm 2
    https://arxiv.org/abs/2309.09117
    """

    cutoff = log(alpha) + expert_logits.amax(dim = -1, keepdim = True)
    diffs = (1 + beta) * expert_logits - beta * amateur_logits
    contrastive_decode_logits = diffs.masked_fill(expert_logits < cutoff, -torch.finfo(expert_logits.dtype).max)
    return contrastive_decode_logits

# autoregressive wrapper class

class AutoregressiveWrapper(Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        add_attn_z_loss = False
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

        # whether to add router z-loss
        self.add_attn_z_loss = add_attn_z_loss

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Optional[Tensor] = None,
        filter_logits_fn: Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Optional[Union[Module, Tuple[Module]]] = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: Union[dict, Tuple[dict]] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = False,
        verbose=True,
        return_prime=False,
        **kwargs
    ):
        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        prompts, ps = pack([prompts], '* n')

        b, t = prompts.shape

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        if verbose:
          print("Generating sequence of max length:", seq_len)

        # kv caches

        cache = None

        # if doing contrastive decoding, turn off filter automatically

        if exists(amateur_model):
            amateur_model = cast_tuple(amateur_model)
            contrastive_decode_kwargs = cast_tuple(contrastive_decode_kwargs)

            assert len(amateur_model) == len(contrastive_decode_kwargs)

            amateur_caches = [None] * len(amateur_model)
            filter_logits_fn = identity

            for i, module in enumerate(amateur_model):
                if isinstance(module, AutoregressiveWrapper):
                    amateur_model[i] = module.net

                module.eval()

        # sampling up to seq_len

        for sl in range(seq_len):

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embeeding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        inter.cached_kv = [t[..., -(max_seq_len - 1):, :] for t in inter.cached_kv]

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            if cache_kv and self.net.can_cache_kv:
                cache = new_cache

            logits = logits[:, -1]

            # handle contrastive decoding, Li et al.
            # https://arxiv.org/abs/2210.15097

            if exists(amateur_model):
                for i, (amateur, amateur_cache, amateur_contrastive_decode_kwargs) in enumerate(zip(amateur_model, amateur_caches, contrastive_decode_kwargs)):
                    amateur_logits, next_amateur_cache = amateur(
                        x,
                        return_intermediates = True,
                        cache = amateur_cache,
                        seq_start_pos = seq_start_pos,
                        **kwargs
                    )

                    amateur_logits = amateur_logits[:, -1]

                    assert amateur_logits.shape == logits.shape, 'logits dimension are not the same between amateur and expert model'
                    logits = contrastive_decode_fn(logits, amateur_logits, **amateur_contrastive_decode_kwargs)

                    if cache_kv and amateur.can_cache_kv:
                        amateur_caches[i] = next_amateur_cache

            # filter by top_k, top_p (nucleus), top_a, or custom

            if greedy:
                sample = logits.argmax(dim = -1, keepdim = True)
            else:
                filtered_logits = filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            # concat sample

            out = torch.cat((out, sample), dim=-1)

            if verbose:
              if sl % 32 == 0:
                print(sl, '/', seq_len)

            if not exists(eos_token):
                continue

            is_eos_tokens = (out == eos_token)

            if is_eos_tokens.any(dim = -1).all():
              if verbose: 
                print('Model called the end of sequence at:', sl, '/', seq_len)
              break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        if return_prime:
          return out[:, :]
        
        else:
          return out[:, t:]

        # out, = unpack(out, ps, '* n')

        # return out

    def compute_accuracy(self, logits, labels): 
        out = torch.argmax(logits, dim=-1) 
        out = out.flatten() 
        labels = labels.flatten() 

        mask = (labels != self.ignore_index) # can also be self.pad_value (your choice)
        out = out[mask] 
        labels = labels[mask] 

        num_right = (out == labels)
        num_right = torch.sum(num_right).type(torch.float32)

        acc = num_right / len(labels) 
        return acc

    def forward(self, x, return_outputs = False, **kwargs):
        seq, ignore_index, add_attn_z_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss

        inp, target = x[:, :-1], x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)

        logits, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            **kwargs
        )

        acc = self.compute_accuracy(logits, target)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if not return_outputs:
            return loss, acc

        return loss, acc, (logits, cache)

################################################################################

################################################################################
# Code for x-transformers Python module xl_autoregressive_wrapper.py
################################################################################

from math import ceil

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, pack, unpack

# helper functions

def exists(val):
    return val is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0 

# xl autoregressive wrapper class

class XLAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        ignore_index = -100,
        pad_value = 0
    ):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        start_tokens,
        seq_len,
        eos_token = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        mems = None,
        **kwargs
    ):
        device, max_seq_len = start_tokens.device, self.max_seq_len

        start_tokens, ps = pack([start_tokens], '* n')

        b, t = start_tokens.shape

        *all_leading_tokens, _ = start_tokens.split(max_seq_len, dim = -1)

        # catch the memory up to the current segment

        for leading_tokens in all_leading_tokens:
            _, mems = self.net(
                leading_tokens,
                mems = mems,
                return_mems = True,
                **kwargs
            )

        # now start sampling from the current segment

        curr_pos = len(all_leading_tokens) * max_seq_len
        curr_mems = mems

        cache = None
        out = start_tokens

        for _ in range(seq_len):
            curr_segment_len = out.shape[-1]
            is_last_segment_tokens = divisible_by(curr_segment_len, max_seq_len)

            x = out[:, curr_pos:]

            logits, cache = self.net(
                x,
                mems = curr_mems,
                cache = cache,
                return_mems = True,
                return_intermediates = True,
                **kwargs
            )

            mems = cache.mems

            logits = logits[:, -1]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            if is_last_segment_tokens:
                curr_pos = curr_segment_len
                curr_mems = mems

            out = torch.cat((out, sample), dim=-1)

            if exists(eos_token):
                is_eos_tokens = (out == eos_token)

                if is_eos_tokens.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

    def forward(
        self,
        x,
        mems = None,
        **kwargs
    ):
        ignore_index, max_seq_len = self.ignore_index, self.max_seq_len

        x, labels = x[:, :-1], x[:, 1:]

        seq_len = x.shape[1]

        # prepare chunks

        split_x = x.split(max_seq_len, dim = -1)
        split_labels = labels.split(max_seq_len, dim = -1)
        loss_weights = tuple(map(lambda t: t.shape[-1] / seq_len, split_x))

        # go through each chunk and derive weighted losses

        total_loss = 0.        

        for chunk, chunk_labels, loss_weight in zip(split_x, split_labels, loss_weights):

            logits, mems = self.net(
                chunk,
                mems = mems,
                return_mems = True,
                **kwargs
            )

            loss = F.cross_entropy(
                rearrange(logits, 'b n c -> b c n'),
                chunk_labels,
                ignore_index = ignore_index
            )

            total_loss = total_loss + loss * loss_weight

        return total_loss

################################################################################

################################################################################
# Code for x-transformers Python module xval.py
################################################################################

"""
regular transformer with discrete tokens, but continuous for number
generalizes better for arithmetic
https://arxiv.org/abs/2310.02989
"""

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from typing import Callable
from collections import namedtuple

from einops import rearrange
from einops.layers.torch import Rearrange

# constants

LossBreakdown = namedtuple('LossBreakdown', ['cross_entropy_loss', 'numerical_mse_loss'])

GenerateReturn = namedtuple('GenerateReturn', ['sampled_token_ids', 'sampled_numbers', 'is_number_mask'])

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# main classes

class XValTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        numerical_token_id,
        attn_layers: AttentionLayers,
        emb_dim = None,
        logits_dim = None,
        tie_embedding = False,
        max_mem_len = 0,
        num_memory_tokens = None,
        emb_dropout = 0.,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
    ):
        super().__init__()
        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.emb_dim = emb_dim
        self.token_emb = TokenEmbedding(emb_dim, num_tokens)

        self.numerical_token_id = numerical_token_id

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

        self.emb_dropout = nn.Dropout(emb_dropout)

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # to logits

        logits_dim = default(logits_dim, num_tokens)
        self.to_logits = nn.Linear(dim, logits_dim) if not tie_embedding else lambda t: t @ self.token_emb.emb.weight.t()

        self.to_numerical_output = nn.Sequential(
            nn.Linear(dim, 1),
            Rearrange('... 1 -> ...')
        )

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        prepend_embeds = None,
        **kwargs
    ):
        assert x.shape == x_num.shape

        batch = x.shape[0]

        is_number_mask = x == self.numerical_token_id

        x = self.token_emb(x)

        scale = torch.where(is_number_mask, x_num, 1.)
        scale = rearrange(scale, '... -> ... 1')

        x = x * scale

        x = x + self.pos_emb(x, pos = pos)

        # memory tokens

        if self.has_memory_tokens:
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            _, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], 'prepended embeddings need to have same dimensions as model dimensions'

            x = torch.cat((prepend_embeds, x), dim = -2)

        x = self.emb_dropout(x)

        # attention layers

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)

        # splice out memory tokens

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        if not return_embeddings:
            logits = self.to_logits(x)
            numerical_pred = self.to_numerical_output(x)
            out = (logits, numerical_pred)
        else:
            out = x

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out

class XValAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net: XValTransformerWrapper,
        ignore_index = -100,
        pad_value = 0,
        numerical_loss_weight = 1.
    ):
        super().__init__()
        self.net = net
        self.max_seq_len = net.max_seq_len
        self.numerical_loss_weight = numerical_loss_weight
        self.ignore_index = ignore_index

    @torch.no_grad()
    def generate(
        self,
        start_tokens: Tensor,
        start_numbers: Tensor,
        seq_len,
        filter_logits_fn: Callable = top_k,
        filter_kwargs: dict = dict(),
        temperature = 1.,
        **kwargs
    ):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        assert num_dims >= 2, 'number of dimensions of your start tokens must be greater or equal to 2'
        assert start_tokens.shape == start_numbers.shape

        b, t, device = *start_tokens.shape, start_tokens.device

        self.net.eval()
        out = start_tokens
        num_out = start_numbers

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            x_num = num_out[:, -self.max_seq_len:]

            logits, numerical_pred = self.net(x, x_num, **kwargs)

            last_logits = logits[:, -1]
            last_num_pred = numerical_pred[:, -1:]

            filtered_logits = filter_logits_fn(last_logits, **filter_kwargs)

            probs = F.softmax(filtered_logits / temperature, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim = -1)
            num_out = torch.cat((num_out, last_num_pred), dim = -1)

        out = out[:, t:]
        num_out = num_out[:, t:]

        is_number = out == self.net.numerical_token_id
        num_out = torch.where(is_number, num_out, float('nan'))

        self.net.train(was_training)
        return GenerateReturn(out, num_out, is_number)

    def forward(
        self,
        x: Tensor,
        x_num: Tensor,
        return_loss_breakdown = False,
        **kwargs
    ):
        inp, target = x[:, :-1], x[:, 1:]
        x_num_inp, x_num_target = x_num[:, :-1], x_num[:, 1:]

        mask = kwargs.get('mask', None)
        if exists(mask) and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        logits, numerical_pred = self.net(inp, x_num_inp, **kwargs)

        logits = rearrange(logits, 'b n c -> b c n')

        cross_entropy_loss = F.cross_entropy(logits, target, reduction = 'none', ignore_index = self.ignore_index)

        target_mask = target != self.ignore_index

        numerical_mse_loss = F.mse_loss(numerical_pred, x_num_target, reduction = 'none')

        numerical_mse_loss = numerical_mse_loss * target_mask

        loss = cross_entropy_loss + numerical_mse_loss * self.numerical_loss_weight

        if exists(mask):
            loss = loss[mask]

        loss = loss.mean()

        if not return_loss_breakdown:
            return loss

        return loss, LossBreakdown(cross_entropy_loss, numerical_mse_loss)

################################################################################

################################################################################
# Code for instantiating, saving and loading a simple model
################################################################################

import os
import importlib

#===============================================================================

def instantiate_x_transformer_model(num_tokens=20000,
                                    max_seq_len=8192,
                                    dim=1024,
                                    depth=32,
                                    heads=32,
                                    attn_flash=True,
                                    ignore_index=-1,
                                    verbose=True):
    if verbose:
      print('=' * 70)
      print('Instantiating x-transformer model...')

    model = TransformerWrapper(
                                num_tokens = num_tokens,
                                max_seq_len = max_seq_len,
                                attn_layers = Decoder(
                                                      dim = dim, 
                                                      depth = depth, 
                                                      heads = heads, 
                                                      attn_flash = attn_flash
                                                      )
                              )

    model = AutoregressiveWrapper(model, 
                                  ignore_index=ignore_index
                                  )

    model.cuda()

    if verbose:
      print('Done!')
      print('=' * 70)

    return model

#===============================================================================

def save_x_transformer_model(model,
                              checkpoint_dir='./',
                              checkpoint_name='model_checkpoint', 
                              number_of_tokens=20000, 
                              ignore_index=-1, 
                              max_seq_len=8192, 
                              dim=1024, 
                              depth=32, 
                              heads=32, 
                              use_flash_attn=True,
                              batch_size=4,
                              grad_acc_rate=4,
                              learning_rate=1e-4,
                              num_epochs=1,
                              num_steps=1,
                              loss=0,
                              accuracy=1,
                              verbose=True
                            ):
    
    if verbose:
      print('=' * 70)
      print('Saving x-transformer model...')

    checkpoint_full_name = ''
    checkpoint_full_name += checkpoint_name + '_' 
    checkpoint_full_name += str(num_epochs) + '_epochs_'
    checkpoint_full_name += str(num_steps) + '_steps_'
    checkpoint_full_name += str(loss) + '_loss_'
    checkpoint_full_name += str(accuracy) + '_acc'
    checkpoint_full_name += '.pth'

    checkpoint_full_path_and_name = os.path.join(checkpoint_dir, checkpoint_full_name)

#===============================================================================

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_name': model.__class__.__name__,
        'class_module': model.__class__.__module__,
        'num_tokens': number_of_tokens,
        'max_seq_len': max_seq_len,
        'attn_layers': {
            'dim': dim,
            'depth': depth,
            'heads': heads,
            'attn_flash': use_flash_attn
            },
        'batch_size': batch_size,
        'grad_acc_rate': grad_acc_rate,
        'learning_rate': learning_rate,
        'ignore_index': ignore_index,
        'num_epochs': num_epochs,
        'num_steps': num_steps,
        'loss': loss,
        'accuracy': accuracy
    }, checkpoint_full_path_and_name
)

    if verbose:
      print('Done!')
      print('=' * 70)
      print('Saved model name:', checkpoint_full_name)
      print('=' * 70)

#===============================================================================

def load_x_transformer_model(checkpoint_file_path,
                             verbose=True
                             ):
    
    if verbose:
      print('=' * 70)
      print('Loading x-transformer model...')

    checkpoint = torch.load(checkpoint_file_path)
    module = importlib.import_module(checkpoint['class_module'])
    class_ = getattr(module, checkpoint['class_name'])
    attn_layers = Decoder(**checkpoint['attn_layers'])
    transformer_model = TransformerWrapper(num_tokens=checkpoint['num_tokens'],
                                          max_seq_len=checkpoint['max_seq_len'],
                                          attn_layers=attn_layers)
    model = class_(transformer_model, ignore_index=checkpoint['ignore_index'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()

    if verbose:
      print('Done!')
      print('=' * 70)
      print('Model stats:')
      print('Number of tokens:', checkpoint['num_tokens'])
      print('Ignore index:', checkpoint['ignore_index'])
      print('Max sequence length::', checkpoint['max_seq_len'])
      print('Dimension:', checkpoint['attn_layers']['dim'])
      print('Depth:', checkpoint['attn_layers']['depth'])
      print('Number of heads:', checkpoint['attn_layers']['heads'])
      print('Flash attention:', checkpoint['attn_layers']['attn_flash'])
      print('Training batch size', checkpoint['batch_size'])
      print('Training gradient accumulation rate:', checkpoint['grad_acc_rate'])
      print('Training learining rate:', checkpoint['learning_rate'])
      print('Number of training epochs:', checkpoint['num_epochs'])
      print('Number of training steps:', checkpoint['num_steps'])
      print('Model loss:', checkpoint['loss'])
      print('Model accuracy:', checkpoint['accuracy'])
      print('=' * 70)

################################################################################
# This is the end of x-transformer Python module
################################################################################