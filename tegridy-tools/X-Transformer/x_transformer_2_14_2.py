#===================================================================================================================
#
# X Trasformer Python Module
#
# Partial x-transformers code With useful modifications as a stand-alone Python module
#
# Version 2.0
#
# Original source code courtesy of lucidrains
# https://github.com/lucidrains/x-transformers
#
# Original source code retrieved on 01/27/2026
# Original version 2.14.2 / Commit 03d11fa
#
# Project Los Angeles
# Tegridy Code 2026
#
#===================================================================================================================
#
# Critical dependencies
#
# !pip install torch
# !pip install einops
# !pip install einx
#
#===================================================================================================================

from __future__ import annotations

import os
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_cudnn_sdp(True)

#==================================================================================================================================
# attend.py
#==================================================================================================================================

from functools import partial
from typing import Tuple, Callable

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from collections import namedtuple
from functools import wraps
from packaging import version
from dataclasses import dataclass

from einops import rearrange, repeat, pack, unpack

#========================================================================================================================

# constants

@dataclass
class Intermediates:
    qk_similarities:    Tensor | None = None
    pre_softmax_attn:   Tensor | None = None
    post_softmax_attn:  Tensor | None = None
    values:             Tensor | None = None
    cached_kv:          tuple[Tensor, Tensor] | None = None
    layer_type:         str | None = None
    hybrid_hidden:      Tensor | None = None

    def to_tuple(self):
        return (self.qk_similarities, self.pre_softmax_attn, self.post_softmax_attn)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def at_most_one_of(*bools):
    return sum([*map(int, bools)]) <= 1

def compact(arr):
    return [*filter(exists, arr)]

@torch.jit.script
def softclamp(t: Tensor, value: float):
    return (t / value).tanh() * value

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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

# gumbel softmax attention related

def log_prob_from_hard_attend(intermeds: Intermediates):
    log_probs = intermeds.pre_softmax_attn.log_softmax(dim = -1)

    one_hot = intermeds.post_softmax_attn.argmax(dim = -1, keepdim = True)
    log_prob = log_probs.gather(-1, one_hot)
    return rearrange(log_prob, 'b h i 1 -> b h i')

# selective attention
# https://arxiv.org/abs/2410.02703 - section 3.3
# it is a technique to allow each token to prevent itself from being attended to by future tokens
# if sim_head_gate not supplied, will use the first head of the attention logits (sim in this framework)

def selective_attn(
    sim,
    sim_head_gate = None,
    no_mask_sos = True
):
    i, j, device = *sim.shape[-2:], sim.device
    sim_head_gate = default(sim_head_gate, sim[:, 0])

    gate = F.relu(sim_head_gate) # only positive

    if no_mask_sos:
        gate = gate.clone()
        gate[..., -i] = 0.

    eye = torch.eye(i, device = device)

    if j > i:
        eye = F.pad(eye, (j - i, 0), value = 1.)

    gate = (1. - eye) * gate
    gate = F.pad(gate, (0, 0, 1, -1), value = 0.) # only allow for masking the future
    gate = gate.cumsum(dim = -2)

    return sim - rearrange(gate, 'b i j -> b 1 i j')

# alternative distance functions

def qk_l2_dist_squared(q, k):
    if k.ndim == 3:
        k = repeat(k, 'b j d -> b h j d', h = q.shape[1])

    q, packed_shape = pack_one(q, '* i d')
    k, _ = pack_one(k, '* j d')

    l2_dist_squared = torch.cdist(q, k) ** 2
    return unpack_one(l2_dist_squared, packed_shape, '* i j')

# one-hot straight through softmax

def one_hot_straight_through(logits, temperature = 1.):
    one_hot_indices = logits.argmax(dim = -1, keepdim = True)
    one_hot = torch.zeros_like(logits).scatter(-1, one_hot_indices, 1.)

    soft_attn = (logits / temperature).softmax(dim = -1)
    return one_hot + soft_attn - soft_attn.detach()

# sparse topk attention - only keep topk attn logits for softmax
# optional straight through with masked out logits by setting `attn_sparse_topk_straight_through = True`

def sparse_topk_attn(
    logits,
    sparse_topk,
    temperature = 1.,
    straight_through = False
):
    orig_logits = logits

    mask_value = -torch.finfo(logits.dtype).max
    top_values, _ = logits.topk(sparse_topk, dim = -1)
    sparse_topk_mask = (logits >= top_values[..., -1:]) & (logits > mask_value)
    logits = logits.masked_fill(~sparse_topk_mask, mask_value)
    topk_attn = logits.softmax(dim = -1)

    if not straight_through:
        return topk_attn

    soft_attn = (orig_logits / temperature).softmax(dim = -1)
    return topk_attn.detach() + soft_attn - soft_attn.detach()

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

class Attend(Module):
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        heads = None,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        sparse_topk = None,
        sparse_topk_straight_through = False, # https://arxiv.org/abs/2505.22074
        scale = None,
        qk_norm = False,
        l2_distance = False,
        sigmoid = False,
        gumbel_softmax = False,
        gumbel_softmax_temp = 1.,
        gumbel_softmax_hard = True,
        cog_signed = False,
        custom_attn_fn: Callable | None = None,
        flash = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        add_zero_kv = False,
        head_learned_sink = False,
        selective = False,
        hard = False,
        cope = None,
        onnxable = False,
        sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
    ):
        super().__init__()
        self.scale = scale

        # causal related

        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask

        # attention type

        is_sparse_topk_attn = exists(sparse_topk)

        assert not (flash and sigmoid), 'sigmoid attention not available for flash'
        assert not (flash and hard), 'hard attention not available for flash'
        assert not (flash and is_sparse_topk_attn), 'topk attention not available for flash'

        assert at_most_one_of(sigmoid, hard, l2_distance, gumbel_softmax, is_sparse_topk_attn)

        if exists(custom_attn_fn):
            self.attn_fn = custom_attn_fn
        elif sigmoid:
            self.attn_fn = F.sigmoid
        elif hard:
            self.attn_fn = one_hot_straight_through
        elif is_sparse_topk_attn:
            self.attn_fn = partial(sparse_topk_attn, sparse_topk = sparse_topk, straight_through = sparse_topk_straight_through)
        elif gumbel_softmax:
            self.attn_fn = partial(F.gumbel_softmax, dim = -1, tau = gumbel_softmax_temp, hard = gumbel_softmax_hard)
        else:
            softmax_fn = partial(F.softmax, dim = -1)
            self.attn_fn = partial(softmax_fn, dtype = torch.float32) if not qk_norm else softmax_fn

        # dropouts

        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        # talking heads

        assert not (flash and (pre_talking_heads or post_talking_heads or pre_scale_post_talking_heads)), 'talking heads not compatible with flash attention'

        self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_talking_heads else None
        self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if post_talking_heads else None
        self.pre_scale_post_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if pre_scale_post_talking_heads else None

        if exists(self.pre_softmax_talking_heads):
            nn.init.dirac_(self.pre_softmax_talking_heads.weight)

        if exists(self.post_softmax_talking_heads):
            nn.init.dirac_(self.post_softmax_talking_heads.weight)

        if exists(self.pre_scale_post_talking_heads):
            # an improvisation where heads are combined pre-softmax attention, then used to scale post-softmax attention
            nn.init.dirac_(self.pre_scale_post_talking_heads.weight)

        # selective attention

        assert not (flash and selective), 'selective attention cannot work on flash attention'
        assert not (selective and not causal), 'selective attention is designed for autoregressive'
        self.selective = selective

        # cog attention - negative weights for expressiveness
        # https://openreview.net/forum?id=ezRrwwbxd0

        assert not (flash and cog_signed), 'cog attention not available for flash'
        self.cog_signed = cog_signed

        # l2 distance attention

        self.l2_distance = l2_distance

        # add a key / value token composed of zeros
        # in case this helps controlling outliers, proposed by https://www.evanmiller.org/attention-is-off-by-one.html

        self.add_zero_kv = add_zero_kv

        # learned sink concatted pre-softmax, working solution from gpt-oss

        assert not (head_learned_sink and flash), f'not supported for flash attention yet'

        self.head_learned_sink = head_learned_sink
        self.head_attn_sink = Parameter(torch.zeros(heads)) if head_learned_sink else None

        # soft clamp attention logit value

        if softclamp_logits:
            assert not flash, 'flash attention not compatible with logit softclamp value yet'
            assert logit_softclamp_value > 0.

        self.softclamp_logits = softclamp_logits
        self.logit_softclamp_value = logit_softclamp_value

        # contextual positional encoding

        self.cope = cope

        # flash attention

        self.flash = flash

        torch_version = version.parse(torch.__version__)
        assert not (flash and torch_version < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # torch 2.3 uses new backend and context manager

        if self.flash:
            if torch_version >= version.parse('2.3'):
                from torch.nn.attention import SDPBackend

                str_to_backend = dict(
                    enable_flash = SDPBackend.FLASH_ATTENTION,
                    enable_mem_efficient = SDPBackend.EFFICIENT_ATTENTION,
                    enable_math = SDPBackend.MATH,
                    enable_cudnn = SDPBackend.CUDNN_ATTENTION
                )

                sdpa_backends = [str_to_backend[enable_str] for enable_str, enable in sdp_kwargs.items() if enable]

                self.sdp_context_manager = partial(torch.nn.attention.sdpa_kernel, sdpa_backends)
            else:
                self.sdp_context_manager = partial(torch.backends.cuda.sdp_kernel, **sdp_kwargs)

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

        # handle maybe l2 distance

        if self.l2_distance:
            k_norm_sq = k.norm(dim = -1, keepdim = True) ** 2
            k = F.pad(k, (0, 1), value = -1.)
            k = cat((k, k_norm_sq), dim = -1)

            q_norm_sq = q.norm(dim = -1, keepdim = True) ** 2
            q = cat((2 * q, q_norm_sq), dim = -1)
            q = F.pad(q, (0, 1), value = -1.)

        # handle scale - by default they scale by dim_head ** -0.5, but need to take care if using cosine sim attention

        if exists(self.scale):
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

        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask
            causal = False

        # protect against an entire row being masked out

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        # handle alibi positional bias
        # convert from bool to float

        if exists(attn_bias):
            attn_bias = attn_bias.expand(batch, heads, -1, -1)

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

        with self.sdp_context_manager():
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.training else 0., 
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
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

        # handle key padding mask

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b j -> b 1 1 j')

        # handle kv cached decoding

        if n == 1 and causal:
            causal = False

        # handle grouped multi-query attention

        if kv_heads == 1:
            k, v = tuple(rearrange(t, 'b 1 n d -> b n d') for t in (k, v))
        elif kv_heads < heads:
            k, v = tuple(repeat(t, 'b kvh n d -> b (r kvh) n d', r = heads // kv_heads) for t in (k, v))

        # handle zero kv, as means for allowing network to attend to nothing

        if self.add_zero_kv:
            k, v = tuple(F.pad(t, (0, 0, 1, 0), value = 0.) for t in (k, v))

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

            if exists(attn_bias):
                attn_bias = F.pad(attn_bias, (1, 0), value = 0.)

        if self.flash:
            assert not exists(prev_attn), 'residual attention not compatible with flash attention'
            return self.flash_attn(q, k, v, mask = mask, attn_bias = attn_bias)

        kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

        if not self.l2_distance:
            sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k)
        else:
            sim = -qk_l2_dist_squared(q, k)

        sim = sim * scale

        if exists(prev_attn):
            sim = sim + prev_attn

        qk_similarities = sim.clone()

        if exists(self.pre_scale_post_talking_heads):
            pre_to_post_scale = self.pre_scale_post_talking_heads(sim)

        if exists(self.pre_softmax_talking_heads):
            sim = sim + self.pre_softmax_talking_heads(sim)

        if exists(attn_bias):
            sim = sim + attn_bias

        if self.softclamp_logits:
            sim = softclamp(sim, self.logit_softclamp_value)

        # pre-masking - handle cog by storing sign

        if self.cog_signed:
            sim_sign = sim.sign()
            sim = sim.abs()

        # masking

        i, j, dtype = *sim.shape[-2:], sim.dtype

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            sim = sim.masked_fill(~mask, mask_value)

        if causal:
            causal_mask = self.create_causal_mask(i, j, device = device)
            sim = sim.masked_fill(causal_mask, mask_value)

        row_is_entirely_masked = None

        if exists(mask):
            row_is_entirely_masked = ~mask.any(dim = -1)

        if exists(self.cope):
            sim = sim + self.cope(q, sim)

        if self.selective:
            sim = selective_attn(sim)

        if self.head_learned_sink:
            # add learned attention sink
            attn_sink = repeat(self.head_attn_sink, 'h -> b h i 1', b = sim.shape[0], i = sim.shape[2])

            if self.cog_signed:
                attn_sink, attn_sink_sign = attn_sink.abs(), attn_sink.sign()
                sim_sign = cat((attn_sink_sign, sim_sign), dim = -1)

            sim = cat((attn_sink, sim), dim = -1)

        pre_softmax_attn = sim

        attn = self.attn_fn(sim)

        attn = attn.type(dtype)

        # add back the sign

        if self.cog_signed:
            attn = attn * sim_sign

        post_softmax_attn = attn

        if self.head_learned_sink:
            # remove attention sink
            attn = attn[..., 1:]

        attn = self.attn_dropout(attn)

        if exists(self.post_softmax_talking_heads):
            attn = self.post_softmax_talking_heads(attn)

        if exists(self.pre_scale_post_talking_heads):
            attn = attn * pre_to_post_scale

        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        intermediates = Intermediates(
            qk_similarities = qk_similarities,
            pre_softmax_attn = pre_softmax_attn,
            post_softmax_attn = post_softmax_attn
        )

        if exists(row_is_entirely_masked) and row_is_entirely_masked.any():
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out, intermediates

#=================================================================================================================================
# x_transformers.py
#=================================================================================================================================

from typing import Callable

import math
from copy import deepcopy
from random import random, randrange
from functools import partial, wraps
from itertools import chain
from collections import namedtuple
from contextlib import nullcontext
from dataclasses import dataclass
from packaging import version

import torch
from torch.amp import autocast
import torch.nn.functional as F
from torch import nn, einsum, tensor, Tensor, cat, stack, arange, is_tensor
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map
from torch.nn import Module, ModuleList, ModuleDict

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, pack, unpack

# einstein notation

# b - batch
# n - sequence
# d - feature dimension
# h - attention heads
# i, j - sequence (source, target)

# constants

DEFAULT_DIM_HEAD = 64

@dataclass
class LayerIntermediates:
    hiddens:            list[Tensor] | None = None   # all hiddens, before the final norm (in pre-norm architecture)
    last_hidden:        Tensor | None = None         # very last hidden after all attention layers, after the final norm
    attn_intermediates: list[Intermediates] | None = None
    layer_hiddens:      list[Tensor] | None = None
    attn_z_loss:        Tensor | None = None
    mems:               Tensor | None = None
    last_layer_hiddens: Tensor | None = None
    initial_embeds:     Tensor | None = None
    attn_pooled_tokens: Tensor | None = None
    memory_tokens:      Tensor | None = None
    logit_entropies:    Tensor | None = None
    logits:             Tensor | None = None
    cache_length:       int = 0

LinearNoBias = partial(nn.Linear, bias = False)

# helpers

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def first(it, default = None):
    return it[0] if len(it) > 0 else default

def is_empty(x):
    return len(x) == 0

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else (val,) * depth

def divisible_by(num, den):
    return (num % den) == 0

def detach_all(obj):
    return tree_map(lambda t: t.detach() if is_tensor(t) and t.requires_grad else t, obj)

def maybe(fn = None):
    if not exists(fn):
        fn = identity

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

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def softclamp(t, value):
    return (t / value).tanh() * value

def masked_mean(t, mask = None, dim = 1):
    if not exists(mask):
        return t.mean(dim = dim)

    dims_append = (1,) * (t.ndim - mask.ndim)
    mask = mask.reshape(*mask.shape, *dims_append)

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim).clamp(min = 1.)
    return num / den

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

def orthog_project(x, y):
    x, packed_shape = pack([x], 'b *')
    y, _ = pack([y], 'b *')

    dtype = x.dtype
    x, y = x.double(), y.double()
    unit = F.normalize(y, dim = -1)

    parallel = (x * unit).sum(dim = -1, keepdim = True) * unit
    orthog = x - parallel

    orthog, = unpack(orthog, packed_shape, 'b *')

    return orthog.to(dtype)

# cache helpers

def get_cached_kvs(
    cache: LayerIntermediates
) -> list[tuple[Tensor, Tensor]]:

    cached_kvs = []

    for attn_intermediate in cache.attn_intermediates:
        cached_kvs.append(attn_intermediate.cached_kv)

    return cached_kvs

# entropy

def calc_entropy(
    t: Tensor,
    is_prob = False
):
    prob = t.softmax(dim = -1) if not is_prob else t
    return -(prob * log(prob)).sum(dim = -1)

# auxiliary loss helpers

def calc_z_loss(
    pre_softmax_attns: list[Tensor],
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
    values = tuple(d.pop(key) for key in  keys)
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return tuple(return_val)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    prefix_len = len(prefix)
    kwargs_without_prefix = {key[prefix_len:]: value for key, value in kwargs_with_prefix.items()}
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

    batch_indices = arange(b, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim = -1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = arange(num_keep, device = device) < rearrange(seq_keep_counts, 'b -> b 1')

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask

# activations

class ReluSquared(Module):
    def forward(self, x):
        return F.relu(x) ** 2

class SoLU(Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim)

    def forward(self, x):
        activated = x.softmax(dim = -1) * x
        return self.norm(activated)

# embedding

class TokenEmbedding(Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return l2norm(token_emb) if self.l2norm_embed else token_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.emb.weight, std=1e-5)
            return
        nn.init.kaiming_normal_(self.emb.weight)

# positional embeddings

class AbsolutePositionalEmbedding(Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(
        self,
        x,
        pos = None,
        seq_start_pos = None,
        offset = 0
    ):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = arange(seq_len, device = device) + offset

        if exists(seq_start_pos):
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ScaledSinusoidalEmbedding(Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert divisible_by(dim, 2)
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(
        self,
        x,
        pos = None,
        seq_start_pos = None,
        offset = 0
    ):
        seq_len, device = x.shape[1], x.device

        if not exists(pos):
            pos = arange(seq_len, device = device) + offset

        if exists(seq_start_pos):
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale

class RelativePositionBias(Module):
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
        q_pos = arange(j - i, j, dtype = torch.long, device = device)
        k_pos = arange(j, dtype = torch.long, device = device)
        rel_pos = einx.subtract('j, i -> i j', k_pos, q_pos)
        rp_bucket = self._relative_position_bucket(rel_pos, causal = self.causal, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> h i j')
        return bias * self.scale

class CoPE(Module):
    """
    Appendix B of https://arxiv.org/abs/2405.18719
    """
    def __init__ (
        self,
        dim,
        heads,
        max_pos,
        soft_onehot = False,
        talking_heads = False,
        soft_onehot_temp = 5e-2
    ):
        super () . __init__ ()
        self.max_pos = max_pos
        self.pos_emb = nn.Parameter(torch.zeros(max_pos, dim))

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None
        self.soft_onehot = soft_onehot
        self.soft_onehot_temp = soft_onehot_temp

        if not soft_onehot:
            return

        self.register_buffer('positions', arange(max_pos))

    def forward(self, query, attn_logits):

        if exists(self.talking_heads):
            i, j = attn_logits.shape[-2:]
            causal_mask = attn_logits.new_ones(i, j).triu_(j - i + 1).bool()

            attn_logits = self.talking_heads(attn_logits)

            attn_logits = attn_logits.masked_fill(causal_mask, -torch.finfo(attn_logits.dtype).max)

        # compute positions

        gates = attn_logits.sigmoid()

        pos = gates.flip(-1).cumsum(dim = -1).flip(-1)
        pos = pos.clamp(max = self.max_pos - 1)

        logits_int = einsum('b h n d, p d -> b h n p', query, self.pos_emb)

        if self.soft_onehot:
            diff_pos = einx.subtract('i, j -> i j', pos, self.positions).abs()
            soft_onehot_pos = F.softmax(-diff_pos / self.soft_onehot_temp, dim = -1)
            cope_pos_emb = einsum('b h i j p, b h i p -> b h i j', soft_onehot_pos, logits_int)
        else:
            # interpolate from integer positions
            pos_ceil = pos.ceil().long()
            pos_floor = pos.floor().long()
            logits_ceil = logits_int.gather(-1, pos_ceil)
            logits_floor = logits_int.gather(-1, pos_floor)

            w = pos - pos_floor
            cope_pos_emb = logits_ceil * w + logits_floor * (1 - w)

        return cope_pos_emb

class DynamicPositionBias(Module):
    def __init__(self, dim, *, heads, depth, log_distance = False, norm = False):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.log_distance = log_distance

        self.mlp = ModuleList([])

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
        n, device = j, self.device

        # get the (n x n) matrix of distances
        seq_arange = arange(j - i, j, device = device)
        context_arange = arange(j, device = device)
        indices = einx.subtract('i, j -> i j', seq_arange, context_arange)
        indices += (j - 1)

        # input to continuous positions MLP
        pos = arange(-j + 1, j, device = device).float()
        pos = rearrange(pos, '... -> ... 1')

        if self.log_distance:
            pos = torch.sign(pos) * torch.log(pos.abs() + 1)  # log of distance is sign(rel_pos) * log(abs(rel_pos) + 1)

        for layer in self.mlp:
            pos = layer(pos)

        # get position biases        
        bias = pos[indices]
        bias = rearrange(bias, 'i j h -> h i j')
        return bias

class AlibiPositionalBias(Module):
    def __init__(
        self,
        heads,
        total_heads = None,
        slopes: list[int] | None = None,
        **kwargs
    ):
        super().__init__()
        self.heads = heads
        self.total_heads = default(total_heads, heads)

        slopes = Tensor(default(slopes, self._get_slopes(heads)))
        slopes = rearrange(slopes, 'h -> h 1 1')

        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)
    
    @property
    def device(self):
        return next(self.buffers()).device

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

    def forward_custom_pos(
        self,
        pos_i: Tensor,
        pos_j: Tensor | None = None
    ):
        h, device = self.total_heads, self.device

        pos_j = default(pos_j, pos_i)
        bias = -einx.subtract('... j, ... i -> ... i j', pos_j, pos_i).abs()

        if bias.ndim == 3:
            bias = rearrange(bias, 'b i j -> b 1 i j')

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        return bias

    def forward(self, i, j):
        h, device = self.total_heads, self.device

        if exists(self.bias) and self.bias.shape[-1] >= j and self.bias.shape[-2] >= i:
            return self.bias[..., -i:, -j:]

        seq_arange = arange(j - i, j, device = device)
        context_arange = arange(j, device = device)
        bias = -einx.subtract('j, i -> 1 i j', context_arange, seq_arange).abs()

        bias = bias * self.slopes
        num_heads_unalibied = h - bias.shape[-3]
        bias = pad_at_dim(bias, (0, num_heads_unalibied), dim = -3)

        self.register_buffer('bias', bias, persistent = False)
        return self.bias

class DataDependentAlibi(Module):
    """ https://openreview.net/forum?id=q2Lnyegkr8 """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        bias_init = 5.,
        post_log_scale = 1.,
    ):
        super().__init__()

        self.causal = causal

        linear = nn.Linear(dim, heads * (1 if causal else 2))

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n h -> b h n'),
            nn.LogSigmoid()
        )

        nn.init.constant_(linear.bias, bias_init)
        self.post_log_scale = post_log_scale

    def forward(self, x):
        bidirectional = not self.causal

        forget_gates = self.to_forget_gates(x) * self.post_log_scale

        forget_gates = forget_gates.cumsum(dim = -1)

        if bidirectional:
            forget_gates, forget_gates_reversed = forget_gates.chunk(2, dim = 1)

        forget_gates = einx.subtract('b h i, b h j -> b h i j', forget_gates, forget_gates)

        if bidirectional:
            forget_gates_reversed = einx.subtract('b h j, b h i -> b h i j', forget_gates_reversed, forget_gates_reversed)
            forget_gates = forget_gates.tril() + forget_gates_reversed.triu()

        return forget_gates

class PerRowDataDependentAlibi(Module):
    """ same as data dependent alibi from forgetting transformer, but the forgetting gates are also derived by a queries and keys with a small head dimension """

    def __init__(
        self,
        dim,
        heads,
        causal = True,
        dim_head = 8,
        post_log_scale = 1.
    ):
        super().__init__()
        assert causal, 'bidirectional not supported yet'

        self.scale = dim_head ** -0.5

        linear = nn.Linear(dim, heads * dim_head * 2, bias = False)

        self.to_forget_gates = nn.Sequential(
            linear,
            Rearrange('b n (qk h d) -> qk b h n d', qk = 2, d = dim_head)
        )

        self.post_log_scale = post_log_scale

    def forward(self, x):
        q, k = self.to_forget_gates(x)
        forget_gates = einsum('... i d, ... j d -> ... i j', q, k) * self.scale

        forget_gates = F.logsigmoid(forget_gates) * self.post_log_scale

        # mask out upper triangle + diagonal

        n = x.shape[-2]
        causal_mask = torch.ones((n, n), dtype = torch.bool, device = x.device).triu()

        forget_gates = forget_gates.masked_fill(causal_mask, 0.)

        # reverse cumsum

        forget_gates = forget_gates.flip(dims = (-1,))
        forget_gates = forget_gates.cumsum(dim = -1)
        forget_gates = forget_gates.flip(dims = (-1,))

        return forget_gates

class RotaryEmbedding(Module):
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

        inv_freq = 1. / (base ** (arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = arange(seq_len, device = device)
        return self.forward(t)

    @autocast('cuda', enabled = False)
    def forward(self, t, offset = 0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = stack((freqs, freqs), dim = -1)
        freqs = rearrange(freqs, '... d r -> ... (d r)')

        if not exists(self.scale):
            return freqs, 1.

        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, '... n -> ... n 1')
        scale = stack((scale, scale), dim = -1)
        scale = rearrange(scale, '... d r -> ... (d r)')

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')

@autocast('cuda', enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype

    freqs = freqs[:, -seq_len:, :]
    scale = scale[:, -seq_len:, :] if is_tensor(scale) else scale

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

        if is_tensor(scale):
            scale = rearrange(scale, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = cat((t, t_unrotated), dim = -1)

    return out.type(orig_dtype)

class PolarEmbedding(Module):
    """ https://arxiv.org/abs/2509.10534 """

    def __init__(
        self,
        dim,
        heads,
        bias_uniform_init = False,
        base = 10000,
    ):
        super().__init__()
        inv_freq = 1. / (base ** (arange(0, dim).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        self.learned_bias = nn.Parameter(torch.zeros(heads, 1, dim))

        if bias_uniform_init:
            self.learned_bias.uniform_(-2. * math.pi, 0.)

    @autocast('cuda', enabled = False)
    def forward(self, t, offset = 0):
        max_pos = t.max() + 1

        if t.ndim == 1:
            t = rearrange(t, 'n -> 1 n')

        freqs = torch.einsum('b i , j -> b i j', t.type_as(self.inv_freq), self.inv_freq)

        bias = self.learned_bias.clamp(-2. * math.pi, 0.)

        return freqs, bias

@autocast('cuda', enabled = False)
def apply_polar_pos_emb(t, freqs):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:]

    t = t.float()

    t = F.softplus(t)
    out = cat((t * freqs.cos(), t * freqs.sin()), dim = -1)

    return out.type(orig_dtype)

# norms

class Scale(Module):
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

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()
        self.unit_offset = unit_offset

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.ones(dim))
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        normed = self.ln(x)
        gamma = self.gamma + float(self.unit_offset)
        return normed * gamma

class AdaptiveLayerNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        dim_condition = default(dim_condition, dim)

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = self.ln(x)
        gamma = self.to_gamma(condition)
        return normed * (gamma + 1.)

class ScaleNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(1))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset
        self.scale = dim ** 0.5

        self.g = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.g, 1. - float(unit_offset))

    def forward(self, x):
        gamma = self.g + float(self.unit_offset)
        return F.normalize(x, dim = -1) * self.scale * gamma

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        dim_condition = None
    ):
        super().__init__()
        self.scale = dim ** 0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = LinearNoBias(dim_condition, dim)
        nn.init.zeros_(self.to_gamma.weight)

    def forward(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        normed = F.normalize(x, dim = -1)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.)

class SimpleRMSNorm(Module):
    def __init__(
        self,
        dim,
        **kwargs
    ):
        super().__init__()
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale

class MultiheadRMSNorm(Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = SimpleRMSNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * (self.gamma + 1.)

class DynamicTanh(Module):
    """ https://arxiv.org/abs/2503.10622 """
    def __init__(
        self,
        dim,
        init_alpha = 1.,
        gamma = 1.,
        beta = 0.,
        unit_offset = False
    ):
        super().__init__()
        self.pre_tanh_scale = nn.Parameter(tensor(init_alpha))

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.pre_tanh_scale_offset = init_alpha if unit_offset else 0.
        self.gamma_offset = float(unit_offset)

        nn.init.constant_(self.pre_tanh_scale, 0 if unit_offset else init_alpha)
        nn.init.constant_(self.gamma, 1. - float(unit_offset))

    def forward(self, x):
        pre_tanh_scale = self.pre_tanh_scale + self.pre_tanh_scale_offset
        gamma = self.gamma + self.gamma_offset
        return (x * pre_tanh_scale).tanh() * gamma + self.beta

class Derf(Module):
    """ https://arxiv.org/abs/2512.10938 """
    def __init__(
        self,
        dim,
        init_alpha = 0.5,
        init_bias = 0.,
        unit_offset = False
    ):
        super().__init__()
        scale_offset = 1. if unit_offset else 0.

        self.alpha = nn.Parameter(tensor(init_alpha) - scale_offset)
        self.s = nn.Parameter(tensor(init_bias))

        self.gamma = nn.Parameter(torch.ones(dim) - scale_offset)
        self.beta = nn.Parameter(torch.zeros(dim))

        self.scale_offset = scale_offset

    def forward(self, x):
        x = x * (self.alpha + self.scale_offset) + self.s
        activated = torch.erf(x)
        return activated * (self.gamma + self.scale_offset) + self.beta

# residual and residual gates

class Residual(Module):
    def __init__(self, dim, scale_residual = False, scale_residual_constant = 1., **kwargs):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None
        self.scale_residual_constant = scale_residual_constant

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        if self.scale_residual_constant != 1:
            residual = residual * self.scale_residual_constant

        return x + residual

class GRUGating(Module):
    def __init__(self, dim, scale_residual = False, **kwargs):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def prepare(self, residual):
        return residual, residual, dict()

    def forward(self, x, residual, **kwargs):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)

# hyper connections

def sinkhorn(t, iters = 20):
    dtype = t.dtype
    t = t.float()

    t = t.softmax(dim = -2)

    for _ in range(iters):
        t = F.normalize(t, p = 1, dim = -1)
        t = F.normalize(t, p = 1, dim = -2)

    return t.to(dtype)

class HyperConnection(Module):
    def __init__(
        self,
        dim,
        *,
        layer_index,
        num_residual_streams,
        num_input_views = 1,
        sinkhorn_iters = 5,
        **kwargs
    ):
        """
        https://arxiv.org/abs/2409.19606
        Appendix J - Algorithm 2, Dynamic only

        https://arxiv.org/abs/2512.24880
        "Manifold constrained" mixing matrices
        """
        super().__init__()

        self.norm = nn.LayerNorm(dim, bias = False)

        self.num_residual_streams = num_residual_streams
        self.layer_index = layer_index

        self.static_beta = nn.Parameter(torch.ones(num_residual_streams))

        init_alpha0 = torch.zeros((num_residual_streams, num_input_views))
        init_alpha0[layer_index % num_residual_streams, :] = 1.

        self.static_alpha = nn.Parameter(cat([init_alpha0, torch.eye(num_residual_streams)], dim = 1))

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, num_residual_streams + num_input_views))
        self.dynamic_alpha_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.num_input_views = num_input_views

        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))
        self.dynamic_beta_scale = nn.Parameter(torch.ones(()) * 1e-2)

        self.sinkhorn_iters = sinkhorn_iters

    def prepare(self, residuals):
        views = self.num_input_views
        streams = self.num_residual_streams

        residuals = rearrange(residuals, '(b s) n d -> b n s d', s = self.num_residual_streams)

        normed = self.norm(residuals)

        wc_weight = normed @ self.dynamic_alpha_fn
        dynamic_alpha = wc_weight * self.dynamic_alpha_scale
        alpha = dynamic_alpha + self.static_alpha

        alpha_input, alpha_residual = alpha[..., :views], alpha[..., views:]

        alpha_input = alpha_input.sigmoid() # constraint Hpre

        # the sinkhorn knopps constraint for the residual mixing

        alpha_residual = rearrange(alpha_residual, '... (s1 s2) -> ... s1 s2', s2 = streams)
        alpha_residual = sinkhorn(alpha_residual, self.sinkhorn_iters)
        alpha_residual = rearrange(alpha_residual, '... s1 s2 -> ... (s1 s2)')

        alpha = cat((alpha_input, alpha_residual), dim = -1)

        dc_weight = (normed @ self.dynamic_beta_fn).sigmoid() * 2
        dynamic_beta = dc_weight * self.dynamic_beta_scale
        beta = dynamic_beta + self.static_beta

        beta = beta.sigmoid() * 2 # constraint Hpost

        # width connection

        mix_h = einsum('... s t, ... s d -> ... t d', alpha, residuals)

        if views == 1:
            branch_input, residuals = mix_h[..., 0, :], mix_h[..., 1:, :]
        else:
            branch_input, residuals = mix_h[..., :views, :], mix_h[..., views:, :]
            branch_input = rearrange(branch_input, '... v d -> v ... d')

        return branch_input, residuals, dict(beta = beta)

    def forward(self, x, residuals, *, beta):
        residuals = einsum('b n d, b n s -> b n s d', x, beta) + residuals
        return rearrange(residuals, 'b n s d -> (b s) n d')

# LIMe - layer integrated memory (dynamic version)

class DynamicLIMe(Module):
    def __init__(
        self,
        dim,
        num_layers,
        num_views = 1,
        norm = True,
        use_softmax = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.multiple_views = num_views > 1

        self.to_weights = Sequential(
            RMSNorm(dim) if norm else None,
            nn.Linear(dim, num_views * num_layers),
            Rearrange('... (views layers) -> views ... layers', views = num_views),
            nn.Softmax(dim = -1) if use_softmax else nn.ReLU()
        )

    def forward(
        self,
        x,
        hiddens
    ):

        if not is_tensor(hiddens):
            hiddens = stack(hiddens)

        assert hiddens.shape[0] == self.num_layers, f'expected hiddens to have {self.num_layers} layers but received {tuple(hiddens.shape)} instead (first dimension must be layers)'

        weights = self.to_weights(x)

        out = einsum('l b n d, v b n l -> v b n d', hiddens, weights)

        if self.multiple_views:
            return out

        return rearrange(out, '1 ... -> ...')

# token shifting

def shift(t, amount, mask = None):
    if amount == 0:
        return t

    amount = min(amount, t.shape[1])

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return pad_at_dim(t, (amount, -amount), dim = - 2, value = 0.)

class ShiftTokens(Module):
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
        segments_to_shift = [shift(*args, mask = mask) for args in zip(segments_to_shift, shifts)]
        x = cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)

class FoldAxially(Module):
    def __init__(
        self,
        axial_dim,
        fn: Module
    ):
        super().__init__()
        self.fn = fn
        self.axial_dim = axial_dim # will fold the sequence as rearrange("b (n axial_dim) ... -> (b axial_dim) n ...")

    def forward(
        self,
        x,
        *args,
        **kwargs
    ):
        if self.axial_dim == 1:
            return self.fn(x, *args, **kwargs)

        seq_len, axial_dim = x.shape[1], self.axial_dim

        next_multiple = math.ceil(seq_len / axial_dim) * axial_dim
        x = pad_at_dim(x, (0, next_multiple - seq_len), dim = 1)

        x = rearrange(x, 'b (n axial_dim) ... -> (b axial_dim) n ...', axial_dim = axial_dim)

        out = self.fn(x, *args, **kwargs)

        (out, *rest_out), tree_spec = tree_flatten(out)

        out = rearrange(out, '(b axial_dim) n ... -> b (n axial_dim) ...', axial_dim = axial_dim)

        out = out[:, :seq_len]
        out = tree_unflatten((out, *rest_out), tree_spec)

        return out

# post branch operator

class LayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        init_value = 0.,
        unit_offset = False
    ):
        super().__init__()
        self.unit_offset = unit_offset

        self.fn = fn
        self.gamma = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.gamma, init_value - float(unit_offset))

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)

        gamma = self.gamma + float(self.unit_offset)

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

class AdaptiveLayerScale(Module):
    def __init__(
        self,
        fn: Module,
        dim,
        dim_condition = None,
        init_bias_value = -2.
    ):
        super().__init__()
        self.fn = fn

        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        nn.init.zeros_(self.to_gamma.weight)
        nn.init.constant_(self.to_gamma.bias, init_bias_value)

    def forward(self, x, *, condition, **kwargs):
        if condition.ndim == 2:
            condition = rearrange(condition, 'b d -> b 1 d')

        out = self.fn(x, **kwargs)
        gamma = self.to_gamma(condition).sigmoid()

        if isinstance(out, Tensor):
            return out * gamma

        out, *rest = out
        return out * gamma, *rest

# skip connection combining

class ConcatCombine(Module):
    def __init__(self, dim, prev_layer_ind):
        super().__init__()
        self.prev_layer_ind = prev_layer_ind
        self.combine = LinearNoBias(dim * 2, dim)

    def forward(self, x, prev_layers: list[Tensor]):
        skip = prev_layers[self.prev_layer_ind]
        concatted_skip = cat((skip, x), dim = -1)
        return self.combine(concatted_skip)

# feedforward

class GLU(Module):
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

class FeedForward(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        glu_mult_bias = False,
        swish = False,
        relu_squared = False,
        solu = False,
        custom_activation = None,
        post_act_ln = False,
        dropout = 0.,
        sublayer_dropout = 0.,
        no_bias = False,
        zero_init_output = False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        assert at_most_one_of(relu_squared, solu)

        if exists(custom_activation):
            activation = deepcopy(custom_activation)
        elif relu_squared:
            activation = ReluSquared()
        elif solu:
            activation = SoLU(inner_dim)
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        if glu:
            proj_in = GLU(dim, inner_dim, activation, mult_bias = glu_mult_bias)
        else:
            proj_in = nn.Sequential(
                nn.Linear(dim, inner_dim, bias = not no_bias),
                activation
            )

        proj_out = nn.Linear(inner_dim, dim_out, bias = not no_bias)

        self.ff = Sequential(
            proj_in,
            LayerNorm(inner_dim) if post_act_ln else None,
            nn.Dropout(dropout),
            proj_out,
            nn.Dropout(sublayer_dropout) if sublayer_dropout > 0. else None
        )

        # init last linear layer to 0

        if zero_init_output:
            init_zero_(proj_out)

    def muon_parameters(self):
        weights = []

        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue

            weights.append(m.weight)

        return weights

    def forward(
        self,
        x,
        deep_embed = None
    ):
        out = self.ff(x)

        if exists(deep_embed):
            out = out * deep_embed

        return out

# attention. it is all we need

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        dim_context = None,
        heads = 8,
        causal = False,
        flash = False,
        pre_talking_heads = False,
        post_talking_heads = False,
        pre_scale_post_talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        sparse_topk_straight_through = False,
        num_mem_kv = 0,
        dropout = 0.,
        sublayer_dropout = 0.,
        on_attn = False,
        gate_value_heads = False,
        swiglu_values = False,
        gate_values = False,
        zero_init_output = False,
        hard = False,
        max_attend_past = None,
        qk_norm = False,
        qk_norm_groups = 1,
        qk_norm_scale = 10,
        qk_norm_dim_scale = False,
        value_rmsnorm = False,      # used in alphagenome and bytedance's GR3 for further stability
        l2_distance = False,
        sigmoid = False,
        gumbel_softmax = False,
        gumbel_softmax_temp = 1.,
        gumbel_softmax_hard = True,
        selective = False,
        cog_signed = False,
        custom_attn_fn: Callable | None = None,
        hybrid_module: Module | None = None,
        hybrid_mask_kwarg: str | None = None,
        hybrid_fold_axial_dim: int | None = None,
        hybrid_learned_mix = False,
        one_kv_head = False,
        kv_heads = None,
        value_dim_head = None,
        dim_out = None,
        add_zero_kv = False,         # same as add_zero_attn in pytorch
        head_learned_sink = False,
        rotate_num_heads = None,
        data_dependent_alibi = False,
        data_dependent_alibi_per_row = False,
        data_dependent_alibi_per_row_dim_head = 8,
        data_dependent_alibi_kwargs: dict = dict(),
        use_cope = False,
        cope_max_pos = 16,
        cope_soft_onehot_pos = False,
        cope_talking_heads = False,
        softclamp_logits = False,
        logit_softclamp_value = 50.,
        learned_value_residual_mix = False,
        orthog_projected_values = False,  # https://openreview.net/forum?id=Ard2QzPAUK
        orthog_projected_values_per_head = False,
        laser = False,                    # https://arxiv.org/abs/2411.03493v1
        laser_softclamp_value = 15.,
        qkv_receive_diff_residuals = False,
        use_latent_q = False,
        dim_latent_q = None,
        use_latent_kv = False,
        dim_latent_kv = None,
        latent_rope_subheads = None,
        onnxable = False,
        attend_sdp_kwargs: dict = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )
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
        self.groups = heads // kv_heads

        q_dim = dim_head * heads
        k_dim = dim_head * kv_heads
        v_dim = value_dim_head * kv_heads
        out_dim = value_dim_head * heads

        # determine input dimensions to qkv based on whether intermediate latent q and kv are being used
        # for eventually supporting multi-latent attention (MLA)

        self.to_latent_q = None
        self.to_latent_kv = None
        self.to_rotateable_k = None # for their "decoupled rope", subheads of keys that comes directly from base sequence (does not go through latents)

        dim_q_input = dim
        dim_kv_input = dim_kv

        if use_latent_q:
            assert exists(dim_latent_q)
            self.to_latent_q = LinearNoBias(dim, dim_latent_q)
            dim_q_input = dim_latent_q

        if use_latent_kv:
            assert exists(dim_latent_kv)
            self.to_latent_kv = LinearNoBias(dim, dim_latent_kv)
            dim_kv_input = dim_latent_kv

        if exists(latent_rope_subheads):
            assert not exists(rotate_num_heads), '`rotate_num_heads` cannot be set when multi-latent attention is being used'
            rotate_num_heads = latent_rope_subheads

            k_dim = dim_head * (kv_heads - latent_rope_subheads)

            self.to_rotateable_k = LinearNoBias(dim, dim_head * latent_rope_subheads)
            self.split_rotateable_k_heads = Rearrange('b n (h d) -> b h n d', h = latent_rope_subheads)

        self.use_latent_q = use_latent_q
        self.use_latent_kv = use_latent_kv

        # query key projection

        self.to_q = LinearNoBias(dim_q_input, q_dim)
        self.to_k = LinearNoBias(dim_kv_input, k_dim)
        self.to_v = LinearNoBias(dim_kv_input, v_dim)

        # split and merge of attention heads

        self.split_q_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_k_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.split_v_heads = Rearrange('b n (h d) -> b h n d', d = value_dim_head)

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # whether qkv receives different residual stream combinations from hyper connections or lime

        self.qkv_receive_diff_residuals = qkv_receive_diff_residuals

        # enhancing gradients to attention through exponentiated values

        self.laser = laser
        self.laser_softclamp_value = laser_softclamp_value

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
            self.qk_norm_k_scale = nn.Parameter(torch.ones(kv_heads, 1, dim_head))

        assert (not qk_norm) or divisible_by(dim_head, qk_norm_groups), 'dimension per attention head must be divisible by the qk norm groups'
        assert not (qk_norm and (dim_head // qk_norm_groups) <= 2), 'the group dimension may be too small (2 was too small in my tests, but 4 still works, surprisingly)'

        # value rms norm

        self.value_rmsnorm = MultiheadRMSNorm(dim_head, heads = heads) if value_rmsnorm else None

        # contextual positional encoding
        # https://arxiv.org/html/2405.18719v2

        cope = None

        if use_cope:
            assert causal, 'CoPE was designed for causal attention'
            assert not flash, 'CoPE is not flash attention compatible'

            cope = CoPE(
                dim = dim_head,
                heads = heads,
                max_pos = cope_max_pos,
                talking_heads = cope_talking_heads,
                soft_onehot = cope_soft_onehot_pos
            )

        # data dependent alibi
        # https://openreview.net/forum?id=q2Lnyegkr8

        self.data_dependent_alibi = None

        if data_dependent_alibi:

            dda_klass = DataDependentAlibi if not data_dependent_alibi_per_row else PerRowDataDependentAlibi
            dda_kwargs = dict(dim = dim, heads = heads, causal = causal)

            if data_dependent_alibi_per_row:
                dda_kwargs.update(dim_head = data_dependent_alibi_per_row_dim_head)

            self.data_dependent_alibi = dda_klass(**dda_kwargs, **data_dependent_alibi_kwargs)

        # attend class - includes core attention algorithm + talking heads

        self.attend = Attend(
            heads = heads,
            causal = causal,
            pre_talking_heads = pre_talking_heads,
            post_talking_heads = post_talking_heads,
            pre_scale_post_talking_heads = pre_scale_post_talking_heads,
            dropout = dropout,
            sparse_topk = sparse_topk,
            sparse_topk_straight_through = sparse_topk_straight_through,
            hard = hard,
            qk_norm = qk_norm,
            scale = qk_norm_scale if qk_norm else self.scale,
            l2_distance = l2_distance,
            sigmoid = sigmoid,
            gumbel_softmax = gumbel_softmax,
            gumbel_softmax_temp = gumbel_softmax_temp,
            gumbel_softmax_hard = gumbel_softmax_hard,
            selective = selective,
            cog_signed = cog_signed,
            custom_attn_fn = custom_attn_fn,
            add_zero_kv = add_zero_kv,
            head_learned_sink = head_learned_sink,
            flash = flash,
            softclamp_logits = softclamp_logits,
            logit_softclamp_value = logit_softclamp_value,
            cope = cope,
            onnxable = onnxable,
            sdp_kwargs = attend_sdp_kwargs
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

        # maybe learned value residual mixer per token

        self.to_value_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1')
         ) if learned_value_residual_mix else always(0.5)

        # attention on attention

        self.attn_on_attn = on_attn

        # return orthogonal projected weighted values on original values
        # "belief attention" - iclr 2026

        self.orthog_projected_values = orthog_projected_values
        self.orthog_projected_values_per_head = orthog_projected_values_per_head

        out_dim *= max(1, int(orthog_projected_values) + int(orthog_projected_values_per_head))

        # hybrid module, in same vein as hymba https://www.arxiv.org/abs/2411.13676

        hybrid_mix = None
        hybrid_norms = None
        hybrid_module = maybe(deepcopy)(hybrid_module)

        if exists(hybrid_module) and exists(hybrid_fold_axial_dim):
            hybrid_module = FoldAxially(axial_dim = hybrid_fold_axial_dim, fn = hybrid_module)
            hybrid_mix = LinearNoBias(dim, heads) if hybrid_learned_mix else None

            hybrid_norms = ModuleList([
                MultiheadRMSNorm(dim_head, heads = heads),
                MultiheadRMSNorm(dim_head, heads = heads)
            ])

        self.hybrid_module = hybrid_module
        self.hybrid_norms = hybrid_norms
        self.hybrid_mix = hybrid_mix
        self.hybrid_mask_kwarg = hybrid_mask_kwarg # for bidirectional, can forward `mask` into the hybrid module and let it handle variable lengths

        # output dimension by default same as input, but can be overridden

        dim_out = default(dim_out, dim)
        self.to_out = nn.Sequential(LinearNoBias(out_dim, dim_out * 2), nn.GLU()) if on_attn else LinearNoBias(out_dim, dim_out)

        # sublayer dropout

        self.sublayer_dropout = nn.Dropout(sublayer_dropout) if sublayer_dropout > 0. else None

        # the number of attention heads to rotate, for decoupled rope in multi-latent attention

        rotate_num_heads = default(rotate_num_heads, heads)

        assert 0 < rotate_num_heads <= heads
        is_partial_rotate_heads = rotate_num_heads < heads
        assert not (is_partial_rotate_heads and kv_heads < heads), 'grouped query attention not compatible with partial rotate heads (decoupled rope for multi-latent attention), yet'

        self.rotate_num_heads = rotate_num_heads

        # whether parent can kv cache

        self.can_cache_kv = not selective

        # init output projection 0

        if zero_init_output:
            init_zero_(self.to_out)

    @torch.no_grad()
    def qk_clip_(
        self,
        pre_softmax_attn: Tensor | Intermediates,
        tau = 100 # this hyperparameter controls how large the attention logits can be
    ):
        """ proposed by the Moonshot AI team as a solution for Muon training instability """

        if not is_tensor(pre_softmax_attn):
            pre_softmax_attn = pre_softmax_attn.pre_softmax_attn

        attn_logit_maxes = reduce(pre_softmax_attn, 'b h i j -> h', 'max')

        qk_weight_scale = (tau / attn_logit_maxes).clamp(max = 1.).sqrt()

        q_weight = self.to_q.weight
        k_weight = self.to_k.weight

        qk_dim, heads = q_weight.shape[0], qk_weight_scale.numel()

        qk_weight_scale = repeat(qk_weight_scale, 'h -> (h expand)', expand = qk_dim // heads)

        q_weight.mul_(qk_weight_scale)
        k_weight.mul_(qk_weight_scale)

    def muon_parameters(self):
        return chain(self.to_v.parameters(), self.to_out.parameters())

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        attn_bias = None,
        rotary_pos_emb = None,
        context_rotary_pos_emb = None,
        polar_pos_emb = None,
        pos = None, # for custom alibi positions
        prev_attn = None,
        mem = None,
        mem_mask = None,
        return_intermediates = False,
        cache: Intermediates | None = None,
        value_residual = None,
        additional_key_values: tuple[Tensor, Tensor] | None = None,
        additional_key_value_mask = None,
        kv_input_residual = None,
    ):
        b, n, h, kv_h, head_scale, num_mem_kv, device, has_context, qkv_receive_diff_residuals, is_multi_latent_attn = x.shape[0], x.shape[1], self.heads, self.kv_heads, self.head_scale, self.num_mem_kv, x.device, exists(context), self.qkv_receive_diff_residuals, self.use_latent_kv

        # an interesting possibility with hyper connections
        # having queries, keys, values be routed from different layers

        assert not (qkv_receive_diff_residuals and has_context), 'qkv receiving different sequences can only be used for self attention'

        if qkv_receive_diff_residuals:
            assert x.ndim == 4 and x.shape[0] == 3

            q_input, k_input, v_input = x
        else:
            kv_input = default(context, x)
            q_input, k_input, v_input = x, kv_input, kv_input

        # done for free transformer

        if exists(kv_input_residual):
            k_input = k_input + kv_input_residual
            v_input = v_input + kv_input_residual

        if exists(mem):
            k_input, mem_packed_shape = pack([mem, k_input], 'b * d')
            v_input, _ = pack([mem, v_input], 'b * d')

        # multi-latent attention logic
        # https://arxiv.org/abs/2405.04434 - Deepseek-AI team

        k_sub_heads = None # the rotateable subheads of keys derived from base sequence

        if self.use_latent_q:
            q_input = self.to_latent_q(q_input)

        if is_multi_latent_attn:
            assert not qkv_receive_diff_residuals
            needs_k_sub_heads = exists(self.to_rotateable_k)

            latent_kv_input = self.to_latent_kv(k_input)

            if needs_k_sub_heads:
                rotateable_k = self.to_rotateable_k(k_input)
                k_sub_heads = self.split_rotateable_k_heads(rotateable_k)

            if exists(cache):
                cached_latent_kv, maybe_cached_k_sub_heads = cache.cached_kv
                latent_kv_input = cat((cached_latent_kv, latent_kv_input), dim = -2)

                if exists(maybe_cached_k_sub_heads):
                    k_sub_heads = cat((maybe_cached_k_sub_heads, k_sub_heads), dim = -2)

            if return_intermediates:
                cached_kv = (latent_kv_input, k_sub_heads)

            k_input = v_input = latent_kv_input

        # query, key, value projection

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        q = self.split_q_heads(q)
        k = self.split_k_heads(k)
        v = self.split_v_heads(v)

        # take care of decoupled rope from multi-latent attention

        if exists(k_sub_heads):
            k = cat((k, k_sub_heads), dim = 1)

        # if previous values passed in for residual, either invoke resformer

        orig_values = v

        # https://arxiv.org/abs/2410.17897v1

        if exists(value_residual):
            value_residual_mix = self.to_value_residual_mix(q_input)
            v = value_residual.lerp(v, value_residual_mix)

        # qk normalization

        if self.qk_norm:
            qk_l2norm = partial(l2norm, groups = self.qk_norm_groups)
            q, k = map(qk_l2norm, (q, k))
            scale = self.qk_norm_scale

            q = q * self.qk_norm_q_scale
            k = k * self.qk_norm_k_scale

        # maybe value rmsnorm

        v = maybe(self.value_rmsnorm)(v)

        # take care of caching

        if not is_multi_latent_attn:
            if exists(cache):
                ck, cv = cache.cached_kv

                if exists(mem):
                    mk, k = unpack(k, mem_packed_shape, 'b h * d')
                    mv, v = unpack(v, mem_packed_shape, 'b h * d')

                k = cat((ck, k), dim = -2)
                v = cat((cv, v), dim = -2)

                if exists(mem):
                    k = cat((mk, k), dim = -2)
                    v = cat((mv, v), dim = -2)

            if return_intermediates:
                mem_len = mem.shape[-2] if exists(mem) else 0
                cached_kv = (k[..., mem_len:, :], v[..., mem_len:, :])

        if exists(rotary_pos_emb):
            rotate_num_heads = self.rotate_num_heads
            partial_rotate_heads = rotate_num_heads < h

            freqs, xpos_scale = rotary_pos_emb
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            if partial_rotate_heads:
                q_rest, q = q[:, :-rotate_num_heads], q[:, -rotate_num_heads:]
                k_rest, k = k[:, :-rotate_num_heads], k[:, -rotate_num_heads:]

            q = apply_rotary_pos_emb(q, freqs, q_xpos_scale)

            if has_context:
                # override with `context_rotary_pos_emb` if provided

                freqs, xpos_scale = context_rotary_pos_emb
                _, k_xpos_scale = (xpos_scale, xpos_scale ** -1.) if exists(xpos_scale) else (1., 1.)

            k = apply_rotary_pos_emb(k, freqs, k_xpos_scale)

            if partial_rotate_heads:
                q = cat((q_rest, q), dim = 1)
                k = cat((k_rest, k), dim = 1)

        if exists(polar_pos_emb):
            freqs, bias = polar_pos_emb
            q = apply_polar_pos_emb(q, freqs)
            k = apply_polar_pos_emb(k, freqs + bias)

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
                    input_mask = cat((mem_mask, input_mask), dim = -1)

        # i, j determined for relative positional bias, excluding memory key / values

        i, j = tuple(t.shape[-2] for t in (q, k))

        # maybe append memory key / values

        if num_mem_kv > 0:
            mem_k, mem_v = tuple(repeat(t, 'h n d -> b h n d', b = b) for t in (self.mem_k, self.mem_v))

            if self.qk_norm:
                mem_k = l2norm(mem_k)
                mem_k = mem_k * self.qk_norm_k_scale

            k = cat((mem_k, k), dim = -2)
            v = cat((mem_v, v), dim = -2)

            if exists(input_mask):
                input_mask = pad_at_dim(input_mask, (self.num_mem_kv, 0), dim = -1, value = True)

        # maybe append additional key / values

        if exists(additional_key_values):
            seq_len = k.shape[-2]

            added_k, added_v = additional_key_values
            added_kv_heads, added_kv_len = added_k.shape[1], added_k.shape[-2]

            # take care of expanding to query heads if mismatch between key / value heads with the ones coming from vlm

            if added_kv_heads != kv_h:
                assert divisible_by(h, added_kv_heads)
                k, v, added_k, added_v = tuple(repeat(t, 'b h ... -> b (r h) ...', r = h // t.shape[1]) for t in (k, v, added_k, added_v))

            k = cat((added_k, k), dim = -2)
            v = cat((added_v, v), dim = -2)

            if (exists(input_mask) or exists(additional_key_value_mask)):

                if not exists(additional_key_value_mask):
                    input_mask = pad_at_dim(input_mask, (added_kv_len, 0), dim = -1, value = True)
                elif not exists(input_mask):
                    input_mask = pad_at_dim(additional_key_value_mask, (0, seq_len), dim = -1, value = True)
                else:
                    input_mask = cat((additional_key_value_mask, input_mask), dim = -1)

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
            range_q = arange(j - i, j, device = device)
            range_k = arange(j, device = device)
            dist = einx.subtract('i, j -> 1 1 i j', range_q, range_k)
            max_attend_past_mask = dist > self.max_attend_past
            max_attend_past_mask = pad_at_dim(max_attend_past_mask, (num_mem_kv, 0), value = False, dim = -1) # handle memory key / values
            masks.append(max_attend_past_mask)

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        # prepare relative positional bias, if needed

        if exists(rel_pos):
            assert not exists(attn_bias)

            if exists(pos):
                assert isinstance(rel_pos, AlibiPositionalBias), 'only alibi allowed for custom positions at the moment'
                # allow for custom positions to be passed in
                attn_bias = rel_pos.forward_custom_pos(pos)
            else:
                attn_bias = rel_pos(i, j)

            attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0)) # handle memory key / values

        # prepare data dependent alibi from forgetting transformers paper, if needed

        if exists(self.data_dependent_alibi):
            attn_bias = self.data_dependent_alibi(x)

            attn_bias = pad_at_dim(attn_bias, (num_mem_kv, 0))

        if self.laser:
            v = softclamp(v, self.laser_softclamp_value)
            v = v.exp()

        # attention is all we need

        out, intermediates = self.attend(
            q, k, v,
            mask = final_attn_mask,
            attn_bias = attn_bias,
            prev_attn = prev_attn
        )

        # laser

        if self.laser:
            out = log(out)

        # store the values for resformer

        intermediates.values = orig_values

        # normformer scaling of heads

        if head_scale:
            out = out * self.head_scale_params

        # per head gating, from https://arxiv.org/abs/2306.12929

        if exists(self.to_v_head_gate):
            head_gate = self.to_v_head_gate(x)
            out = einx.multiply('b n h, b h n d ->b h n d', head_gate.sigmoid(), out)

        # if exists hybrid module, must do a normalization

         # hybrid module

        if exists(self.hybrid_module):

            # hybrid input

            hybrid_forward_kwargs = dict()

            if not self.causal and exists(self.hybrid_mask_kwarg):
                hybrid_forward_kwargs = {self.hybrid_mask_kwarg: mask}

            # handle maybe hybrid cache

            hybrid_forward_args = ()

            if exists(cache) and exists(cache.hybrid_hidden):
                hybrid_hiddens = cache.hybrid_hidden
                hybrid_forward_args = (hybrid_hiddens,)

            # hybrid forward

            hybrid_outputs = self.hybrid_module(x, *hybrid_forward_args, **hybrid_forward_kwargs)

            # handle hybrid out

            (hybrid_out, *rest_hybrid_outs), _ = tree_flatten(hybrid_outputs)

            # handle variable hybrid output and multi rmsnorm before summing to main attention output (also normed)

            if hybrid_out.ndim == 3:
                hybrid_out = rearrange(hybrid_out, 'b n (h d) -> b h n d', h = h)

            if len(rest_hybrid_outs) > 0:
                hybrid_hidden = first(rest_hybrid_outs)
                intermediates.hybrid_hidden = hybrid_hidden

            out_norm, hybrid_out_norm = self.hybrid_norms

            out = out_norm(out)
            hybrid_out = hybrid_out_norm(hybrid_out)

            if exists(self.hybrid_mix):
                mix = self.hybrid_mix(x)
                mix = rearrange(mix, 'b n h -> b h n 1')
                out = out.lerp(hybrid_out, mix.sigmoid())
            else:
                out = 0.5 * (out + hybrid_out)

        # merge heads

        out = self.merge_heads(out)

        # alphafold2 styled gating of the values

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * self.to_v_gate_activation(gates)

        # maybe orthogonal projected weighted values - "belief" attention

        if self.orthog_projected_values or self.orthog_projected_values_per_head:
            orthog_projected = []
            v_for_proj = repeat(orig_values, 'b h n d -> b n (g h d)', g = self.groups)

            if self.orthog_projected_values:
                projected = orthog_project(out, v_for_proj)
                orthog_projected.append(projected)

            if self.orthog_projected_values_per_head:
                v_for_proj = rearrange(v_for_proj, 'b n (h d) -> b n h d', h = h)
                out = rearrange(out, 'b n (h d) -> b n h d', h = h)
                projected = orthog_project(out, v_for_proj)
                projected = rearrange(projected, 'b n h d -> b n (h d)')
                orthog_projected.append(projected)

            out = cat(orthog_projected, dim = -1)

        # combine the heads

        out = self.to_out(out)

        # maybe sublayer dropout

        out = maybe(self.sublayer_dropout)(out)

        if exists(mask) and not exists(cache):
            out = einx.where('b n, b n d, -> b n d', mask, out, 0.)

        if not return_intermediates:
            return out

        intermediates.cached_kv = cached_kv

        return out, intermediates

class AttentionLayers(Module):
    def __init__(
        self,
        dim,
        depth = None,
        heads = 8,
        causal = False,
        cross_attend = False,
        only_cross = False,
        use_scalenorm = False,
        use_rmsnorm = False,
        use_dynamic_tanh = False,
        use_derf = False,
        dynamic_tanh_init_alpha = 1.,
        use_simple_rmsnorm = False,
        use_adaptive_layernorm = False,
        use_adaptive_rmsnorm = False,
        use_adaptive_layerscale = False, # paired with use_adaptive_layernorm for ada-ln-zero from DiT paper
        norm_add_unit_offset = True,
        dim_condition = None,
        adaptive_condition_mlp = False,
        adaptive_condition_mlp_expansion = 4,
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
        rotate_num_heads = None,
        polar_pos_emb = False,
        polar_bias_uniform_init = False,
        weight_tie_layers = False,
        custom_layers: tuple[str, ...] | None = None,
        layers_execute_order: tuple[int, ...] | None = None,
        sandwich_coef = None,
        par_ratio = None,
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
        softclamp_output = False,
        softclamp_output_value = 30.,
        zero_init_branch_output = False,
        layer_dropout = 0.,
        cross_attn_tokens_dropout = 0.,
        disable_abs_pos_emb = None,
        use_layerscale = False,
        layerscale_init_value = 0.,
        unet_skips = False,
        integrate_layers = False,
        layer_integrate_use_softmax = True,
        num_residual_streams = 1,
        qkv_receive_diff_residuals = False,
        reinject_input = False,              # seen first in DEQ paper https://arxiv.org/abs/1909.01377, but later used in a number of papers trying to achieve depthwise generalization https://arxiv.org/abs/2410.03020v1
        learned_reinject_input_gate = False,
        add_value_residual = False,          # resformer from Zhou et al - https://arxiv.org/abs/2410.17897v1 - further corroboration by https://arxiv.org/abs/2412.15113 (faster emergence of ICL) - looks like this setting may becoming a necessity for every transformer soon
        learned_value_residual_mix = True,   # seeing big improvements when the value residual mix value is learned per token - credit goes to @faresobeid for taking the first step with learned scalar mix, then @Blinkdl for taking it a step further with data dependent. here we will use per token learned
        rel_pos_kwargs: dict = dict(),
        residual_fn_kwargs: dict = dict(),
        hyper_conn_sinkhorn_iters = 5,
        verbose = True,
        **kwargs
    ):
        super().__init__()
        rotary_pos_emb = rotary_pos_emb or rotary_xpos

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)
        cross_attn_kwargs, kwargs = groupby_prefix_and_trim('cross_attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)
        data_dependent_alibi = attn_kwargs.get('data_dependent_alibi', False)

        assert len(kwargs) == 0, f'unrecognized kwargs passed in {kwargs.keys()}'

        self.dim = dim
        self.causal = causal
        self.layers = ModuleList([])

        self.attn_heads = heads
        self.attn_dim_head = dim_head

        # routing related
        # 1. greater than one residual stream, proposed in Hyper-Connections paper https://arxiv.org/abs/2409.19606
        # 2. integrating more than one past layer, from LIMe paper https://arxiv.org/abs/2502.09245

        qkv_receive_diff_residuals |= integrate_layers # qkv always receives different views if integrating layers

        # hyper connections

        assert num_residual_streams > 0
        has_hyper_connections = num_residual_streams > 1

        self.num_residual_streams = num_residual_streams
        self.stream_emb = nn.Parameter(torch.zeros(num_residual_streams, dim)) if num_residual_streams > 1 else None

        assert not (has_hyper_connections and gate_residual)

        hyper_conn_produce_diff_views = qkv_receive_diff_residuals and not integrate_layers

        # LIMe

        self.layer_integrators = ModuleList([])

        assert not (qkv_receive_diff_residuals and not (hyper_conn_produce_diff_views or integrate_layers))

        # positions related

        self.disable_abs_pos_emb = default(disable_abs_pos_emb, (rel_pos_bias or rotary_pos_emb or polar_pos_emb))

        rotary_emb_dim = default(rotary_emb_dim, dim_head // 2)

        assert rotary_emb_dim <= dim_head, f'rotary emb dim {rotary_emb_dim} must be less than or equal to attention head dimension {dim_head}'

        if verbose and rotary_emb_dim < 32:
            print('when training language model, rotary embedding dimension should be at least 32')

        assert at_most_one_of(rotary_pos_emb, polar_pos_emb), f'either rotary positional embedding or polar positional embedding can be turned on'
        assert not (rotary_xpos and not causal), 'rotary xpos is not compatible with bidirectional attention'
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim, use_xpos = rotary_xpos, scale_base = rotary_xpos_scale_base, interpolation_factor = rotary_interpolation_factor, base_rescale_factor = rotary_base_rescale_factor) if rotary_pos_emb else None

        # polar positional embedding (PoPE) - https://arxiv.org/abs/2509.10534

        self.polar_pos_emb = PolarEmbedding(dim_head, heads, polar_bias_uniform_init) if polar_pos_emb else None

        assert at_most_one_of(alibi_pos_bias, rel_pos_bias, data_dependent_alibi), 'you can only choose one of Alibi positional bias, data dependent Alibi (forgetting transformers), dynamic tanh, or T5 relative positional bias'
        assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'

        # relative positional bias

        flash_attn = attn_kwargs.get('flash', False)
        assert at_most_one_of(rel_pos_bias, dynamic_pos_bias, alibi_pos_bias), 'you can only choose up to one of t5, alibi, or dynamic positional bias'

        self.rel_pos = None

        if rel_pos_bias:
            assert not flash_attn, 'flash attention not compatible with t5 relative positional bias'
            self.rel_pos = RelativePositionBias(scale = dim_head ** 0.5, causal = causal, heads = heads, num_buckets = rel_pos_num_buckets, max_distance = rel_pos_max_distance, **rel_pos_kwargs)
        elif dynamic_pos_bias:
            assert not flash_attn, 'flash attention not compatible with dynamic positional bias'
            self.rel_pos = DynamicPositionBias(dim = dim // 4, heads = heads, log_distance = dynamic_pos_bias_log_distance, depth = dynamic_pos_bias_mlp_depth, norm = dynamic_pos_bias_norm, **rel_pos_kwargs)
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            self.rel_pos = AlibiPositionalBias(heads = alibi_num_heads, total_heads = heads, **rel_pos_kwargs)

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'

        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        assert not (flash_attn and (residual_attn or cross_residual_attn)), 'flash attention is not compatible with residual attention'

        self.cross_attend = cross_attend

        # determine norm

        assert at_most_one_of(use_scalenorm, use_rmsnorm, use_dynamic_tanh, use_derf, use_simple_rmsnorm, use_adaptive_layernorm, use_adaptive_rmsnorm), 'you can only use either scalenorm, rmsnorm, adaptive layernorm, adaptive rmsnorm, or simple rmsnorm'

        norm_need_condition = False
        dim_condition = default(dim_condition, dim)
        dim_condition_mult = 1

        if adaptive_condition_mlp:
            dim_condition_mult = adaptive_condition_mlp_expansion

        if use_scalenorm:
            norm_class = ScaleNorm
        elif use_rmsnorm:
            norm_class = RMSNorm
        elif use_simple_rmsnorm:
            norm_class = SimpleRMSNorm
        elif use_dynamic_tanh:
            assert pre_norm, 'dynamic tanh norm only tested for pre-norm'
            norm_class = partial(DynamicTanh, init_alpha = dynamic_tanh_init_alpha)
        elif use_derf:
            norm_class = Derf
        elif use_adaptive_layernorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveLayerNorm, dim_condition = dim_condition * dim_condition_mult)
        elif use_adaptive_rmsnorm:
            norm_need_condition = True
            norm_class = partial(AdaptiveRMSNorm, dim_condition = dim_condition * dim_condition_mult)
        else:
            norm_class = LayerNorm

        norm_fn = partial(norm_class, dim)

        if not norm_need_condition and norm_add_unit_offset:
            # researcher Ohad Rubin shares in a blog post by adding an offset to gammas, they can be subjected to weight decay safely
            norm_fn = partial(norm_fn, unit_offset = True)

        self.norm_need_condition = norm_need_condition
        self.dim_condition = dim_condition

        # determine default block layer type order

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # determine post branch wrapper

        assert at_most_one_of(use_layerscale, use_adaptive_layerscale)

        post_branch_fn = None
        post_branch_fn_needs_condition = False

        if use_layerscale:
            post_branch_fn = partial(LayerScale, dim = dim, init_value = layerscale_init_value)
        elif use_adaptive_layerscale:
            post_branch_fn = partial(AdaptiveLayerScale, dim = dim, dim_condition = dim_condition * dim_condition_mult)
            post_branch_fn_needs_condition = True

        self.post_branch_fn_needs_condition = post_branch_fn_needs_condition

        if exists(post_branch_fn) and not post_branch_fn_needs_condition and norm_add_unit_offset:
            post_branch_fn = partial(post_branch_fn, unit_offset = True)

        # setup mlp for conditioning

        self.need_condition = norm_need_condition or post_branch_fn_needs_condition

        self.adaptive_mlp = nn.Identity()

        if self.need_condition and adaptive_condition_mlp:
            self.adaptive_mlp = nn.Sequential(
                LinearNoBias(dim_condition, dim_condition * dim_condition_mult),
                nn.SiLU()
            )

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output':  True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output':  True}

        # setup weight tying, which is a special case of `layer_execute_order`

        assert not (exists(layers_execute_order) and exists(custom_layers) and exists(depth)), 'depth should not be passed in if using custom layers and custom layer execution order'

        assert not (weight_tie_layers and any([*map(exists, (custom_layers, par_ratio, sandwich_coef))]))

        if weight_tie_layers:
            assert exists(depth), 'depth must be passed in with `weight_tie_layers` = True'
            assert not exists(layers_execute_order)
            layers_execute_order = tuple(range(len(default_block))) * depth
            depth = 1

        # calculate layer block order

        len_default_block = 1

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
            assert exists(depth), '`depth` must be passed in for `Decoder` or `Encoder`'
            layer_types = default_block * depth
            len_default_block = len(default_block)

        self.layer_types = layer_types
        self.layers_execute_order = default(layers_execute_order, tuple(range(len(layer_types))))

        assert all([i < len(self.layer_types) for i in self.layers_execute_order])

        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # set the depth

        depth = default(depth, len(self.layers_execute_order))
        self.depth = depth

        # stochastic depth

        self.layer_dropouts = cast_tuple(layer_dropout, len(layer_types))

        # structured dropout for cross attending

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # optional soft clamping just before the final norm
        # used in gemma 2

        self.softclamp_output = softclamp_output
        self.softclamp_output_value = softclamp_output_value

        # whether it has post norm

        self.final_norm = norm_fn() if pre_norm and pre_norm_has_final_norm else nn.Identity()

        # whether unet or not

        self.unet_skips = unet_skips
        num_skips = self.depth // len_default_block

        assert not (unet_skips and num_skips == 0), 'must have depth of at least 2 for unet skip connections'

        skip_indices = [i * len_default_block for i in range(num_skips)]

        self.skip_combines = ModuleList([])

        # whether there is reinjection of input at every layer

        self.reinject_input = reinject_input
        self.reinject_input_proj = nn.Linear(dim, dim, bias = False) if reinject_input else None
        self.learned_reinject_input_gate = nn.Linear(dim, 1, bias = False) if learned_reinject_input_gate else None

        # add the value from the first self attention block to all latter projected self attention values as a residual

        self.add_value_residual = add_value_residual

        is_first_self_attn = True
        is_first_cross_attn = True
        learned_value_residual_mix &= add_value_residual

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):

            # `ind` is the index of each module - attention, feedforward, cross attention
            # but `block_ind` refers to the typical enumeration of a transformer block (attn + ff + [optional] cross attn)

            block_begin = divisible_by(ind, len_default_block)
            block_ind = ind // len_default_block

            is_last_layer = ind == (len(self.layer_types) - 1)

            # attention, cross attention, feedforward

            layer_qkv_receives_diff_view = layer_type == 'a' and qkv_receive_diff_residuals and not (is_first_self_attn and integrate_layers)

            if layer_type == 'a':
                self_attn_learned_value_residual = learned_value_residual_mix and not is_first_self_attn

                layer = Attention(dim, heads = heads, causal = causal, qkv_receive_diff_residuals = layer_qkv_receives_diff_view, learned_value_residual_mix = self_attn_learned_value_residual, rotate_num_heads = rotate_num_heads, **attn_kwargs)
                is_first_self_attn = False

            elif layer_type == 'c':
                layer = Attention(dim, heads = heads, **{**attn_kwargs, **cross_attn_kwargs})
                is_first_cross_attn = False

            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)

            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(post_branch_fn):
                layer = post_branch_fn(layer)

            layer_integrate = None

            if integrate_layers:
                num_layer_hiddens = ind + 1
                layer_integrate_num_view = 3 if layer_qkv_receives_diff_view else 1

                layer_integrate = DynamicLIMe(dim, num_layer_hiddens, num_views = layer_integrate_num_view, use_softmax = layer_integrate_use_softmax)

            if has_hyper_connections:
                residual_fn = partial(HyperConnection, num_residual_streams = num_residual_streams, sinkhorn_iters = hyper_conn_sinkhorn_iters)

                if layer_type == 'a' and hyper_conn_produce_diff_views:
                    residual_fn = partial(residual_fn, num_input_views = 3)

            elif gate_residual:
                residual_fn = GRUGating
            else:
                residual_fn = Residual

            residual = residual_fn(dim, layer_index = ind, scale_residual = scale_residual, scale_residual_constant = scale_residual_constant, **residual_fn_kwargs)

            # handle unet skip connection

            skip_combine = None
            is_latter_half = block_begin and block_ind >= (self.depth / 2)

            if self.unet_skips and is_latter_half:
                skip_combine = ConcatCombine(dim, skip_indices.pop())

            # all normalizations of the layer

            pre_branch_norm = norm_fn() if pre_norm else None
            post_branch_norm = norm_fn() if sandwich_norm else None
            post_main_norm = norm_fn() if not pre_norm else None

            norms = ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.skip_combines.append(skip_combine)

            self.layer_integrators.append(layer_integrate)

            self.layers.append(ModuleList([
                norms,
                layer,
                residual
            ]))

        # determine whether can cache kv

        self.can_cache_kv = all([module.can_cache_kv for module in self.modules() if isinstance(module, Attention)])

    def attn_qk_clip_(
        self,
        intermediates: LayerIntermediates,
        tau = 100.
    ):
        # pairs up the attention intermediates with each attention module and does qk clip proposed by kimi team

        layer_and_layer_types = (self.layers, self.layer_types)

        attn_layers = [layer for (_, layer, _), layer_type in zip(self.layers, self.layer_types) if layer_type in ('a', 'c')]
        attn_intermeds = intermediates.attn_intermediates

        assert len(attn_layers) == len(attn_intermeds)

        for attn_layer, attn_inter in zip(attn_layers, attn_intermeds):
            attn_layer.qk_clip_(attn_inter, tau = tau)

    def muon_parameters(self):
        params = []

        for m in self.modules():
            if not isinstance(m, (Attention, FeedForward)):
                continue

            params.extend(list(m.muon_parameters()))

        return params

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
        seq_start_pos: Tensor | None = None,
        seq_pos_offset: int = 0,
        cache: LayerIntermediates | None = None,
        input_not_include_cache = False,
        cache_age = 1,
        return_hiddens = False,
        rotary_pos_emb = None,
        polar_pos_emb = None,
        pos = None,
        context_pos = None,
        attn_bias = None,
        deep_embeds_and_ids: tuple[nn.Parameter, Tensor] | None = None,
        self_attn_additional_kv: (
            LayerIntermediates |
            list[tuple[Tensor, Tensor]]
            | None
        ) = None,
        additional_kv_mask = None,
        detach_additional_kv = False,
        route_additional_kv_to_top = True,
        condition = None,
        in_attn_cond = None, # https://arxiv.org/abs/2105.04090
        layers_execute_order: tuple[int, ...] | None = None,
        self_attn_kv_residuals: Tensor | None = None,
        cross_attn_kv_residuals: Tensor | None = None
    ):
        assert not (self.cross_attend ^ exists(context)), 'context must be passed in if cross_attend is set to True'
        assert not (exists(condition) ^ self.need_condition), 'condition needs to be passed in if using adaptive layernorm or vice versa'

        # handle condition

        if exists(condition):
            assert condition.shape[-1] == self.dim_condition, f'expected condition dimension of {self.dim_condition} but received {condition.shape[-1]}'

            assert condition.ndim in {2, 3}

            if condition.ndim == 2:
                condition = rearrange(condition, 'b d -> b 1 d')

            condition = self.adaptive_mlp(condition)

        # setup maybe layernorm kwarg

        norm_kwargs = dict()

        if self.norm_need_condition:
            norm_kwargs.update(condition = condition)

        # maybe post branch fn conditioning (DiT paper's ada-ln-zero)

        block_forward_kwargs = dict()

        if self.post_branch_fn_needs_condition:
            block_forward_kwargs.update(condition = condition)

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
            seq_arange = arange(x.shape[-2], device = x.device, dtype = torch.long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        cross_attn_rotary_pos_emb = dict()

        if exists(self.rotary_pos_emb):
            if not exists(rotary_pos_emb):
                maybe_mem = first(mems, None) # todo - handle edge case where different layers get different memory lengths. don't think this will ever come up but who knows
                mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

                if not exists(pos):
                    pos = arange(x.shape[1] + mem_len + seq_pos_offset, device = x.device) - mem_len

                rotary_pos_emb = self.rotary_pos_emb(pos)

            # allow for rotary positions for context if provided

            if exists(context_pos):
                assert self.cross_attend
                context_rotary_pos_emb = self.rotary_pos_emb(context_pos)

                cross_attn_rotary_pos_emb.update(
                    rotary_pos_emb = rotary_pos_emb,
                    context_rotary_pos_emb = context_rotary_pos_emb
                )

        # polar positions

        if exists(self.polar_pos_emb):
            if not exists(polar_pos_emb):
                if not exists(pos):
                    pos = arange(x.shape[1] + seq_pos_offset, device = x.device)

                polar_pos_emb = self.polar_pos_emb(pos)

        # assume cached key / values

        prev_cache_length = 0

        attn_cache = []

        if exists(cache):
            assert self.causal and not exists(attn_mask)

            prev_cache_length = cache.cache_length

            if exists(context):
                context = context[:, :0]

            if cache_age > 0:
                x = x[:, -cache_age:] # for spec decoding, may be greater than 1

                if exists(deep_embeds_and_ids):
                    deep_embeds, token_ids = deep_embeds_and_ids
                    token_ids = token_ids[:, -cache_age:]
                    deep_embeds_and_ids = (deep_embeds, token_ids)

            attn_cache = cache.attn_intermediates

        next_cache_length = x.shape[1]

        iter_attn_cache = iter(attn_cache)

        # handle deep embeds if needed

        deep_embeds = []

        if exists(deep_embeds_and_ids):
            deep_embeds, token_ids = deep_embeds_and_ids
            deep_embeds_across_depth = deep_embeds[token_ids]
            deep_embeds = rearrange(deep_embeds_across_depth, 'b n l d -> l b n d')

        deep_embeds_iter = iter(deep_embeds)

        # setup multistreams if needed

        streams = self.num_residual_streams
        is_multistream = streams > 1

        if is_multistream:
            x = einx.add('b n d, s d -> (b s) n d', x, self.stream_emb)

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.skip_combines,
            self.layers,
            self.layer_dropouts,
            self.layer_integrators
        )

        # able to override the layers execution order on forward, for trying to depth extrapolate

        layers_execute_order = default(layers_execute_order, self.layers_execute_order)
        layer_variables = tuple(tuple(layer_variable[i] for i in layers_execute_order) for layer_variable in layer_variables)

        # additional self attn key / values - say coming from vlm

        if exists(self_attn_additional_kv) and route_additional_kv_to_top:

            if isinstance(self_attn_additional_kv, LayerIntermediates):
                self_attn_additional_kv = get_cached_kvs(self_attn_additional_kv)

            if detach_additional_kv:
                self_attn_additional_kv = detach_all(self_attn_additional_kv)

            num_self_attns = sum([layer_type == 'a' for layer_type in first(layer_variables)])

            self_attn_additional_kv = self_attn_additional_kv[-num_self_attns:]
            self_attn_additional_kv = [None] * (num_self_attns - len(self_attn_additional_kv)) + self_attn_additional_kv

        iter_self_attn_kv = iter(default(self_attn_additional_kv, ()))

        # derived input for reinjection if needed

        inp_inject = None

        if self.reinject_input:
            assert not exists(in_attn_cond)
            inp_inject = self.reinject_input_proj(x)

        elif exists(in_attn_cond):
            # handle in-attention conditioning, which serves the same purpose of having the network learn the residual
            inp_inject = in_attn_cond if in_attn_cond.ndim == 3 else rearrange(in_attn_cond, 'b d -> b 1 d')

        if exists(inp_inject) and exists(self.learned_reinject_input_gate):
            inp_inject_gate = self.learned_reinject_input_gate(x).sigmoid()
            inp_inject = inp_inject * inp_inject_gate

        # store all hiddens for skips

        skip_hiddens = []

        # for residuals to key value inputs for self and cross attention

        self_attn_kv_residuals_iter = iter((None,))
        cross_attn_kv_residuals_iter = iter((None,))

        if exists(self_attn_kv_residuals):
            if self_attn_kv_residuals.ndim == 3:
                self_attn_kv_residuals = rearrange(self_attn_kv_residuals, '... ->  1 ...')

            self_attn_kv_residuals_iter = iter(self_attn_kv_residuals)

        if exists(cross_attn_kv_residuals):
            if cross_attn_kv_residuals.ndim == 3:
                cross_attn_kv_residuals = rearrange(cross_attn_kv_residuals, '... ->  1 ...')

            cross_attn_kv_residuals_iter = iter(cross_attn_kv_residuals)

        # for value residuals

        first_self_attn_inter = None
        first_cross_attn_inter = None

        # go through the attention and feedforward layers

        for ind, (layer_type, skip_combine, (norm, block, residual_fn), layer_dropout, layer_integrator) in enumerate(zip(*layer_variables)):
            is_last = ind == (len(self.layers) - 1)

            # handle skip connections

            skip_hiddens.append(x)

            if exists(skip_combine):
                x = skip_combine(x, skip_hiddens)

            # layer dropout

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

            x, inner_residual, residual_kwargs = residual_fn.prepare(x)

            layer_hiddens.append(x)

            if exists(layer_integrator):
                x = layer_integrator(x, layer_hiddens)

            pre_norm, post_branch_norm, post_main_norm = norm

            if self.need_condition:
                pre_norm = maybe(partial)(pre_norm, **norm_kwargs)
                post_branch_norm = maybe(partial)(post_branch_norm, **norm_kwargs)
                post_main_norm = maybe(partial)(post_main_norm, **norm_kwargs)

            if exists(inp_inject):
                x = x + inp_inject

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == 'a' and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)

            block = partial(block, **block_forward_kwargs)

            # handle maybe value residuals

            maybe_self_attn_value_residual = None
            maybe_cross_attn_value_residual = None

            if self.add_value_residual:
                if exists(first_self_attn_inter):
                    maybe_self_attn_value_residual = first_self_attn_inter.values

                if exists(first_cross_attn_inter):
                    maybe_cross_attn_value_residual = first_cross_attn_inter.values

            # forward depending on layer type

            if layer_type == 'a':
                out, inter = block(x, mask = mask, context_mask = self_attn_kv_mask, attn_mask = attn_mask, rel_pos = self.rel_pos, pos = pos, rotary_pos_emb = rotary_pos_emb, polar_pos_emb = polar_pos_emb, additional_key_values = next(iter_self_attn_kv, None), additional_key_value_mask = additional_kv_mask, prev_attn = prev_attn, cache = next(iter_attn_cache, None), mem = layer_mem, mem_mask = layer_mem_mask, attn_bias = attn_bias, kv_input_residual = next(self_attn_kv_residuals_iter, None), value_residual = maybe_self_attn_value_residual, return_intermediates = True)
            elif layer_type == 'c':
                out, inter = block(x, context = context, mask = mask, context_mask = context_mask, prev_attn = prev_cross_attn, cache = next(iter_attn_cache, None), kv_input_residual = next(cross_attn_kv_residuals_iter, None), value_residual = maybe_cross_attn_value_residual, **cross_attn_rotary_pos_emb, return_intermediates = True)
            elif layer_type == 'f':
                out = block(x, deep_embed = next(deep_embeds_iter, None))

            # store first self or cross attention intermediate for value residual

            if not exists(first_self_attn_inter) and layer_type == 'a':
                first_self_attn_inter = inter

            if not exists(first_cross_attn_inter) and layer_type == 'c':
                first_cross_attn_inter = inter

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual, **residual_kwargs)

            if layer_type in ('a', 'c') and return_hiddens:
                inter.layer_type = layer_type
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.softclamp_output:
            x = softclamp(x, self.softclamp_output_value)

        final_norm = self.final_norm

        if self.need_condition:
            final_norm = maybe(partial)(final_norm, **norm_kwargs)

        # take care of multistreams if needed, use sum for now

        if is_multistream:
            x = reduce(x, '(b s) n d -> b n d', 'sum', s = streams)

        x = final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens = hiddens,
            last_hidden = x,
            attn_intermediates = intermediates,
            layer_hiddens = layer_hiddens,
            cache_length = next_cache_length + prev_cache_length
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

            prefix_mask = arange(n, device = device) < rearrange(prefix_attn_len, 'b -> b 1 1 1')
            forwarded_mask = forwarded_mask | prefix_mask

        if exists(attn_mask):
            forwarded_mask = forwarded_mask & attn_mask

        return super().forward(x, *args, attn_mask = forwarded_mask, **kwargs)

class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend = True, only_cross = True, **kwargs)

class AttentionPool(Module):
    def __init__(
        self,
        dim,
        num_pooled_tokens = 1,
        dim_context = None,
        add_residual = False,
        depth = 1,
        heads = 8,
        dim_head = 64,
        use_transformer_blocks = None,
        squeeze_output = None,
        attn_kwargs: dict = dict()
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        squeeze_output = default(squeeze_output, False)
        assert not (squeeze_output and num_pooled_tokens > 1)

        use_transformer_blocks = default(use_transformer_blocks, depth > 1)
        assert use_transformer_blocks or depth == 1

        self.queries = nn.Parameter(torch.randn(num_pooled_tokens, dim) * 1e-2)

        if use_transformer_blocks:
            assert not add_residual, 'residual already in effect when doing a full cross attention based transformer for pooling'
            attn_kwargs = {f'attn_{k}': v for k, v in attn_kwargs.items()}

            self.pooler = CrossAttender(dim = dim, cross_attn_dim_context = dim_context, depth = depth, heads = heads, attn_dim_head = dim_head, )
        else:
            self.pooler = Attention(dim = dim, dim_context = dim_context, heads = heads, dim_head = dim_head, **attn_kwargs)

        self.add_residual = add_residual
        self.squeeze_output = squeeze_output

    def forward(self, context, mask = None):
        batch = context.shape[0]

        queries = repeat(self.queries, 'n d -> b n d', b = batch)

        pooled = self.pooler(queries, context, context_mask = mask)

        if self.add_residual:
            pooled = pooled + queries

        if self.squeeze_output:
            pooled = rearrange(pooled, 'b 1 d -> b d')

        return pooled

class ViTransformerWrapper(Module):
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

class TransformerWrapper(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: dict[str, int] = dict(),
        emb_dim = None,
        max_mem_len = 0,
        shift_mem_down = 0,
        emb_dropout = 0.,
        post_emb_norm = False,
        num_memory_tokens = None,
        memory_tokens_interspersed_every = None,
        tie_embedding = False,
        logits_dim = None,
        return_only_embed = False,
        num_output_heads = 1,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
        l2norm_embed = False,
        recycling = False,            # from Jumper et al. - Alphafold2
        train_max_recycle_steps = 4,  # saw a benefit for language modeling up to 3 recycling steps, so let's default this to 4
        emb_frac_gradient = 1.,       # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight = 1e-4,
        average_pool_embed = False,
        use_cls_token = False,
        num_cls_tokens = 1,
        attn_pool = False,
        num_pooled_tokens = 1,
        attn_pool_depth = 1,
        dim_pooled_tokens = None,
        squeeze_out_last_dim = False,
        token_emb: TokenEmbedding | None = None,
        mixture_of_softmax = False,
        mixture_of_softmax_k = 4,
        sigsoftmax_logits = False,
        ff_deep_embed = False,
        to_logits: Module | None = None,
        add_continuous_pred_head = False
    ):
        super().__init__()

        dim = attn_layers.dim
        depth = attn_layers.depth

        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens
        self.num_cls_tokens = num_cls_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed

        if not exists(token_emb):
            token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed = l2norm_embed)

        self.token_emb = token_emb

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
            self.embeds = ModuleDict({f'{name}_embed': nn.Embedding(num_tokens, emb_dim) for name, num_tokens in embed_num_tokens.items()})

        # deep embed

        # credit goes to Braden Koszarsky for first devising value embeddings in nanogpt-speedrun project
        # then Bo Peng for coming up with this alternate design in feedforward for RWKV 8
        # improvements were clearest to me (on my toy setup) with multiplying on output of feedforward, will try with attention at future date

        self.ff_deep_embed = None
        if ff_deep_embed:
            self.ff_deep_embed = nn.Parameter(torch.ones(num_tokens, depth, dim))

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        assert num_output_heads > 0

        assert at_most_one_of(average_pool_embed, use_cls_token)

        # maybe recycling

        self.recycling = recycling
        self.recycled_proj = LinearNoBias(dim, dim) if recycling else None

        self.train_max_recycle_steps = train_max_recycle_steps

        # either cls token or attn pool, but not both

        assert not (use_cls_token and attn_pool)

        # classic cls token from the bert days

        self.cls_token = None

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(num_cls_tokens, dim))
            nn.init.normal_(self.cls_token, std = 0.02)

        # attn pool

        self.attn_pool = None

        if attn_pool:
            self.attn_pool = AttentionPool(dim = default(dim_pooled_tokens, dim), dim_context = dim, num_pooled_tokens = num_pooled_tokens, depth = attn_pool_depth, heads = self.attn_layers.attn_heads, dim_head = self.attn_layers.attn_dim_head)

        # whether to average pool the embed (`global average pool`)

        self.average_pool_embed = average_pool_embed

        # output type

        self.output_is_log_prob = mixture_of_softmax

        self.to_mixture = None
        self.combine_mixture = None

        if mixture_of_softmax:
            assert num_output_heads == 1

            self.to_mixture = Sequential(
                LinearNoBias(dim, dim * mixture_of_softmax_k),
                Rearrange('... (k d) -> ... k d', k = mixture_of_softmax_k)
            )

            self.combine_mixture = LinearNoBias(dim, mixture_of_softmax_k)

        # sig softmax

        self.sigsoftmax_logits = sigsoftmax_logits

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, num_tokens)

        self.has_multiple_heads = num_output_heads > 1

        if return_only_embed:
            self.to_logits = None
        elif tie_embedding:
            assert isinstance(token_emb, TokenEmbedding), 'can only tie embedding if using `TokenEmbedding`'
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.to_logits = ModuleList([LinearNoBias(dim, logits_dim) for _ in range(num_output_heads)])
        else:
            self.to_logits = LinearNoBias(dim, logits_dim) if not exists(to_logits) else to_logits

        # add a head that predicts the embedding of the next step

        self.add_continuous_pred_head = add_continuous_pred_head

        if add_continuous_pred_head:

            self.to_next_embed_pred = nn.Sequential(
                LinearNoBias(dim, dim),
                nn.SiLU(),
                LinearNoBias(dim, dim)
            )

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # squeeze out last dimension if possible

        self.squeeze_out_last_dim = squeeze_out_last_dim

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0 and not recycling and self.attn_layers.can_cache_kv
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if hasattr(self.token_emb, 'init_'):
            self.token_emb.init_()

        if self.l2norm_embed:
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std = 1e-5)

    def attn_qk_clip_(
        self,
        intermediates: LayerIntermediates,
        tau = 100.
    ):
        self.attn_layers.attn_qk_clip_(intermediates, tau = tau)

    def muon_parameters(self):
        return self.attn_layers.muon_parameters()

    def forward(
        self,
        x,
        return_embeddings = False,
        return_logits_and_embeddings = False,
        return_intermediates = False,
        return_embeddings_and_intermediates = False,
        return_logit_entropies = False,
        return_next_embed_pred = False,
        mask = None,
        return_mems = False,
        return_attn = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds = None,
        prepend_mask = None,
        embed_ids: dict[str, Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss = False,
        attn_z_loss_weight = 1e-4,
        seq_start_pos = None,
        cache: LayerIntermediates | None = None,
        input_not_include_cache = False,
        token_emb_kwargs = dict(),
        to_logits_kwargs = dict(),
        **kwargs,
    ):

        # if sequence is None, auto create an empty one if `prepend_embeds` was supplied

        if not exists(x):
            assert exists(prepend_embeds)
            x = prepend_embeds.new_empty((prepend_embeds.shape[0], 0), dtype = torch.long)

        # shapes and variables

        b, n, device, token_ids, num_mems, has_memory_tokens, emb_frac_gradient, orig_mask = x.shape[0], x.shape[1], x.device, x, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient, mask

        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss | return_embeddings_and_intermediates
        return_embeddings = return_embeddings | (not exists(self.to_logits)) | return_embeddings_and_intermediates

        # take care of position embedding offsets in the presence of cache and sequence is less than cache length (not full sequence)

        seq_pos_offset = 0

        if exists(cache) and input_not_include_cache:
            seq_pos_offset = cache.cache_length

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos = pos, seq_start_pos = seq_start_pos, offset = seq_pos_offset) if not external_pos_emb else pos
        x = self.token_emb(x, **token_emb_kwargs) + pos_emb

        # add additional embeddings

        assert not (exists(self.embeds) ^ (len(embed_ids) > 0)), '`embed_num_tokens` must be defined on `TransformerWrapper`'

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

            x = cat((prepend_embeds, x), dim = -2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device = device, dtype = torch.bool))
                prepend_mask = default(prepend_mask, lambda: torch.ones((b, prepend_seq), device = device, dtype = torch.bool))

                mask = cat((prepend_mask, mask), dim = -1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # init embed

        init_embed = x

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        # maybe deep embeds

        deep_embed_and_ids = None

        if exists(self.ff_deep_embed):
            deep_embed_and_ids = (self.ff_deep_embed, token_ids)

        # maybe cls token

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, '... -> b ...', b = b)
            x, cls_packed_shape = pack([cls_tokens, x], 'b * d')

            if exists(mask):
                mask = F.pad(mask, (self.num_cls_tokens, 0), value = True)

        # maybe memory / register tokens

        if has_memory_tokens:
            mem_seq = x.shape[-2]
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

        # handle maybe shifting of memories

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        # attn layers kwargs

        kwargs = dict(
            **kwargs,
            pos = pos,
            seq_pos_offset = seq_pos_offset,
            seq_start_pos = seq_start_pos,
            input_not_include_cache = input_not_include_cache
        )

        # attention layers

        if not self.recycling:
            assert not exists(recycle_steps) or recycle_steps == 1, 'you did not train with recycling'

            # regular

            attended, intermediates = self.attn_layers(x, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, deep_embeds_and_ids = deep_embed_and_ids, return_hiddens = True, **kwargs)

        else:
            # recycling

            recycle_steps = default(recycle_steps, (randrange(self.train_max_recycle_steps) + 1) if self.training else None)
            assert exists(recycle_steps) and recycle_steps > 0, '`recycle_steps` must be provided on forward if recycling is turned on and not training'

            for i in range(recycle_steps):
                first_step = i == 0
                last_step = i == (recycle_steps - 1)

                context = nullcontext if last_step else torch.no_grad

                with context():
                    maybe_recycled = self.recycled_proj(attended.detach()) if not first_step else 0.

                    attended, intermediates = self.attn_layers(x + maybe_recycled, mask = mask, mems = mems, mem_masks = mem_masks, cache = cache, return_hiddens = True, **kwargs)

        x = attended

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m = (mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b = b)

            x = x[:, :mem_seq]

        # store last layer hiddens, for access in case of cls token or attention pooling

        intermediates.last_layer_hiddens = x

        # store initial embed

        intermediates.initial_embed = init_embed

        # global average pool

        if self.average_pool_embed:
            x = masked_mean(x, mask = orig_mask, dim = 1)

        # cls token(s)

        if exists(self.cls_token):
            x, last_layer_hiddens = unpack(x, cls_packed_shape, 'b * d')

            intermediates.last_layer_hiddens = last_layer_hiddens

            if x.shape[1] == 1:
                x = rearrange(x, 'b 1 d -> b d')  # Remove sequence dimension if num_cls_tokens=1 to keep previous behavior

        # attention pool

        is_encoder = not self.attn_layers.causal
        return_pooled_tokens = exists(self.attn_pool) and is_encoder

        if (
            exists(self.attn_pool) and
            (return_intermediates or is_encoder) # in a new paper, they use attention pooling on decoder - so we'll default to returning pooled tokens if encoder, but for decoder, they must set `return_intermediates`
        ):

            attn_pooled_tokens = self.attn_pool(x, mask = mask)

            intermediates.attn_pooled_tokens = attn_pooled_tokens

        # handle expansion to mixture if needed (for mixture of softmax)

        combine_mixture = None

        if exists(self.to_mixture):
            combine_mixture = self.combine_mixture(x).softmax(dim = -1)
            x = self.to_mixture(x)

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x, **to_logits_kwargs) for fn in self.to_logits)
            else:
                logits = self.to_logits(x, **to_logits_kwargs)

        # maybe sig softmax

        if self.sigsoftmax_logits:
            logits = logits + logits.sigmoid().log()

        # handle maybe combine mixture

        if exists(combine_mixture):
            with autocast('cuda', enabled = False):
                prob = logits.softmax(dim = -1)
                mos = einsum('... k d, ... k -> ... d', prob, combine_mixture)
                logits = log(mos)

        # maybe squeeze out last dimension of logits

        if self.squeeze_out_last_dim:
            logits = tuple((rearrange(t, '... 1 -> ...') if t.shape[-1] == 1 else t) for t in cast_tuple(logits))

            if not self.has_multiple_heads:
                logits = first(logits)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings_and_intermediates:
            out = (x, intermediates)
        elif return_embeddings:
            out = x
        elif return_pooled_tokens:
            intermediates.logits = logits
            out = attn_pooled_tokens
        else:
            out = logits

        # maybe next embed pred

        if return_next_embed_pred:
            assert self.add_continuous_pred_head
            next_embed_out = self.to_next_embed_pred(x)

            out = (out, (next_embed_out, init_embed))

        # logit entropies

        if return_logit_entropies:
            intermediates.logit_entropies = calc_entropy(logits)
            return_intermediates = True

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = [t.pre_softmax_attn for t in  intermediates.attn_intermediates]
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight = attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = [cat(pair, dim = -2) for pair in zip(mems, hiddens)] if exists(mems) else hiddens
            new_mems = [t[..., -self.max_mem_len:, :].detach() for t in new_mems]

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = [t.post_softmax_attn for t in intermediates.attn_intermediates]
            return out, attn_maps

        return out

class XTransformer(Module):
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
            return_only_embed = True,
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

#=================================================================================================================================
# autoregressive_wrapper.py
#=================================================================================================================================

from math import ceil, log
from typing import Tuple, Callable

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, repeat, pack, unpack

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t, *args, **kwargs):
    return t

def join(arr, delimiter = ', '):
    return delimiter.join(arr)

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

# gumbel topk

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    return -log(-log(torch.rand_like(t)))

def gumbel_sample(logits, temperature = 1., eps = 1e-6):
    noise = gumbel_noise(logits)
    return ((logits / max(temperature, eps)) + noise).argmax(dim = -1)

# function for modifying all the cached key / values

def modify_cached_kv(cache, fn):
    for inter in cache.attn_intermediates:
        if inter.layer_type == 'a':
            inter.cached_kv = [fn(t) for t in inter.cached_kv]

# for variable lengthed prefixes

def pad_at_dim(t, pad: tuple[int, int], dim = -1, value = 0.):
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def align_right(t, lens, pad_id = 0):
    batch, seq_len, device, dtype = *t.shape[:2], t.device, t.dtype

    assert lens.ndim == 1 and lens.shape[0] == batch
    assert lens.amax() <= seq_len

    pad_lens = seq_len - lens
    max_pad_len = pad_lens.amax()

    batch_arange = torch.arange(batch, device = device, dtype = torch.long)[..., None]
    prompt_len_arange = torch.arange(seq_len, device = device, dtype = torch.long)

    t = pad_at_dim(t, (max_pad_len, 0), value = pad_id, dim = 1)
    offset = max_pad_len - pad_lens

    aligned = t[batch_arange, prompt_len_arange + offset[..., None], ...]
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
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = torch.pow(max_probs, min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# filter logits functions dict[str -> Callable]

FILTER_LOGITS_FN = dict(
    top_p = top_p,
    top_k = top_k,
    top_a = top_a,
    min_p = min_p
)

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
        add_attn_z_loss = False,
        next_embed_loss_weight = 0.1
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

        # whether to add a continuous loss
        self.add_continuous_pred_head = net.add_continuous_pred_head
        self.next_embed_loss_weight = next_embed_loss_weight

    @torch.no_grad()
    @eval_decorator
    def beam_search(
        self,
        prompts,
        seq_len,
        beams = 4,
        return_beams_and_scores = False,
        eos_token = None,
        temperature = 1.,
        stochastic = False,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable = identity,
        restrict_to_max_seq_len = True,
        filter_kwargs: dict = dict(),
        cache_kv = True,
        **kwargs
    ):
        assert not exists(eos_token), 'eos token not supported yet'

        max_seq_len, greedy, device = self.max_seq_len, temperature == 0., prompts.device

        prompts, packed_shape = pack([prompts], '* n')

        batch, orig_seq_len = prompts.shape

        # handle filter logits fn given as string

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = orig_seq_len - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

        # kv caches

        cache = None

        should_cache = cache_kv and self.net.can_cache_kv

        # scores for the beams

        scores = torch.zeros((batch,), device = device)

        batch_arange = torch.arange(batch, device = device)

        # sampling up to seq_len

        for i in range(seq_len):
            is_first = i == 0

            if restrict_to_max_seq_len:
                max_len_exceeded = out.shape[-1] > max_seq_len

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    modify_cached_kv(cache, lambda t: t[..., -(max_seq_len - 1):, :])

            logits, new_cache = self.net(
                x,
                return_intermediates = True,
                cache = cache,
                seq_start_pos = seq_start_pos,
                **kwargs
            )

            if should_cache:
                cache = new_cache

            logits = logits[:, -1]

            # to add to the scores

            log_probs = logits.log_softmax(dim = -1)

            # maybe filter by top_k, top_p (nucleus) for stochastic beam search

            if stochastic and not greedy:
                logits = filter_logits_fn(logits, **filter_kwargs)
                logits = (logits / temperature) + gumbel_noise(logits)

            # (gumbel) topk

            samples = logits.topk(beams, dim = -1).indices

            # get the scores for keeping track of beams

            next_scores = log_probs.gather(-1, samples)

            # expand beam times

            scores = repeat(scores, 'b -> b beams', beams = beams)
            scores = scores + next_scores

            out = repeat(out, 'b ... -> (b beams) ...', beams = beams)
            samples = rearrange(samples, 'b beams -> (b beams) 1')

            if should_cache and is_first:
                modify_cached_kv(cache, lambda t: repeat(t, 'b ... -> (b beams) ...', beams = beams))

            # concat sample

            out = torch.cat((out, samples), dim=-1)

            # sort by score and excise
            # excise out the beams

            scores = rearrange(scores, '(b prev_beams) next_beams -> b (prev_beams next_beams)', b = batch)
            curr_num_beams = scores.shape[-1]

            if curr_num_beams > beams:
                scores, sort_indices = scores.sort(dim = -1, descending = True)

                scores = scores[:, :beams]
                top_beams_indices = sort_indices[:, :beams]

                top_beams_indices = curr_num_beams * batch_arange[:, None] + top_beams_indices

                flattened_beam_indices = rearrange(top_beams_indices, 'b beams -> (b beams)')

                out = out[flattened_beam_indices]

            scores = rearrange(scores, 'b beams -> (b beams)')

            if not exists(eos_token):
                continue

            is_eos_tokens = (out == eos_token)

            if is_eos_tokens.any(dim = -1).all():
                break

        if exists(eos_token):
            # mask out everything after the eos tokens
            shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
            mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
            out = out.masked_fill(mask, self.pad_value)

        # select out the top beam

        out = rearrange(out, '(b beams) seq -> b beams seq', b = batch)

        out = out[..., orig_seq_len:]

        out, = unpack(out, packed_shape, '* beams n') # prompt may have no batch dimension

        if not return_beams_and_scores:
            return out[..., 0, :]

        scores = rearrange(scores, '(b beams) -> beams b', b = batch)
        out = rearrange(out, 'b beams n -> beams b n')

        return out, scores

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        prompts: list[Tensor] | Tensor,
        seq_len,
        eos_token = None,
        temperature = 1.,
        prompt_lens: Tensor | None = None,
        filter_logits_fn: str | Callable = top_k,
        restrict_to_max_seq_len = True,
        amateur_model: Module | Tuple[Module] | None = None,
        filter_kwargs: dict = dict(),
        contrastive_decode_kwargs: dict | Tuple[dict] = dict(
            beta = 0.5,
            alpha = 0.1
        ),
        cache_kv = True,
        return_prime=False,
        verbose=True,
        **kwargs
    ):

        if verbose:
            print("Generating sequence of max length:", seq_len)

        max_seq_len, greedy = self.max_seq_len, temperature == 0.

        # handle prompts given as list of variable lengthed token ids

        if isinstance(prompts, list):
            assert len(prompts) > 0, 'prompts cannot be empty list'
            assert not exists(prompt_lens), '`prompt_len` will be auto derived if prompts are passed in as list of Tensors'

            prompt_lens = tensor([t.shape[0] for t in prompts], device = prompts[0].device)

            prompts = pad_sequence(prompts, batch_first = True)

        # pack maybe no batch

        prompts, ps = pack([prompts], '* n')

        b, t, device = *prompts.shape, prompts.device

        # handle filter logits fn given as string

        if isinstance(filter_logits_fn, str):
            assert filter_logits_fn in FILTER_LOGITS_FN, f"only {join(FILTER_LOGITS_FN.keys())} are available"

            filter_logits_fn = FILTER_LOGITS_FN[filter_logits_fn]

        # handle variable lengthed prompts (prefixes)

        seq_start_pos = None
        if exists(prompt_lens):
            prompts = align_right(prompts, prompt_lens, pad_id = self.pad_value)
            seq_start_pos = t - prompt_lens

        # output from which sampled tokens appended to

        out = prompts

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

                assert not (cache_kv and max_len_exceeded and not self.net.can_cache_kv_outside_max_seq_len), 'the network cannot use cached key values when decoding outside the max sequence length. most likely because you are using absolute positional embedding. you can switch to rotary embeddings to resolve this issue'

                x = out[:, -max_seq_len:]

                if exists(cache):
                    for inter in cache.attn_intermediates:
                        if inter.layer_type == 'a':
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
            out = out[:, :]

        else:
            out = out[:, t:]

        out, = unpack(out, ps, '* n')

        return out

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

    def forward(
        self,
        x,
        return_outputs = False,
        prepend_embeds = None,
        **kwargs
    ):
        seq, ignore_index, add_attn_z_loss, add_next_embed_loss = x.shape[1], self.ignore_index, self.add_attn_z_loss, self.add_continuous_pred_head

        inp, target = x, x[:, 1:]
        inp = torch.where(inp == ignore_index, self.pad_value, inp)

        if self.mask_prob > 0.:
            rand = torch.randn(inp.shape, device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            num_mask = min(int(seq * self.mask_prob), seq - 1)
            indices = rand.topk(num_mask, dim = -1).indices
            mask = ~torch.zeros_like(inp).scatter(1, indices, 1.).bool()
            kwargs.update(self_attn_kv_mask = mask)

        out, cache = self.net(
            inp,
            return_intermediates = True,
            return_attn_z_loss = add_attn_z_loss,
            return_next_embed_pred = add_next_embed_loss,
            prepend_embeds = prepend_embeds,
            **kwargs
        )

        # destruct differently if doing continuous pred

        if add_next_embed_loss:
            logits, (next_embed_pred, init_embeds) = out
        else:
            logits = out

        # if there are prepended embeds, excise it out

        if exists(prepend_embeds):
            prepend_len = prepend_embeds.shape[1]
            logits = logits[:, prepend_len:]

        # take all tokens but the last

        logits = logits[:, :-1]

        # Compute accuracy
        
        acc = self.compute_accuracy(logits, target)      

        # loss function

        loss_fn = F.cross_entropy if not self.net.output_is_log_prob else F.nll_loss

        # cross entropy loss

        loss = loss_fn(
            rearrange(logits, 'b n c -> b c n'),
            target,
            ignore_index = ignore_index
        )

        if add_attn_z_loss:
            loss = loss + cache.attn_z_loss

        if add_next_embed_loss:
            mask = target != ignore_index
            embed_pred = next_embed_pred[:, :-1]
            cont_targets = init_embeds[:, 1:].detach()

            cont_loss = F.l1_loss(embed_pred, cont_targets, reduction = 'none')
            cont_loss = cont_loss[mask].mean()

            loss = loss + cont_loss * self.next_embed_loss_weight

        # Return
        
        if not return_outputs:
            return loss, acc

        return loss, acc, logits, cache

#=================================================================================================================================
# gpt_vae.py
#=================================================================================================================================

# applying the cvae + detr design from ACT (Zhou et al.) to GPT
# for steering, diversity rlvr, map-elites in epo, and other possibilities

import torch
from torch import nn, Tensor, is_tensor, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class GPTVAE(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        enc_depth,
        max_seq_len,
        dim_latent = None,
        attn_dim_head = 64,
        heads = 8,
        enc_kwargs: dict = dict(),
        dec_kwargs: dict = dict(),
        vae_kl_loss_weight = 1.,
        vae_kl_div_floor = 0.,      # what was done in free transformer, which in turn came from Kingma 2016
        latents_dropout_prob = 0.5, # what percentage of the time to dropout the latents completely
        pad_id = -1,
        encoder: Module | None = None,
        **kwargs
    ):
        super().__init__()
        dim_latent = default(dim_latent, dim)

        if not exists(encoder):
            encoder = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len + 1,
                return_only_embed = True,
                average_pool_embed = True,
                attn_layers = Encoder(
                    dim = dim,
                    depth = enc_depth,
                    attn_dim_head = attn_dim_head,
                    heads = heads,
                    **kwargs,
                    **enc_kwargs
                ),
            )

        self.encoder = encoder

        self.to_latent_mean_log_variance = nn.Sequential(
            nn.Linear(dim, dim_latent * 2),
            Rearrange('b (two d) -> two b d', two = 2)
        )

        self.from_latent_to_prepend_token = nn.Sequential(
            nn.Linear(dim_latent, dim),
            Rearrange('b d -> b 1 d')
        )

        self.decoder = TransformerWrapper(
            num_tokens = num_tokens,
            max_seq_len = max_seq_len,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                attn_dim_head = attn_dim_head,
                heads = heads,
                **kwargs,
                **dec_kwargs
            ),
        )

        self.ar_wrapped_decoder = AutoregressiveWrapper(self.decoder, ignore_index = pad_id)

        self.pad_id = pad_id

        # loss weights - vae kl loss

        self.vae_kl_div_floor = vae_kl_div_floor
        self.vae_kl_loss_weight = vae_kl_loss_weight

        self.latents_dropout = nn.Dropout(latents_dropout_prob)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode_to_latents(
        self,
        seq,
        return_mean_log_var = False
    ):
        mask = seq != self.pad_id
        pooled = self.encoder(seq, mask = mask)

        latents_mean, latents_log_var = self.to_latent_mean_log_variance(pooled)
        latents_std = (0.5 * latents_log_var).exp()

        # reparam trick

        latents = latents_mean + latents_std * torch.randn_like(latents_mean)

        if not return_mean_log_var:
            return latents

        return latents, (latents_mean, latents_log_var)

    @torch.no_grad()
    def generate(
        self,
        prompts,
        seq_len,
        latents = None,
        seq_for_latents = None,
        **generate_kwargs
    ):
        assert prompts.ndim in {1, 2}
        batch = prompts.shape[0] if prompts.ndim == 2 else 1

        # if seq_for_latents passed in, derive latents from it

        if exists(seq_for_latents):
            assert not exists(latents), 'latents should not be passed in if given the seq from which to derive them'

            latents = self.encode_to_latents(seq_for_latents)

        # prepend embeds

        prepend_embeds = None
        if exists(latents):
            if not is_tensor(latents):
                latents = tensor(latents, device = self.device)

            if latents.ndim == 1: # repeat latents
                latents = repeat(latents, 'd -> b d', b = batch)

            prepend_embeds = self.from_latent_to_prepend_token(latents)

        # generated

        generated = self.ar_wrapped_decoder.generate(
            prompts,
            seq_len,
            prepend_embeds = prepend_embeds,
            **generate_kwargs
        )

        return generated

    def forward(
        self,
        seq,
        seq_for_latents = None,
        return_all_losses = False
    ):
        batch, device = seq.shape[0], seq.device

        seq_for_latents = default(seq_for_latents, seq)

        latents, (latents_mean, latents_log_var) = self.encode_to_latents(seq_for_latents, return_mean_log_var = True)

        dropped_latents = ~self.latents_dropout(torch.ones((batch,), device = device)).bool()

        prepend_embeds = self.from_latent_to_prepend_token(latents)

        ar_loss = self.ar_wrapped_decoder(
            seq,
            prepend_embeds = prepend_embeds,
            seq_start_pos = dropped_latents.long() # sequence starts at 1 and does not attend to the first style latent
        )

        # vae kl loss

        vae_kl_loss = 0.5 * (
            latents_log_var.exp()
            + latents_mean.square()
            - latents_log_var
            - 1.
        )

        vae_kl_loss = F.relu(vae_kl_loss - self.vae_kl_div_floor)

        vae_kl_loss = vae_kl_loss.sum(dim = -1).mean()

        # return losses

        total_loss = (
            ar_loss +
            vae_kl_loss * self.vae_kl_loss_weight
        )

        if not return_all_losses:
            return total_loss

        losses = (ar_loss, vae_kl_loss)

        return total_loss, losses

#=================================================================================================================================
# Binary classifier fuctions
# https://github.com/lucidrains/x-transformers/pull/264
#=================================================================================================================================

class ClsInferenceDataset(Dataset):
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

def build_cls_model(num_tokens=18819,
                    max_seq_len=1024,
                    logits_dim=1,
                    use_cls_token=True,
                    squeeze_out_last_dim=True,
                    dim=1024,
                    depth=8,
                    heads=8,
                    device='cuda'
                   ):

    """
    Constructs the Transformer model that outputs a single logit per input.
    """

    model = TransformerWrapper(
        num_tokens=num_tokens,
        max_seq_len=max_seq_len,
        logits_dim=logits_dim,
        use_cls_token=use_cls_token,
        squeeze_out_last_dim = squeeze_out_last_dim,
        attn_layers=Encoder(dim=dim,
                            depth=depth,
                            heads=heads
                           )
    )

    return model.to(device)

def load_cls_model(checkpoint_path, device='cuda'):
    
    """
    Rebuilds the architecture, loads weights.
    """
    
    model = build_cls_model(device=device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    
    return model

def cls_predict(model,
                seqs,
                batch_size=8,
                threshold=0.5,
                seq_len=1024,
                pad_token=18818,
                device='cuda'
               ):
    
    """
    Returns two lists:
      - probs: float probabilities  
      - preds: int 0/1 predictions  
    """
    
    def collate_fn(batch):
        # batch: list of sequences (list/1D-tensor)
        tensors = [s[:seq_len].detach().clone() for s in batch]
        max_len = min(seq_len, max(t.size(0) for t in tensors))
        padded = torch.full((len(tensors), max_len), pad_token, dtype=torch.long)
        for i, t in enumerate(tensors):
            L = t.size(0)
            padded[i, :L] = t
        return padded

    ds = ClsInferenceDataset(seqs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    all_probs = []
    all_preds = []

    model.to(device)
    model.eval()
    
    with torch.inference_mode():
        for x in loader:
            
            x = x.to(device)                       # [B, L] (truncated & padded)
            
            logits = model(x).squeeze()            # [B]
            
            probs = torch.sigmoid(logits)         # [B]
            
            preds = (probs >= threshold).long()

            probs = probs.cpu().tolist()
            preds = preds.cpu().tolist()

            if type(preds) == list:
                all_probs.extend(probs)
                all_preds.extend(preds)

            else:
                all_probs.append(probs)
                all_preds.append(preds)                

    return all_preds, all_probs

#=================================================================================================================================
# Sequences probabilities and scores functions
#=================================================================================================================================

import inspect
import math
from typing import Callable, Optional, Dict, Any, List, Tuple
import torch
import torch.nn.functional as F

def print_probs_scoring_guide():
    print(inspect.getdoc(probs_scoring_guide))

def probs_scoring_guide():

    """
    Return dictionary structure and metric descriptions for generate_with_probs / score_sequences.
    
    Returns
    -------
    result : dict
        A dictionary containing token-level and sequence-level scoring information.
    
        Keys
        ----
        tokens : torch.Tensor
            Tensor of token ids for each batch entry. Shape (batch, seq_len).
            - Meaning: Generated tokens (for generate_with_probs) or the original
              input sequences (for score_sequences).
            - Interpretation: Map ids to text with your tokenizer to inspect outputs.
    
        token_probs : List[List[float]]
            Per-batch lists of probabilities assigned to each chosen token at the time
            it was produced. Values in [0, 1].
            - Meaning: Softmax probability for the selected token at each step.
            - Interpretation: Higher → model more confident about that token. Do not
              multiply many token_probs directly (underflow risk); use log-probs.
    
        token_logprobs : List[List[float]]
            Per-batch lists of natural log probabilities (nats) for each chosen token:
            log p(x_t | x_<t).
            - Meaning: Numerically stable per-token log-probabilities.
            - Interpretation: Less negative = more likely. Sum these to get sequence_logprobs.
    
        token_scores : List[List[float]]
            Per-batch lists of token negative log-probabilities (NLL) computed as -log p.
            - Meaning: Token-level loss (positive).
            - Interpretation: Lower = model found token less surprising. Useful to spot spikes.
    
        sequence_logprobs : List[float]
            Sum of token log-probabilities for each sequence (nats): sum_t log p(x_t | x_<t).
            - Meaning: Canonical sequence score; additive and numerically stable.
            - Interpretation: Use this to compare sequences. Higher (less negative) is better.
    
        nll : List[float]
            Negative sequence log-probabilities (nats): -sequence_logprobs.
            - Meaning: Sequence-level negative log-likelihood (loss).
            - Interpretation: Lower NLL indicates a sequence the model finds more probable.
    
        sequence_probs : List[float]
            Numeric probabilities computed as exp(sequence_logprobs) (float64 when possible).
            - Meaning: Absolute probability of the full sequence.
            - Interpretation: Often underflows to 0.0 for realistic lengths; prefer sequence_logprobs.
    
        sequence_prob_display : List[str]
            Human-readable string for sequence probability. If numeric underflow occurs,
            this shows an approximate scientific form (e.g., "~10^-550.65").
            - Meaning: Readable magnitude of the sequence probability.
            - Interpretation: Use this for reporting instead of raw sequence_probs when it is 0.0.
    
        mask : torch.Tensor
            Boolean tensor indicating which positions were included in scoring.
            Shape (batch, scored_len). False for padded positions or tokens after the first EOS.
            - Meaning: Aligns token-level lists with original sequence positions.
            - Interpretation: Use to ignore padded or post-EOS tokens in aggregates.
    
        metadata : dict
            Miscellaneous run information such as:
            - prompt_len : int or list[int] — length of prompt tokens (if applicable)
            - generated_len : int — number of generated tokens (generate_with_probs)
            - temperature : float — sampling temperature used
            - seq_len : int — original sequence length (score_sequences)
            - Interpretation: Useful for reproducing runs and normalizing comparisons.
    
        metrics : dict
            Per-sequence derived diagnostics (under result["metrics"]["per_sequence"]).
            Each entry contains:
            - sequence_index : int
            - token_count : int
            - sequence_logprob_nats : float
                Sum of log-probs (nats). Primary canonical score.
            - sequence_log10 : float
                Log10 of the sequence probability (for display).
            - sequence_prob_display : str
                Human-friendly scientific display of the sequence probability.
            - avg_logprob_per_token_nats : float
                Average log-prob per token (nats): (1/T) * sum_t log p.
                - Interpretation: Normalizes for length; higher (less negative) is better.
            - avg_logprob_per_token_bits : float
                Average log-prob per token in bits (divide nats by ln(2)).
                - Interpretation: Intuitive unit; lower bits = easier prediction.
            - geometric_mean_token_prob : str
                Geometric mean of token probabilities (display).
                - Interpretation: Typical per-token probability; quick sense of per-token confidence.
            - perplexity : float
                exp(-avg_logprob_per_token). Standard LM metric; lower is better.
    
    Notes
    -----
    - Use `sequence_logprobs` (or `nll`) as the authoritative score for comparisons and ranking.
    - Avoid relying on `sequence_probs` for comparisons because of floating-point underflow.
    - Prefer `avg_logprob_per_token_nats` or `perplexity` when comparing sequences of different lengths.
    - Token-level spikes in `token_scores` (large -log p) indicate surprising tokens and are useful
      for debugging prompts or model behavior.
    
    Examples
    --------
    # Example usage after calling generate_with_probs or score_sequences:
    res = generate_with_probs(...)
    print("Sequence logprob (nats):", res["sequence_logprobs"][0])
    print("Sequence prob (display):", res["sequence_prob_display"][0])
    print("Avg logprob/token (nats):", res["metrics"]["per_sequence"][0]["avg_logprob_per_token_nats"])
    print("Perplexity:", res["metrics"]["per_sequence"][0]["perplexity"])
    """

    return inspect.getdoc(probs_scoring_guide)

# --- helpers ---
def _safe_exp64(logp: float) -> Tuple[float, str]:
    lp64 = torch.tensor(logp, dtype=torch.float64)
    try:
        p64 = float(torch.exp(lp64).item())
    except Exception:
        p64 = 0.0
    if p64 == 0.0:
        log10_prob = float(lp64.item() / math.log(10.0))
        display = f"~10^{log10_prob:.2f}"
    else:
        display = f"{p64:.6e}"
    return p64, display

def _attach_metrics_to_result(result: Dict[str, Any]) -> Dict[str, Any]:
    seq_logprobs: List[float] = result.get("sequence_logprobs", [])
    token_logprobs: List[List[float]] = result.get("token_logprobs", [])
    token_probs: List[List[float]] = result.get("token_probs", [])
    metrics = {"per_sequence": []}
    for i, seq_lp in enumerate(seq_logprobs):
        toks_lp = token_logprobs[i] if i < len(token_logprobs) else []
        token_count = len(toks_lp)
        avg_lp = float(sum(toks_lp) / token_count) if token_count > 0 else 0.0
        avg_lp_bits = avg_lp / math.log(2.0)
        try:
            perplexity = math.exp(-avg_lp)
        except OverflowError:
            perplexity = float("inf")
        log10_prob = seq_lp / math.log(10.0)
        seq_prob_display = result.get("sequence_prob_display", [None]*len(seq_logprobs))[i]
        if seq_prob_display is None:
            seq_prob_display = f"~10^{log10_prob:.2f}"
        if token_count > 0:
            try:
                geom_mean = math.exp(avg_lp)
                geom_mean_display = f"{geom_mean:.6e}"
            except OverflowError:
                geom_mean_display = f"exp({avg_lp:.3f})"
        else:
            geom_mean_display = "n/a"
        metrics["per_sequence"].append({
            "sequence_index": i,
            "token_count": token_count,
            "sequence_logprob_nats": float(seq_lp),
            "sequence_log10": float(log10_prob),
            "sequence_prob_display": seq_prob_display,
            "avg_logprob_per_token_nats": float(avg_lp),
            "avg_logprob_per_token_bits": float(avg_lp_bits),
            "geometric_mean_token_prob": geom_mean_display,
            "perplexity": float(perplexity)
        })
    result["metrics"] = metrics
    return result

def _decode_token(tokenizer, tok_id: int) -> str:
    if tokenizer is None:
        return str(tok_id)
    try:
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode([tok_id])
        if hasattr(tokenizer, "convert_ids_to_tokens"):
            return tokenizer.convert_ids_to_tokens([tok_id])[0]
    except Exception:
        pass
    return str(tok_id)

# ---------------------------
# generate_with_probs (with diff)
# ---------------------------
@torch.inference_mode()
def generate_with_probs(
    model,
    prompts: torch.Tensor,
    seq_len: int,
    eos_token: Optional[int] = None,
    temperature: float = 1.0,
    prompt_lens: Optional[torch.Tensor] = None,
    filter_logits_fn: Optional[Callable] = None,
    filter_kwargs: Optional[Dict[str, Any]] = None,
    pad_value: Optional[int] = None,
    tokenizer = None,
    print_table: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    include_top1: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate sequences from an autoregressive model while collecting per-token probabilities,
    log-probabilities, scores and an optional "diff" view comparing sampled tokens to the
    model's top-1 (greedy) tokens.

    This function runs the model in inference mode and appends sampled tokens to the provided
    prompts until `seq_len` tokens have been generated (or until an `eos_token` ends all
    sequences). It supports temperature sampling, optional logits filtering, and returns
    detailed diagnostics useful for evaluation, debugging and analysis (per-token probs,
    cumulative sequence log-probabilities, NLL, perplexity, and a diff of sampled vs top-1).

    Key behaviors
    - Operates under `torch.inference_mode()` (no gradients).
    - If `prompt_lens` is provided, prompts are right-aligned into a padded buffer of the
      same shape as `prompts` before generation (useful when prompts are suffixes).
    - If `filter_logits_fn` is provided it is applied to raw logits before softmax.
    - If `temperature == 0.0` the function performs greedy decoding (argmax).
    - If `include_top1` is True, the function computes the top-1 token and its probability
      at each step (after optional filtering) and records whether the sampled token matched it.
    - If `eos_token` is provided, generation stops early when every batch item has produced
      an EOS; generated outputs after the first EOS are optionally padded with `pad_value`.
    - Returned numeric log-probabilities are in natural log (nats) and converted to float64
      for sequence-level aggregation to reduce numerical error.

    Parameters
    - model: A model object exposing a `net` callable with signature
      `logits = model.net(tokens, return_intermediates=True, cache=None, seq_start_pos=None, **kwargs)`.
      `logits` must be a tensor of shape (batch, seq, vocab) or a tuple/list whose first
      element is that tensor.
    - prompts (torch.Tensor): Integer token tensor of shape (batch, prompt_len) containing
      prompt tokens. Prompts are copied and extended in-place to produce generated sequences.
    - seq_len (int): Maximum number of tokens to generate per example (not counting prompt).
    - eos_token (Optional[int]): Token id that marks end-of-sequence. If provided, generation
      may stop early and outputs after the first EOS are optionally replaced with `pad_value`.
    - temperature (float): Sampling temperature. `0.0` forces greedy decoding.
    - prompt_lens (Optional[torch.Tensor]): Optional per-batch prompt lengths (int or tensor)
      used to right-align prompts into the generation buffer when prompts are suffixes.
    - filter_logits_fn (Optional[Callable]): Function applied to raw logits before softmax.
      Signature should accept `(logits, **filter_kwargs)` and return logits of same shape.
    - filter_kwargs (Optional[Dict[str, Any]]): Keyword arguments forwarded to `filter_logits_fn`.
    - pad_value (Optional[int]): Token id used to pad generated outputs after EOS (if any).
    - tokenizer: Optional tokenizer used to decode token ids for human-readable diffs and
      printed tables. If absent, token ids are stringified.
    - print_table (bool): If True, prints a human-readable table summarizing per-token stats.
    - device (Optional[torch.device]): Device to run generation on. Defaults to `prompts.device`.
    - verbose (bool): If True, prints progress messages during generation.
    - include_top1 (bool): If True, compute and return top-1 tokens, their probs/logprobs,
      and a `diff` structure listing positions where sampled != top-1.
    - **kwargs: Additional keyword arguments forwarded to `model.net`.

    Returns
    A dictionary with the following keys (types shown informally):
    - "tokens" (torch.Tensor): Generated tokens (batch, generated_len) as CPU tensor.
    - "token_probs" (List[List[float]]): Per-batch list of per-token sampling probabilities.
    - "token_logprobs" (List[List[float]]): Per-batch list of per-token log-probabilities (nats).
    - "token_scores" (List[List[float]]): Per-token scores (negative log-probabilities).
    - "sequence_logprobs" (List[float]): Sum of token log-probabilities per generated sequence.
    - "sequence_probs" (List[float]): Sequence probabilities (exp of sequence_logprobs) where
      numerically possible; extremely small values may be represented as 0.0.
    - "sequence_prob_display" (List[str]): Human-friendly display of sequence probability
      (either decimal or approximate 10^x form for tiny values).
    - "nll" (List[float]): Negative log-likelihood per sequence (i.e., -sequence_logprob).
    - "metadata" (dict): Contains "prompt_len", "generated_len", and "temperature".
    - "diff" (List[List[Dict]]): Per-batch list of dictionaries for positions where the sampled
      token differed from the top-1 token. Each dict contains:
        - "pos": position index within the generated span (0-based)
        - "token": sampled token id
        - "token_str": decoded sampled token (or id string)
        - "token_prob": sampled token probability
        - "top1_token": top-1 token id
        - "top1_token_str": decoded top-1 token
        - "top1_prob": top-1 probability
        - "match": boolean (always False for entries in diff)
    - If `include_top1` is True, additional keys are included:
      - "top1_tokens", "top1_token_probs", "top1_token_logprobs", "top1_matches"

    After the primary result is assembled the function attaches a "metrics" entry with:
    - "per_sequence": list of per-sequence metric dicts containing:
      - "sequence_index", "token_count", "sequence_logprob_nats", "sequence_log10",
        "sequence_prob_display", "avg_logprob_per_token_nats", "avg_logprob_per_token_bits",
        "geometric_mean_token_prob", "perplexity"

    Notes and caveats
    - Numerical stability: very small probabilities are clamped before log to avoid -inf;
      sequence probabilities that underflow are represented with an approximate 10^x string.
    - The function assumes the model's logits correspond to the next-token distribution for
      the last position of the provided input; it uses `logits[:, -1, :]` for sampling.
    - The function may raise exceptions if `model.net` returns tensors of unexpected shape
      or if device/dtype mismatches occur.
    - This function is intended for analysis and debugging; it is not optimized for maximal
      throughput in production sampling loops.

    Example (conceptual)
    >>> res = generate_with_probs(model, prompts, seq_len=20, temperature=0.8, tokenizer=tok)
    >>> print(res["metrics"]["per_sequence"][0]["perplexity"])
    """
    if filter_kwargs is None:
        filter_kwargs = {}
    if device is None:
        device = prompts.device
    if pad_value is None:
        pad_value = getattr(model, "pad_value", None)

    model.eval()
    with torch.inference_mode():
        prompts_in = prompts.to(device)
        b, t = prompts_in.shape

        if prompt_lens is not None:
            aligned = torch.full_like(prompts_in, pad_value)
            for i in range(b):
                L = int(prompt_lens[i].item()) if isinstance(prompt_lens[i], torch.Tensor) else int(prompt_lens[i])
                if L > 0:
                    aligned[i, -L:] = prompts_in[i, -L:]
            prompts_in = aligned

        out = prompts_in.clone()

        token_probs: List[List[float]] = [[] for _ in range(b)]
        token_logprobs: List[List[float]] = [[] for _ in range(b)]
        token_scores: List[List[float]] = [[] for _ in range(b)]
        seq_logprob_tensors = [torch.tensor(0.0, dtype=torch.float64) for _ in range(b)]

        top1_tokens: List[List[int]] = [[] for _ in range(b)]
        top1_token_probs: List[List[float]] = [[] for _ in range(b)]
        top1_token_logprobs: List[List[float]] = [[] for _ in range(b)]
        top1_matches: List[List[bool]] = [[] for _ in range(b)]

        greedy = (temperature == 0.0)

        if verbose:
            print("Generating sequence of max length:", seq_len)

        for sl in range(seq_len):
            max_seq_len = getattr(model, "max_seq_len", None)
            x = out if max_seq_len is None else out[:, -max_seq_len:]

            logits_out = model.net(x, return_intermediates=True, cache=None, seq_start_pos=None, **kwargs)
            logits = logits_out[0] if isinstance(logits_out, (tuple, list)) else logits_out
            logits = logits[:, -1, :]

            # top1 (greedy) from raw logits
            if include_top1:
                top1_ids = logits.argmax(dim=-1, keepdim=True)  # (batch,1)
                filtered_for_top1 = logits if filter_logits_fn is None else filter_logits_fn(logits, **filter_kwargs)
                probs_for_top1 = F.softmax(filtered_for_top1 / (temperature if temperature > 0 else 1.0), dim=-1)
                top1_p = probs_for_top1.gather(1, top1_ids).squeeze(1)
                top1_lp = torch.log(top1_p.clamp_min(1e-45)).to(dtype=torch.float64)

            if greedy:
                filtered_logits = logits if filter_logits_fn is None else filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / (temperature if temperature > 0 else 1.0), dim=-1)
                sample = logits.argmax(dim=-1, keepdim=True)
            else:
                filtered_logits = logits if filter_logits_fn is None else filter_logits_fn(logits, **filter_kwargs)
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)

            picked_probs = probs.gather(1, sample).squeeze(1)
            picked_logprobs = torch.log(picked_probs.clamp_min(1e-45)).to(dtype=torch.float64)

            out = torch.cat((out, sample), dim=-1)

            for i in range(b):
                p = float(picked_probs[i].cpu().item())
                lp = float(picked_logprobs[i].cpu().item())
                token_probs[i].append(p)
                token_logprobs[i].append(lp)
                token_scores[i].append(-lp)
                seq_logprob_tensors[i] = seq_logprob_tensors[i] + torch.tensor(lp, dtype=torch.float64)

                if include_top1:
                    tid = int(top1_ids[i].item())
                    tp = float(top1_p[i].cpu().item())
                    tlp = float(top1_lp[i].cpu().item())
                    top1_tokens[i].append(tid)
                    top1_token_probs[i].append(tp)
                    top1_token_logprobs[i].append(tlp)
                    top1_matches[i].append(int(sample[i].item()) == tid)

            if verbose and (sl % 32 == 0):
                print(f"{sl} / {seq_len}")

            if eos_token is not None:
                last_tokens = out[:, -1]
                if (last_tokens == eos_token).any(dim=-1).all():
                    if verbose:
                        print('Model called the end of sequence at:', sl, '/', seq_len)
                    break

        gen = out[:, t:].cpu()

        if eos_token is not None:
            for i in range(b):
                seq_full = out[i].cpu()
                eos_positions = (seq_full == eos_token).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first_eos_idx = int(eos_positions[0].item())
                    gen_len_before_eos = max(0, first_eos_idx - t)
                    token_probs[i] = token_probs[i][:gen_len_before_eos]
                    token_logprobs[i] = token_logprobs[i][:gen_len_before_eos]
                    token_scores[i] = token_scores[i][:gen_len_before_eos]
                    seq_logprob_tensors[i] = torch.tensor(sum(token_logprobs[i]), dtype=torch.float64)
                    if include_top1:
                        top1_tokens[i] = top1_tokens[i][:gen_len_before_eos]
                        top1_token_probs[i] = top1_token_probs[i][:gen_len_before_eos]
                        top1_token_logprobs[i] = top1_token_logprobs[i][:gen_len_before_eos]
                        top1_matches[i] = top1_matches[i][:gen_len_before_eos]
                    if pad_value is not None:
                        start_mask = max(0, first_eos_idx - t)
                        if start_mask < gen.shape[1]:
                            gen[i, start_mask:] = pad_value

        sequence_logprobs: List[float] = [float(x.item()) for x in seq_logprob_tensors]
        sequence_probs: List[float] = []
        sequence_prob_display: List[str] = []
        nll: List[float] = []

        for lp in sequence_logprobs:
            pnum, disp = _safe_exp64(lp)
            sequence_probs.append(pnum)
            sequence_prob_display.append(disp)
            nll.append(-lp)

        result = {
            "tokens": gen,
            "token_probs": token_probs,
            "token_logprobs": token_logprobs,
            "token_scores": token_scores,
            "sequence_logprobs": sequence_logprobs,
            "sequence_probs": sequence_probs,
            "sequence_prob_display": sequence_prob_display,
            "nll": nll,
            "metadata": {
                "prompt_len": t,
                "generated_len": gen.shape[1],
                "temperature": temperature
            }
        }

        if include_top1:
            result.update({
                "top1_tokens": top1_tokens,
                "top1_token_probs": top1_token_probs,
                "top1_token_logprobs": top1_token_logprobs,
                "top1_matches": top1_matches
            })

        # build diff view: sampled != top1
        diff_all: List[List[Dict[str, Any]]] = [[] for _ in range(b)]
        if include_top1:
            for i in range(b):
                for pos, (sample_tok, sample_p, t1_tok, t1_p, match) in enumerate(zip(
                    [int(x) for x in gen[i].tolist()],
                    token_probs[i],
                    top1_tokens[i],
                    top1_token_probs[i],
                    top1_matches[i]
                )):
                    if not match:
                        diff_all[i].append({
                            "pos": pos,
                            "token": sample_tok,
                            "token_str": _decode_token(tokenizer, sample_tok),
                            "token_prob": sample_p,
                            "top1_token": int(t1_tok),
                            "top1_token_str": _decode_token(tokenizer, int(t1_tok)),
                            "top1_prob": t1_p,
                            "match": bool(match)
                        })
        result["diff"] = diff_all

        result = _attach_metrics_to_result(result)

        if print_table:
            for i in range(b):
                print("="*110)
                print(f"Batch {i}  (prompt_len={t})")
                print("-"*110)
                print(" idx | token        |   prob    |   logprob   |   cum_logp   | token_nll | top1_token (p) | match")
                print("-"*110)
                cum_logp = 0.0
                for idx, (p, lp, sc) in enumerate(zip(token_probs[i], token_logprobs[i], token_scores[i])):
                    cum_logp += lp
                    tok_id = int(gen[i, idx].item()) if idx < gen.shape[1] else -1
                    tok_display = _decode_token(tokenizer, tok_id)
                    if include_top1:
                        t1_id = top1_tokens[i][idx]
                        t1_p = top1_token_probs[i][idx]
                        match_mark = "*" if top1_matches[i][idx] else " "
                        print(f"{idx:3d} | {tok_display:>12s} | {p:9.6f} | {lp:11.6f} | {cum_logp:12.6f} | {sc:10.6f} | {_decode_token(tokenizer, t1_id):>12s} ({t1_p:5.3f}){match_mark}")
                    else:
                        print(f"{idx:3d} | {tok_display:>12s} | {p:9.6f} | {lp:11.6f} | {cum_logp:12.6f} | {sc:10.6f}")
                print("-"*110)
                print(f"Sequence logprob (nats): {result['sequence_logprobs'][i]:.6f} | Sequence prob: {result['sequence_prob_display'][i]} | NLL: {result['nll'][i]:.6f}")
                m = result["metrics"]["per_sequence"][i]
                print(f"Avg logprob/token: {m['avg_logprob_per_token_nats']:.6f} nats ({m['avg_logprob_per_token_bits']:.4f} bits) | Perplexity: {m['perplexity']:.6f}")
                if result["diff"][i]:
                    print("DIFF (sampled != top1) positions:")
                    for d in result["diff"][i]:
                        print(f" pos={d['pos']} token={d['token_str']}({d['token']}) p={d['token_prob']:.6f} | top1={d['top1_token_str']}({d['top1_token']}) p={d['top1_prob']:.6f}")
                else:
                    print("No diffs: sampled tokens matched top1 at every step.")
                print("="*110)

        return result

# ---------------------------
# score_sequences (with diff)
# ---------------------------
@torch.inference_mode()
def score_sequences(
    model,
    sequences: torch.Tensor,
    prompt_lens: Optional[torch.Tensor] = None,
    eos_token: Optional[int] = None,
    pad_value: Optional[int] = None,
    filter_logits_fn: Optional[Callable] = None,
    filter_kwargs: Optional[Dict[str, Any]] = None,
    tokenizer = None,
    print_table: bool = False,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    include_top1: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute per-token and per-sequence likelihood statistics for given full sequences
    under an autoregressive model, optionally comparing each target token to the model's
    top-1 prediction and producing a diff of mismatches.

    This function scores provided sequences by computing the model's next-token distribution
    for each position and extracting the probability and log-probability assigned to the
    actual target token (i.e., the token that follows each input prefix). It supports
    masking of padding tokens, optional EOS-based truncation, and an optional logits filter.
    The function returns detailed per-token lists, aggregated sequence log-probabilities,
    NLLs, human-friendly probability displays, and diagnostic "diff" entries where the
    target token differs from the model's greedy top-1.

    Key behaviors
    - Operates under `torch.inference_mode()` (no gradients).
    - Expects `sequences` shaped (batch, seq_len). The function scores tokens at positions
      1..(seq_len-1) where each target is `sequences[:, pos]` and the corresponding input
      is `sequences[:, :pos]`.
    - If `filter_logits_fn` is provided it is applied to the model logits before softmax.
    - If `pad_value` is provided, positions where the target equals `pad_value` are masked
      out and not counted in sequence sums or per-token lists.
    - If `eos_token` is provided, tokens after the first EOS in each sequence are masked out.
    - If `include_top1` is True, the function computes top-1 ids and probabilities and
      records whether the target matched the top-1 at each scored position.

    Parameters
    - model: A model object exposing a `net` callable with signature
      `logits = model.net(tokens, return_intermediates=True, cache=None, seq_start_pos=None, **kwargs)`.
      `logits` must be a tensor of shape (batch, seq, vocab) or a tuple/list whose first
      element is that tensor.
    - sequences (torch.Tensor): Integer token tensor of shape (batch, seq_len) containing
      full sequences to be scored. The first token of each sequence is treated as context
      and scoring begins at the second token.
    - prompt_lens (Optional[torch.Tensor]): Optional per-batch prompt lengths; included in
      returned metadata for bookkeeping (does not change scoring logic).
    - eos_token (Optional[int]): Token id that marks end-of-sequence. If provided, tokens
      after the first EOS are excluded from scoring.
    - pad_value (Optional[int]): Token id used to indicate padding; masked positions are
      excluded from per-token lists and sequence aggregates.
    - filter_logits_fn (Optional[Callable]): Function applied to raw logits before softmax.
      Signature should accept `(logits, **filter_kwargs)` and return logits of same shape.
    - filter_kwargs (Optional[Dict[str, Any]]): Keyword arguments forwarded to `filter_logits_fn`.
    - tokenizer: Optional tokenizer used to decode token ids for human-readable diffs and
      printed tables. If absent, token ids are stringified.
    - print_table (bool): If True, prints a human-readable table summarizing per-token stats.
    - device (Optional[torch.device]): Device to run scoring on. Defaults to `sequences.device`.
    - verbose (bool): If True, prints progress or extra information (currently minimal).
    - include_top1 (bool): If True, compute and return top-1 tokens, their probs/logprobs,
      and a `diff` structure listing positions where target != top-1.
    - **kwargs: Additional keyword arguments forwarded to `model.net`.

    Returns
    A dictionary with the following keys:
    - "tokens" (torch.Tensor): The input `sequences` returned as a CPU tensor.
    - "token_probs" (List[List[float]]): Per-batch lists of probabilities assigned to each
      scored target token (masked positions removed).
    - "token_logprobs" (List[List[float]]): Per-batch lists of log-probabilities (nats).
    - "token_scores" (List[List[float]]): Per-token scores (negative log-probabilities).
    - "sequence_logprobs" (List[float]): Sum of log-probabilities over unmasked target tokens.
    - "sequence_probs" (List[float]): Sequence probabilities where numerically representable.
    - "sequence_prob_display" (List[str]): Human-friendly display of sequence probability.
    - "nll" (List[float]): Negative log-likelihood per sequence (i.e., -sequence_logprob).
    - "mask" (torch.BoolTensor): Boolean mask (batch, scored_len) indicating which target
      positions were included in scoring (True = scored).
    - "diff" (List[List[Dict]]): Per-batch list of dicts for positions where the target
      token did not match the model's top-1. Each dict contains:
        - "pos": index within the scored positions (0-based)
        - "token": target token id
        - "token_str": decoded target token (or id string)
        - "token_prob": probability assigned to the target token
        - "top1_token": top-1 token id
        - "top1_token_str": decoded top-1 token
        - "top1_prob": top-1 probability
        - "match": boolean (False for entries in diff)
    - "metadata" (dict): Contains "prompt_len" (if provided), "seq_len" (original sequence
      length), and "scored_len_per_batch" (number of scored tokens per batch item).
    - If `include_top1` is True, additional keys are included:
      - "top1_tokens", "top1_token_probs", "top1_token_logprobs", "top1_matches"

    After assembling the primary result the function attaches a "metrics" entry with:
    - "per_sequence": list of per-sequence metric dicts containing:
      - "sequence_index", "token_count", "sequence_logprob_nats", "sequence_log10",
        "sequence_prob_display", "avg_logprob_per_token_nats", "avg_logprob_per_token_bits",
        "geometric_mean_token_prob", "perplexity"

    Notes and caveats
    - The function expects `sequences` to contain at least two tokens per batch item; if
      `seq_len < 2` a minimal result with empty scored lists is returned.
    - Numerical stability: probabilities are clamped before log to avoid -inf; extremely
      small sequence probabilities are represented in approximate 10^x form.
    - The function may raise ValueError if `model.net` returns logits of unexpected shape.
    - This routine is intended for evaluation and analysis of model likelihoods rather than
      high-performance batched scoring in production.

    Example (conceptual)
    >>> res = score_sequences(model, sequences, pad_value=0, eos_token=2, tokenizer=tok)
    >>> print(res["metrics"]["per_sequence"][0]["avg_logprob_per_token_nats"])
    """
    if filter_kwargs is None:
        filter_kwargs = {}
    if device is None:
        device = sequences.device

    model.eval()
    with torch.inference_mode():
        sequences = sequences.to(device)
        b, L = sequences.shape

        if L < 2:
            empty = [[] for _ in range(b)]
            return {
                "tokens": sequences.cpu(),
                "token_probs": empty,
                "token_logprobs": empty,
                "token_scores": empty,
                "sequence_probs": [1.0 for _ in range(b)],
                "sequence_prob_display": [f"{1.0:.6e}" for _ in range(b)],
                "sequence_logprobs": [0.0 for _ in range(b)],
                "nll": [0.0 for _ in range(b)],
                "mask": torch.zeros((b, 0), dtype=torch.bool),
                "diff": [[] for _ in range(b)],
                "metadata": {"prompt_len": None if prompt_lens is None else (prompt_lens.tolist() if isinstance(prompt_lens, torch.Tensor) else prompt_lens),
                             "seq_len": L,
                             "scored_len": 0}
            }

        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]

        logits_out = model.net(inputs, return_intermediates=True, cache=None, seq_start_pos=None, **kwargs)
        logits = logits_out[0] if isinstance(logits_out, (tuple, list)) else logits_out

        if logits.dim() != 3:
            raise ValueError(f"Expected logits with shape (b, seq, vocab), got {logits.shape}")

        filtered_logits = logits if filter_logits_fn is None else filter_logits_fn(logits, **(filter_kwargs or {}))
        probs = F.softmax(filtered_logits, dim=-1)
        targets_unsq = targets.unsqueeze(-1)
        picked_probs = probs.gather(dim=-1, index=targets_unsq).squeeze(-1)
        picked_logprobs = torch.log(picked_probs.clamp_min(1e-45)).to(dtype=torch.float64)

        if include_top1:
            top1_ids = probs.argmax(dim=-1)            # (b, seq)
            top1_p = probs.gather(-1, top1_ids.unsqueeze(-1)).squeeze(-1)
            top1_lp = torch.log(top1_p.clamp_min(1e-45)).to(dtype=torch.float64)

        mask = torch.ones_like(picked_probs, dtype=torch.bool)
        if pad_value is not None:
            mask = mask & (targets != pad_value)

        if eos_token is not None:
            for i in range(b):
                seq_full = sequences[i]
                eos_positions = (seq_full == eos_token).nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first_eos = int(eos_positions[0].item())
                    cutoff = max(0, first_eos - 1)
                    if cutoff + 1 < mask.shape[1]:
                        mask[i, cutoff+1:] = False

        token_probs: List[List[float]] = []
        token_logprobs: List[List[float]] = []
        token_scores: List[List[float]] = []
        sequence_logprobs: List[float] = []
        sequence_probs: List[float] = []
        sequence_prob_display: List[str] = []
        nll: List[float] = []

        top1_tokens: List[List[int]] = [[] for _ in range(b)]
        top1_token_probs: List[List[float]] = [[] for _ in range(b)]
        top1_token_logprobs: List[List[float]] = [[] for _ in range(b)]
        top1_matches: List[List[bool]] = [[] for _ in range(b)]

        diff_all: List[List[Dict[str, Any]]] = [[] for _ in range(b)]

        for i in range(b):
            row_mask = mask[i]
            row_probs = picked_probs[i]
            row_logps = picked_logprobs[i]
            kept_probs = row_probs[row_mask].cpu().tolist()
            kept_logps = row_logps[row_mask].cpu().tolist()
            kept_scores = [-lp for lp in kept_logps]
            token_probs.append([float(x) for x in kept_probs])
            token_logprobs.append([float(x) for x in kept_logps])
            token_scores.append([float(x) for x in kept_scores])

            if include_top1:
                t1_row = top1_ids[i]
                t1_p_row = top1_p[i]
                t1_lp_row = top1_lp[i]
                kept_t1_ids = t1_row[row_mask].cpu().tolist()
                kept_t1_ps = t1_p_row[row_mask].cpu().tolist()
                kept_t1_lps = t1_lp_row[row_mask].cpu().tolist()
                top1_tokens[i] = [int(x) for x in kept_t1_ids]
                top1_token_probs[i] = [float(x) for x in kept_t1_ps]
                top1_token_logprobs[i] = [float(x) for x in kept_t1_lps]
                kept_targets = targets[i][row_mask].cpu().tolist()
                top1_matches[i] = [int(t == top1) for t, top1 in zip(kept_targets, kept_t1_ids)]

                # build diff entries where target != top1
                for pos_idx, (tgt, tgt_p, t1, t1_p, match) in enumerate(zip(kept_targets, kept_probs, kept_t1_ids, kept_t1_ps, top1_matches[i])):
                    if not match:
                        diff_all[i].append({
                            "pos": pos_idx,
                            "token": int(tgt),
                            "token_str": _decode_token(tokenizer, int(tgt)),
                            "token_prob": float(tgt_p),
                            "top1_token": int(t1),
                            "top1_token_str": _decode_token(tokenizer, int(t1)),
                            "top1_prob": float(t1_p),
                            "match": bool(match)
                        })

            seq_lp_tensor = torch.tensor(sum(kept_logprobs := kept_logps), dtype=torch.float64)
            seq_lp = float(seq_lp_tensor.item())
            pnum, disp = _safe_exp64(seq_lp)
            sequence_logprobs.append(seq_lp)
            sequence_probs.append(pnum)
            sequence_prob_display.append(disp)
            nll.append(-seq_lp)

        result = {
            "tokens": sequences.cpu(),
            "token_probs": token_probs,
            "token_logprobs": token_logprobs,
            "token_scores": token_scores,
            "sequence_logprobs": sequence_logprobs,
            "sequence_probs": sequence_probs,
            "sequence_prob_display": sequence_prob_display,
            "nll": nll,
            "mask": mask.cpu(),
            "diff": diff_all,
            "metadata": {
                "prompt_len": None if prompt_lens is None else (prompt_lens.tolist() if isinstance(prompt_lens, torch.Tensor) else prompt_lens),
                "seq_len": L,
                "scored_len_per_batch": [int(m.sum().item()) for m in mask]
            }
        }

        if include_top1:
            result.update({
                "top1_tokens": top1_tokens,
                "top1_token_probs": top1_token_probs,
                "top1_token_logprobs": top1_token_logprobs,
                "top1_matches": top1_matches
            })

        result = _attach_metrics_to_result(result)

        if print_table:
            for i in range(b):
                print("=" * 120)
                header = f"Batch {i}  (seq_len={L})"
                if prompt_lens is not None:
                    header += f"  prompt_len={int(prompt_lens[i].item()) if isinstance(prompt_lens[i], torch.Tensor) else prompt_lens[i]}"
                print(header)
                print("-" * 120)
                print(" idx | token        |   prob    |   logprob   |   cum_logp   |   token_nll | top1_token (p) | match")
                print("-" * 120)
                cum_logp = 0.0
                pos_idx = 0
                for pos in range(1, L):
                    if not mask[i, pos-1]:
                        continue
                    tok_id = int(sequences[i, pos].item())
                    p = float(picked_probs[i, pos-1].cpu().item())
                    lp = float(picked_logprobs[i, pos-1].cpu().item())
                    cum_logp += lp
                    sc = -lp
                    if include_top1:
                        t1_id = top1_tokens[i][pos_idx]
                        t1_p = top1_token_probs[i][pos_idx]
                        match = top1_matches[i][pos_idx]
                        print(f"{pos_idx:3d} | {_decode_token(tokenizer, tok_id):>12s} | {p:9.6f} | {lp:11.6f} | {cum_logp:12.6f} | {sc:10.6f} | {_decode_token(tokenizer, t1_id):>12s} ({t1_p:5.3f}) | {match}")
                    else:
                        print(f"{pos_idx:3d} | {_decode_token(tokenizer, tok_id):>12s} | {p:9.6f} | {lp:11.6f} | {cum_logp:12.6f} | {sc:10.6f}")
                    pos_idx += 1
                print("-" * 120)
                print(f"Sequence logprob (nats): {result['sequence_logprobs'][i]:.6f} | Sequence prob: {result['sequence_prob_display'][i]} | NLL: {result['nll'][i]:.6f}")
                m = result["metrics"]["per_sequence"][i]
                print(f"Avg logprob/token: {m['avg_logprob_per_token_nats']:.6f} nats ({m['avg_logprob_per_token_bits']:.4f} bits) | Perplexity: {m['perplexity']:.6f}")
                if result["diff"][i]:
                    print("DIFF (target != top1) positions:")
                    for d in result["diff"][i]:
                        print(f" pos={d['pos']} token={d['token_str']}({d['token']}) p={d['token_prob']:.6f} | top1={d['top1_token_str']}({d['top1_token']}) p={d['top1_prob']:.6f}")
                else:
                    print("No diffs: target tokens matched top1 at every scored position.")
                print("=" * 120)

        return result

#=================================================================================================================================
# ETA functions
#=================================================================================================================================

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

def calculate_eta(
    hours_until_done: float,
    *,
    tz: str = "America/Los_Angeles",
    now: datetime | None = None,
    return_dict: bool = False,
    verbose: bool = True,
    ):
    
    """
    Compute an ETA timestamp based on the current time (or a provided time)
    in a specified timezone.

    Parameters
    ----------
    hours_until_done : float
        Number of hours remaining until completion.
    tz : str, optional
        IANA timezone name (default: "America/Los_Angeles").
    now : datetime or None, optional
        If provided, use this datetime as the starting point.
        If None, the current time in the given timezone is used.
    return_dict : bool, optional
        If True, return a dictionary with ETA components.
    verbose : bool, optional
        If True, print a formatted ETA string.

    Returns
    -------
    datetime or dict
        ETA as a datetime object or a dictionary (if return_dict=True).

    Examples
    --------
    
    # Simple ETA 5.5 hours from now
    calculate_eta(5.5)
    
    # ETA using a custom starting time in Tokyo
    from datetime import datetime
    calculate_eta(
        12,
        tz="Asia/Tokyo",
        now=datetime(2026, 1, 29, 8, 30),
    )
    
    # Get ETA as a dict without printing
    info = calculate_eta(3, verbose=False, return_dict=True)
    print(info["pretty"])
    """

    # Resolve timezone
    zone = ZoneInfo(tz)

    # Determine current time
    current_time = now.astimezone(zone) if now else datetime.now(zone)

    # Compute ETA
    eta = current_time + timedelta(hours=hours_until_done)

    # Format for printing
    pretty = eta.strftime("ETA: %A, %B %d %Y @ %H:%M")

    if verbose:
        print(pretty)

    if return_dict:
        return {
            "eta_datetime": eta,
            "year": eta.year,
            "month": eta.month,
            "day": eta.day,
            "hour": eta.hour,
            "minute": eta.minute,
            "second": eta.second,
            "timezone": tz,
            "pretty": pretty,
        }

    return eta

def calculate_training_run_eta(
    num_epochs: int,
    num_steps_per_epoch: int,
    sec_per_iter: float,
    *,
    cost_per_hr: float = 0.0,
    tz: str = "America/Los_Angeles",
    now: datetime | None = None,
    return_dict: bool = False,
    verbose: bool = True,
):
    """
    Compute ETA and cost for a full training run based on:
    - number of epochs
    - number of steps per epoch
    - seconds per iteration
    - optional cost per hour of compute

    Prints:
        - start time
        - ETA timestamp
        - per-epoch runtime (h/m/s)
        - total runtime (h/m/s)
        - cost per epoch
        - total run cost

    Returns:
        datetime or dict (if return_dict=True)

    Examples:

    # 2 epochs, 7770 steps each, 15.07 sec/iter, $5.3 per/hr
    calculate_training_run_eta(
        num_epochs=2,
        num_steps_per_epoch=7771,
        cost_per_hr=5.3,
        sec_per_iter=15.07,
    )
    
    # Get structured info without printing
    info = calculate_training_run_eta(
        3, 1000, 0.5,
        verbose=False,
        return_dict=True
    )
    print(info["eta_str"])
    """

    zone = ZoneInfo(tz)
    start_time = now.astimezone(zone) if now else datetime.now(zone)

    # Core calculations
    total_iters = num_epochs * num_steps_per_epoch
    total_seconds = total_iters * sec_per_iter
    epoch_seconds = num_steps_per_epoch * sec_per_iter

    eta = start_time + timedelta(seconds=total_seconds)

    # Formatting helpers
    def fmt(seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h}h {m}m {s}s"

    # Cost calculations
    total_hours = total_seconds / 3600
    epoch_hours = epoch_seconds / 3600

    cost_epoch = epoch_hours * cost_per_hr
    cost_total = total_hours * cost_per_hr

    # Pretty strings
    start_str = start_time.strftime("%A, %B %d %Y @ %H:%M")
    eta_str = eta.strftime("%A, %B %d %Y @ %H:%M")

    if verbose:
        print(f"Start Time: {start_str}")
        print(f"ETA:        {eta_str}")
        print(f"Per Epoch:  {fmt(epoch_seconds)}")
        print(f"Total Run:  {fmt(total_seconds)}")
        print(f"Cost/Epoch: ${cost_epoch:,.2f}")
        print(f"Cost/Run:   ${cost_total:,.2f}")

    if return_dict:
        return {
            "start_time": start_time,
            "eta": eta,
            "start_str": start_str,
            "eta_str": eta_str,
            "epoch_seconds": epoch_seconds,
            "total_seconds": total_seconds,
            "epoch_runtime_hms": fmt(epoch_seconds),
            "total_runtime_hms": fmt(total_seconds),
            "epoch_hours": epoch_hours,
            "total_hours": total_hours,
            "cost_per_hr": cost_per_hr,
            "cost_epoch": cost_epoch,
            "cost_total": cost_total,
            "timezone": tz,
        }

    return eta

#=================================================================================================================================
# MuseNet-style MusicTransformer class and functions
#=================================================================================================================================

import warnings
import torch
import torch.nn as nn
from typing import Optional, Any, Callable, Dict, Tuple, List

# -------------------------
# Helpers
# -------------------------
def exists(x: Any) -> bool:
    return x is not None

def default_token_type_ids(tokens: torch.Tensor) -> torch.Tensor:
    """
    Vectorized mapping of token ids -> type ids in {0,1,2,3}.
    Mirrors original behavior:
      0: tokens < 128
      1: 128 <= tokens < 256
      2: 256 <= tokens < 384
      3: tokens >= 384
    """
    t0 = (tokens < 128).to(torch.long) * 0
    t1 = ((tokens >= 128) & (tokens < 256)).to(torch.long) * 1
    t2 = ((tokens >= 256) & (tokens < 384)).to(torch.long) * 2
    t3 = (tokens >= 384).to(torch.long) * 3
    return t0 + t1 + t2 + t3

def build_type_map_from_ranges(vocab_size: int, ranges: List[Tuple[int, int, int]], default_type: int = 0) -> torch.Tensor:
    """
    Build a 1D torch.LongTensor of length vocab_size mapping token_id -> type_id.
    ranges: list of (start, end, type_id) inclusive ranges. Values outside 0..vocab_size-1 are clamped.
    default_type: type id for tokens not covered by any range.
    """
    type_map = torch.full((vocab_size,), fill_value=int(default_type), dtype=torch.long)
    for (start, end, type_id) in ranges:
        s = max(0, int(start))
        e = min(vocab_size - 1, int(end))
        if e < s:
            continue
        type_map[s:e+1] = int(type_id)
    return type_map

def infer_num_types_from_ranges_or_map(vocab_size: int, type_ranges: Optional[List[Tuple[int,int,int]]], type_map: Optional[torch.Tensor]) -> Optional[int]:
    """
    Return inferred num_types if possible, else None.
    """
    if type_map is not None:
        if type_map.numel() != vocab_size:
            raise ValueError("type_map length must equal vocab_size for inference")
        return int(type_map.max().item()) + 1
    if type_ranges is not None and len(type_ranges) > 0:
        max_type = max(int(tr[2]) for tr in type_ranges)
        return max_type + 1
    return None

# -------------------------
# Shared + delta embeddings (with padding handling)
# -------------------------
class SharedDeltaEmbedding(nn.Module):
    """
    MuseNet-style factorized embedding:
        final_emb = base[token] + delta[type]

    Two ways to supply token types:
      - token_type_fn: callable(tokens) -> type_ids (vectorized torch ops)
      - type_map: 1D tensor of length vocab_size mapping token_id -> type_id (fast indexing)

    If padding_idx is provided, the base embedding row at padding_idx is
    handled by nn.Embedding(padding_idx=...), and the delta contribution
    is explicitly zeroed at pad positions so the final pad embedding is zero.
    Returns (B, L, dim).
    """
    def __init__(
        self,
        dim: int,
        vocab_size: int = 718,
        num_types: Optional[int] = None,
        padding_idx: Optional[int] = None,
        token_type_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        type_map: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.padding_idx = padding_idx

        # If a type_map is provided, register it as a buffer and infer num_types if not given
        if type_map is not None:
            tm = type_map.to(dtype=torch.long).flatten()
            if tm.numel() != vocab_size:
                raise ValueError("type_map length must equal vocab_size")
            self.register_buffer("type_map", tm, persistent=True)
            inferred = int(tm.max().item()) + 1
            if num_types is None:
                num_types = inferred
            else:
                if int(num_types) < inferred:
                    raise ValueError(f"Provided num_types={num_types} is smaller than max(type_map)+1={inferred}")
        else:
            self.type_map = None

        # If num_types still None, leave it None for now; will be resolved below or defaulted
        if num_types is None:
            # If no type_map but user provided token_type_fn, we cannot infer reliably
            # fall back to default 4 and warn
            num_types = 4
            warnings.warn("num_types not provided and cannot be inferred from ranges/map; defaulting to 4. Provide num_types if your token_type_fn produces a different number of types.", UserWarning)

        self.num_types = int(num_types)

        # base embedding: padding_idx will ensure base[padding_idx] stays zero and not updated
        self.base = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        # small delta per token type; must be zeroed at pad positions manually
        self.delta = nn.Embedding(self.num_types, dim)

        # token_type_fn should be vectorized and use torch ops only
        self.token_type_fn = token_type_fn if token_type_fn is not None else default_token_type_ids

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, L) with integer token ids
        returns: (B, L, dim)
        """
        if hasattr(self, "type_map") and self.type_map is not None:
            type_ids = self.type_map[tokens]
        else:
            type_ids = self.token_type_fn(tokens).to(torch.long)

        base_emb  = self.base(tokens)                  # (B, L, dim)
        delta_emb = self.delta(type_ids)               # (B, L, dim)

        if self.padding_idx is not None:
            pad_mask = tokens == self.padding_idx     # (B, L) boolean
            delta_emb = delta_emb.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        return base_emb + delta_emb

# -------------------------
# MusicTransformer compatible with x-transformers AutoregressiveWrapper
# -------------------------
class MusicTransformer(nn.Module):
    
    """
    Transformer model adapted to be wrapped by x-transformers' AutoregressiveWrapper.

    Constructor args:
      - token_type_fn: optional callable(tokens) -> type_ids tensor
      - type_ranges: optional list of (start, end, type_id) tuples (inclusive)
      - type_map: optional prebuilt 1D tensor mapping token_id -> type_id
      - num_token_types: optional int; if omitted, will be inferred from ranges or type_map when possible


    Use Example
    -------

    vocab_size = 718
    pad_value = 717
    SEQ_LEN = 1024

    # Example: specify contiguous ranges for token types (inclusive)
    type_ranges = [
        (0, 127, 0),
        (128, 255, 1),
        (256, 383, 2),
        (384, vocab_size - 1, 3),
    ]

    # You can omit num_token_types and it will be inferred from type_ranges
    model = MusicTransformer(
        dim=2048,
        depth=12,
        heads=16,
        max_seq_len=SEQ_LEN,
        vocab_size=vocab_size,
        pad_value=pad_value,
        token_type_fn=None,
        type_ranges=type_ranges,
        num_token_types=None,  # will be inferred as 4
        decoder_kwargs = dict(
            attn_orthog_projected_values = True,
            attn_orthog_projected_values_per_head = True
        ),
        tw_kwargs = dict()
    )

    wrapped = AutoregressiveWrapper(model, ignore_index=pad_value, pad_value=pad_value)

    # Recommended compile & device order:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapped.to(device)
    wrapped = torch.compile(wrapped)

    tokens = torch.randint(0, vocab_size, (2, 16), device=device)
    logits, cache = wrapped.net(tokens, return_intermediates=True)
    print("Logits shape:", logits.shape)
    """
    
    def __init__(
        self,
        dim: int = 2048,
        depth: int = 8,
        heads: int = 16,
        max_seq_len: int = 4096,
        vocab_size: int = 718,
        pad_value: Optional[int] = None,
        num_token_types: Optional[int] = None,
        token_type_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        type_ranges: Optional[List[Tuple[int, int, int]]] = None,
        type_map: Optional[torch.Tensor] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
        tw_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if decoder_kwargs is None:
            decoder_kwargs = {}
        if tw_kwargs is None:
            tw_kwargs = {}

        # Attributes used by AutoregressiveWrapper
        self.max_seq_len = max_seq_len
        self.add_continuous_pred_head = False
        self.output_is_log_prob = False
        self.can_cache_kv = True
        self.can_cache_kv_outside_max_seq_len = False

        # If type_ranges provided, build a type_map and infer num types if needed
        built_type_map = None
        if type_ranges is not None:
            built_type_map = build_type_map_from_ranges(vocab_size, type_ranges, default_type=0)

        # If user provided a type_map argument, prefer it (but ensure shape)
        if type_map is not None:
            tm = type_map.to(dtype=torch.long).flatten()
            if tm.numel() != vocab_size:
                raise ValueError("Provided type_map length must equal vocab_size")
            built_type_map = tm

        # Try to infer num_token_types if not provided
        inferred = infer_num_types_from_ranges_or_map(vocab_size, type_ranges, built_type_map)
        if num_token_types is None and inferred is not None:
            num_token_types = inferred

        # If still None and token_type_fn provided, we cannot infer; warn and default to 4
        if num_token_types is None:
            if token_type_fn is not None:
                warnings.warn("num_token_types not provided and cannot be inferred from ranges/map; defaulting to 4. Provide num_token_types if your token_type_fn produces a different number of types.", UserWarning)
                num_token_types = 4
            else:
                # no token_type_fn and no ranges/map: default to 4 to preserve original behavior
                num_token_types = 4

        # Embeddings and transformer
        self.token_emb = SharedDeltaEmbedding(
            dim=dim,
            vocab_size=vocab_size,
            num_types=num_token_types,
            padding_idx=pad_value,
            token_type_fn=token_type_fn,
            type_map=built_type_map
        )

        # Build Decoder with decoder_kwargs but ensure required args are present/overridden
        decoder_init_kwargs = dict(
            dim = dim,
            depth = depth,
            heads = heads,
            rotary_pos_emb = True,
            attn_flash = True
        )
        decoder_init_kwargs.update(decoder_kwargs)
        decoder = Decoder(**decoder_init_kwargs)

        # TransformerWrapper requires num_tokens even when using a custom token_emb.
        tw_init_kwargs = dict(
            num_tokens = vocab_size,
            max_seq_len = max_seq_len,
            token_emb = self.token_emb,
            attn_layers = decoder
        )
        tw_init_kwargs.update(tw_kwargs)

        self.transformer = TransformerWrapper(**tw_init_kwargs)
        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False,
        return_logits_and_embeddings: bool = False,
        return_intermediates: bool = False,
        return_embeddings_and_intermediates: bool = False,
        return_logit_entropies: bool = False,
        return_next_embed_pred: bool = False,
        mask = None,
        return_mems: bool = False,
        return_attn: bool = False,
        mems = None,
        mem_masks = None,
        recycle_steps = None,
        pos = None,
        prepend_embeds: Optional[torch.Tensor] = None,
        prepend_mask = None,
        embed_ids: Dict[str, torch.Tensor] = dict(),
        sum_embeds = None,
        return_attn_z_loss: bool = False,
        attn_z_loss_weight: float = 1e-4,
        seq_start_pos: Optional[torch.Tensor] = None,
        cache: Any = None,
        input_not_include_cache: bool = False,
        token_emb_kwargs: dict = dict(),
        to_logits_kwargs: dict = dict(),
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        Forward adapted to TransformerWrapper's signature.

        Returns:
          - (logits, cache) when return_intermediates=True
          - (logits, None) otherwise
        """
        tw_out = self.transformer(
            x,
            return_embeddings = return_embeddings,
            return_logits_and_embeddings = return_logits_and_embeddings,
            return_intermediates = return_intermediates,
            return_embeddings_and_intermediates = return_embeddings_and_intermediates,
            return_logit_entropies = return_logit_entropies,
            return_next_embed_pred = return_next_embed_pred,
            mask = mask,
            return_mems = return_mems,
            return_attn = return_attn,
            mems = mems,
            mem_masks = mem_masks,
            recycle_steps = recycle_steps,
            pos = pos,
            prepend_embeds = prepend_embeds,
            prepend_mask = prepend_mask,
            embed_ids = embed_ids,
            sum_embeds = sum_embeds,
            return_attn_z_loss = return_attn_z_loss,
            attn_z_loss_weight = attn_z_loss_weight,
            seq_start_pos = seq_start_pos,
            cache = cache,
            input_not_include_cache = input_not_include_cache,
            token_emb_kwargs = token_emb_kwargs,
            to_logits_kwargs = to_logits_kwargs,
            **kwargs
        )

        # Normalize TW outputs into (embeddings_or_logits, new_cache)
        new_cache = None
        embeddings_or_logits = tw_out

        if isinstance(tw_out, tuple):
            possible_cache = tw_out[-1]
            if not isinstance(possible_cache, torch.Tensor):
                new_cache = possible_cache
                embeddings_or_logits = tw_out[0]
            else:
                embeddings_or_logits = tw_out[0]

        # Unpack embeddings_or_logits if it's a tuple like (logits, embeddings) or (embeddings, logits)
        embeddings = None
        logits_candidate = None

        if isinstance(embeddings_or_logits, tuple):
            first, second = embeddings_or_logits[0], embeddings_or_logits[1]
            if isinstance(second, torch.Tensor) and second.dim() == 3 and second.shape[-1] == self.to_logits.in_features:
                embeddings = second
                logits_candidate = first
            elif isinstance(first, torch.Tensor) and first.dim() == 3 and first.shape[-1] == self.to_logits.in_features:
                embeddings = first
                logits_candidate = second
            else:
                embeddings = first if isinstance(first, torch.Tensor) else None
                logits_candidate = second if isinstance(second, torch.Tensor) else None
        else:
            candidate = embeddings_or_logits
            if isinstance(candidate, torch.Tensor) and candidate.dim() == 3 and candidate.shape[-1] == self.to_logits.in_features:
                embeddings = candidate
            else:
                logits_candidate = candidate

        if exists(embeddings):
            logits = self.to_logits(embeddings)
        elif exists(logits_candidate):
            logits = logits_candidate
        else:
            raise RuntimeError("Unable to determine logits or embeddings from TransformerWrapper output. Check flags passed to TW.")

        if return_intermediates:
            return logits, new_cache

        return logits, None
        
#=================================================================================================================================
# This is the end of x_transformer_2_14_2 Python module
#=================================================================================================================================