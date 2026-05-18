# PyTorch [x-transformers](https://github.com/lucidrains/x-transformers) implementation by lucidrains
## with useful modifications as a stand-alone Python module

***

## Original source code retrieved on 01/27/2026
## Original version 2.14.2 / Commit 03d11fa

***

## Critical dependencies:
### [PyTorch](https://github.com/pytorch/pytorch)
### [einops](https://github.com/arogozhnikov/einops)
### [einx](https://github.com/fferflo/einx)

```sh
!pip install torch
!pip install einops
!pip install einx
```

***

## Documentation

### [Main docs](https://github.com/lucidrains/x-transformers/blob/main/README.md)
### [Classification docs](https://github.com/lucidrains/x-transformers/pull/264)

***

## Useful examples

```python
import torch
from torch import nn

from x_transformers import (
    TransformerWrapper,
    Encoder
)

# CLS token test
transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=2, # num_classes 
    use_cls_token=True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10))
y = torch.tensor([0, 1])

print(x.shape)
logits = transformer(x)
print(logits.shape)
loss = nn.CrossEntropyLoss()(logits, y)

print(loss)

# BCE cls token

transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=1, # num_classes 
    use_cls_token=True,
    squeeze_out_last_dim = True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10)).float()
y = torch.tensor([0, 1]).float()

print(x.shape)
logits = transformer(x).squeeze()
loss = nn.BCEWithLogitsLoss()(logits, y)

print(loss)

# pooling test
transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=2, # num_classes 
    average_pool_embed = True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10))
y = torch.tensor([0, 1])

print(x.shape)
logits = transformer(x)
print(logits.shape)
loss = nn.CrossEntropyLoss()(logits, y)

print(loss)

# pooling BCE test

# pooling test
transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=1, # num_classes 
    average_pool_embed = True,
    squeeze_out_last_dim = True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (2, 10)).float()
y = torch.tensor([0, 1]).float()

print(x.shape)
logits = transformer(x).squeeze()
print(logits.shape)
loss = nn.BCEWithLogitsLoss()(logits, y)

print(loss)

# normal test 

transformer = TransformerWrapper(
    num_tokens=6,
    max_seq_len=10,
    logits_dim=2, # num_classes 
    average_pool_embed = True,
    attn_layers = Encoder(
        dim = 6,
        depth = 1,
        heads = 2,
    )
)

x = torch.randint(0, 5, (1, 10))
y = torch.tensor([0])

print(x.shape)
logits = transformer(x)
print(logits.shape)
```

***

### Project Los Angeles
### Tegridy Code 2026
