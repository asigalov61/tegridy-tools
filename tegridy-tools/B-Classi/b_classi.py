r'''############################################################################
################################################################################
#
#
#	    Binary Classifier (B Classi)
#	    Version 1.0
#
#	    Project Los Angeles
#
#	    Tegridy Code 2025
#
#       https://github.com/asigalov61/tegridy-tools
#
#
################################################################################
#
#       Copyright 2024 Project Los Angeles / Tegridy Code
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
#
################################################################################
################################################################################
#
#       Critical dependencies
#
#       !pip install torch
#       !pip install -U scikit-learn
#
################################################################################
'''

################################################################################

print('=' * 70)
print('Loading B Classi module...')
print('Please wait...')
print('=' * 70)

################################################################################

import sys
import os
import random
import numpy as np
import tqdm

################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)

################################################################################

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, vocab_size):
        """
        sequences: List[List[int]] of shape (N, seq_len)
        labels:    List[int] of length N, values 0 or 1
        vocab_size: int, maximum token value + 1
        """
        self.x = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float)
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

################################################################################

def make_dataloader(sequences, labels, vocab_size,
                    batch_size=64, shuffle=True, num_workers=4):
    ds = MidiSequenceDataset(sequences, labels, vocab_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True
    )

################################################################################

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)
        nn.init.xavier_uniform_(self.attn.weight)

    def forward(self, x, mask=None):
        # x: [B, L, E]
        scores = self.attn(x).squeeze(-1)               # [B, L]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (x * weights).sum(dim=1)                # [B, E]

################################################################################

class BClassi(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=8,
        ff_dim=1024,
        num_layers=2,
        dropout=0.2,
        max_seq_len=1024
    ):
        super().__init__()
        # token + positional embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # hybrid pooling & classification head
        self.pool = AttentionPooling(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x, mask=None):
        # x: [B, L], mask: [B, L] (True for valid tokens)
        B, L = x.shape

        # embed tokens + prepend CLS
        tok_emb = self.token_embed(x)                            # [B, L, E]
        cls_emb = self.cls_token.expand(B, -1, -1)               # [B, 1, E]
        emb = torch.cat([cls_emb, tok_emb], dim=1)               # [B, L+1, E]
        emb = emb + self.pos_embed[:, : L + 1]                   # add pos

        # transformer encoding
        enc = self.transformer(
            emb,
            src_key_padding_mask=(~mask) if mask is not None else None
        )                                                         # [B, L+1, E]

        # pooling: CLS + attention on tokens
        cls_rep = enc[:, 0]                                       # [B, E]
        attn_rep = self.pool(enc[:, 1:], mask)                   # [B, E]
        rep = cls_rep + attn_rep

        # classification head
        rep = self.norm(rep)
        rep = self.dropout(rep)
        logits = self.fc(rep).squeeze(-1)
        
        return logits
        
################################################################################

def train_model(model, train_loader, val_loader, epochs=20,
               lr=1e-3, weight_decay=1e-5, patience=10):
    
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_train_loss = 0

        for x_batch, y_batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            # Forward + backward in bfloat16 autocast
            with autocast('cuda', dtype=torch.float16):
                logits = model(x_batch)
                loss = criterion(logits, y_batch)

            # Scale the loss, call backward, step optimizer, update scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item() * x_batch.size(0)

        scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch} Train loss: {avg_train_loss:.4f}")

        # --- Validate ---
        model.eval()
        total_val_loss = 0
        preds, trues = [], []

        with torch.no_grad():
            for x_batch, y_batch in tqdm.tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # Use autocast to speed up validation
                with autocast('cuda', dtype=torch.float16):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                total_val_loss += loss.item() * x_batch.size(0)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds.extend(probs)
                trues.extend(y_batch.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch} Val loss:   {avg_val_loss:.4f}")

        # Early stopping & checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_val_loss_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch}")
                torch.save(model.state_dict(), 'best_train_loss_model.pth')
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_train_loss_model.pth', map_location=DEVICE))
    return evaluate(model, val_loader)

################################################################################

def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            with autocast('cuda', dtype=torch.float16):
                logits = model(x_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y_batch.numpy())

    preds = [1 if p >= threshold else 0 for p in all_probs]
    metrics = {
        'accuracy': accuracy_score(all_labels, preds),
        'roc_auc': roc_auc_score(all_labels, all_probs),
        'precision': precision_score(all_labels, preds),
        'recall': recall_score(all_labels, preds),
        'f1': f1_score(all_labels, preds)
    }
    return metrics
    
################################################################################

print('Module is loaded!')
print('Enjoy! :)')
print('=' * 70)

################################################################################
# This is the end of the B Classi Python module
################################################################################