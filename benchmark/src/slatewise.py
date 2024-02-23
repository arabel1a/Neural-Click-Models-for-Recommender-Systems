import numpy as np
import os
import pandas as pd
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from tqdm.notebook import tqdm
from torchmetrics import AUROC
from datetime import datetime, timedelta
from torch.nn.utils import clip_grad_norm_
from torchmetrics.functional import f1_score, accuracy

current_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.append(current_dir)

from datasets import RecommendationData
from utils import train, collate_recommendation_datasets
from embeddings import RecsysEmbedding



class MF(torch.nn.Module):
    def __init__(self, embedding):
        super().__init__()
        self.embedding = embedding
    
    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        scores = item_embs * user_embs[:, :, None, :].repeat(1, 1, item_embs.size(-2), 1)
        scores = scores.sum(-1)
        return scores

class LogisticRegression(torch.nn.Module):
    def __init__(self, embedding, output_dim=1):
        super().__init__()
        self.embedding = embedding
        self.linear = torch.nn.Linear(2 * embedding.embedding_dim, output_dim)
    
    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)

        features = torch.cat(
            [
                item_embs,
                user_embs[:, :, None, :].repeat(1, 1, item_embs.size(-2), 1)
            ],
            dim = -1
        )
        return self.linear(features).squeeze(-1)

class SlatewiseAttention(torch.nn.Module):
    """
    No recurrent dependency, just slate-wise attention.
    """
    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention= torch.nn.MultiheadAttention(
            2 * embedding.embedding_dim,
            num_heads=nheads,
            batch_first=True
        )
        
        self.out_layer = torch.nn.Linear(2 * embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape
        
        # let model attent to first padd token if slate is empty to avoid NaN gradients
        # (anyway they does not contrinute into metrics computation)
        key_padding_mask = batch['slates_mask'].clone()
        key_padding_mask[:,:, 0] = True 
        
        # repeating user embedding for each item embeding
        features = torch.cat(
            [
                item_embs,
                user_embs[:, :, None, :].repeat(1, 1, shp[-2], 1).reshape(shp)
            ],
            dim = -1
        )
        
        # all recomendations in session are scored independently
        features = features.flatten(0,1)
        
        features, attn_map = self.attention(
            features, features, features,
            key_padding_mask=~key_padding_mask.flatten(0, 1)
        )
        
        # transforming back to sequence shape
        shp = list(shp)
        shp[-1] *= 2
        return self.out_layer(features.reshape(shp)).squeeze(-1)

class SlatewiseGRU(nn.Module):
    def __init__(self, embedding, readout=False):
        super().__init__()
        
        self.embedding = embedding
        
        self.emb_dim = embedding.embedding_dim
        self.rnn_layer = nn.GRU(
            input_size=self.emb_dim * 2, 
            hidden_size=self.emb_dim, 
            batch_first=True
        )
        self.out_layer = nn.Linear(self.emb_dim, 1)
        
        self.thr = -1.5
        self.readout = readout
        self.readout_mode = 'threshold' # ['soft' ,'threshold', 'sample', 'diff_sample']
        
        self.calibration = False
        self.w = 1
        self.b = 0

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape
        max_sequence = item_embs.size(1)
        # ilya format:
        # 'items': (batch, slate, 2*embedding_dim ) 2, нужно для readout, по умолчанию ноли на половине эмбеддинга
        # 'clicks': (batch, slate)
        # 'users': (1, batch, embedding_dim), 
        # 'mask': (batch, slate)
        
        x = {}
        x['items'] = torch.cat(
            [
                item_embs.flatten(0,1),
                torch.zeros_like(item_embs.flatten(0,1)),
            ],
            dim = -1
        )
                
        if self.training:
            indices = (batch['length'] - 1)
        else:
            indices = (batch['in_length'] - 1)
        indices[indices<0] = 0
        indices = indices[:, None, None].repeat(1, 1, user_embs.size(-1))
        user_embs = user_embs.gather(1, indices).squeeze(-2).unsqueeze(0)
        x['users'] = user_embs.repeat_interleave(max_sequence, 1)
        x['clicks'] = (batch['responses'].flatten(0,1) > 0 ).int().clone()
        x['mask'] = batch['slates_mask'].flatten(0,1).clone()
        
        items = x['items']
        h = x['users']
        
        if self.readout:
            res = []
            seq_len = items.shape[1]
            for i in range(seq_len):
#                 print(items[:,[i],:])
                output, h = self.rnn_layer(items[:,[i],:], h)
                y = self.out_layer(output)[:, :, 0]
                
                # readout
                if i + 1 < seq_len:
                    if self.readout_mode == 'threshold':
                        items[:, [i+1], self.emb_dim:] *= (y.detach()[:, :, None] > self.thr).to(torch.float32)
                    elif self.readout_mode == 'soft':
                        items[:, [i+1], self.emb_dim:] *= torch.sigmoid(y)[:, :, None]
                    elif self.readout_mode == 'diff_sample' or self.readout_mode == 'sample':
                        eps = 1e-8
                        gumbel_sample = -( (torch.rand_like(y) + eps).log() / (torch.rand_like(y) + eps).log() + eps).log()
                        T = 0.3
                        bernoulli_sample = torch.sigmoid( (nn.LogSigmoid()(self.w * y + self.b) + gumbel_sample) / T )
                        if self.readout_mode == 'sample':
                            bernoulli_sample = bernoulli_sample.detach()
                        items[:, [i+1], self.emb_dim:] *= bernoulli_sample[:, :, None]
                    else:
                        raise
                    
                res.append(y)
        
            y = torch.cat(res, axis=1)
            
        else:
            items[:, 1:, self.emb_dim:] *= x['clicks'][:, :-1, None]
            rnn_out, _ = self.rnn_layer(items, h)
            y = self.out_layer(rnn_out)[:, :, 0]
        
        
        if self.calibration and self.training:
            clicks_flat, logits_flat = flatten(x['clicks'], y.detach(), x['mask'])
            logreg = LogisticRegression()
            logreg.fit(logits_flat[:, None], clicks_flat)
            γ = 0.3
            self.w = (1 - γ) * self.w + γ * logreg.coef_[0, 0]
            self.b = (1 - γ) * self.b + γ * logreg.intercept_[0]
            y = self.w * y + self.b
        else:
            y = self.w * y + self.b
            
        return y.reshape(shp[:-1])
