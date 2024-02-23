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
from collections import namedtuple




class SessionwiseGRU(torch.nn.Module):
    def __init__(self, embedding, output_dim=1, dropout = 0.1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.rnn_layer = torch.nn.GRU(
            input_size = embedding.embedding_dim, 
            hidden_size = embedding.embedding_dim, 
            batch_first = True,
            dropout=dropout
        )
        self.out_layer = torch.nn.Linear(embedding.embedding_dim, output_dim)


    def forward(self, batch):
        shp = batch['slates_item_indexes'].shape
        item_embs, user_embs = self.embedding(batch)
        item_embs = item_embs.flatten(-3, -2)
        
        # while training, let out model see the future 
        # while testing, it can see only the 
        if self.training:
            indices = (batch['length'] - 1)
        else:
            indices = (batch['in_length'] - 1)
        
        indices[indices<0] = 0
        indices = indices[:, None, None].repeat(1, 1, user_embs.size(-1))
        user_embs = user_embs.gather(1, indices).squeeze(-2).unsqueeze(0)

#         print(indices.shape, user_embs.shape, item_embs.shape, )
        rnn_out, _ = self.rnn_layer(
            item_embs,
            user_embs,
        )
        return self.out_layer(rnn_out).reshape(shp)

class SCOT(torch.nn.Module):
    """
    No recurrent dependency, just slate-wise attention.
    """
    def __init__(self, embedding, nheads=2, output_dim=1, debug=False):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.nheads = nheads
        self.debug = debug
        self.attention= torch.nn.MultiheadAttention(
            self.embedding_dim,
            num_heads=nheads,
            batch_first=True
        )
        
        self.out_layer = torch.nn.Sequential(
                # torch.nn.LayerNorm(2* embedding.embedding_dim),
                # torch.nn.Linear(embedding.embedding_dim * 2, embedding.embedding_dim * 2),
                # torch.nn.GELU(),
                torch.nn.Linear(embedding.embedding_dim * 2, output_dim)
        )

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape
        device = item_embs.device
        
        if self.debug: print('responses', batch['responses'])
        
        # flattening slates into long sequences
        item_embs = item_embs.flatten(1, 2)
        
        # getting user embedding (mean consumed items)
        if self.training:
            indices = (batch['length'] - 1)
        else:
            indices = (batch['in_length'] - 1)
        indices[indices<0] = 0
        indices = indices[:, None, None].repeat(1, 1, user_embs.size(-1))
        user_embs = user_embs.gather(1, indices).squeeze(-2)
        
        # adding a user embedding as a 'zero item' to 
        # make predictions if nothing is observed
        keys = torch.cat([
                user_embs[:, None, :],
                item_embs
            ],
            dim = 1
        )
        
        clicked_mask = batch['responses'].flatten(1, 2) > 0
        if self.debug: print(clicked_mask, clicked_mask.shape)
        clicked_mask = torch.cat([
                torch.ones_like(clicked_mask[:, 0, None]).bool(),
                clicked_mask
            ],
            dim=-1
        )
        if self.debug: print('clicked_mask:', clicked_mask, clicked_mask.shape)
        
        clicked_items = [
            keys[i][clicked_mask[i], :]
            for i in range(shp[0])
        ]
        keys = torch.nn.utils.rnn.pad_sequence(
            clicked_items,
            batch_first=True,
            padding_value=float('nan')
        )
        key_padding_mask = keys.isnan().all(-1)
        keys = keys.nan_to_num(0)
        
        if self.debug: print('key', keys.shape, keys, key_padding_mask)
        
        # forbid model looking into future (and into current iteraction)
        # at the end, mask will be (num_heads * bsize, slate_size * sequence_size, max_len_clicked_items)
        future_mask = torch.ones((item_embs.size(-2) + 1, item_embs.size(-2) + 1 ), device=device)
        future_mask = torch.triu(future_mask, diagonal=0)[1:, :].bool()
        if self.debug: print('future_mask', future_mask.shape, future_mask)
        ####### TODOTODOTODO ########
        # change future masl to be slatewise, not sequencewise
        # build it without large matrix generaton if possible
        # chunk previous iteracrtion
        ############################
        
        if self.debug: print('click_mask_repeated', clicked_mask[None, 0, ...].repeat(item_embs.size(-2), 1))
        
        if self.debug: print(
            'first user click & past', 
            future_mask[clicked_mask[None, 0, ...].repeat(item_embs.size(-2), 1)].reshape(item_embs.size(-2), keys.size(-2) )
        )
        
        attn_mask = [
            future_mask[
                clicked_mask[None, i, ...].repeat(item_embs.size(-2), 1)
            ].reshape(item_embs.size(-2), clicked_mask[None, i, ...].sum() ).T
            for i in range(shp[0])
        ]
        if self.debug: print('attn_mask', attn_mask[0].shape, attn_mask[1].shape )
                
        
        attn_mask = torch.nn.utils.rnn.pad_sequence(
            attn_mask,
            batch_first=True,
            padding_value=True
        ).permute(0, 2, 1)
        if self.debug: print('attn_mask_stacked', attn_mask)
        
        if self.debug: print(item_embs, keys, key_padding_mask, attn_mask.repeat_interleave(self.nheads, 0))
        features, attn_map = self.attention(
            item_embs, keys, keys,
            key_padding_mask=key_padding_mask,
            attn_mask = attn_mask.repeat_interleave(self.nheads, 0)
        )
        
        if self.debug: print(shp, item_embs.shape)
        features = torch.cat(
            [
                features.reshape(shp),
                item_embs.reshape(shp)
            ],
            dim = -1
        )
        
        return self.out_layer(features).squeeze(-1)

# torch.autograd.set_detect_anomaly(True)

class AttentionGRU(torch.nn.Module):
    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention = torch.nn.MultiheadAttention(
            2 * embedding.embedding_dim,
            num_heads=nheads,
            batch_first=True
        )
        
        self.rnn_layer = torch.nn.GRU(
            input_size = embedding.embedding_dim, 
            hidden_size = embedding.embedding_dim, 
            batch_first=True
        )
        
        self.out_layer = torch.nn.Linear(3 * embedding.embedding_dim, output_dim)
    
    
    def get_attention_embeddings(self, item_embs, user_embs, slate_mask):
        shp = item_embs.shape      
        key_padding_mask = slate_mask
        key_padding_mask[:,:, 0] = True # let model attent to first padd token if slate is empty 
        features = torch.cat(
            [
                item_embs,
                user_embs[:, :, None, :].repeat(1, 1, item_embs.size(-2), 1).reshape(shp)
            ],
            dim = -1
        ).flatten(0,1)
        
        features, attn_map = self.attention(
            features, features, features,
            key_padding_mask=~key_padding_mask.flatten(0, 1)
        )
        shp = list(shp)
        shp[-1] *= 2
        features = features.reshape(shp)
        return features
    
    def forward(self, batch):
        # consider sequential clicks, hence need to flatten slates
        item_embs, user_embs = self.embedding(batch)
        slate_mask = batch['slates_mask'].clone()
        
        att_features = self.get_attention_embeddings(item_embs, user_embs, slate_mask)
        
        gru_features, _ = self.rnn_layer(item_embs.flatten(-3, -2))
        gru_features = gru_features.reshape(item_embs.shape)
        
        features = torch.cat(
            [att_features, gru_features],
            dim=-1
        )
        
        return self.out_layer(features).squeeze(-1)


class SessionwiseAttention(torch.nn.Module):
    """
    No recurrent dependency, just slate-wise attention.
    """
    def __init__(self, embedding, nheads=2, output_dim=1):
        super().__init__()
        self.embedding_dim = embedding.embedding_dim
        self.embedding = embedding
        self.attention= torch.nn.MultiheadAttention(
            self.embedding_dim,
            num_heads=nheads,
            batch_first=True
        )
        
        self.out_layer = torch.nn.Linear(2 * embedding.embedding_dim, output_dim)

    def forward(self, batch):
        item_embs, user_embs = self.embedding(batch)
        shp = item_embs.shape
        device = item_embs.device
        
        # flattening slates in to long sequences
        item_embs = item_embs.flatten(1, 2)
        
        # let model attent to first padd token if slate is empty to avoid NaN gradients
        # (anyway they does not contrinute into metrics computation)
        key_padding_mask = batch['slates_mask'].clone()
        key_padding_mask[:,:, 0] = True # let model attent to first padd token if slate is empty 
        key_padding_mask = ~key_padding_mask.flatten(1,2)
        
        # forbid model looking into future (and into current iteraction)
        future_mask = torch.ones((item_embs.size(-2), item_embs.size(-2))).to(device)
        future_mask = torch.triu(future_mask, diagonal=1).bool()
        
        features, attn_map = self.attention(
            item_embs, item_embs, item_embs,
            key_padding_mask=key_padding_mask,
            attn_mask = future_mask
        )
#         print(features.shape, user_embs.shape, shp)
        features = torch.cat(
            [
                features.reshape(shp),
                user_embs[:, :, None, :].repeat(1, 1, shp[-2], 1)
            ],
            dim = -1
        )
        
        return self.out_layer(features).squeeze(-1)

