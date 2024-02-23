from sklearn.utils.extmath import randomized_svd
import os, sys
import torch
import random
import datetime
import pandas as pd
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset

current_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.append(current_dir)

from utils import train, get_dummy_data, get_train_val_test_tmatrix_tnumitems, get_svd_encoder


class RecsysEmbedding(torch.nn.Module):
    """
    Old embedding module, without support of categorical features.
    """
    def __init__ (self, n_items, user_item_matrix, agg='mean', embeddings='svd', embedding_dim=32):
        super().__init__()
        self.embeddings = embeddings
        self.embedding_dim = embedding_dim
        self.agg = agg
        self.n_items = n_items
        
        if embeddings == 'svd':
            _, _, self.item_embeddings = randomized_svd(
                user_item_matrix, 
                n_components=embedding_dim, 
                n_iter=4,
                power_iteration_normalizer='QR'
            )
            self.item_embeddings = torch.tensor(self.item_embeddings.T).float()        
        elif embeddings == 'neural':
            self.item_embeddings = torch.nn.Embedding(n_items, embedding_dim)
                
        
    def forward(self, batch):
        """
            Item embeddings are just embeddings(svd, or neural, or explicit)
            User embeddings on step i are aggregated embeddings of all user consumed
            items on steps before i        
        """
        
        batch_size, max_sequence = batch['responses'].shape[:2]
        device = batch['responses'].device
        
        seen_items = batch['slates_item_indexes'] < self.n_items
        item_embeddings = torch.zeros_like(batch['slates_item_indexes'].float() )[..., None].repeat(
            * ([1] * seen_items.ndim),
            self.embedding_dim
        )
        if self.embeddings == 'svd':
            item_embeddings[seen_items] = self.item_embeddings.to(device)[batch['slates_item_indexes'][seen_items]]
        elif self.embeddings == 'neural':
            item_embeddings[seen_items] = self.item_embeddings(batch['slates_item_indexes'][seen_items])
        elif self.embeddings == 'explicit':
            item_embeddings = batch['slates_item_embeddings']
        else:
            raise NotImplemented
        
        # item_embeddigns: (batch, sequence_pos, slate_pos, embedding)
        
#         print("batch_items", item_embeddings.shape)
        
        ## roll a batch for 1 item or it there would be a leakage
        
        item_embeddings_shifted = torch.cat(
            [
                torch.zeros_like(item_embeddings[:,0,None,:,:]),
                item_embeddings[:,:-1,:,:]
            ], 
            dim=1
        )
        
        responses_shifted = torch.cat(
            [
                torch.zeros_like(batch['responses'][:, None, 0,:]),
                batch['responses'][:,:-1,:]
            ], 
            dim=1
        )
        
        batchwise_num_items = responses_shifted.sum(-1).cumsum(-1)
#         print("batch nclicks", batchwise_num_items)
#         print("batch responses", responses_shifted)
#         print("item_embeddings", item_embeddings_shifted)
#         print('shapes', item_embeddings_shifted.shape, responses_shifted.shape )
        consumed_embedding = item_embeddings_shifted * responses_shifted[..., None]
#         print("batch consumed", consumed_embedding, consumed_embedding.sum(-2), consumed_embedding.sum(-2).cumsum(-2) )
        
        consumed_embedding = consumed_embedding.sum(-2).cumsum(-2)
#         print(consumed_embedding)
#         print("batch users", consumed_embedding.shape)

        if self.agg == 'mean':
            nnz = batchwise_num_items > 0
#             print('nnz', nnz.shape)
#             print('consumed', consumed_embedding.shape, consumed_embedding[nnz].shape)
#             print('bwise nitems',  batchwise_num_items.shape, batchwise_num_items[nnz].shape )
            consumed_embedding[nnz] /= batchwise_num_items[nnz].unsqueeze(-1)
    
        return item_embeddings, consumed_embedding.nan_to_num(0.)


class ItemEmbedding(torch.nn.Module):
    """
    Family of embeddings for items, base class corresponds to explicit item embeddings.
    User embeddings are obtained by aggregating previously consumed item embeddings
    by this user.
    """
    def __init__ (self, user_aggregate='mean', embedding_dim=32):
        super().__init__()
        self.agg = user_aggregate
        self.embedding_dim = embedding_dim

    def forward(self, batch):
        """
            Dummy zero embeddings
        """
        item_embeddings = batch['slates_item_embeddings']

        return self.embed(item_embeddings, batch)

    def embed (self, item_embeddings, batch):
        batch_size, max_sequence = batch['responses'].shape[:2]
        device = batch['responses'].device

        ## roll a batch for 1 item or it there would be a leakage
        item_embeddings_shifted = torch.cat(
            [
                torch.zeros_like(item_embeddings[:,0,None,:,:]),
                item_embeddings[:,:-1,:,:]
            ],
            dim=1
        )

        responses_shifted = torch.cat(
            [
                torch.zeros_like(batch['responses'][:, None, 0,:]),
                batch['responses'][:,:-1,:]
            ],
            dim=1
        )

        batchwise_num_items = responses_shifted.sum(-1).cumsum(-1)
        consumed_embedding = item_embeddings_shifted * responses_shifted[..., None]
        consumed_embedding = consumed_embedding.sum(-2).cumsum(-2)

        if self.agg == 'sum':
            pass
        elif self.agg == 'mean':
            nnz = batchwise_num_items > 0
            consumed_embedding[nnz] /= batchwise_num_items[nnz].unsqueeze(-1)
        else:
            raise ValueError(f'Unknown aggregation {self.agg}')

        return item_embeddings, consumed_embedding.nan_to_num(0.)

class IndexItemEmbeddings(ItemEmbedding):
    def __init__(self, n_items, embedding_dim):
        """
            Embeddings for item indexes.
        """
        super().__init__(embedding_dim=embedding_dim)
        self.embeddings = torch.nn.Embedding(
                    n_items + 1,
                    embedding_dim
                )
    def forward(self, batch):
        seen_items = batch['slates_item_indexes'] < self.embeddings.num_embeddings
        # index 0 for previously unseen tokens
        item_embeddings = torch.zeros_like(batch['slates_item_indexes'].float() )[..., None].repeat(
            * ([1] * seen_items.ndim),
            self.embeddings.embedding_dim
        )
        item_embeddings[seen_items] = self.embeddings(batch['slates_item_indexes'][seen_items])

        return self.embed(item_embeddings, batch)

class CategoricalItemEmbeddings(ItemEmbedding):
    """
    Used to embed categorical stuff, like item indexes and category codes.
    """
    def __init__(self, train_data, user_aggregate='mean', columnwise_embedding_dim=8):
        super().__init__()
        """
        train_data: np.array of shape (SOMETHING, num_categorical_features INDEXES)
                    for all items features available. Values not presented
                    in this array are threaten as unknown items.
        """
        self.embeddings = torch.nn.ModuleList([])
        for i in range(train_data.shape[-1]):
            ids = [-1] + list(np.unique(train_data[..., i]))
            self.embeddings.append(
                torch.nn.Embedding(
                    len(ids),
                    columnwise_embedding_dim
                )
            )
        self.embedding_dim = sum([
            module.embedding_dim
            for module in self.embeddings
        ])

    def forward(self, batch):
        """
            Embedding = concatentaion of all categorical embeddings
        """
        bce = batch['slates_item_categorical'].shape[-1]
        ece = len(self.embeddings)
        assert bce==ece, f"number of categorical embeddings {bce} is not the same as in train data {ece}"
        item_embeddings = []
        for i in range(len(self.embeddings)):
            seen_items = batch['slates_item_categorical'][...,i] < self.embeddings[i].num_embeddings

            # index 0 for previously unseen tokens
            this_feature_embeddings = torch.zeros_like(batch['slates_item_indexes'].float() )[..., None].repeat(
                * ([1] * seen_items.ndim),
                self.embeddings[i].embedding_dim
            )
            this_feature_embeddings[seen_items] = self.embeddings[i](batch['slates_item_categorical'][...,i][seen_items])
            item_embeddings.append(this_feature_embeddings)

        item_embeddings = torch.cat(item_embeddings, dim=-1)

        return self.embed(item_embeddings, batch)

class SVDItemEmbeddings(ItemEmbedding):
    """
    Static embeddings from svd decomposition of train matrix.
    """
    def __init__(self, train_matrix, user_aggregate='mean', embedding_dim = 32):
        super().__init__(embedding_dim=embedding_dim)
        _, _, self.item_embeddings = randomized_svd(
            train_matrix,
            n_components=embedding_dim,
            n_iter=4,
            power_iteration_normalizer='QR'
        )
        self.encoder = torch.tensor(self.item_embeddings.T).float()
        self.n_items = self.item_embeddings.shape[-1]

    def forward(self, batch):
        """
            Embedding = concatentaion of all categorical embeddings
        """
        device = batch['slates_item_indexes'].device
        seen_items = batch['slates_item_indexes'] < self.n_items
        item_embeddings = torch.zeros_like(batch['slates_item_indexes'].float() )[..., None].repeat(
            * ([1] * seen_items.ndim),
            self.embedding_dim
        )
        item_embeddings[seen_items] = self.encoder.to(device)[batch['slates_item_indexes'][seen_items]]

        return self.embed(item_embeddings, batch)

class MixedEmbeddings(torch.nn.Module):
    def __init__ (self, *embedding_modules):
        super().__init__()
        self.embeddings = nn.ModuleList(embedding_modules)
        self.embedding_dim = sum([module.embedding_dim for module in self.embeddings])

    def forward(self, batch):
        """
            Concatenate embeddings from given embedding layers
        """
        item_embeddings = []
        user_embeddings = []

        for module in self.embeddings:
            items, users = module(batch)
            item_embeddings.append(items)
            user_embeddings.append(users)

        return torch.cat(item_embeddings, axis=-1), torch.cat(user_embeddings, axis=-1)


if __name__ == '__main__':
    from datasets import RL4RS, ContentWise, DummyData
    d = DummyData()
    dummy_loader, dummy_matrix = get_dummy_data(d)
    for batch in dummy_loader:
        break
    external = ItemEmbedding()
    index_embeddings = IndexItemEmbeddings(d.n_items, embedding_dim = 32)
    category_embeddings = CategoricalItemEmbeddings(d.item_cateforical)
    svd_embeddings = SVDItemEmbeddings(dummy_matrix, embedding_dim=2)

    me = MixedEmbeddings(
        external,
        index_embeddings,
        svd_embeddings,
        category_embeddings
    )
    print(me(batch)[0].shape, me(batch)[1].shape)
