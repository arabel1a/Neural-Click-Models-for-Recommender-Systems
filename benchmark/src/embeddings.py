from sklearn.utils.extmath import randomized_svd
import os
import torch
import random
import datetime
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from src.utils import train, get_dummy_data, get_train_val_test_tmatrix_tnumitems, get_svd_encoder


class RecsysEmbedding(torch.nn.Module):
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
            item_embeddings[seen_items] = self.item_embeddings[batch['slates_item_indexes'][seen_items]]
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

if __name__ == '__main__':    
    from src.datasets import RL4RS, ContentWise, DummyData
    d = DummyData()
    dummy_loader, dummy_matrix = get_dummy_data(d)
    emb = RecsysEmbedding(d.n_items, dummy_matrix, embeddings='explicit')

    for batch in dummy_loader:
        break

    emb(batch)[0], emb(batch)[1]