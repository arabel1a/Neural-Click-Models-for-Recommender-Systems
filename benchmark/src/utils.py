import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import pickle
import re
import gc
from collections import Counter, OrderedDict
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from sklearn.utils.extmath import randomized_svd


from torchmetrics import AUROC, F1Score, Accuracy
from torchmetrics.functional.classification import binary_f1_score, binary_accuracy  

def get_dummy_data(d):
    dummy_loader = torch.utils.data.DataLoader(
            d,
            batch_size=2,
            collate_fn=collate_recommendation_datasets,
            shuffle=False,
    )
    return dummy_loader, d.user_item_matrix

def get_train_val_test_tmatrix_tnumitems(dataset, train_frac=0.8, val_vs_test_frac=0.5, batch_size=256, seed=None, **kwargs):
    """
        Splits datset into train, test and val parts by given fractions.
        The `train_frac` of users in dataset will be train set, and the rest is splitted 
        into val and test in `val_vs_test_frac` proportion.
    """
    train, rem = dataset.split_by_users(train_frac, seed=seed)
    val, test = rem.split_by_users(val_vs_test_frac, seed=seed)
    
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        collate_fn=collate_recommendation_datasets,
        shuffle=True,
        **kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val,
        batch_size=batch_size,
        collate_fn=collate_recommendation_datasets,
        shuffle=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        collate_fn=collate_recommendation_datasets,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader,  train.user_item_matrix, train.n_items


def map_array(arr, mapping):
    """
        Map values of array with dict
    """
    k = np.array(list(mapping.keys()))
    v = np.array(list(mapping.values()))
    sidx = k.argsort()

    k = k[sidx]
    v = v[sidx]
    
    idx = np.searchsorted(k,arr.ravel()).reshape(arr.shape)
    idx[idx==len(k)] = 0
    mask = k[idx] == arr
    out = np.where(mask, v[idx], 0)
    return out

def get_svd_encoder(dataset, embedding_dim):
    """
        Get item embeddings from matrix factorization.
    """
    _, _, item_embeddings = randomized_svd(dataset.user_item_matrix, 
                                           n_components=embedding_dim, 
                                           n_iter=4,
                                           power_iteration_normalizer='QR')
    return item_embeddings.T


def collate_recommendation_datasets(batch, padding_value = 0):
    """
    Defines, how data samples are batched (respecting different session lengths)
    Simply padded for now and mask is returned.
    """
    batch_size = len(batch)
    
    # lengths
    batch_lengths = [i['length'] for i in batch]
    batch_lengths = torch.tensor(np.stack(batch_lengths), dtype=torch.long)
    max_sequence_len = batch_lengths.max().item()
    
    in_lengths = [i['in_length'] for i in batch]
    in_lengths = torch.tensor(np.stack(in_lengths), dtype=torch.long)

    
    # slates: sequences, recommendation, recommended_items)
    # shape: batch_size, max_sequence_len, max_slate_size
    batch_slates_item_ids = [
        torch.tensor(i['slates_item_ids'][:max_sequence_len, :], dtype=torch.long)
        for i in batch 
    ]
    batch_slates_item_ids = torch.nn.utils.rnn.pad_sequence(
        batch_slates_item_ids, padding_value=padding_value, batch_first=True)

    batch_slates_item_indexes = [
        torch.tensor(i['slates_item_indexes'][:max_sequence_len, :], dtype=torch.long)
        for i in batch 
    ]
    batch_slates_item_indexes = torch.nn.utils.rnn.pad_sequence(
        batch_slates_item_indexes, padding_value=padding_value, batch_first=True)
    
    # slate masks: True recommended items, False for padding tokens (both sequence padding and slate padding)
    # shape: batch_size, max_sequence_len, max_slate_size
    batch_slates_masks = [
        torch.tensor(i['slates_mask'][:max_sequence_len, :], dtype=torch.bool)
        for i in batch 
    ]
    batch_slates_masks = torch.nn.utils.rnn.pad_sequence(
        batch_slates_masks, 
        padding_value=False,
        batch_first=True
    )

    # in_out mask, same as previous
    batch_in_mask = torch.zeros_like(batch_slates_masks, dtype=bool)
    batch_out_mask = torch.zeros_like(batch_slates_masks, dtype=bool)
    for i, session in enumerate(batch):
        batch_in_mask[i, : session['in_length'], :] = True
        batch_out_mask[i, session['in_length'] : , :] = True
    batch_in_mask = batch_slates_masks & batch_in_mask
    batch_out_mask = batch_slates_masks & batch_out_mask

    # slate_item_embeddings (like slate, but embeddings instead of item ids)
    # resulting shape (batch_size, max_sequence_len, max_slate_size, embedding dim)
    batch_slates_item_embeddings = torch.empty(1)
    if batch[0]['slates_item_embeddings'] is not None:
        batch_slates_item_embeddings = [
            torch.tensor(i['slates_item_embeddings'][:max_sequence_len, :, :], dtype=torch.float)
            for i in batch 
        ]
        batch_slates_item_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_slates_item_embeddings, padding_value=padding_value, batch_first=True)

    # slate_item_categorical (like slate, but embeddings instead of item ids)
    # resulting shape (batch_size, max_sequence_len, max_slate_size, embedding dim)
    batch_slates_item_categorical = torch.empty(1)
    if batch[0]['slates_item_categorical'] is not None:
        batch_slates_item_categorical = [
            torch.tensor(i['slates_item_categorical'][:max_sequence_len, :, :], dtype=torch.long)
            for i in batch
        ]
        batch_slates_item_categorical = torch.nn.utils.rnn.pad_sequence(
            batch_slates_item_categorical, padding_value=padding_value, batch_first=True)

    # responses: number of clicks per recommended item
    # slates: resulting shape (batch_size, max_sequence_len, max_slate_size)
    batch_responses = [
        torch.tensor(i['responses'][:max_sequence_len, :], dtype=torch.long)
        for i in batch 
    ]
    batch_responses = torch.nn.utils.rnn.pad_sequence(
        batch_responses, padding_value=padding_value, batch_first=True)

    # user_embeddings
    # resulting shape (batch_size, max_sequence_len, embedding dim)
    batch_user_embeddings = torch.empty(1)
    if batch[0]['user_embeddings'] is not None:
        batch_user_embeddings = [
            torch.tensor(i['user_embeddings'][:max_sequence_len, :], dtype=torch.float)
            for i in batch 
        ]
        batch_user_embeddings = torch.nn.utils.rnn.pad_sequence(
            batch_user_embeddings, padding_value=padding_value, batch_first=True)

    # user_ids & indexes
    user_ids = [
        torch.tensor(i['user_ids'][: max_sequence_len], dtype = torch.long)
        for i in batch
    ]
    user_ids = torch.nn.utils.rnn.pad_sequence(
        user_ids, padding_value=padding_value, batch_first=True
    )

    # reccommendation_indexes
    recommendation_indexes = [
        torch.tensor(i['recommendation_idx'][: max_sequence_len], dtype = torch.long)
        for i in batch
    ]
    recommendation_indexes = torch.nn.utils.rnn.pad_sequence(
        recommendation_indexes, padding_value=-1, batch_first=True
    )
    
    user_indexes = [
        torch.tensor(i['user_indexes'][: max_sequence_len], dtype = torch.long)
        for i in batch
    ]
    user_indexes = torch.nn.utils.rnn.pad_sequence(
            user_indexes, padding_value=padding_value, batch_first=True)

    return {
          'slates_item_ids': batch_slates_item_ids,
          'slates_item_indexes': batch_slates_item_indexes,
          'slates_item_embeddings': batch_slates_item_embeddings,
          'slates_item_categorical': batch_slates_item_categorical,
          'slates_mask': batch_slates_masks,
          'responses': batch_responses,
          'in_mask' : batch_in_mask,
          'out_mask' : batch_out_mask,
          # 'responses_masks': self.responses[session_metadata['recommendation_idx']] > 0,
          'length': batch_lengths,
          'in_length': in_lengths,
          'user_embeddings': batch_user_embeddings,
          'user_ids': user_ids,
          'user_indexes' : user_indexes,
          'recommendation_indexes': recommendation_indexes
        }

def evaluate_model(model, data_loader, device='cuda', threshold=0.5, silent=False, debug=False, **kwargs):
    # run model on dataloader, compute metrics
    f1 = F1Score(task='binary',average='macro', threshold = threshold).to(device)
    acc = Accuracy(task='binary', threshold = threshold).to(device)
    auc = AUROC(task='binary').to(device)
    
    model.to(device)
    model.eval()
    
    for batch in tqdm(data_loader, desc='evaluating...', disable=silent):
        batch = {k:v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            prediction_scores = torch.sigmoid(model(batch))
        corrects = (batch['responses'] > 0).float()
        mask = batch['out_mask']
        
        # # prediction_shape: (batch_size, max_sequence, 'max_slate, 2)   
        f1(prediction_scores[mask], corrects[mask])
        auc(prediction_scores[mask], corrects[mask])
        acc(prediction_scores[mask], corrects[mask])
        if debug: 
            print('\r', prediction_scores[mask], corrects[mask])


    gc.collect()
    return {
        'f1': f1.compute().item(), 
        'roc-auc': auc.compute().item(),
        'accuracy': acc.compute().item()
    }


def fit_treshold(labels, scores):
    best_f1, best_thold, acc = 0.0, 0.01, 0.0
    for thold in np.arange(1e-2, 1 - 1e-2, 0.01):
        preds_labels = scores > thold
        f1 = binary_f1_score(preds_labels, labels)
        # print(f"{thold}: {f1}")
        if f1 > best_f1:
            acc = binary_accuracy(preds_labels, labels)
            best_f1, best_thold = f1, thold
    return best_f1, acc, best_thold

def train(model, train_loader, val_loader, test_loader,  
          device='cuda', lr=1e-3, num_epochs=50, silent=False, early_stopping=None, debug=False, **kwargs):   
    if early_stopping is None:
        early_stopping = num_epochs
    model.to(device)
    best_model = model
    
    auc = AUROC(task='binary').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epochs_without_improvement = 0
    best_val_scores = evaluate_model(model, val_loader, device=device, silent=silent, debug=debug)
    best_test_scores = evaluate_model(model, test_loader, device=device, silent=silent, debug=debug)
    best_loss = 999.
    
    print(f"Test before learning: {best_test_scores}")
    ebar = tqdm(range(num_epochs), desc='train')
    
    for epoch in ebar:
        loss_accumulated = 0.
        mean_grad_norm = 0.
        model.train()

        labels = []
        preds = []
        
        gc.collect()        
        # torch.cuda.empty_cache()

        for batch in tqdm(train_loader, desc=f'epoch {epoch}', disable=silent):
            batch = {k:v.to(device) for k, v in batch.items()}
            raw_scores = model(batch)
            prediction_scores = torch.sigmoid(raw_scores)
            corrects = (batch['responses'] > 0).float()
            mask = batch['slates_mask']
            if debug: print('\n\nTest predictions:', mask, raw_scores[mask], prediction_scores[mask], corrects[mask])
            # prediction_shape: (batch_size, max_sequence, 'max_slate, 2)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                raw_scores[mask], 
                corrects[mask],
            )
            
            loss.backward()
            # print(loss.item())
            mean_grad_norm += clip_grad_norm_(model.parameters(), 1).sum().item()
            optimizer.step()
            
            loss_accumulated += loss.detach().cpu().item()
            labels.append(corrects[batch['out_mask']].detach().cpu())
            preds.append(prediction_scores[batch['out_mask']].detach().cpu())
            auc(prediction_scores[batch['out_mask']].detach().cpu(), corrects[batch['out_mask']].detach().cpu())
        
        f1, acc, thold = fit_treshold(torch.cat(labels), torch.cat(preds))
        # print(torch.concat(labels), torch.concat(preds))
        # for name, p in model.named_parameters():
        #     print(f"{name}:{p}")
        ebar.set_description(f"train... loss:{loss_accumulated}")
        val_m = evaluate_model(model, val_loader, device=device, threshold=thold, silent=silent,debug=debug, **kwargs)
        if not silent:
            print(f"Train: epoch: {epoch} | accuracy: {acc} | "
                  f"f1: {f1} | loss: {loss_accumulated} | "
                  f"auc: {auc.compute()}  | thld {thold} | grad_norm: {mean_grad_norm / len(train_loader)}")
            print(f"Val: epoch: {epoch} | accuracy: {val_m['accuracy']} | f1: {val_m['f1']} | auc: {val_m['roc-auc']}")

        epochs_without_improvement += 1
        if (val_m['roc-auc'], val_m['f1'], val_m['accuracy']) > (best_val_scores['roc-auc'], best_val_scores['f1'], best_val_scores['accuracy']) :
            best_model = deepcopy(model)
            best_val_scores = val_m
            best_test_scores = evaluate_model(model, test_loader, device=device, threshold=thold, silent=silent )
            print(f"Val update: epoch: {epoch} |"
                  f"accuracy: {best_val_scores['accuracy']} | "
                  f"f1: {best_val_scores['f1']} | "
                  f"auc: {best_val_scores['roc-auc']} | "
                  f"treshold: {thold}"
            )
            print(f"Test: "
                  f"accuracy: {best_test_scores['accuracy']} | "
                  f"f1: {best_test_scores['f1']} | "
                  f"auc: {best_test_scores['roc-auc']} | "
            )
                
        auc.reset()
        
        if best_loss > loss_accumulated:
            epochs_without_improvement = 0
            best_loss = loss_accumulated 
       
        if epochs_without_improvement >= early_stopping or (
            best_val_scores['roc-auc'] == 1. and
            best_val_scores['f1'] == 1. and
            best_val_scores['accuracy'] == 1.):
            break
    return model, best_test_scores

class Indexer:
    """
        Index <--> Id register
        default index: 0
    """
    def __init__(self, default_token=None):
        self.default_token = default_token
        self.index2id = [ default_token ]
        self.id2index = {default_token:0}
        self.counter = Counter()

    def update(self, value):
        if value in self.id2index:
            return self.id2index[value]
        else:
            idx = len(self.index2id)
            self.index2id.append(value)
            self.id2index[value] = idx
            return idx

    def get(self, value):
        if value not in self.id2index:
            return 0
        return self.id2index[value]

    def from_iter(self, column, min_occurences=5):
        counter = self.counter.update(column)
        frequent = {
            id for id in self.counter if self.counter[id] > min_occurences
        }
        frequent.discard(self.default_token)
        self.index2id = [ self.default_token ] + list(frequent)
        self.id2index = { id:index for (index, id) in enumerate(self.index2id) }
        return self

