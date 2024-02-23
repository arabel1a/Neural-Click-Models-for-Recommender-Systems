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
from slatewise import MF, LogisticRegression, SlatewiseGRU, SlatewiseAttention
from sessionwise import SessionwiseGRU, SessionwiseAttention, AttentionGRU, SCOT

from collections import namedtuple
Metrics = namedtuple('metrics', ['rocauc', 'f1', 'accuracy'])


class ResponseModel:
    def __init__(self, model, embeddings, **kwargs):
        self._embeddings = embeddings
        self.model_name = model
        self.threshold = 0.5
        self.auc = AUROC(task='binary')

        if model == 'MF':
            self._model = MF(embeddings, **kwargs)
        elif model == 'LogisticRegression':
            self._model = LogisticRegression(embeddings, **kwargs)
        elif model == 'SlatewiseAttention':
            self._model = SlatewiseAttention(embeddings, **kwargs)
        elif model == 'SessionwiseAttention':
            self._model = SessionwiseAttention(embeddings, **kwargs)
        elif model == 'AttentionGRU':
            self._model = AttentionGRU(embeddings, **kwargs)
        elif model == 'SCOT':
            self._model = SCOT(embeddings, **kwargs)
        elif model == 'SlatewiseGRU':
            self._model = SlatewiseGRU(embeddings, **kwargs)
        elif model == 'SessionwiseGRU':
            self._model = SessionwiseGRU(embeddings, **kwargs)
        else:
            raise ValueError(f'unknown model {model}')
    
    def _val_epoch(self, data_loader, silent=True):
        # run model on dataloader, compute auc
        self.auc.reset()
        self._model.eval()
        for batch in tqdm(data_loader, desc='evaluating...', disable=silent):
            batch = {k:v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                prediction_scores = torch.sigmoid(self._model(batch))
            corrects = (batch['responses'] > 0).float()
            mask = batch['out_mask']
            self.auc(prediction_scores[mask].cpu(), corrects[mask].cpu())

    def _train_epoch(self, data_loader, optimizer, criterion):
        loss_accumulated = 0.
        for batch in data_loader:
            batch = { k:v.to(self.device) for k, v in batch.items() }
            mask = batch['slates_mask']
            corrects = (batch['responses'] > 0).float()

            scores = self._model(batch)        
            loss = criterion(scores[mask], corrects[mask],)
            loss_accumulated += loss.detach().cpu().item()
            loss.backward()
            clip_grad_norm_(self._model.parameters(), 1.)
            optimizer.step()

            metric_mask = batch['out_mask']
            self.auc(
                torch.sigmoid(scores[metric_mask]).detach().cpu(), 
                corrects[metric_mask].detach().cpu()
            )

        return loss_accumulated

    def _train(self, train_loader, val_loader, device='cuda', lr=1e-3, 
        num_epochs=100, silent=True, early_stopping=7, debug=False, **kwargs):   
    
        if early_stopping == 0:
            early_stopping = num_epochs

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        epochs_without_improvement = 0
        criterion = torch.nn.functional.binary_cross_entropy_with_logits
        self.ebar = tqdm(range(num_epochs), desc='train')
        self.device = device
        self._model.to(device)
        self.best_val_scores = self.evaluate(val_loader, silent=silent)
        best_train_loss = 999999999.
        
        for epoch in self.ebar:
            self._model.train()
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.ebar.set_description(f"train... loss:{train_loss}")
            preds, target = torch.cat(self.auc.preds), torch.cat(self.auc.target)
            train_f1 = self._fit_threshold_f1(preds, target)
            self.auc.reset()

            val_scores = self.evaluate(val_loader, silent=silent)
            epochs_without_improvement += 1
            if val_scores >= self.best_val_scores:
                self.best_model = deepcopy(self._model)
                best_val_scores = val_scores            
            
            if best_train_loss > train_loss:
                epochs_without_improvement = 0
                best_train_loss = train_loss 
            
            # early stopping
            if (epochs_without_improvement >= early_stopping 
                # or val_scores == (1., 1., 1.)
                ):
                print('Early stopping')
                break
        self._model = self.best_model

    def evaluate(self, datalaoder, silent=False):
        self._val_epoch(datalaoder, silent=silent)
        preds, target = torch.cat(self.auc.preds), torch.cat(self.auc.target)
        return Metrics(
            self.auc.compute(),
            f1_score(preds, target, task='binary', threshold = self.threshold, average='macro'),
            accuracy(preds, target, task='binary', threshold = self.threshold)
        )

    def fit(self, dataset: RecommendationData, batch_size, val_frac=0.8, **kwargs):
        if self.model_name == 'MF':
            return self
        train_data, val_data = dataset.split_by_users(val_frac, seed=123)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size, collate_fn=collate_recommendation_datasets, shuffle=False)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size, collate_fn=collate_recommendation_datasets,  shuffle=True)

        self._train(train_loader, val_loader, silent=True, **kwargs )

    def transform(self, dataset: RecommendationData, batch_size, **kwargs):
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_recommendation_datasets,
            shuffle=False, **kwargs
        )        
        self._model.eval()
        scores, recommendation_indexes = [], []
        for batch in tqdm(loader, desc='transform...'):
            batch = {k:v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                prediction_scores = torch.sigmoid(self._model(batch))
            scores.append(prediction_scores.detach().cpu().squeeze(0))
            recommendation_indexes.append(
                batch['recommendation_indexes'].cpu().squeeze(0)
            )
        predicted_probs = np.zeros(dataset.recommendations.shape)
        for batch_score, batch_index in zip(scores, recommendation_indexes):
            for session, r_index in zip(batch_score, batch_index):
                for recommendation_no, ind in enumerate(r_index):
                    ind = ind.item()
                    predicted_probs[ind] = session[recommendation_no].numpy()
        
        dataset.predicted_probs = predicted_probs
        dataset.predicted_responses = (predicted_probs >= self.threshold).astype(int)
        return dataset

    def to(self, device: str):
        self._model = self._model.to(device)
        self.device = device

    def _fit_threshold_f1(self, preds, target):
        best_f1 = 0.0
        for thold in np.arange(0., 1., 0.01):
            f1 = f1_score(
                preds, target, task='binary', threshold = thold, average='macro'
            ).item()
            if f1 > best_f1:
                self.threshold = thold
                best_f1 = f1
        return best_f1

if __name__ == '__main__':
    from embeddings import IndexItemEmbeddings
    from datasets import DummyData

    dataset = DummyData()
    embeddings = IndexItemEmbeddings(dataset.n_items, embedding_dim=32)
    model = ResponseModel('LogisticRegression', embeddings)
    model.fit(dataset, batch_size=256)
    dataset = model.transform(dataset)
    slates_mask = dataset.recommendations != dataset.NO_ITEM
    print('Real responses: \n', (dataset.responses[slates_mask] > 0).astype(int))
    print('Predicted probabilities:\n', dataset.predicted_probs[slates_mask])
    print('Predicted responses: \n', dataset.predicted_responses[slates_mask])
