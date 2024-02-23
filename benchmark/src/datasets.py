import numpy as np
import os
import pandas as pd
import pickle
import re
import sys
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from scipy.sparse import dok_matrix, csr_matrix
from tqdm import tqdm
from copy import deepcopy

from datetime import datetime, timedelta
from torchmetrics import AUROC, F1Score, Accuracy

current_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.append(current_dir)
from utils import map_array, Indexer

class RecommendationData(Dataset):    
    # special id (in recommendation slate) to represent 'nothing'
    # useful for situations, where slate size may vary
    NO_ITEM = -1
    
    def __init__(self, recommendations: np.array, responses: np.array, metadata:pd.DataFrame, 
                 user_item_matrix = None, user_id2index = None, item_id2index=None,
                 item_features: np.array =None, user_features: np.array =None,
                 item_categorical: np.array =None, user_categorical: np.array =None, **kwargs):
        """
        :param recommendations: NumPy array with impressions. If slate size 
                                can differ, fill last X values must contain
                                NO_ITEM. Expected shape (num_recomendations, slate_size)
                                
        :param response: NumPy array with the same shape as recommendation_id.
                         Counts a responses for each item in slate.
                         Expected shape (num_recomendations, slate_size)
                         
        :param metadata: DataFrame with response metadata. Must have the same number of 
                         rows as response. Expected shape (num_recommendations, D) where
                         D is number of columns. 4 columns requred, but can contain any
                         other information (for further usage, for like coloring graphs, etc)
                         
                         Contain following columns:
                         1. 'recommendation_idx'. Reccomendation index between 0 and 
                             num_recommendayions.
                         3,4. 'session_id' and 'user_id'. If one of session or user ids is 
                             not represented in data, let them be the euqal.
                         5. 'timestamp' for sequence-based approaches. If no timestamp 
                             in dataset, fill enumeration (i.e. sorting responses by
                             this column might produce correct order over responses.
                         6,7.  [optional] 'user_feature_idx' and 'item_feature_idx', meaning 
                             indexes of embeddings in user and item is used by current
                             response. Taken intro account only if embeddings are presented.
                             Expected values are in (0, num_user_features) for 
                             user_emb_idx and (0, num_item_featuress) for item_emb_idx.
                             
        :param user_features: [optional] Numpy array with embeddings for user.
                                Expected shape (num_user_features, ...).
        :param user_categorical: [optional] Numpy array with category features for user.
                                Expected shape (num_user_features, ...).

        :param item_features: [optional] Numpy array with embeddings for item.
                                Expected shape (num_item_features, slate_size, ...).
        :param item_categorical: [optional] Numpy array with category features for item.
                                Expected shape (num_item_features, slate_size, ...).
        """
        super().__init__()
        self.predicted_probs, self.predicted_responses = None, None
        assert recommendations.shape == responses.shape, "responses and recommmendations shapes differ"
        assert metadata.shape[0] == responses.shape[0], "metadata's row count is not equal to numer of recommendations"
        self.num_recommendations = responses.shape[0]
        
        assert 'recommendation_idx' in metadata, "recommendation_idx field is missing"
        assert (metadata['recommendation_idx'].min() == 0 and
                metadata['recommendation_idx'].max() == self.num_recommendations - 1)
        assert 'session_id' in metadata, "session_id field is missing"
        assert 'user_id' in metadata, "session_id field is missing"
        assert 'timestamp' in metadata, "timestamp field is missing"
        
        self.prepared_data_cache = {}
        
        # to avoid corruption if metadata is a view
        self.metadata = metadata.copy()

        # responses as number of clicks for each item in recommendation slate
        self.responses = responses
        
        # mask for clicked items
        self.response_mask = self.responses > 0

        # recommended item ids
        self.recommendations = recommendations
                
        # features
        self.user_features = user_features
        self.item_features = item_features
        self.item_categorical = item_categorical
        self.user_categorical = user_categorical
        if self.item_features is not None:
            self.embedding_dim = item_features.shape[-1]
        
        # filter small sequences
        self._from_metadata(self.metadata, inplace=True, **kwargs)
                
        self.n_users = len(self.metadata['user_id'].unique())
        self.n_items = len(np.unique(self.recommendations)) 

        
        self.user_item_matrix = user_item_matrix
        self.user_id2index = user_id2index
        self.item_id2index = item_id2index
        if user_item_matrix is None:
            self.build_affinity_matrix()
            
        self.recommendations_indexes = map_array(self.recommendations, self.item_id2index)
        self.metadata['user_idx'] = map_array(self.metadata['user_id'].to_numpy(), self.user_id2index)

        # sorting & indexing for faster loading
        self.sessions = self.metadata['session_id'].unique().tolist()
        self.metadata.set_index(['session_id', 'timestamp'], drop=False, inplace=True)
        self.metadata.sort_index(inplace=True)
        
    def dump(self, path):
        """
        Saves dataset on a disk.
        :param path: where data is saved.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """
        Loads dataset from file, creqated by `dump` method.
        :param path: where data is located
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def _from_metadata(self, metadata, inplace=False, **kwargs):
        """
        Filters dataset based on metadata. All recommendations not
        in the metadata are dropped, and the rest is reindexed.
        
        :param metadata: subset of self.metadata
        :param inplace: If True, self is modified. 
                        Else return another object.
                        
        :param keep_indexes: If True, does not update affinity matrix 
                             indexes. Defaults to False.
        """

        # first, filter sessions
        m = self._filter_session_length(metadata, **kwargs).copy()
                
        # new recomendations, responses & metadata
        rec = self.recommendations[m['recommendation_idx'],:]
        resp = self.responses[m['recommendation_idx'],:]
        
        # drop items & users without impressions
        user_features, item_features, user_categorical, item_categorical = (
            None, None, None, None
        )
        if self.item_features is not None:
            item_features = self.item_features[m['item_feature_idx'],:]
        if self.item_categorical is not None:
            item_categorical = self.item_categorical[m['item_feature_idx'],:]
        if self.user_features is not None:
            user_features = self.user_features[m['user_feature_idx'],:]
        if self.user_categorical is not None:
            user_categorical = self.user_categorical[m['user_feature_idx'],:]

            
        m.reset_index(drop=True, inplace=True)
        m['recommendation_idx'] = m.index
        m['user_feature_idx'] = m.index
        m['item_feature_idx'] = m.index
                
        if inplace:
            self.metadata = m
            self.recommendations = rec
            self.responses = resp
            self.item_features = item_features
            self.user_features = user_features
            self.item_categorical = item_categorical
            self.user_categorical = user_categorical
        else:
            return RecommendationData(
                rec,
                resp,
                m,
                item_features = item_features,
                user_features = user_features,
                item_categorical = item_categorical,
                user_categorical = user_categorical,
                user_item_matrix = self.user_item_matrix,
                item_id2index = self.item_id2index,
                user_id2index = self.user_id2index,
                **kwargs
            )
        
    
    def _filter_session_length(self, metadata, min_session_len = 1, max_session_len=256, **kwargs):
        """
        Removes too short and too long sessions FROM METADATA ONLY.
        Use with 'from_metadata' to filter responses and recommendations too.
        """
        sequence_lengths = metadata['session_id'].value_counts()
        good_sessions = sequence_lengths[ 
            (sequence_lengths >= min_session_len) & 
            (sequence_lengths <= max_session_len)
        ].index
        new_metadata = metadata[ metadata.session_id.isin(good_sessions)]
        return new_metadata
                
    def filter_slate_size(self, size, drop=False):
        if drop:
            raise NotImplementedError
        else:
            self.recommendations = self.recommendations[...,:size]
            self.response_mask = self.response_mask[...,:size]
            self.responses = self.responses[...,:size]
            
    def split_by_users(self, ratio = 0.8, seed=None, **kwargs):
        """
        :param float ratio:  frac of users TODO change to frac of rows
        :return: two RecommendationData classes, obtained 
                from current by splitting current by users.
        """
        permutation = self.metadata['user_id'].unique()
        np.random.seed(seed)
        np.random.shuffle(permutation)
        train_users = permutation[:int(self.n_users * ratio)]
        test_users = permutation[int(self.n_users * ratio):]
        m1 = self.metadata[self.metadata['user_id'].isin(train_users)].copy()
        m2 = self.metadata[self.metadata['user_id'].isin(test_users)].copy()
        return (
            self._from_metadata(m1, **kwargs), 
            self._from_metadata(m2, **kwargs)
        )
    
    def build_affinity_matrix(self):
        """ 
        Builds an user-item iteraction matrix for further embrddings extraction.
        """
        print('biulding affinity matrix...')
        self.user_index2id = np.unique(self.metadata.user_id).tolist()
        self.user_id2index = {id:ind for ind, id in enumerate(self.user_index2id) }
                
        self.item_index2id = np.unique(self.recommendations).tolist()
        self.item_id2index = {id:ind for ind, id in enumerate(self.item_index2id) }
        
        user_item_matrix = dok_matrix((self.n_users, self.n_items))
        for i, row in tqdm(self.metadata.iterrows()):
            user = self.user_id2index[row['user_id']]
            idx = row['recommendation_idx']
            for position, item in enumerate(self.recommendations[idx,:]):
                if item == self.NO_ITEM: 
                    continue
                item = self.item_id2index[item]
                if (user, item) not in user_item_matrix:
                    user_item_matrix[user, item] = 0
                user_item_matrix[user, item] += self.responses[idx, position]
        self.user_item_matrix = user_item_matrix.tocsr()

    def __getitem__(self, idx):
        """
            Gets one datapoint (a history of interactions for single session).
            In what follows, define:
                * R -- number of recommendations for this
                * S - slate size
                * Eu, Ei - embedding dim for users and items
            Datapoint is a dictionary with the following content:

            Item data:
                'slates_item_ids': np.array with shape (R, S). Cell (i, j)
                                contains an id of item which was recommended
                                at j-th position of i-th slate.
                'slates_item_indexes': np.array with shapr (R, S). Same as previous,
                                but with indexes (0...N) instead if ids. Used to
                                index embeddings: nn.Embeddings nor scipy.sparse
                                can not be used with custom index.
                'slates_item_embeddings': np.array with shape (R, S, Ei).
                                Contains embeddings for each recommended item.
            User data:
                'user_ids': np.array of shape (R). User id, repeaten R times.
                'user_indexes': np.array of shape (R). User index, repeaten R times.
                'user_embeddings': np.array with shape (R, Eu). User embeddings for
                                   each iteraction.
            Iteraction data:
                'slates_mask': np.array with shapr (R, S). True for recommended items,
                                False for placeholder.
                'responses': np.array with shape (R, S). Cell (i, j)
                                contains an id number of iteractions item
                                at j-th position of i-th slate.
                'session': unique session identifier. Not used.
                'length': int. R.
                'in_length': int. Used in test datasets to fit on user before evaluation.
                             0.8 * R usually.
        """
        if idx in self.prepared_data_cache:
            return self.prepared_data_cache[idx]
        session = self.sessions[idx]
        if session in self.metadata.index:
            session_metadata = self.metadata.loc[[session]]
        else:
            session_metadata = self.metadata[
                self.metadata['session_id'] == session
            ].sort_values('timestamp')
        in_size = int(0.8 * len(session_metadata))
        self.prepared_data_cache[idx] = {
            'slates_item_ids': self.recommendations[session_metadata['recommendation_idx']],
            'slates_item_indexes':self.recommendations_indexes[session_metadata['recommendation_idx']],
            'slates_item_embeddings': None if self.item_features is None else self.item_features[session_metadata['item_feature_idx']],
            'slates_item_categorical': None if self.item_categorical is None else self.item_categorical[session_metadata['item_feature_idx']],
            'slates_mask': (self.recommendations[session_metadata['recommendation_idx']] != self.NO_ITEM),
            'responses': self.responses[session_metadata['recommendation_idx']],
            'response_timestamps': None, # TODO
            'in_length': in_size,
            'length': len(session_metadata),
            'user_embeddings': None if self.user_features is None else self.user_features[session_metadata['user_feature_idx']],
            'user_categorical': None if self.user_categorical is None else self.user_categorical[session_metadata['user_feature_idx']],
            'user_ids': session_metadata['user_id'].to_numpy(),
            'user_indexes': session_metadata['user_idx'].to_numpy(),
            'session':session,
            'recommendation_idx': session_metadata['recommendation_idx'].to_numpy()
        }

        return self.prepared_data_cache[idx]

    def __len__(self):
        return len(self.sessions)
    
    def __repr__(self):
        return (
            "\nRecomendation Dataset\n"
            f"users: \t{self.n_users}\n"
            f"items: \t{self.n_items}\n"
            f"recommendations: {self.recommendations.shape[0]}\n"
            # f"mean session length: {0}\n"
            # f"mean clicks per recommendation length: {0}\n"
            # f"mean recommendation length: {0}\n"
        )

    def to_iteraction_matrix(self):
        result = []
        for _, row in self.metadata.iterrows():
            slate = self.recommendations[row.recommendation_idx]
            for slate_pos, item_id in enumerate(slate):
                if item_id == self.NO_ITEM: break
                res = row.to_dict()
                res['response'] = self.responses[row.recommendation_idx, slate_pos]
                if self.predicted_probs is not None:
                    res['predicted_prob'] = self.predicted_probs[row.recommendation_idx, slate_pos]
                if self.predicted_responses is not None:
                    res['predicted_response'] = self.predicted_responses[row.recommendation_idx, slate_pos]
                if self.item_features is not None:
                    res['item_features'] = list(self.item_features[row.item_feature_idx, slate_pos])
                if self.item_categorical is not None:
                    res['item_categorical'] = list(self.item_categorical[row.item_feature_idx, slate_pos])
                if self.user_categorical is not None:
                    res['user_categorical'] = list(self.user_categorical[row.user_feature_idx, slate_pos])
                if self.user_features is not None:
                    res['user_features'] = list(self.user_features[row.user_feature_idx])
                result.append(res)
        return pd.DataFrame(result)

class DummyData(RecommendationData):
    """
    Simple data for self-check and play around
    """
    def __init__(self):
        recommendations = np.array([
            [1, 999, 3],
            [3, 999, -1],
            [0, 0, -1] 
        ])
        # число кликов
        responses = np.array([
            [0, 1, 1],
            [4, 0, 0],
            [0, 0, 0]
        ])
        
        user_embeddings = np.array([
            [1., 1.],
            [-1., -1.],
        ])

        item_embeddings = np.array([
            [
                [1., 1], 
                [999, 999], 
                [3, 3]
            ],
            [
                [3, 3], 
                [999, 999],
                [0, 0]
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0]
            ] 
        ])
        item_categorical = np.array([
            [
                [1, 3],
                [1, 3],
                [3, 3]
            ],
            [
                [3, 3],
                [1, 1],
                [0, 0]
            ],
            [
                [1, 3],
                [2, 2],
                [3, 1]
            ]
        ])

        metadata = pd.DataFrame({
            'recommendation_idx' : [0, 1, 2],
            'user_id': [13, 13, 0],
            'session_id': [0, 0, 777],
            'timestamp': [1,2,3],
            'user_feature_idx': [0, 0, 1],
            'item_feature_idx': [0, 1, 2],
        })
        super().__init__(
            recommendations,
            responses,
            metadata,
            user_features=user_embeddings,
            item_features=item_embeddings,
            item_categorical=item_categorical
        )


class RL4RS(RecommendationData):
    """
    Data structure: 
    1. Item embeddings (obfuscated features) file:
        'item_id' -- item id, to join with recommendations file
        'item_vec' -- string of features, separated by ','
        'price' -- price (constant for item)
        'special_item', 'location' -- ignored for now, some boolean features
    2. Recomendations and responses single csv file. 
        'timestamp' -- TODO check format, it is not utc
        'exposed_items' -- string with recommended item ids, split by ','
        'user_feedback' -- string with 1 for 'clicked', 0 for 'not clicked
                            for each item in exposed items.
        'user_protrait' -- obfuscated persistent features
         
        Other fields are ignored, but some of'em sounds interesting:
        user click history and context 
    """
    def __init__(self, path, which='rl4rs_dataset_a_rl.csv', nrows=None, **kwargs):
    # if True:
        # metadata
        print('reading files ...')
        metadata = pd.read_csv(os.path.join(path, which), nrows=nrows, delimiter='@')
        metadata['user_id'] = metadata['session_id']

        # recommendations & responses
        print('preprocessing ...')
        recommendations = np.array(metadata.exposed_items.str.split(',', expand=True)).astype(int)
        clicks = np.array(metadata.user_feedback.str.split(',', expand=True)).astype(int)
        metadata['recommendation_idx'] = metadata.index
        metadata.reset_index(inplace=True)
       
        # user features
        user_features = metadata['user_protrait'].str.split(',', expand=True).astype(float).to_numpy()
        metadata['user_feature_idx'] = metadata.index

        # item features
        item_info = pd.read_csv(os.path.join(path, 'item_info.csv'), delimiter=' ')
        item_features = item_info.item_vec.str.split(',', expand=True).astype(float).to_numpy()
        item_id2index = item_info['item_id'].reset_index().set_index('item_id').to_dict()['index']
        id2index = np.vectorize(lambda x: item_id2index[x], cache=True)
        recommended_item_indexes = id2index(recommendations)
        item_features = item_features[recommended_item_indexes]
        metadata['item_feature_idx'] = metadata.index

        print('preprocessing done')
        super().__init__(recommendations, clicks, metadata, user_features=user_features, item_features=item_features, **kwargs)


class ContentWise(RecommendationData):
    """
    Consists of following files: 
    1. interactions.csv.gz:
        'utc_ts_milliseconds' -- timestamp
        'user_id' -- user id
        'item_id', 'series_id' -- video id. video serie id
        'recommendation_id' - key to match with recommendations
        
        Columns 'episode_number', 'series_length', 'item_type', 'interaction_type',	
        'vision_factor', 'explicit_rating' are ignored, but may be usefull in future
        especially last three of them

    2. impressions-direct-link.csv.gz:
            'recommendation_id' -- impression id to match with responses	
         	'recommendation_list_length'
            'recommended_series_list' -- string represenation of python list of serie ids
            'row_position' - ignored
            
    3. impressions-non-direct-link.csv.gz:
        Part user-items interaction without concrete recommendations.
        Can be used to produce embeddings, ignored for now.
    """
    def __init__(self, path, nrows=None, **kwargs):
        print('reading files. . . ')
        
        ratings_log = pd.read_csv(os.path.join(path, 'interactions.csv.gz'), 
                                      compression='gzip', delimiter=',')
        slates_log = pd.read_csv(os.path.join(path, 'impressions-direct-link.csv.gz'),
                                      compression='gzip', delimiter=',',nrows=nrows)

        # drop rows with recommendation_id == -1 as early as possible
        ratings_log = ratings_log[
            ratings_log['recommendation_id'].isin(slates_log['recommendation_id'])
        ]
        print('preprocessing. . . ')
        
        # recommendations 
        trimmed = np.char.strip(slates_log.recommended_series_list.to_numpy(dtype='str'),'][').tolist()
        trimmed = pd.Series([re.sub('\s+', '#', x, flags=re.ASCII).strip('#') for x in trimmed], dtype='str')
        recommendations = trimmed.str.split('#', expand=True).fillna(-1).astype('int').to_numpy()

        # recommendation_id -> recommendation index
        index = slates_log['recommendation_id'].copy()
        index = index.reset_index().rename(columns={'index':'recommendation_idx'})
        ratings_log = ratings_log.merge(
            index,
            on='recommendation_id',
            validate = "many_to_one",
        )

        # responses
        # Number of clicks on `serie` we define total number of clicks on item in 
        # such `serie`
        responses = np.zeros_like(recommendations)
        for (rec, item), group in tqdm(ratings_log.groupby(['recommendation_idx', 'series_id'])):
            position = np.where(recommendations[rec.item()] == item.item())
            responses[rec.item()][position] += group.shape[0]

        # metadata
        # timestamp for `serie' will be a first timetamp for item
        metadata = ratings_log.groupby('recommendation_idx' ).agg(
            {
                "utc_ts_milliseconds": "min", 
                "user_id": lambda x: x.unique()[0], 
            }
        )
        metadata.reset_index(inplace=True)
        metadata['session_id'] = metadata['user_id']
        metadata.rename(columns={
            'index': 'recommendation_idx',
            'utc_ts_milliseconds' : 'timestamp'
            }, inplace=True)
        self.metadata = metadata

        super().__init__(recommendations, responses, metadata, **kwargs)


class OpenCDP(RecommendationData):
    def __init__(self, data_dir, files=None, min_session_len=24, window_size=8, max_active_hours=24, **kwargs):
        """
        Reads all .csv from data_dir directory, and concatenates them to one larga table 
        :param files: if specified, loads only specified subset of data files
        :param min_session_len: drops sessions with fewer number of events
        :param window_size: how many consequent events are threaten as one recommendation
        :param max_active_hours: max session duration in hours.
        """
        # reading files
        print('reading files...')
        files = os.listdir(data_dir) if files is None else files
        events = [pd.read_csv(os.path.join(data_dir, file)) for file in files]
        raw_events = pd.concat(events)
        print(f"total {len(raw_events)} raw events")
        
        # dropping too short session
        c = raw_events[["user_session","event_time"]].groupby('user_session').count()
        good_sessions = c[c.event_time > 30].index
        raw_events = raw_events[raw_events.user_session.isin(good_sessions)]
        print(f"after filtering total {len(raw_events)} events")

        # aggregating sequence of events into "recommendations"
        recommendations = []
        responses = []
        metadata = []
        category_features = []
        brand_indexer = Indexer(np.nan).from_iter(raw_events.brand)
        cat_indexer = Indexer(np.nan).from_iter(raw_events.category_code)
        for session, group in tqdm(raw_events.groupby('user_session')):
            rec = []
            cat_features = []
            clicks = []
            for timestamp, row in group.sort_values(by='event_time').iterrows():
                # purchases and removing from catr are ignored
                if row.event_type == 'remove_from_cart' or row.event_type == 'purchase':
                    continue

                if len(rec) == 0:
                    first_action = datetime.strptime(row.event_time, "%Y-%m-%d %H:%M:%S UTC")

                if row.event_type == 'cart':
                    if row.product_id not in rec:
                        rec.append(row.product_id)
                        cat_features.append(
                            [
                                cat_indexer.get(row.category_code),
                                brand_indexer.get(row.brand)
                            ]
                        )
                        clicks.append(1)
                    else:
                        clicks[rec.index(row.product_id)] += 1

                elif row.event_type == 'view':
                    rec.append(row.product_id)
                    cat_features.append(
                        [
                            cat_indexer.get(row.category_code),
                            brand_indexer.get(row.brand)
                        ]
                    )
                    clicks.append(0)
                else:
                    raise ValueError
                if (datetime.strptime(row.event_time, "%Y-%m-%d %H:%M:%S UTC") -
                        first_action >
                        timedelta(hours=max_active_hours)):
                    rec.extend([self.NO_ITEM] * (window_size - len(rec)))
                    clicks.extend([0] * (window_size - len(clicks)))
                    cat_features.extend(
                        [[0, 0]] * (window_size - len(cat_features))
                    )

                if len(rec) == window_size:
                    metadata.append({
                        'recommendation_idx': len(recommendations),
                        'item_feature_idx': len(recommendations),
                        'session_id': row.user_session,
                        'user_id': row.user_id,
                        'timestamp': row.event_time,
                    })
                    recommendations.append(rec)
                    responses.append(clicks)
                    category_features.append(cat_features)
                    rec, clicks, cat_features = [], [], []
        super().__init__(
            recommendations = np.array(recommendations).astype(int),
            item_categorical = np.array(category_features),
            responses = np.array(responses).astype(int),
            metadata = pd.DataFrame(metadata),
            **kwargs
        )
