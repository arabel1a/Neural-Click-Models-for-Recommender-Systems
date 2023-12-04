import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import pickle
import re
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F

from torchmetrics import AUROC, F1Score, Accuracy


from src.utils import map_array

class RecommendationData(Dataset):
    # special id (in recommendation slate) to represent 'nothing'
    # useful for situations, where slate size may vary
    NO_ITEM = -1
    
    def __init__(self, recommendations: np.array, responses: np.array, metadata:pd.DataFrame, 
                 user_item_matrix = None, user_id2index = None, item_id2index=None,
                 item_features: np.array =None, user_features: np.array =None, **kwargs):
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
                             indexes of embeddings in user and item is used ay current 
                             response. Taken intro account only if embeddings are presented.
                             Expected values are in (0, num_user_features) for 
                             user_emb_idx and (0, num_item_featuress) for item_emb_idx.
                             
        :param user_features: [optional] Numpy array with embeddings for user. 
                                Expected shape (num_user_features, ...). 
                                
        :param item_features: [optional] Numpy array with embeddings for item. 
                                Expected shape (num_item_features, slate_size, ...). 
        """
        super().__init__
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
        
        # filter small sequences
        self._from_metadata(self.metadata, inplace=True, **kwargs)
                
        self.n_users = len(self.metadata['user_id'].unique())
        self.n_items = len(np.unique(self.recommendations)) 
        # if self.NO_ITEM in self.recommendations:
        #     self.n_items -= 1
        
        self.user_item_matrix = user_item_matrix
        self.user_id2index = user_id2index
        self.item_id2index = item_id2index
        if user_item_matrix is None:
            self.build_affinity_matrix()
            
        self.recommendations_indexes = map_array(self.recommendations, self.item_id2index)
        self.metadata['user_idx'] = map_array(self.metadata['user_id'].to_numpy(), self.user_id2index)

        self.sessions = self.metadata['session_id'].unique().tolist()

        # print("caching prepared data...")
        # for i in tqdm(range(self.__len__())):
        #     self.__getitem__(i)
        
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
        
        # filter user&items
        user_features, item_features = None, None
        if self.item_features is not None:
            # drop items without impressions
            item_features = self.item_features[m['item_feature_idx'],:]
        if self.user_features is not None:
            # drop users without impressions
            user_features = self.user_features[m['user_feature_idx'],:]
            
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

        else:
            return RecommendationData(
            rec, 
            resp,
            m, 
            item_features = self.item_features, 
            user_features = self.user_features,
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
            
    def split_by_users(self, ratio = 0.8, **kwargs):
        """
        :param float ratio:  frac of users TODO change to frac of rows
        :return: two RecommendationData classes, obtained 
                from current by splitting current by users.
        """
        permutation = self.metadata['user_id'].unique()
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

    def update_embeddings(user_embeddings, item_embedidngs):
        # TODO
        pass

    def __getitem__(self, idx):
        if idx in self.prepared_data_cache:
            return self.prepared_data_cache[idx]
        session = self.sessions[idx]
        session_metadata = self.metadata[self.metadata['session_id'] == session].sort_values('timestamp')
        in_size = int(0.8 * len(session_metadata))
        out_size = len(session_metadata) - in_size
        self.prepared_data_cache[idx] = {
            'slates_item_ids': self.recommendations[session_metadata['recommendation_idx']],
            'slates_item_indexes':self.recommendations_indexes[session_metadata['recommendation_idx']],
            'slates_item_embeddings': None if self.item_features is None else self.item_features[session_metadata['item_feature_idx']],
            'slates_mask': (self.recommendations[session_metadata['recommendation_idx']] != self.NO_ITEM),
            'responses': self.responses[session_metadata['recommendation_idx']],
            'in_length': in_size,
            'length': len(session_metadata),
            'user_embeddings': None if self.user_features is None else self.user_features[session_metadata['user_feature_idx']],
            'user_ids': session_metadata['user_id'].to_numpy(),
            'user_indexes': session_metadata['user_idx'].to_numpy(),
        }
        return self.prepared_data_cache[idx]

    def __len__(self):
        return len(self.sessions)
    
    def __repr__(self):
        return ("\nRecomendation Dataset\n"
                f"users: \t{self.n_users}\n"
                f"items: \t{self.n_items}\n"
                f"recommendations: {self.recommendations.shape[0]}\n"
        )



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
                                   
        metadata = pd.DataFrame({
            'recommendation_idx' : [0, 1, 2],
            'user_id': [13, 13, 0],
            'session_id': [0, 0, 777],
            'timestamp': [1,2,3],
            'user_feature_idx': [0, 0, 1],
            'item_feature_idx': [0, 1, 2],
        })
        super().__init__(recommendations, responses, metadata, user_features=user_embeddings, item_features=item_embeddings)


class RL4RS(RecommendationData):
# if True:
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


### FIXME
# class OpenCDP_one_purchase_from_cart(RecommendationData):
#     """
#     Consist of several file (event logs splitted by months).
#         'event_time' -- timestamp in iso format
#         'event_type' -- one of 'view', 'cart', 'remove_from_cart', 'purchase'
#         'product_id' -- id of item
#         'user_id', 'user_ssession' -- user data
#         'price' -- the only feature of item used
        
#         'brand', 'category_code', 'category_id', -- ignored

    
#     """
#     def __init__(self, dir, files, events_per_user_thold=20, max_slate_size=50):
#         events = []
#         for file in files:
#             events.append( pd.read_csv(os.path.join(dir, file)))
#         self.raw_events = pd.concat(events)
#         logger.info(f'read {self.raw_events.shape} lines')
#         users_event_cnt = self.raw_events.groupby('user_id').agg(
#         {'event_time':'count'}
#          ).reset_index()
#         good_users = users_event_cnt[users_event_cnt['event_time'] > events_per_user_thold].user_id
#         self.events = self.raw_events[self.raw_events.user_id.isin(good_users)]

#         recommendations = []
#         responses = []
#         metadata = []
#         idx = 0   
        
#         for user, group in tqdm(self.events.groupby('user_id')):
#             currernt_rec = set()
#             group = group[group.event_type.isin(['purchase', 'cart', 'remove_from_cart'])]
#             for timestamp, row in group.sort_values(by='event_time').iterrows():
#                 if row.event_type == 'cart':
#                     currernt_rec.add(row.product_id)
#                 elif row.event_type == 'remove_from_cart':
#                      currernt_rec.discard(row.product_id)
#                 else:
#                     if not row.product_id in currernt_rec:
#                         continue
#                     cr = list(currernt_rec)
                        
#                     # bad way to drop too long session
#                     if len(cr) > max_slate_size or len(cr) < 2:
#                         continue
                        
#                     recommendations.append(cr + [-1] * (max_slate_size - len(cr)))
#                     resp = [0 for x in currernt_rec]
#                     resp[cr.index(row.product_id)]
#                     responses.append(resp + [-1] * (max_slate_size - len(cr)) )
#                     metadata.append({
#                         'user_id': row.user_id,
#                         'recommendation_idx': idx,
#                         'session_id':row.user_id,
#                         'timestamp': timestamp
#                     })
#                     idx += 1
#                     currernt_rec.remove(row.product_id) 
#         super().__init__(
#             np.array(recommendations).astype(int),
#             np.array(responses).astype(int),
#             pd.DataFrame(metadata)
#         )

# o = OpenCDP_one_purchase_from_cart(opencdp_dir,m )
  
