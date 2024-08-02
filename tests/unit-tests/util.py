"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import torch as th

from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX
from graphstorm.dataloading import (GSgnnEdgeDataLoaderBase,
                                    GSgnnLinkPredictionDataLoaderBase,
                                    GSgnnNodeDataLoaderBase)
from graphstorm.model import GSOptimizer
from graphstorm.model import GSgnnModelBase
from graphstorm.model.multitask_gnn import GSgnnMultiTaskModelInterface
from graphstorm.model.gnn_encoder_base import GSgnnGNNEncoderInterface

class Dummy:
    """dummy object used to create config objects
    constructor"""
    def __init__(self, arg_dict):
        self.__dict__.update(arg_dict)

def create_tokens(tokenizer, input_text, max_seq_length, num_node, return_token_type_ids=False):
    input_text = input_text * num_node
    tokens = tokenizer(input_text,  max_length=max_seq_length,
                        truncation=True, padding='max_length', return_tensors='pt')
    # we only use TOKEN_IDX and VALID_LEN_IDX
    input_ids = tokens[TOKEN_IDX]
    attention_mask = tokens[ATT_MASK_IDX]
    valid_len = tokens[ATT_MASK_IDX].sum(dim=1)
    token_type_ids = tokens.get(TOKEN_TID_IDX, th.zeros_like(input_ids))\
        if return_token_type_ids else None
    return input_ids, valid_len, attention_mask, token_type_ids

class DummyGSgnnData():
    def __init__(self):
        pass # do nothing

    @property
    def g(self):
        return None

class DummyGSgnnNodeDataLoader(GSgnnNodeDataLoaderBase):
    def __init__(self):
        pass # do nothing

    def __len__(self):
        return 10

    def __iter__(self):
        return self

    @property
    def fanout(self):
        return [10]

class DummyGSgnnEdgeDataLoader(GSgnnEdgeDataLoaderBase):
    def __init__(self):
        pass # do nothing

    def __len__(self):
        return 10

    def __iter__(self):
        return self

    @property
    def fanout(self):
        return [10]

class DummyGSgnnLinkPredictionDataLoader(GSgnnLinkPredictionDataLoaderBase):
    def __init__(self):
        pass # do nothing

    def __len__(self):
        return 10

    def __iter__(self):
        return self

    @property
    def fanout(self):
        return [10]

class DummyGSgnnModel(GSgnnModelBase):
    def __init__(self, encoder_model, has_sparse=False):
        self.gnn_encoder = encoder_model
        self._has_sparse = has_sparse

    def has_sparse_params(self):
        return self._has_sparse

    def create_optimizer(self):
        return GSOptimizer(dense_opts=["dummy"])

class DummyGSgnnMTModel(DummyGSgnnModel, GSgnnMultiTaskModelInterface):
    def __init__(self, encoder_model, decoders, has_sparse=False):
        super(DummyGSgnnMTModel, self).__init__(encoder_model, has_sparse)
        self._decoders = decoders

    @property
    def node_embed_norm_methods(self):
        return {}

    def normalize_task_node_embs(self, task_id, embs, inplace=False):
        return embs

    def forward(self, task_mini_batches):
        pass

    def predict(self, task_id, mini_batch):
        pass

    def train(self, train=True):
        pass

    @property
    def task_decoders(self):
        return self._decoders

class DummyGSgnnEncoderModel(GSgnnGNNEncoderInterface):
    def skip_last_selfloop(self):
        pass

    def reset_last_selfloop(self):
        pass
