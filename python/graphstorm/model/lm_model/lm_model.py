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

    Builtin language model support
"""
from abc import abstractmethod

import numpy as np
from torch import nn

TOKEN_IDX = 'input_ids'
VALID_LEN = 'valid_len'
ATT_MASK_IDX = 'attention_mask'
TOKEN_TID_IDX = 'token_type_ids'

class GSFLanguageModelWrapper(nn.Module):
    """ Wrapper of language model

    Parameters
    ----------
    lm_model:
        language model
    num_train: int
        Number of trainable texts
    bert_infer_bs: int
        Batch size used for computing text embeddings for static bert
    lm_output_size: int
        The deminsion of the LM output embedding size.
    lm_fnames: list of str
        Names of features required by a language model to compute LM embeddings.
    profile: bool
        If True, compute flops statistics.
    """
    def __init__(self,
                 lm_model,
                 num_train,
                 lm_output_size,
                 lm_fnames,
                 lm_infer_batch_size=32,
                 profile=False):
        super(GSFLanguageModelWrapper, self).__init__()
        self.lm_model = lm_model
        self.num_train = num_train
        self.infer_bs = lm_infer_batch_size
        self.profile = profile

        self.num_params = -1
        self.max_train_seq_lens = []
        self.max_static_seq_lens = []
        self.train_flops = []
        self.static_flops = []

        self._feat_size = lm_output_size
        self._lm_fnames = lm_fnames

    def get_avg_train_seq_len(self):
        """ Get average sequence length used in training bert
        """
        assert self.profile is True, "Please turn on profile flag"
        return np.mean(self.max_train_seq_lens) \
            if len(self.max_train_seq_lens) > 0 else -1

    def get_avg_train_flops(self):
        """ Get average flops during training bert
        """
        assert self.profile is True, "Please turn on profile flag"
        return np.mean(self.train_flops) \
            if len(self.train_flops) > 0 else -1

    def get_avg_static_seq_len(self):
        """ Get average sequence length used in static bert forward
        """
        assert self.profile is True, "Please turn on profile flag"
        return np.mean(self.max_static_seq_lens) \
            if len(self.max_static_seq_lens) > 0 else -1

    def get_avg_static_flops(self):
        """ Get average flops during static bert forward
        """
        assert self.profile is True, "Please turn on profile flag"
        return np.mean(self.static_flops) \
            if len(self.static_flops) > 0 else -1

    @abstractmethod
    def forward(self, input_ntypes, input_lm_feats):
        """ Forward

        Parameters
        ----------
        input_ntypes: list of str
            A list of input node types
        input_lm_feats: dict of dict of tensors
            Input language model related node features

        Return
        ------
        dict of tensor
            Node type -> text embedding (torch tensor)
        """

    @property
    def feat_size(self):
        """ Output feature size
        """
        return self._feat_size

    @property
    def lm_fnames(self):
        """ Names of features required by a language model to compute LM embeddings.
            For example: [TOKEN_IDX, ATT_MASK_IDX, TOKEN_TID_IDX]

            Return:
            A list of str
        """
        return self._lm_fnames
