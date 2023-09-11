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

    GNN student model class
"""

import torch as th
from torch import nn
from transformers import DistilBertConfig, DistilBertModel, AutoTokenizer

SUPPORTED_MODEL = {
"DistilBertModel": (DistilBertConfig, DistilBertModel, 768),
}

PRETRAINED_MODEL = [
"pre_trained_name",
]

# TODO: move this into a lm-distilled folder
class GSDistilledModel(nn.Module):
    """ GraphStorm GNN distilled model class
    """
    def __init__(self, transformer_name, node_type, gnn_embed_dim, pre_trained_name=None):
        super(GSDistilledModel, self).__init__()
        if transformer_name not in SUPPORTED_MODEL:
            raise ValueError(f'Model class {transformer_name} is not supported.')
        if pre_trained_name is not None and pre_trained_name not in PRETRAINED_MODEL:
            raise ValueError(f'Pre-trained model {pre_trained_name} is not supported.')

        self.node_type = node_type

        # initiate Transformer-based model
        configuration = SUPPORTED_MODEL[transformer_name][0]()
        self.transformers = SUPPORTED_MODEL[transformer_name][1](configuration)
        self.tsf_embed_dim = SUPPORTED_MODEL[transformer_name][2]

        # load pre-trained parameters if any
        if pre_trained_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_name)
            self.transformers = self.transformers.from_pretrained("distilbert-base-uncased")
        else:
            # default tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # initiate embedding project layer
        self.proj = nn.Parameter(th.Tensor(self.tsf_embed_dim, gnn_embed_dim))
        nn.init.xavier_uniform_(self.proj)

        # loss
        self.loss = nn.MSELoss()


    def forward(self, inputs, attention_mask, labels):
        tsf_outputs = self.transformers(inputs, attention_mask=attention_mask)
        h = tsf_outputs.last_hidden_state
        # get pooled h
        h = h[:, 0]
        h = th.matmul(h, self.proj)
        loss = self.loss(h, labels)
        return loss