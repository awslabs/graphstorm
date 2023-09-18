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
"distilbert-base-uncased",
]

class GSDistilledModel(nn.Module):
    """ GraphStorm GNN distilled model class.
    User specified Transformer-based model 
    and a projection matrix will be initialized.

    Parameters
    ----------
    lm_name : str
        Model name for Transformer-based student model.
    gnn_embed_dim : int
        Dimension of GNN embeddings.
    pre_trained_name : str
        Name of pre-trained model.
    """
    def __init__(self, lm_name, pre_trained_name=None):
        super(GSDistilledModel, self).__init__()

        # TODO (HZ): need to test other HF models.
        if lm_name not in SUPPORTED_MODEL:
            raise ValueError(f'Model class {lm_name} is not supported.')
        if pre_trained_name is not None and pre_trained_name not in PRETRAINED_MODEL:
            raise ValueError(f'Pre-trained model {pre_trained_name} is not supported.')

        self.lm_name = lm_name

        # initiate Transformer-based model
        self.lm_config = SUPPORTED_MODEL[lm_name][0]()
        self.lm = SUPPORTED_MODEL[lm_name][1](self.lm_config)
        self.lm_embed_dim = SUPPORTED_MODEL[lm_name][2]

        # load pre-trained parameters if any
        if pre_trained_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_name)
            self.lm = self.lm.from_pretrained(pre_trained_name)
        else:
            # default tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


        # TODO (HZ): support more distance-based loss 
        self.loss = nn.MSELoss()

    def init_proj_layer(self, gnn_embed_dim):
        """ initiate embedding project layer."""
        self.gnn_embed_dim = gnn_embed_dim
        self.proj = nn.Parameter(th.Tensor(self.lm_embed_dim, gnn_embed_dim))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, inputs, attention_mask, labels):
        """ Forward function for student model.

        Parameters
        ----------
        inputs : dict
            A batch from dataloader.
        attention_mask : torch.tensor
            Masks for self attention.
        labels : torch.tensor
            GNN embeddings.
        
        Returns
        -------
        torch.tensor : MSE loss for the batch
        """
        tsf_outputs = self.lm(inputs, attention_mask=attention_mask)
        h = tsf_outputs.last_hidden_state
        # get pooled h
        h = h[:, 0]
        # project to the same dimension
        h = th.matmul(h, self.proj)
        loss = self.loss(h, labels)
        return loss