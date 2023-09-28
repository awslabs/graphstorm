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
import os
import torch as th
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    DistilBertConfig,
    DistilBertModel,
)

SUPPORTED_MODEL = {
"DistilBertModel": (DistilBertConfig, DistilBertModel),
}

PRETRAINED_MODEL = [
"distilbert-base-uncased",
"distilbert-base-cased",
]

class GSDistilledModel(nn.Module):
    """ GraphStorm GNN distilled model class.
    User specified Transformer-based model 
    and a projection matrix will be initialized.

    Parameters
    ----------
    lm_type : str
        Model type for Transformer-based student model.
    pre_trained_name : str
        Name of pre-trained model.
    checkpoint_path : str
        Path of model checkpoint.
    """
    def __init__(self, lm_type=None, pre_trained_name=None, checkpoint_path=None):
        super(GSDistilledModel, self).__init__()

        # TODO: Try to unify between loading HF checkpoints and loading GS checkpoints.
        if not checkpoint_path:
            assert lm_type is not None, \
                "HuggingFace Name of model architecture needs to be specified"
            assert pre_trained_name is not None, \
                "HuggingFace Name of pre-trained weights needs to be specified"
            # TODO (HZ): need to test other HF models.
            if lm_type not in SUPPORTED_MODEL:
                raise ValueError(f'Model class {lm_type} is not supported.')
            if pre_trained_name not in PRETRAINED_MODEL:
                raise ValueError(f'Pre-trained model {pre_trained_name} is not supported.')

            # initiate Transformer-based model
            self.lm_config = SUPPORTED_MODEL[lm_type][0]()
            self.lm = SUPPORTED_MODEL[lm_type][1](self.lm_config)

            # load pre-trained tokenizers
            self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_name)

            # load pre-trained parameters
            self.lm = self.lm.from_pretrained(pre_trained_name)
        else:
            self.load_gs_checkpoint(checkpoint_path)

        # TODO (HZ): support more distance-based loss
        self.loss = nn.MSELoss()

    def save_gs_checkpoint(self, checkpoint_path):
        """ Save distilled checkpoint.
        Parameters
        ----------
        checkpoint_path : str
            Saved path for checkpoint.
        """
        proj_dir_loc = os.path.join(checkpoint_path, "proj")
        os.makedirs(proj_dir_loc, exist_ok=True)
        tokenizer_dir_loc = os.path.join(checkpoint_path, "tokenizer")
        lm_dir_loc = os.path.join(checkpoint_path, "lm")
        self.tokenizer.save_pretrained(tokenizer_dir_loc)
        self.lm.save_pretrained(lm_dir_loc)
        th.save(self.state_dict()["proj"], os.path.join(proj_dir_loc, "pytorch_model.bin"))

    def load_gs_checkpoint(self, checkpoint_path):
        """ Load student moddel from checkpoint_path.

        Parameters
        ----------
        checkpoint_path : str
            Path for student checkpoint.
        """
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer")
        lm_path = os.path.join(checkpoint_path, "lm")
        proj_path = os.path.join(checkpoint_path, "proj", "pytorch_model.bin")

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # load lm model
        self.lm_config = AutoConfig.from_pretrained(lm_path)
        self.lm = AutoModel.from_pretrained(lm_path)
        proj_weights = th.load(proj_path, "cpu")
        self.init_proj_layer(weights=proj_weights)

    def init_proj_layer(self, gnn_embed_dim=None, weights=None):
        """ initiate embedding project layer.
        Parameters
        ----------
        gnn_embed_dim : int
            Dimension of GNN embeddings.
        weights : torch.tensor
            Weights for projection matrix.
        """
        assert gnn_embed_dim is not None or weights is not None, \
            "Either gnn_embed_dim or weights needs to be provided."

        if weights is not None:
            self.proj = nn.Parameter(weights)
        else:
            self.proj = nn.Parameter(th.Tensor(self.lm_config.dim, gnn_embed_dim))
            nn.init.xavier_uniform_(self.proj)
            self.gnn_embed_dim = gnn_embed_dim

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
