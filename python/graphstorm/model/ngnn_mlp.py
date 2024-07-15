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

    Ref Paper: https://arxiv.org/abs/2111.11638
    NGNN MLP Layer.
"""
import torch as th
from torch import nn
import torch.nn.functional as F

class NGNNMLP(nn.Module):
    r"""NGNN MLP Implementation

    NGNN Layer is consisted of combination of a MLP Layer, an activation layer and dropout

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    layer_number: int
        Number of NGNN layers
    activation: torch.nn.functional
        Type of NGNN activation layer
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 layer_number=0,
                 activation=F.relu,
                 dropout=0.0):
        super(NGNNMLP, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.layer_number = 0
        self.dropout = nn.Dropout(dropout)
        self.ngnn_gnn = nn.ParameterList()
        for _ in range(0, layer_number):
            mlp_layer = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(mlp_layer, gain=nn.init.calculate_gain('relu'))
            self.ngnn_gnn.append(mlp_layer)

    # pylint: disable=invalid-name
    def forward(self, emb):
        """Forward computation

        Parameters
        ----------
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        h = emb
        for layer in self.ngnn_gnn:
            h = th.matmul(h, layer)
            h = self.activation(h)
        return self.dropout(h)
