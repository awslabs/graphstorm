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

    RGCN layer implementation.
"""
import torch.nn.functional as F

from .gnn_encoder_base import GraphConvEncoder

class GNNEncoderWithReconstructedEmbed(GraphConvEncoder):
    """ A GNN encoder wrapper.

    This wrapper reconstructs node features on the featureless nodes.
    It will use the first block to reconstruct the node features and pass
    the reconstructed node features to the remaining GNN layers.

    Parameters
    ----------
    gnn_encoder : GraphConvEncoder
        The GNN layers.
    input_gnn : nn module
        The GNN layer to construct node features.
    """
    def __init__(self, gnn_encoder, input_gnn):
        self._gnn_encoder = gnn_encoder
        self._input_gnn = input_gnn

    def forward(self, blocks, h):
        outs = self._input_gnn(blocks[0], h)
        for ntype, out in outs.items():
            h[ntype] = out
        return self._gnn_encoder(blocks[1:], h)
