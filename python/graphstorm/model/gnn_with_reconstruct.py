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
import dgl
from dgl.distributed import DistTensor, node_split

from ..utils import barrier
from .gnn_encoder_base import GraphConvEncoder, dist_inference_one_layer

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
    input_rel_names : list of tuples
        The relation types used for computing the node features.
    """
    def __init__(self, gnn_encoder, input_gnn, input_rel_names):
        super(GNNEncoderWithReconstructedEmbed, self).__init__(gnn_encoder.h_dims,
                                                               gnn_encoder.out_dims,
                                                               gnn_encoder.num_layers)
        self._gnn_encoder = gnn_encoder
        self._input_gnn = input_gnn
        self._input_rel_names = input_rel_names

    def forward(self, blocks, h):
        """ The forward function.

        Parameters
        ----------
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        outs = self._input_gnn(blocks[-1], h)
        for ntype, out in outs.items():
            h[ntype] = out
        return self._gnn_encoder(blocks[:-1], h)

    def dist_inference(self, g, get_input_embeds, batch_size, fanout,
                       edge_mask=None, task_tracker=None):
        """Distributed inference of final representation over all node types.

        Parameters
        ----------
        g : DistGraph
            The distributed graph.
        gnn_encoder : GraphConvEncoder
            The GNN encoder on the graph.
        get_input_embeds : callable
            Get the node features of the input nodes.
        batch_size : int
            The batch size for the GNN inference.
        fanout : list of int
            The fanout for computing the GNN embeddings in a GNN layer.
        edge_mask : str
            The edge mask indicates which edges are used to compute GNN embeddings.
        task_tracker : GSTaskTrackerAbc
            The task tracker.

        Returns
        -------
        dict of Tensor : the final GNN embeddings of all nodes.
        """
        device = self._gnn_encoder.device
        target_ntypes = set([rel_name[2] for rel_name in self._input_rel_names])
        with th.no_grad():
            y = {}
            for k in target_ntypes:
                h_dim = self._gnn_encoder.h_dims
                y[k] = DistTensor((g.number_of_nodes(k), h_dim),
                                  dtype=th.float32, name='h-reconstruct',
                                  part_policy=g.get_node_partition_policy(k),
                                  # TODO(zhengda) this makes the tensor persistent in memory.
                                  persistent=True)
            infer_nodes = {}
            for ntype in target_ntypes:
                infer_nodes[ntype] = node_split(th.ones((g.number_of_nodes(ntype),),
                                                        dtype=th.bool),
                                                partition_book=g.get_partition_book(),
                                                ntype=ntype, force_even=False)
            # We use all neighbors to reconstruct node features.
            sampler = dgl.dataloading.MultiLayerNeighborSampler([-1], mask=edge_mask)
            dataloader = dgl.dataloading.DistNodeDataLoader(g, infer_nodes, sampler,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            drop_last=False)
            dist_inference_one_layer(g, dataloader, self._input_gnn,
                                     get_input_embeds, y, device, task_tracker)
        barrier()
        def get_input_embeds1(input_nodes):
            orig_inputs = {}
            embeds = {}
            for ntype, nodes in input_nodes.items():
                if ntype in y:
                    embeds[ntype] = y[ntype][nodes]
                else:
                    orig_inputs[ntype] = nodes
            embeds1 = get_input_embeds(orig_inputs)
            embeds.update(embeds1)
            return {ntype: embed.to(device) for ntype, embed in embeds.items()}
        return self._gnn_encoder.dist_inference(g, get_input_embeds1, batch_size, fanout,
                                                edge_mask=edge_mask, task_tracker=task_tracker)
