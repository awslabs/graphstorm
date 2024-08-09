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

from functools import partial

import torch as th
import dgl
from dgl.distributed import node_split

from ..utils import barrier
from .gnn_encoder_base import GraphConvEncoder, dist_inference_one_layer

def construct_node_feat(g, rel_names, input_gnn, get_input_embeds, batch_size,
                        edge_mask=None, device="cpu", task_tracker=None):
    """ Construct node features with the input layer in the full-graph inference.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    rel_names : list
        The relation types used to construct node features.
    input_encoder : GraphConvEncoder
        The encoder to construct node features.
    get_input_embeds : callable
        Get the node features of the input nodes.
    batch_size : int
        The batch size for the GNN inference.
    edge_mask : str
        The edge mask indicates which edges are used to compute GNN embeddings.
    device : Pytorch Device
        The device where to perform the computation.
    task_tracker : GSTaskTrackerAbc
        The task tracker.

    Returns
    -------
    dict of Tensor : the constructed node features.
    """
    # Here we only need to compute embeddings for the destination node types
    # of the required relation types.
    target_ntypes = {rel_name[2] for rel_name in rel_names}
    with th.no_grad():
        infer_nodes = {}
        for ntype in target_ntypes:
            infer_nodes[ntype] = node_split(th.ones((g.number_of_nodes(ntype),),
                                                    dtype=th.bool),
                                            partition_book=g.get_partition_book(),
                                            ntype=ntype, force_even=False)
        # We use all neighbors to reconstruct node features.
        fanout = {rel_name: 0 for rel_name in g.canonical_etypes}
        for rel_name in rel_names:
            fanout[rel_name] = -1
        sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout], mask=edge_mask)
        dataloader = dgl.dataloading.DistNodeDataLoader(g, infer_nodes, sampler,
                                                        batch_size=batch_size,
                                                        shuffle=False,
                                                        drop_last=False)
        return dist_inference_one_layer('input', g, dataloader, target_ntypes, input_gnn,
                                        get_input_embeds, device, task_tracker)

def get_input_embeds_combined(input_nodes, feats, get_input_embeds, device='cpu'):
    """ This gets the node embeddings from feats or get_input_embeds.

    If required node embeddings are in feats, it reads from feats directly.
    Otherise, it reads from get_input_embeds.

    Parameters
    ----------
    input_nodes : dict of Tensors
        The input node IDs.
    feats : dict of Tensors
        The node features.
    get_input_embeds : callable
        The function to get input node embeddings.
    device : Pytorch device
        The device where the output tensors are stored.

    Returns
    -------
    dict of Tensors : The node embeddings of the input nodes.
    """
    orig_inputs = {}
    embeds = {}
    for ntype, nodes in input_nodes.items():
        if ntype in feats:
            embeds[ntype] = feats[ntype][nodes]
        else:
            orig_inputs[ntype] = nodes
    embeds1 = get_input_embeds(orig_inputs)
    assert len(embeds1) == len(orig_inputs)
    embeds.update(embeds1)
    return {ntype: embed.to(device) for ntype, embed in embeds.items()}

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
        self._required_src_type = {etype[0] for etype in input_rel_names}
        self._constructed_ntype = {etype[2] for etype in input_rel_names}

    def forward(self, blocks, h):
        """ The forward function.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graph (MFG) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        """
        assert len(blocks) == self._gnn_encoder.num_layers + 1, \
                f'There are {len(blocks)}, but there are {self._gnn_encoder.num_layers} GNN layers.'
        h = self.construct_node_feat(blocks[0], h)
        return self._gnn_encoder(blocks[1:], h)

    def construct_node_feat(self, block, h):
        """ Construct node features in a mini-batch.

        It uses the input GNN layer to construct node features on the specified
        node types. It reads the remaining node features directly.

        Parameters
        ----------
        block : DGLBlock
            The subgraph for node construction.
        h : dict of Tensors
            The input node features.

        Returns
        -------
        dict of Tensors : the output node embeddings.
        """
        input_h = {}
        for ntype in self._required_src_type:
            assert ntype in h, f"The features of node type {ntype} are required."
            input_h[ntype] = h[ntype]
        out_h = self._input_gnn(block, input_h)
        for ntype in h:
            if ntype not in self._constructed_ntype:
                out_h[ntype] = h[ntype][0:block.num_dst_nodes(ntype)]
        return out_h

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
        constructed_feats = construct_node_feat(g, self._input_rel_names,
                                                self._input_gnn, get_input_embeds,
                                                batch_size, edge_mask=edge_mask,
                                                device=device, task_tracker=task_tracker)
        barrier()
        get_input_embeds = partial(get_input_embeds_combined,
                                   feats=constructed_feats,
                                   get_input_embeds=get_input_embeds,
                                   device=device)
        return self._gnn_encoder.dist_inference(g, get_input_embeds,
                                                batch_size, fanout, edge_mask=edge_mask,
                                                task_tracker=task_tracker)
