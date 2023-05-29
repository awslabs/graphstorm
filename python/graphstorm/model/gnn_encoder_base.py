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

    Relational GNN
"""
import tqdm

import dgl
import torch as th
from torch import nn
from dgl.distributed import DistTensor, node_split
from .gs_layer import GSLayer

class GraphConvEncoder(GSLayer):     # pylint: disable=abstract-method
    r"""General encoder for graph data.

    Parameters
    ----------
    h_dim : int
        Hidden dimension
    out_dim : int
        Output dimension
    num_hidden_layers : int
        Number of hidden layers. Total GNN layers is equal to num_hidden_layers + 1. Default 1
    """
    def __init__(self,
                 h_dim,
                 out_dim,
                 num_hidden_layers=1):
        super(GraphConvEncoder, self).__init__()
        self._h_dim = h_dim
        self._out_dim = out_dim
        self._num_hidden_layers = num_hidden_layers
        self._layers = nn.ModuleList()  # GNN layers.

    @property
    def in_dims(self):
        return self._h_dim

    @property
    def out_dims(self):
        return self._out_dim

    @property
    def h_dims(self):
        """ The hidden dimension size.
        """
        return self._h_dim

    @property
    def num_layers(self):
        """ The number of GNN layers.
        """
        # The number of GNN layer is the number of hidden layers + 1
        return self._num_hidden_layers + 1

    @property
    def layers(self):
        """ GNN layers
        """
        return self._layers

def dist_inference(g, gnn_encoder, get_input_embeds, batch_size, fanout,
                   edge_mask=None, task_tracker=None):
    """Distributed inference of final representation over all node types.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    gnn_encoder : GraphConvEncoder
        The GNN encoder on the graph.
    node_feats : dict of Tensors
        The node features of the graph.
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
    device = gnn_encoder.device
    with th.no_grad():
        for i, layer in enumerate(gnn_encoder.layers):
            # get the fanout for this layer
            y = {}
            for k in g.ntypes:
                h_dim = gnn_encoder.h_dims \
                        if i < len(gnn_encoder.layers) - 1 else gnn_encoder.out_dims
                y[k] = DistTensor((g.number_of_nodes(k), h_dim),
                                  dtype=th.float32, name='h-' + str(i),
                                  part_policy=g.get_node_partition_policy(k),
                                  # TODO(zhengda) this makes the tensor persistent in memory.
                                  persistent=True)

            infer_nodes = {}
            for ntype in g.ntypes:
                infer_nodes[ntype] = node_split(th.ones((g.number_of_nodes(ntype),),
                                                        dtype=th.bool),
                                                partition_book=g.get_partition_book(),
                                                ntype=ntype, force_even=False)
            # need to provide the fanout as a list, the number of layers is one obviously here
            fanout_i = [-1] if fanout is None or len(fanout) == 0 else [fanout[i]]
            sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout_i, mask=edge_mask)
            dataloader = dgl.dataloading.DistNodeDataLoader(g, infer_nodes, sampler,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            drop_last=False)

            for iter_l, (input_nodes, output_nodes, blocks) in enumerate(tqdm.tqdm(dataloader)):
                if task_tracker is not None:
                    task_tracker.keep_alive(report_step=iter_l)
                block = blocks[0].to(device)
                if not isinstance(input_nodes, dict):
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}

                if not isinstance(output_nodes, dict):
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    output_nodes = {g.ntypes[0]: output_nodes}

                if i == 0:
                    h = get_input_embeds(input_nodes)
                else:
                    h = {k: x[k][input_nodes[k]].to(device) for k in input_nodes.keys()}
                h = layer(block, h)

                for k in h.keys():
                    # some ntypes might be in the tensor h but are not in the output nodes
                    # that have empty tensors
                    if k in output_nodes:
                        y[k][output_nodes[k]] = h[k].cpu()

            x = y
            th.distributed.barrier()
    return y
