"""Relational GNN"""
import tqdm

import dgl
import torch as th
from dgl.distributed import DistTensor, node_split
from .gs_layer import GSLayer

class RelGraphConvEncoder(GSLayer):     # pylint: disable=abstract-method
    r"""General encoder for heterogeneous graph conv encoder.

    Parameters
    ----------
    g : DGLHeteroGraph
        Input graph.
    h_dim : int
        Hidden dimension
    out_dim : int
        Output dimension
    num_hidden_layers : int
        Number of hidden layers. Total GNN layers is equal to num_hidden_layers + 1. Default 1
    dropout : float
        Dropout. Default 0.
    use_self_loop : bool
        Whether to add selfloop. Default True
    last_layer_act : torch.function
        Activation for the last layer. Default None
    """
    def __init__(self,
                 g,
                 h_dim,
                 out_dim,
                 num_hidden_layers=1,
                 dropout=0.,
                 use_self_loop=True,
                 last_layer_act=None):
        super(RelGraphConvEncoder, self).__init__()
        self.g = g
        self._h_dim = h_dim
        self._out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.last_layer_act = last_layer_act
        self.layers = None # GNN layers

        self.init_encoder()

    def init_encoder(self):
        """ Initialize GNN encoder
        """

    def forward(self, blocks, h):
        """Forward computation

        Parameters
        ----------
        h: dict[str, torch.Tensor]
            Input node feature for each node type.
        blocks: DGL MFGs
            Sampled subgraph in DGL MFG
        """

    def dist_inference(self, g, batch_size, device, num_workers,
                       x, fanout, task_tracker=None):
        """Distributed inference of final representation over all node types.
        ***NOTE***
        For node classification, the model is trained to predict on only one node type's
        label.  Therefore, only that type's final representation is meaninful.
        """
        print("Full graph RGCN inference")

        with th.no_grad():
            for i, layer in enumerate(self.layers):
                # get the fanout for this layer
                y = {}
                for k in g.ntypes:
                    y[k] = DistTensor((g.number_of_nodes(k),
                                       self._h_dim if i != len(self.layers) - 1\
                                                   else self._out_dim),
                                       dtype=th.float32, name='h-' + str(i),
                                       part_policy=g.get_node_partition_policy(k),
                                       # TODO(zhengda) this makes the tensor persistent
                                       # in memory.
                                       persistent=True)

                infer_nodes = {}
                for ntype in g.ntypes:
                    infer_nodes[ntype] = node_split(th.ones((g.number_of_nodes(ntype),),
                                                            dtype=th.bool),
                                                    partition_book=g.get_partition_book(),
                                                    ntype=ntype, force_even=False)
                # need to provide the fanout as a list, the number of layers is one obviously here
                sampler = dgl.dataloading.MultiLayerNeighborSampler([fanout])
                dataloader = dgl.dataloading.DistNodeDataLoader(g, infer_nodes, sampler,
                                                                batch_size=batch_size,
                                                                shuffle=True,
                                                                drop_last=False,
                                                                num_workers=num_workers)

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

    @property
    def in_dims(self):
        return self._h_dim

    @property
    def out_dims(self):
        return self._out_dim

    @property
    def h_dims(self):
        """ The hidden dimension.
        """
        return self._h_dim

    @property
    def n_layers(self):
        """ The number of GNN layers.
        """
        # The number of GNN layer is the number of hidden layers + 1
        return self.num_hidden_layers + 1
