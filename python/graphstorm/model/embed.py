"""Embedding layer implementation"""
import torch as th

from torch import nn
from dgl.distributed import DistEmbedding, DistTensor, node_split
from .gs_layer import GSLayer

def init_emb(shape, dtype):
    """Create a tensor with the given shape and date type.

    This function is used to initialize the data in the distributed embedding layer
    and set their value with uniformly random values.

    Parameters
    ----------
    shape : tuple of int
        The shape of the tensor.
    dtype : Pytorch dtype
        The data type

    Returns
    -------
    Tensor : the tensor with random values.
    """
    arr = th.zeros(shape, dtype=dtype)
    nn.init.uniform_(arr, -1.0, 1.0)
    return arr

class GSNodeInputLayer(GSLayer):
    """The input embedding layer for all nodes in a heterogeneous graph.

    Parameters
    ----------
    g: DistGraph
        The distributed graph
    feat_size : dict of int
        The original feat sizes of each node type
    embed_size : int
        The embedding size
    activation : func
        The activation function
    dropout : float
        The dropout parameter
    use_node_embeddings : bool
        Whether we will use the node embeddings for individual nodes even when node features are
        available.
    """
    def __init__(self,
                 g,
                 feat_size,
                 embed_size,
                 activation=None,
                 dropout=0.0,
                 use_node_embeddings=False):
        super(GSNodeInputLayer, self).__init__()
        self.g = g
        self.embed_size = embed_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.use_node_embeddings = use_node_embeddings

        # create weight embeddings for each node for each relation
        self.sparse_embeds = {}
        self.proj_matrix = nn.ParameterDict() if self.use_node_embeddings else None
        self.input_projs = nn.ParameterDict()
        embed_name = 'embed'
        for ntype in g.ntypes:
            feat_dim = 0
            if feat_size[ntype] > 0:
                feat_dim += feat_size[ntype]
            if feat_dim > 0:
                if g.rank() == 0:
                    print('Node {} has {} features.'.format(ntype, feat_dim))
                input_projs = nn.Parameter(th.Tensor(feat_dim, self.embed_size))
                nn.init.xavier_uniform_(input_projs, gain=nn.init.calculate_gain('relu'))
                self.input_projs[ntype] = input_projs
                if self.use_node_embeddings:
                    if g.rank() == 0:
                        print('Use sparse embeddings on node {}'.format(ntype))
                    part_policy = g.get_node_partition_policy(ntype)
                    self.sparse_embeds[ntype] = DistEmbedding(g.number_of_nodes(ntype),
                                                              self.embed_size,
                                                              embed_name + '_' + ntype,
                                                              init_emb,
                                                              part_policy)
                    proj_matrix = nn.Parameter(th.Tensor(2 * self.embed_size, self.embed_size))
                    nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain('relu'))
                    # nn.ParameterDict support this assignment operation if not None,
                    # so disable the pylint error
                    self.proj_matrix[ntype] = proj_matrix   # pylint: disable=unsupported-assignment-operation
            else:
                part_policy = g.get_node_partition_policy(ntype)
                if g.rank() == 0:
                    print('Use sparse embeddings on node {}'.format(ntype))
                self.sparse_embeds[ntype] = DistEmbedding(g.number_of_nodes(ntype),
                                self.embed_size,
                                embed_name + '_' + ntype,
                                init_emb,
                                part_policy)

    def has_dense_params(self):
        """ test if the module has dense parameters.
        """
        return len(self.input_projs) > 0

    def get_sparse_params(self):
        """ get the sparse parameters.

        Returns
        -------
        list of Tensors: the sparse embeddings.
        """
        if self.sparse_embeds is not None and len(self.sparse_embeds) > 0:
            return list(self.sparse_embeds.values())
        else:
            return []

    def forward(self, input_feats, input_nodes):
        """Forward computation

        Parameters
        ----------
        input_feats: dict
            input features
        input_nodes: dict
            input node ids

        Returns
        -------
        a dict of Tensor: the node embeddings.
        """
        embs = {}
        for ntype in input_nodes:
            if ntype in input_feats:
                assert ntype in self.input_projs, \
                        f"We need a projection for node type {ntype}"
                # If the input data is not float, we need to convert it t float first.
                emb = input_feats[ntype].float() @ self.input_projs[ntype]
                if self.use_node_embeddings:
                    assert ntype in self.sparse_embeds, \
                            f"We need sparse embedding for node type {ntype}"
                    node_emb = self.sparse_embeds[ntype](input_nodes[ntype]).to(emb.device)
                    concat_emb=th.cat((emb, node_emb),dim=1)
                    emb = concat_emb @ self.proj_matrix[ntype]
            else: # nodes do not have input features
                # If the number of the input node of a node type is 0,
                # return an empty tensor with shape (0, emb_size)
                if len(input_nodes[ntype]) == 0:
                    dtype = self.sparse_embeds[ntype].weight.dtype
                    embs[ntype] = th.zeros((0, self.sparse_embeds[ntype].embedding_dim),
                                           dtype=dtype)
                    continue
                emb = self.sparse_embeds[ntype](input_nodes[ntype])
            if self.activation is not None:
                emb = self.activation(emb)
            emb = self.dropout(emb)
            embs[ntype] = emb

        return embs

    @property
    def in_dims(self):
        """ The number of input dimensions.

        The input dimension can be different for different node types.
        """
        return None

    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return self.embed_size

def prepare_batch_input(g, input_nodes,
                        dev='cpu', feat_field='feat'):
    """ Prepare minibatch input features

    Note: The output is stored in dev.

    Parameters
    ----------
    g: DGLGraph
        The graph.
    input_nodes: dict of tensor
        Input nodes.
    dev: th.device
        Device to put output in.
    feat_field: str or dict of str
        Fields to extract features

    Return:
    -------
    Dict of tensors.
        If a node type has features, it will get node features.
    """
    feat = {}
    for ntype, nid in input_nodes.items():
        feat_name = None if feat_field is None else \
            feat_field if isinstance(feat_field, str) \
            else feat_field[ntype] if ntype in feat_field else None

        if feat_name is not None:
            feat[ntype] = g.nodes[ntype].data[feat_name][nid].to(dev)
    return feat

def compute_node_input_embeddings(g, batch_size, embed_layer,
                                  task_tracker=None, feat_field='feat'):
    """
    This function computes the input embeddings of all nodes in a distributed graph
    either from the node features or from the embedding layer.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    batch_size: int
        The mini-batch size for computing the input embeddings of nodes.
    embed_layer : Pytorch model
        The encoder of the input nodes.
    task_tracker : GSTaskTrackerAbc
        The task tracker.
    feat_field : str or dict of str
        The fields that contain the node features.

    Returns
    -------
    dict of Tensors : the node embeddings.
    """
    assert embed_layer is not None, "The input embedding layer is needed"
    embed_layer.eval()

    n_embs = {}
    th.cuda.empty_cache()
    with th.no_grad():
        for ntype in g.ntypes:
            embed_size = embed_layer.out_dims
            # TODO(zhengda) we need to be careful about this. Here it creates a persistent
            # distributed tensor to store the node embeddings. This can potentially consume
            # a lot of memory.
            if 'input_emb' not in g.nodes[ntype].data:
                g.nodes[ntype].data['input_emb'] = DistTensor(
                        (g.number_of_nodes(ntype), embed_size),
                        dtype=th.float32, name='{}_input_emb'.format(ntype),
                        part_policy=g.get_node_partition_policy(ntype),
                        persistent=True)
            else:
                assert g.nodes[ntype].data['input_emb'].shape[1] == embed_size
            input_emb = g.nodes[ntype].data['input_emb']
            # TODO(zhengda) this is not a memory efficient way of implementing this.
            infer_nodes = node_split(th.ones((g.number_of_nodes(ntype),), dtype=th.bool),
                                     partition_book=g.get_partition_book(),
                                     ntype=ntype, force_even=False)
            node_list = th.split(infer_nodes, batch_size)
            dev = embed_layer.device
            for iter_l, input_nodes in enumerate(node_list):
                if iter_l % 10000 == 0:
                    print ("extract_all_embeddings_dist on {}: {} of {}".format(ntype,
                                                                                iter_l,
                                                                                len(node_list)))
                if task_tracker is not None:
                    task_tracker.keep_alive(iter_l)

                feat = prepare_batch_input(g, {ntype: input_nodes}, dev=dev, feat_field=feat_field)
                emb = embed_layer(feat, {ntype: input_nodes})
                input_emb[input_nodes] = emb[ntype].to('cpu')
            n_embs[ntype] = input_emb
        if g.rank() == 0:
            print("Extract node embeddings")
    if embed_layer is not None:
        embed_layer.train()
    th.distributed.barrier()
    return n_embs
