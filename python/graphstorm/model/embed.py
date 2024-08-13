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

    Embedding layer implementation
"""

import time
import logging
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.distributed import DistEmbedding, node_split

from .gs_layer import GSLayer
from ..dataloading.dataset import prepare_batch_input
from ..utils import (
    get_rank,
    barrier,
    is_distributed,
    get_backend,
    create_dist_tensor,
)
from .ngnn_mlp import NGNNMLP
from ..wholegraph import WholeGraphDistTensor
from ..wholegraph import is_wholegraph_init


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


class GSNodeInputLayer(GSLayer):  # pylint: disable=abstract-method
    """The input layer for all nodes in a heterogeneous graph.

    Parameters
    ----------
    g: DistGraph
        The distributed graph
    """
    def __init__(self, g):
        super(GSNodeInputLayer, self).__init__()
        self.g = g
        # By default, there is no learnable embeddings (sparse_embeds)
        self._sparse_embeds = {}

    def prepare(self, _):
        """ Preparing input layer for training or inference.

        The input layer can pre-compute node features in the preparing step
        if needed. For example pre-compute all BERT embeddings

        Default action: Do nothing
        """

    def freeze(self, _):
        """ Freeze the models in input layer during model training

        Default action: Do nothing
        """

    def unfreeze(self):
        """ Unfreeze the models in input layer during model training

        Default action: Do nothing
        """

    def get_general_dense_parameters(self):
        """ Get dense layers' parameters.

        Returns
        -------
        list of Tensors: the dense parameters
        """
        return self.parameters()

    def get_lm_dense_parameters(self):
        """ Get the language model related parameters
        Returns
        -------
        list of Tensors: the language model parameters.
        """
        # By default, there is no language model
        return []

    def get_sparse_params(self):
        """ Get the sparse parameters.

        Returns
        -------
        list of Tensors: the sparse embeddings.
        """
        # By default, there is no sparse_embeds
        return []

    def require_cache_embed(self):
        """ Whether to cache the embeddings for inference.

        If the input encoder has heavy computations, such as BERT computations,
        it should return True and the inference engine will cache the embeddings
        from the input encoder.

        Returns
        -------
        Bool : True if we need to cache the embeddings for inference.
        """
        return False

    @property
    def sparse_embeds(self):
        """ Get sparse embeds
        """
        return self._sparse_embeds

    @property
    def in_dims(self):
        """ The number of input dimensions.

        The input dimension can be different for different node types.
        """
        return None

    @property
    def use_wholegraph_sparse_emb(self):
        """ Whether or not to use WholeGraph to host embeddings for sparse updates.

            Note: By default, a GSNodeInputLayer does not support WholeGraph
            sparse embedding, unless implemented specifically.

            Note: GSNodeEncoderInputLayer supports WholeGraph sparse embedding.
        """
        return False


class GSNodeEncoderInputLayer(GSNodeInputLayer):
    """ The node encoder input layer for all nodes in a heterogeneous graph.

    The input layer adds a linear layer on nodes with node features and the linear layer
    projects the node features into a specified dimension. 
    It also adds learnable embeddings on nodes that do not have features. Users can add
    learnable embeddings on the nodes with node features by setting ``use_node_embeddings``
    to True. In this case, the input layer combines the node features with the learnable
    embeddings and project them to the specified dimension.

    Parameters
    ----------
    g: DistGraph
        The input DGL distributed graph.
    feat_size : dict of int
        The original feat size of each node type in the format of {ntype: size}.
    embed_size : int
        The output embedding size.
    activation : callable
        The activation function applied to the output embeddigns. Default: None.
    dropout : float
        The dropout parameter. Default: 0.
    use_node_embeddings : bool
        Whether to use learnable embeddings for nodes even when node features are
        available. Default: False.
    force_no_embeddings : list of str
        The list node types that are forced to not use learnable embeddings. Default:
        None.
    num_ffn_layers_in_input: int
        (Optional) Number of layers of feedforward neural network for each node type
        in the input layer. Default: 0.
    ffn_activation : callable
        The activation function for the feedforward neural networks. Default: relu.
    cache_embed : bool
        Whether or not to cache the embeddings. Default: False.
    use_wholegraph_sparse_emb : bool
        Whether or not to use WholeGraph to host embeddings for sparse updates. Default:
        False.

    Examples:
    ----------

    .. code:: python

        from graphstorm import get_node_feat_size
        from graphstorm.model import GSgnnNodeModel, GSNodeEncoderInputLayer
        from graphstorm.dataloading import GSgnnData

        np_data = GSgnnData(...)

        model = GSgnnEdgeModel(alpha_l2norm=0)
        feat_size = get_node_feat_size(np_data.g, "feat")
        encoder = GSNodeEncoderInputLayer(g, feat_size, 
                                          embed_size=4,
                                          use_node_embeddings=True)
        model.set_node_input_encoder(encoder)
    """
    def __init__(self,
                 g,
                 feat_size,
                 embed_size,
                 activation=None,
                 dropout=0.0,
                 use_node_embeddings=False,
                 force_no_embeddings=None,
                 num_ffn_layers_in_input=0,
                 ffn_activation=F.relu,
                 cache_embed=False,
                 use_wholegraph_sparse_emb=False):
        super(GSNodeEncoderInputLayer, self).__init__(g)
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)
        self.use_node_embeddings = use_node_embeddings
        self._use_wholegraph_sparse_emb = use_wholegraph_sparse_emb
        self.feat_size = feat_size
        if force_no_embeddings is None:
            force_no_embeddings = []

        self.activation = activation
        self.cache_embed = cache_embed

        if self._use_wholegraph_sparse_emb:
            assert get_backend() == "nccl",  \
                "WholeGraph sparse embedding is only supported on NCCL backend."
            assert is_wholegraph_init(), \
                "WholeGraph is not initialized yet."
        if (
            dgl.__version__ <= "1.1.2"
            and is_distributed()
            and get_backend() == "nccl"
            and not self._use_wholegraph_sparse_emb
        ):
            if self.use_node_embeddings:
                raise NotImplementedError(
                    "NCCL backend is not supported for utilizing "
                    + "node embeddings. Please use DGL version >=1.1.2 or gloo backend."
                )
            for ntype in g.ntypes:
                if not feat_size[ntype]:
                    raise NotImplementedError(
                        "NCCL backend is not supported for utilizing "
                        + "learnable embeddings on featureless nodes. Please use DGL version "
                        + ">=1.1.2 or gloo backend."
                    )

        # create weight embeddings for each node for each relation
        self.proj_matrix = nn.ParameterDict()
        self.input_projs = nn.ParameterDict()
        embed_name = "embed"
        for ntype in g.ntypes:
            feat_dim = 0
            if feat_size[ntype] > 0:
                feat_dim += feat_size[ntype]
            if feat_dim > 0:
                if get_rank() == 0:
                    logging.debug("Node %s has %d features.", ntype, feat_dim)
                input_projs = nn.Parameter(th.Tensor(feat_dim, self.embed_size))
                nn.init.xavier_uniform_(input_projs, gain=nn.init.calculate_gain("relu"))
                self.input_projs[ntype] = input_projs
                if self.use_node_embeddings:
                    if self._use_wholegraph_sparse_emb:
                        if get_rank() == 0:
                            logging.debug(
                                "Use WholeGraph to host additional sparse embeddings on node %s",
                                ntype,
                            )
                        self._sparse_embeds[ntype] = WholeGraphDistTensor(
                            (g.number_of_nodes(ntype), self.embed_size),
                            th.float32,  # to consistent with distDGL's DistEmbedding dtype
                            embed_name + "_" + ntype,
                            use_wg_optimizer=True,  # no memory allocation before opt available
                        )
                    else:
                        if get_rank() == 0:
                            logging.debug("Use additional sparse embeddings on node %s", ntype)
                        part_policy = g.get_node_partition_policy(ntype)
                        self._sparse_embeds[ntype] = DistEmbedding(
                            g.number_of_nodes(ntype),
                            self.embed_size,
                            embed_name + "_" + ntype,
                            init_emb,
                            part_policy,
                        )
                    proj_matrix = nn.Parameter(th.Tensor(2 * self.embed_size, self.embed_size))
                    nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain("relu"))
                    # nn.ParameterDict support this assignment operation if not None,
                    # so disable the pylint error
                    self.proj_matrix[ntype] = proj_matrix

            elif ntype not in force_no_embeddings:
                if self._use_wholegraph_sparse_emb:
                    if get_rank() == 0:
                        logging.debug(
                            "Use WholeGraph to host sparse embeddings on node %s:%d",
                            ntype,
                            g.number_of_nodes(ntype),
                        )
                    self._sparse_embeds[ntype] = WholeGraphDistTensor(
                        (g.number_of_nodes(ntype), self.embed_size),
                        th.float32,  # to consistent with distDGL's DistEmbedding dtype
                        embed_name + "_" + ntype,
                        use_wg_optimizer=True,  # no memory allocation before opt available
                    )
                else:
                    if get_rank() == 0:
                        logging.debug('Use sparse embeddings on node %s:%d',
                                    ntype, g.number_of_nodes(ntype))
                    part_policy = g.get_node_partition_policy(ntype)
                    self._sparse_embeds[ntype] = DistEmbedding(g.number_of_nodes(ntype),
                                    self.embed_size,
                                    embed_name + '_' + ntype,
                                    init_emb,
                                    part_policy=part_policy)

                proj_matrix = nn.Parameter(th.Tensor(self.embed_size, self.embed_size))
                nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain('relu'))
                self.proj_matrix[ntype] = proj_matrix

        # ngnn
        self.num_ffn_layers_in_input = num_ffn_layers_in_input
        self.ngnn_mlp = nn.ModuleDict({})
        for ntype in g.ntypes:
            self.ngnn_mlp[ntype] = NGNNMLP(embed_size, embed_size,
                            num_ffn_layers_in_input, ffn_activation, dropout)

    def forward(self, input_feats, input_nodes):
        """ Input layer forward computation.

        Parameters
        ----------
        input_feats: dict of Tensor
            The input features in the format of {ntype: feats}.
        input_nodes: dict of Tensor
            The input node indexes in the format of {ntype: indexes}.

        Returns
        -------
        embs: dict of Tensor
            The projected node embeddings in the format of {ntype: emb}.
        """
        assert isinstance(input_feats, dict), 'The input features should be in a dict.'
        assert isinstance(input_nodes, dict), 'The input node IDs should be in a dict.'
        embs = {}
        for ntype in input_nodes:
            if isinstance(input_nodes[ntype], np.ndarray):
                # WholeGraphSparseEmbedding requires the input nodes (indexing tensor)
                # to be a th.Tensor
                input_nodes[ntype] = th.from_numpy(input_nodes[ntype])
            emb = None
            if ntype in input_feats:
                assert ntype in self.input_projs, \
                    f"We need a projection for node type {ntype}"
                # If the input data is not float, we need to convert it t float first.
                emb = input_feats[ntype].float() @ self.input_projs[ntype]
                if self.use_node_embeddings:
                    assert ntype in self.sparse_embeds, \
                        f"We need sparse embedding for node type {ntype}"
                    # emb.device: target device to put the gathered results
                    if self._use_wholegraph_sparse_emb:
                        node_emb = self.sparse_embeds[ntype].module(input_nodes[ntype].cuda())
                        node_emb = node_emb.to(emb.device, non_blocking=True)
                    else:
                        node_emb = self.sparse_embeds[ntype](input_nodes[ntype], emb.device)
                    concat_emb = th.cat((emb, node_emb), dim=1)
                    emb = concat_emb @ self.proj_matrix[ntype]
            elif ntype in self.sparse_embeds:  # nodes do not have input features
                # If the number of the input node of a node type is 0,
                # return an empty tensor with shape (0, emb_size)
                device = self.proj_matrix[ntype].device
                # If DistEmbedding supports 0-size input, we can remove this if statement.
                if isinstance(self.sparse_embeds[ntype], WholeGraphDistTensor):
                    # Need all procs pass the following due to nccl all2lallv in wholegraph
                    emb = self.sparse_embeds[ntype].module(input_nodes[ntype].cuda())
                    emb = emb.to(device, non_blocking=True)
                else:
                    if len(input_nodes[ntype]) == 0:
                        dtype = self.sparse_embeds[ntype].weight.dtype
                        embs[ntype] = th.zeros((0, self.sparse_embeds[ntype].embedding_dim),
                                        device=device, dtype=dtype)
                        continue
                    emb = self.sparse_embeds[ntype](input_nodes[ntype], device)

                emb = emb @ self.proj_matrix[ntype]

            if emb is not None:
                if self.activation is not None:
                    emb = self.activation(emb)
                    emb = self.dropout(emb)
                embs[ntype] = emb

        def _apply(t, h):
            if self.num_ffn_layers_in_input > 0:
                h = self.ngnn_mlp[t](h)
            return h

        embs = {ntype: _apply(ntype, h) for ntype, h in embs.items()}
        return embs

    def require_cache_embed(self):
        """ Whether to cache the embeddings for inference.

        If the input layer encoder includes heavy computations, such as BERT computations,
        it should return ``True`` and the inference engine will cache the embeddings
        from the input layer encoder.

        Returns
        -------
        bool : ``True`` if we need to cache the embeddings for inference.
        """
        return self.cache_embed

    def get_sparse_params(self):
        """ Get the sparse parameters of this input layer.

        This function is normally called by optimizers to update sparse model parameters,
        i.e., learnable node embeddings.

        Returns
        -------
        list of Tensors: the sparse embeddings, or empty list if no sparse parameters.
        """
        if self.sparse_embeds is not None and len(self.sparse_embeds) > 0:
            return list(self.sparse_embeds.values())
        else:
            return []

    @property
    def in_dims(self):
        """ Return the input feature size, which is given in class initialization.
        """
        return self.feat_size

    @property
    def out_dims(self):
        """ Return the number of output dimensions, which is given in class initialization.
        """
        return self.embed_size

    @property
    def use_wholegraph_sparse_emb(self):
        """ Return whether or not to use WholeGraph to host embeddings for sparse updates,
        which is given in class initialization.
        """
        return self._use_wholegraph_sparse_emb


def _gen_emb(g, feat_field, embed_layer, ntype):
    """ Test if the embed layer can generate embeddings on the node type.

    If a node type doesn't have node features, running the input embedding layer
    on the node type may not generate any tensors. This function is to check
    it by attempting getting the embedding of node 0. If we cannot get
    the embedding of node 0, we believe that the embedding layer cannot
    generate embeddings for this node type.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    feat_field : str
        The field of node features.
    embed_layer : callable
        The function to generate the embedding.
    ntype : str
        The node type that we will test if it generates node embeddings.

    Returns
    -------
    bool : whether embed_layer can generate embeddings on the given node type.
    """
    input_nodes = th.tensor([0])
    dev = embed_layer.device
    feat = prepare_batch_input(g, {ntype: input_nodes}, dev=dev, feat_field=feat_field)
    emb = embed_layer(feat, {ntype: input_nodes})
    return ntype in emb


def compute_node_input_embeddings(g, batch_size, embed_layer,
                                  task_tracker=None, feat_field='feat',
                                  target_ntypes=None):
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
    target_ntypes: list of str
        Node types that need to compute input embeddings.

    Returns
    -------
    dict of Tensors : the node embeddings.
    """
    if get_rank() == 0:
        logging.debug("Compute the node input embeddings.")
    assert embed_layer is not None, "The input embedding layer is needed"
    embed_layer.eval()

    n_embs = {}
    target_ntypes = g.ntypes if target_ntypes is None else target_ntypes
    th.cuda.empty_cache()
    start = time.time()
    with th.no_grad():
        for ntype in target_ntypes:
            # When reconstructed_embed is enabled, we may not be able to generate
            # embeddings on some node types. We will skip the node types.
            if not _gen_emb(g, feat_field, embed_layer, ntype):
                continue

            embed_size = embed_layer.out_dims
            # TODO(zhengda) we need to be careful about this. Here it creates a persistent
            # distributed tensor to store the node embeddings. This can potentially consume
            # a lot of memory.
            if 'input_emb' not in g.nodes[ntype].data:
                g.nodes[ntype].data['input_emb'] = create_dist_tensor(
                    (g.number_of_nodes(ntype), embed_size),
                    dtype=th.float32, name=f'{ntype}_input_emb',
                    part_policy=g.get_node_partition_policy(ntype),
                    persistent=True)
            else:
                assert g.nodes[ntype].data['input_emb'].shape[1] == embed_size
            input_emb = g.nodes[ntype].data['input_emb']
            # TODO(zhengda) this is not a memory efficient way of implementing this.
            # Here `force_even` is set to False, this means that we compute the input node
            # embeddings for the nodes in the local partition and save the embeddings to
            # the local partition with shared memory. Therefore, we don't need to call
            # flush at the end of inference.
            infer_nodes = node_split(th.ones((g.number_of_nodes(ntype),), dtype=th.bool),
                                     partition_book=g.get_partition_book(),
                                     ntype=ntype, force_even=False)
            node_list = th.split(infer_nodes, batch_size)
            dev = embed_layer.device
            for iter_l, input_nodes in enumerate(node_list):
                iter_start = time.time()
                if task_tracker is not None:
                    task_tracker.keep_alive(iter_l)

                feat = prepare_batch_input(g, {ntype: input_nodes}, dev=dev, feat_field=feat_field)
                emb = embed_layer(feat, {ntype: input_nodes})
                input_emb[input_nodes] = emb[ntype].to('cpu')
                if iter_l % 200 == 0 and g.rank() == 0:
                    logging.debug("compute input embeddings on %s: %d of %d, takes %.3f s",
                                  ntype, iter_l, len(node_list), time.time() - iter_start)
            n_embs[ntype] = input_emb
    if embed_layer is not None:
        embed_layer.train()
    barrier()
    if get_rank() == 0:
        logging.info("Computing input embeddings takes %.3f seconds", time.time() - start)
    return n_embs
