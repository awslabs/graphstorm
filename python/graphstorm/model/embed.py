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
import torch as th
from torch import nn
from dgl.distributed import DistEmbedding, DistTensor, node_split

from .gs_layer import GSLayer
from ..dataloading.dataset import prepare_batch_input
from ..utils import get_rank

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

class GSNodeInputLayer(GSLayer): # pylint: disable=abstract-method
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


class GSNodeEncoderInputLayer(GSNodeInputLayer):
    """The input encoder layer for all nodes in a heterogeneous graph.

    The input layer adds learnable embeddings on nodes if the nodes do not have features.
    It adds a linear layer on nodes with node features and the linear layer projects the node
    features to a specified dimension. A user can add learnable embeddings on the nodes
    with node features. In this case, the input layer combines the node features with
    the learnable embeddings and project them to the specified dimension.

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
        super(GSNodeEncoderInputLayer, self).__init__(g)
        self.embed_size = embed_size
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.use_node_embeddings = use_node_embeddings

        # create weight embeddings for each node for each relation
        self.proj_matrix = nn.ParameterDict()
        self.input_projs = nn.ParameterDict()
        embed_name = 'embed'
        for ntype in g.ntypes:
            feat_dim = 0
            if feat_size[ntype] > 0:
                feat_dim += feat_size[ntype]
            if feat_dim > 0:
                if get_rank() == 0:
                    print('Node {} has {} features.'.format(ntype, feat_dim))
                input_projs = nn.Parameter(th.Tensor(feat_dim, self.embed_size))
                nn.init.xavier_uniform_(input_projs, gain=nn.init.calculate_gain('relu'))
                self.input_projs[ntype] = input_projs
                if self.use_node_embeddings:
                    if get_rank() == 0:
                        print('Use additional sparse embeddings on node {}'.format(ntype))
                    part_policy = g.get_node_partition_policy(ntype)
                    self._sparse_embeds[ntype] = DistEmbedding(g.number_of_nodes(ntype),
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
                if get_rank() == 0:
                    print(f'Use sparse embeddings on node {ntype}:{g.number_of_nodes(ntype)}')
                proj_matrix = nn.Parameter(th.Tensor(self.embed_size, self.embed_size))
                nn.init.xavier_uniform_(proj_matrix, gain=nn.init.calculate_gain('relu'))
                self.proj_matrix[ntype] = proj_matrix
                self._sparse_embeds[ntype] = DistEmbedding(g.number_of_nodes(ntype),
                                self.embed_size,
                                embed_name + '_' + ntype,
                                init_emb,
                                part_policy=part_policy)

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
        assert isinstance(input_feats, dict), 'The input features should be in a dict.'
        assert isinstance(input_nodes, dict), 'The input node IDs should be in a dict.'
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
                    node_emb = self.sparse_embeds[ntype](input_nodes[ntype], emb.device)
                    concat_emb=th.cat((emb, node_emb),dim=1)
                    emb = concat_emb @ self.proj_matrix[ntype]
            else: # nodes do not have input features
                # If the number of the input node of a node type is 0,
                # return an empty tensor with shape (0, emb_size)
                device = self.proj_matrix[ntype].device
                if len(input_nodes[ntype]) == 0:
                    dtype = self.sparse_embeds[ntype].weight.dtype
                    embs[ntype] = th.zeros((0, self.sparse_embeds[ntype].embedding_dim),
                                           device=device, dtype=dtype)
                    continue
                emb = self.sparse_embeds[ntype](input_nodes[ntype], device)
                emb = emb @ self.proj_matrix[ntype]
            if self.activation is not None:
                emb = self.activation(emb)
            emb = self.dropout(emb)
            embs[ntype] = emb

        return embs

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

    @property
    def out_dims(self):
        """ The number of output dimensions.
        """
        return self.embed_size

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
                if iter_l % 10000 == 0 and g.rank() == 0:
                    print ("extract_all_embeddings_dist on {}: {} of {}".format(ntype,
                                                                                iter_l,
                                                                                len(node_list)))
                if task_tracker is not None:
                    task_tracker.keep_alive(iter_l)

                feat = prepare_batch_input(g, {ntype: input_nodes}, dev=dev, feat_field=feat_field)
                emb = embed_layer(feat, {ntype: input_nodes})
                input_emb[input_nodes] = emb[ntype].to('cpu')
            n_embs[ntype] = input_emb
        if get_rank() == 0:
            print("Extract node embeddings")
    if embed_layer is not None:
        embed_layer.train()
    th.distributed.barrier()
    return n_embs
