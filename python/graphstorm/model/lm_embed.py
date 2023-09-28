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

    Embedding layer with Language model
    Some node features are generated by a language model, e.g., BERT
"""

import logging

import torch as th
from torch import nn
import dgl

from .embed import GSNodeInputLayer
from .embed import GSNodeEncoderInputLayer
from .lm_model import init_lm_model
from .lm_model import get_lm_node_feats
from ..utils import get_rank, barrier

def update_bert_cache(g, lm_models, lm_emb_cache, lm_infer_batch_size, use_fp16=True):
    """ Update the lm_emb_cache using lanaguage models.

    Parameters
    ----------
    lm_models: LMModels
        A collection of LM models and related information.
    lm_emb_cache: dict
        Language model embedding cache
    lm_infer_batch_size: int
        Language model inference batch size
    use_fp16 : bool
        Use float16 to store BERT embeddings.
    """
    for ntype in lm_models.ntypes:
        lm_model = lm_models.get_lm_model(ntype)
        lm_node_feat = lm_models.get_lm_node_feat(ntype)
        lm_model.eval()
        if get_rank() == 0:
            logging.info('Compute bert embedding on node %s.', ntype)
        hidden_size = lm_model.feat_size
        # TODO we should not save the BERT embeddings on the graph data in the future.
        if 'bert_emb' not in g.nodes[ntype].data:
            g.nodes[ntype].data['bert_emb'] = \
                    dgl.distributed.DistTensor(
                        (g.number_of_nodes(ntype), hidden_size),
                        name="bert_emb",
                        dtype=th.float16 if use_fp16 else th.float32,
                        part_policy=g.get_node_partition_policy(ntype),
                        persistent=True)
        input_emb = g.nodes[ntype].data['bert_emb']
        infer_nodes = dgl.distributed.node_split(
                th.ones((g.number_of_nodes(ntype),), dtype=th.bool),
                partition_book=g.get_partition_book(),
                ntype=ntype, force_even=False)
        logging.debug("node %s, local infer set: %d, batch size: %d",
                      ntype, len(infer_nodes), lm_infer_batch_size)

        node_list = th.split(infer_nodes, lm_infer_batch_size)
        input_ntypes = [ntype]
        for input_nodes in node_list:
            input_lm_feats = {}
            input_lm_feats[ntype] = {
                    fname: feat[input_nodes] for fname, feat in lm_node_feat.items()
            }
            text_embs = lm_model(input_ntypes, input_lm_feats)
            if use_fp16:
                input_emb[input_nodes] = text_embs[ntype].half().to('cpu')
            else:
                input_emb[input_nodes] = text_embs[ntype].to('cpu')
        barrier()
        lm_emb_cache[ntype] = input_emb
        lm_model.train()

class LMModels(nn.Module):
    """ LM model collection

    This class maintains the mapping between node type and the LM model
    as well as the corresponding information of the LM model.

    The class ensures that the connection between the LM model and
    the node type is fixed. We always get the right LM model for a given node type.

    Parameters
    ----------
    g : DistGraph
        The distributed graph.
    node_lm_configs:
        A list of language model configurations.
    num_train: int
        Number of trainable texts
    lm_infer_batch_size: int
        Batch size used for computing text embeddings for static lm model
    """
    def __init__(self, g, node_lm_configs, num_train, lm_infer_batch_size):
        super(LMModels, self).__init__()
        # More than one node type may share the same BERT model.
        # To avoid duplicate BERT models, we save add the BERT model
        # in the module dict once, whose key is the node types of the BERT model.
        # We then maintain a mapping from the node type to the key
        # to help find the right BERT model for a node type.
        self._lm_map = {}
        self._lm_models = nn.ModuleDict()
        self._lm_node_feats = {}
        for lm_config in node_lm_configs:
            lm_model = init_lm_model(lm_config,
                                     num_train=num_train,
                                     lm_infer_batch_size=lm_infer_batch_size)
            # A list of node types sharing the same lm model
            lm_ntypes = lm_config["node_types"]
            lm_node_feats = get_lm_node_feats(g, lm_model, lm_ntypes)
            for ntype, feats in lm_node_feats.items():
                assert ntype not in self._lm_node_feats, \
                        f"More than one BERT model runs on Node {ntype}."
                self._lm_node_feats[ntype] = feats
            # We should sort the node type list before converting it to the key.
            lm_ntypes.sort()
            key = ','.join(lm_ntypes)
            self._lm_models[key] = lm_model
            for ntype in lm_ntypes:
                self._lm_map[ntype] = key

    def forward(self, input_nodes, lm_emb_cache=None):
        """ Do language model forward pass on input_nodes

        Parameters
        ----------
        input_nodes: dict
            Input nodes for different node types
        lm_emb_cache: dict
            Language model embedding cache for different node types
        """
        lm_feats = {}

        # Get the device from lm_models
        # The cached BERT embedding should be moved to the same device
        # as lm_models.
        dev = self.device
        if lm_emb_cache is not None:
            # No bert training, Get cached LM embedding
            # Note: self.lm_emb_cache is initialized by calling warmup
            for ntype, idx in input_nodes.items():
                if ntype in lm_emb_cache:
                    lm_feats[ntype] = lm_emb_cache[ntype][idx].to(dev).float()
        else:
            # TODO: Release the bert cache properly
            #       This may need support from DistDGL
            # Need bert training
            for ntype in self.ntypes:
                lm_node_feat = self.get_lm_node_feat(ntype)
                lm_model = self.get_lm_model(ntype)
                if ntype in input_nodes:
                    input_lm_feat = {
                            fname: feat[input_nodes[ntype]].to(dev) \
                                    for fname, feat in lm_node_feat.items()
                        }
                    lm_feats.update(lm_model([ntype], {ntype: input_lm_feat}))
        return lm_feats

    def get_lm_model(self, ntype):
        """ Get a LM model for a node type

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        nn.Module : the LM model for the node type.
        """
        idx = self._lm_map[ntype]
        return self._lm_models[idx]

    def get_lm_node_feat(self, ntype):
        """ Get the LM node features.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        dict : the node features of the node type.
        """
        return self._lm_node_feats[ntype]

    @property
    def ntypes(self):
        """ Get all node types with text features.

        Returns
        -------
        list of str : the node types with text features.
        """
        return list(self._lm_map.keys())

    @property
    def feat_size(self):
        """ The feature size of the BERT model.
        """
        assert len(self._lm_models) > 0
        for model in self._lm_models.values():
            return model.feat_size
        return -1

    @property
    def device(self):
        """ The device where the model is on.
        """
        assert len(self._lm_models) > 0
        for model in self._lm_models.values():
            return next(model.parameters()).device
        return None

    @property
    def lm_models(self):
        """ The list of LM models.
        """
        return list(self._lm_models.values())

class GSPureLMNodeInputLayer(GSNodeInputLayer):
    """The input embedding layer with language model only for all nodes in a
    heterogeneous graph.

    The input layer only has the language model layer and each node type should
    have text feature. The output dimension will be the same as the output
    dimension of the language model.

    Use GSLMNodeEncoderInputLayer if there are extra node features or a different
    output dimension is required.

    Parameters
    ----------
    g: DistGraph
        The distributed graph.
    node_lm_configs:
        A list of language model configurations.
    num_train: int
        Number of trainable texts. Default: 0
    lm_infer_batch_size: int
        Batch size used for computing text embeddings for static lm model. Default: 16
    use_fp16 : bool 
        Use float16 to store BERT embeddings. Default: True
    
    Examples:
    ----------

    .. code:: python

        from graphstorm.model import GSgnnNodeModel, GSPureLMNodeInputLayer
        from graphstorm.dataloading import GSgnnNodeTrainData

        node_lm_configs = [
            {
                "lm_type": "bert",
                "model_name": "bert-base-uncased",
                "gradient_checkpoint": True,
                "node_types": ['a']
            }
        ]
        np_data = GSgnnNodeTrainData(...)
        model = GSgnnNodeModel(...)
        lm_train_nodes=10
        encoder = GSPureLMNodeInputLayer(g=np_data.g, node_lm_configs=node_lm_configs,
                                        num_train=lm_train_nodes)
        model.set_node_input_encoder(encoder)
    """
    def __init__(self,
                 g,
                 node_lm_configs,
                 num_train=0,
                 lm_infer_batch_size=16,
                 use_fp16=True):
        super(GSPureLMNodeInputLayer, self).__init__(g)
        assert node_lm_configs is not None and len(node_lm_configs) > 0, \
            "language model configurations must be provided"

        self._lm_models = LMModels(g, node_lm_configs, num_train, lm_infer_batch_size)
        self.num_train = num_train
        self.lm_infer_batch_size = lm_infer_batch_size
        self.use_fp16 = use_fp16
        self.use_cache = False
        self.lm_emb_cache = {}

        self._feat_size = self._lm_models.feat_size
        for lm_model in self._lm_models.lm_models:
            assert self.out_dims == lm_model.feat_size, \
                "All Language models should have the same output embedding " \
                "dimension, otherwise please use GSLMNodeEncoderInputLayer " \
                "(--model-encoder-type mlp) instead of GSLMNodeLMInputLayer " \
                "(--model-encoder-type lm)"

    def get_general_dense_parameters(self):
        """ Get dense layers' parameters.

        Returns
        -------
        list of Tensors: the dense parameters
        """
        # There is no dense parameters
        return []

    def get_lm_dense_parameters(self):
        """ get the language model related parameters

        Returns
        -------
        list of Tensors: the language model parameters.
        """
        return self._lm_models.parameters()

    def prepare(self, g):
        # If there is no trainable nodes, freeze Bert layer.
        if self.num_train == 0:
            self.freeze(g)

    def freeze(self, g):
        """ Generate Bert caching if needed

        Parameters
        ----------
        g : DistGraph
            The distributed graph object.
        """
        # The lm_emb_cache is used in following cases:
        # 1) We don't need to fine-tune Bert, i.e., train_nodes == 0.
        #    In this case, we only generate bert lm_emb_cache once before model training.
        #
        # 2) GNN warnup when lm_freeze_epochs > 0 (controlled by trainer)
        #    We generate the bert emb_cache before model training.
        #    In the first lm_freeze_epochs epochs, the number of trainable text
        #    nodes are set to 0 and the lm_emb_cache is not refreshed.
        #
        # 3) if train_nodes > 0, no emb_cache is used unless Case 2.
        update_bert_cache(g,
                          self._lm_models,
                          self.lm_emb_cache,
                          self.lm_infer_batch_size,
                          use_fp16=self.use_fp16)
        self.use_cache = True

    def require_cache_embed(self):
        """ Whether to cache the embeddings for inference.

        Returns
        -------
        Bool : return True to cache the embeddings for inference.
        """
        return True

    #pylint: disable=keyword-arg-before-vararg
    #pylint: disable=unused-argument
    def forward(self, input_feats, input_nodes):
        """Forward computation

        Parameters
        ----------
        input_feats: dict
            input features, ignored
        input_nodes: dict
            input node ids

        Returns
        -------
        a dict of Tensor: the node embeddings.
        """
        assert isinstance(input_nodes, dict), 'The input node IDs should be in a dict.'

        cache = self.lm_emb_cache if len(self.lm_emb_cache) > 0 and self.use_cache else None
        embs = self._lm_models(input_nodes, lm_emb_cache=cache)
        # This module is only used for computing the BERT embeddings on the node types
        # with text features. If it is asked to compute embeddings for some nodes without
        # text features, it should report an error.
        for ntype in input_nodes:
            assert ntype in embs, f"Cannot compute BERT embeddings for node {ntype}."
        return embs

    @property
    def out_dims(self):
        return self._feat_size

class GSLMNodeEncoderInputLayer(GSNodeEncoderInputLayer):
    """The input encoder layer with language model for all nodes in a heterogeneous graph.

    The input layer adds language model layer on nodes with textual node features and
    generate LM embeddings using the LM model. The LM embeddings are then treated
    as node features.

    The input layer adds learnable embeddings on nodes if the nodes do not have features.
    It adds a linear layer on nodes with node features and the linear layer
    projects the node features to a specified dimension. A user can add learnable
    embeddings on the nodes with node features. In this case, the input layer
    combines the node features with the learnable embeddings and project them to
    the specified dimension.

    Parameters
    ----------
    g: DistGraph
        The distributed graph
    node_lm_configs:
        A list of language model configurations.
    feat_size : dict of int
        The original feat sizes of each node type
    embed_size : int
        The embedding size
    num_train: int
        Number of trainable texts. Default: 0
    lm_infer_batch_size: int
        Batch size used for computing text embeddings for static lm model. Default: 16
    activation : func
        The activation function. Default: None
    dropout : float
        The dropout parameter. Default: 0.0
    use_node_embeddings : bool
        Whether we will use the node embeddings for individual nodes even when node features are
        available. Default: False
    use_fp16 : bool
        Use float16 to store the BERT embeddings. Default: True

    Examples:
    ----------

    .. code:: python

        from graphstorm import get_feat_size
        from graphstorm.model import GSgnnNodeModel, GSLMNodeEncoderInputLayer
        from graphstorm.dataloading import GSgnnNodeTrainData
        np_data = GSgnnNodeTrainData(...)
        model = GSgnnNodeModel(...)
        feat_size = get_feat_size(np_data.g, 'feat')
        node_lm_configs = [{"lm_type": "bert",
                        "model_name": "bert-base-uncased",
                        "gradient_checkpoint": True,
                        "node_types": ['a']}]
        lm_train_nodes=10

        encoder = GSLMNodeEncoderInputLayer(
            g=np_data.g, 
            node_lm_configs=node_lm_configs,
            feat_size=feat_size, 
            embed_size=128, 
            num_train=lm_train_nodes
        )
        model.set_node_input_encoder(encoder)
    """
    def __init__(self,
                 g,
                 node_lm_configs,
                 feat_size,
                 embed_size,
                 num_train=0,
                 lm_infer_batch_size=16,
                 activation=None,
                 dropout=0.0,
                 use_node_embeddings=False,
                 use_fp16=True):
        assert node_lm_configs is not None and len(node_lm_configs) > 0, \
            "language model configurations must be provided"

        lm_models = LMModels(g, node_lm_configs, num_train, lm_infer_batch_size)
        adjust_feat_size = dict(feat_size)
        for lm_config in node_lm_configs:
            # A list of node types sharing the same lm model
            lm_ntypes = lm_config["node_types"]
            # Update feature size
            for ntype in lm_ntypes:
                adjust_feat_size[ntype] += lm_models.feat_size
                if get_rank() == 0:
                    logging.debug('Node %s adds lm %s features %d->%d',
                                  ntype, lm_config["lm_type"], feat_size[ntype],
                                  adjust_feat_size[ntype])

        self.num_train = num_train
        self.use_fp16 = use_fp16
        self.lm_infer_batch_size = lm_infer_batch_size
        self.use_cache = False
        self.lm_emb_cache = {}

        super(GSLMNodeEncoderInputLayer, self).__init__(
            g, adjust_feat_size, embed_size,
            activation, dropout, use_node_embeddings)
        self._lm_models = lm_models

    def get_general_dense_parameters(self):
        """ Get dense layers' parameters.

        Returns
        -------
        list of Tensors: the dense parameters
        """
        params = list(self.proj_matrix.parameters()) \
            if self.proj_matrix is not None else []
        params += list(self.input_projs.parameters())
        return params

    def get_lm_dense_parameters(self):
        """ get the language model related parameters

        Returns
        -------
        list of Tensors: the language model parameters.
        """
        return self._lm_models.parameters()

    def prepare(self, g):
        # If there is no trainable nodes, freeze Bert layer.
        if self.num_train == 0:
            self.freeze(g)

    def freeze(self, g):
        """ Generate Bert caching if needed

        Parameters
        ----------
        g : DistGraph
            The distributed graph object.
        """
        # The lm_emb_cache is used in following cases:
        # 1) We don't need to fine-tune Bert, i.e., train_nodes == 0.
        #    In this case, we only generate bert lm_emb_cache once before model training.
        #
        # 2) GNN warnup when lm_freeze_epochs > 0 (controlled by trainer)
        #    We generate the bert emb_cache before model training.
        #    In the first lm_freeze_epochs epochs, the number of trainable text
        #    nodes are set to 0 and the lm_emb_cache is not refreshed.
        #
        # 3) if train_nodes > 0, no emb_cache is used unless Case 2.
        update_bert_cache(g,
                          self._lm_models,
                          self.lm_emb_cache,
                          self.lm_infer_batch_size,
                          use_fp16=self.use_fp16)
        self.use_cache = True

    def unfreeze(self):
        """ Disable Bert caching
        """
        if self.num_train != 0:
            self.use_cache = False

    def require_cache_embed(self):
        """ Whether to cache the embeddings for inference.

        Returns
        -------
        Bool : return True to cache the embeddings for inference.
        """
        return True

    #pylint: disable=keyword-arg-before-vararg
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

        # Compute language model features first
        cache = self.lm_emb_cache if len(self.lm_emb_cache) > 0 and self.use_cache else None
        lm_feats = self._lm_models(input_nodes, lm_emb_cache=cache)

        for ntype, lm_feat in lm_feats.items():
            # move lm_feat to the right device
            # we assume input_feats has already been moved to that device.
            lm_feat = lm_feat.to(next(self.parameters()).device)
            if ntype in input_feats:
                input_feats[ntype] = th.cat((input_feats[ntype].float(), lm_feat), dim=-1)
            else:
                input_feats[ntype] = lm_feat

        return super(GSLMNodeEncoderInputLayer, self).forward(input_feats, input_nodes)
