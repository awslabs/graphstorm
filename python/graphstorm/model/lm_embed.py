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
import time
import os
import hashlib

import numpy as np
import torch as th
from torch import nn
import dgl

from .embed import GSNodeInputLayer
from .embed import GSNodeEncoderInputLayer
from .lm_model import init_lm_model
from .lm_model import get_lm_node_feats
from .utils import load_pytorch_embedding, save_embeddings
from ..utils import get_rank, get_world_size, barrier, create_dist_tensor

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
        self._lm_model_names = {}
        self._lm_node_feats = {}
        for lm_config in node_lm_configs:
            lm_model = init_lm_model(lm_config,
                                     num_train=num_train,
                                     lm_infer_batch_size=lm_infer_batch_size)
            # A list of node types sharing the same lm model
            lm_ntypes = lm_config["node_types"]
            for ntype in lm_ntypes:
                self._lm_model_names[ntype] = lm_config["model_name"]
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
                if ntype in lm_emb_cache.ntypes:
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

    def get_lm_model_name(self, ntype):
        """ Get the LM model name on a node type.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        str : the model name
        """
        return self._lm_model_names[ntype]

    def get_lm_model_hash(self, ntype):
        """ Compute the hash code of a LM model on a given node type.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        str : the hash code of the model.
        """
        weights = [th.flatten(param.data) for param in self.get_lm_model(ntype).parameters()]
        weights = th.cat(weights).cpu().numpy()
        return hashlib.sha1(weights.view(np.uint8)).hexdigest()

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

    def get_feat_size(self, ntype):
        """ Get the LM output feature size for a node type.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        int : The feature size of the LM model
        """
        assert len(self._lm_models) > 0
        lm_type = self._lm_map[ntype]
        return self._lm_models[lm_type].feat_size

    @property
    def ntypes(self):
        """ Get all node types with text features.

        Returns
        -------
        list of str : the node types with text features.
        """
        return list(self._lm_map.keys())

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

class LMCache:
    """ Cache for the LM embeddings.

    Parameters
    ----------
    lm_models: LMModels
        A collection of LM models and related information.
    embed_path : str
        The path where the embedding files are stored.
    """
    def __init__(self, g, lm_models, embed_path=None):
        self._g = g
        self._lm_models = lm_models
        self._lm_emb_cache = {}
        self._embed_path = embed_path

    def _get_model_hash(self, ntype):
        """ Get the hash code of a model.

        If necessary, we may cache the hash code in the future.
        """
        return self._lm_models.get_lm_model_hash(ntype)

    def _get_model_name(self, ntype):
        """ Get the model name

        The model name should be the original model name followed with a hash code:
        "model_name"-"hash_code"

        """
        model_name = self._lm_models.get_lm_model_name(ntype)
        assert "/" not in model_name, \
                f"We only support builtin LM models for now. The model name is {model_name}."
        model_hash = self._get_model_hash(ntype)
        # We only take the first 10 characters of the hash code to construct the model name.
        return model_name + "-" + model_hash[0:10]

    def _load_embeddings(self):
        """ Load LM embeddings from files.
        """
        embed_ndata_names = self.embed_ndata_name
        for ntype in self._lm_models.ntypes:
            embed_path = os.path.join(os.path.join(self._embed_path, ntype),
                    self._get_model_name(ntype))
            if os.path.exists(embed_path):
                if get_rank() == 0:
                    logging.info("load LM embedding from %s for node type %s",
                            embed_path, ntype)
                embed_name = embed_ndata_names[ntype]
                self._lm_emb_cache[ntype] = load_pytorch_embedding(embed_path,
                        self._g.get_node_partition_policy(ntype), embed_name)

    def _save_embeddings(self):
        """ Save LM embeddings.
        """
        for ntype in self._lm_models.ntypes:
            embed_path = os.path.join(os.path.join(self._embed_path, ntype),
                    self._get_model_name(ntype))
            save_embeddings(embed_path, self._lm_emb_cache[ntype], get_rank(), get_world_size())

    def __len__(self):
        return len(self._lm_emb_cache)

    def __getitem__(self, ntype):
        """ Get the cached embedding of a node type.
        """
        return self._lm_emb_cache[ntype]

    def clear_cache(self):
        """ Clear up the cached embeddings.
        """
        self._lm_emb_cache = {}

    @property
    def ntypes(self):
        """ Get the node types with embedding cache.
        """
        return self._lm_models.ntypes

    @property
    def embed_ndata_name(self):
        """ The embed name of the node data
        """
        return {ntype: "bert_emb" for ntype in self.ntypes}

    def update_cache(self, lm_infer_batch_size, use_fp16=True):
        """ Update the LM embedding cache.

        Parameters
        ----------
        lm_infer_batch_size: int
            Language model inference batch size
        use_fp16 : bool
            Use float16 to store BERT embeddings.
        """
        # If the embeddings have been cached, we just load them instead of
        # computing them from scratch.
        if self._embed_path is not None:
            self._load_embeddings()

        # If all embeddings are cached, don't compute the embeddings again.
        if np.all([ntype in self._lm_emb_cache for ntype in self._lm_models.ntypes]):
            return

        embed_ndata_names = self.embed_ndata_name
        for ntype in self._lm_models.ntypes:
            if get_rank() == 0:
                logging.debug("compute embedding for node type %s", ntype)
            start = time.time()
            lm_model = self._lm_models.get_lm_model(ntype)
            lm_node_feat = self._lm_models.get_lm_node_feat(ntype)
            lm_model.eval()
            hidden_size = lm_model.feat_size
            if ntype not in self._lm_emb_cache:
                embed_name = embed_ndata_names[ntype]
                self._lm_emb_cache[ntype] = create_dist_tensor(
                        (self._g.number_of_nodes(ntype), hidden_size),
                        name=embed_name,
                        dtype=th.float16 if use_fp16 else th.float32,
                        part_policy=self._g.get_node_partition_policy(ntype),
                        persistent=True)
            emb = self._lm_emb_cache[ntype]
            infer_nodes = dgl.distributed.node_split(
                    th.ones((self._g.number_of_nodes(ntype),), dtype=th.bool),
                    partition_book=self._g.get_partition_book(),
                    ntype=ntype, force_even=False)
            logging.debug("node %s, local infer set: %d, batch size: %d",
                          ntype, len(infer_nodes), lm_infer_batch_size)

            node_list = th.split(infer_nodes, lm_infer_batch_size)
            input_ntypes = [ntype]
            with th.no_grad():
                for input_nodes in node_list:
                    input_lm_feats = {}
                    input_lm_feats[ntype] = {
                            fname: feat[input_nodes] for fname, feat in lm_node_feat.items()
                    }
                    text_embs = lm_model(input_ntypes, input_lm_feats)
                    if use_fp16:
                        emb[input_nodes] = text_embs[ntype].half().to('cpu')
                    else:
                        emb[input_nodes] = text_embs[ntype].to('cpu')
            barrier()
            if get_rank() == 0:
                logging.info('Computing bert embedding on node %s takes %.3f seconds',
                             ntype, time.time() - start)
            lm_model.train()

        if self._embed_path is not None:
            self._save_embeddings()

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
        Use float16 to store LM embeddings. Default: True
    cached_embed_path : str
        The path where the LM embeddings are cached.
    
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
                 use_fp16=True,
                 cached_embed_path=None):
        super(GSPureLMNodeInputLayer, self).__init__(g)
        assert node_lm_configs is not None and len(node_lm_configs) > 0, \
            "language model configurations must be provided"

        self._lm_models = LMModels(g, node_lm_configs, num_train, lm_infer_batch_size)
        self.num_train = num_train
        self.lm_infer_batch_size = lm_infer_batch_size
        self.use_fp16 = use_fp16
        self.use_cache = False
        self.lm_emb_cache = LMCache(g, self._lm_models, embed_path=cached_embed_path)

        self._feat_size = self._lm_models.get_feat_size(self._lm_models.ntypes[0])
        for lm_model in self._lm_models.lm_models:
            assert self._feat_size == lm_model.feat_size, \
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
        self.lm_emb_cache.update_cache(self.lm_infer_batch_size, use_fp16=self.use_fp16)
        self.use_cache = True

    def unfreeze(self):
        """ Disable Bert caching
        """
        if self.num_train != 0:
            self.use_cache = False
        self.lm_emb_cache.clear_cache()

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

        The forward function only computes the BERT embeddings and
        ignore the input node features if there are node features.

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
    def in_dims(self):
        """ The input feature size.

        The BERT embeddings are usually pre-computed as node features.
        So we consider the BERT embedding size as input node feature size.
        """
        return self._feat_size

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
    cached_embed_path : str
        The path where the LM embeddings are cached.

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
                 use_fp16=True,
                 cached_embed_path=None,
                 force_no_embeddings=None):
        assert node_lm_configs is not None and len(node_lm_configs) > 0, \
            "language model configurations must be provided"

        lm_models = LMModels(g, node_lm_configs, num_train, lm_infer_batch_size)
        adjust_feat_size = dict(feat_size)
        for lm_config in node_lm_configs:
            # A list of node types sharing the same lm model
            lm_ntypes = lm_config["node_types"]
            # Update feature size
            for ntype in lm_ntypes:
                adjust_feat_size[ntype] += lm_models.get_feat_size(ntype)
                if get_rank() == 0:
                    logging.debug('Node %s adds lm %s features %d->%d',
                                  ntype, lm_config["lm_type"], feat_size[ntype],
                                  adjust_feat_size[ntype])

        self.num_train = num_train
        self.use_fp16 = use_fp16
        self.lm_infer_batch_size = lm_infer_batch_size
        self.use_cache = False
        self.lm_emb_cache = LMCache(g, lm_models, embed_path=cached_embed_path)

        super(GSLMNodeEncoderInputLayer, self).__init__(
            g, adjust_feat_size, embed_size,
            activation, dropout, use_node_embeddings,
            force_no_embeddings=force_no_embeddings)
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

    def freeze(self, _):
        """ Generate Bert caching if needed
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
        self.lm_emb_cache.update_cache(self.lm_infer_batch_size, use_fp16=self.use_fp16)
        self.use_cache = True

    def unfreeze(self):
        """ Disable Bert caching
        """
        if self.num_train != 0:
            self.use_cache = False
        self.lm_emb_cache.clear_cache()

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

        The forward function computes the BERT embeddings and combine them with
        the input node features.

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
            lm_feat = lm_feat.to(self.device)
            if ntype in input_feats:
                input_feats[ntype] = th.cat((input_feats[ntype].float(), lm_feat), dim=-1)
            else:
                input_feats[ntype] = lm_feat

        return super(GSLMNodeEncoderInputLayer, self).forward(input_feats, input_nodes)
