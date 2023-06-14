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

    GNN model in GraphStorm
"""

import abc
import time
import torch as th
import dgl
from torch import nn

from .utils import load_model as load_gsgnn_model
from .utils import save_model as save_gsgnn_model
from .utils import save_opt_state, load_opt_state
from .utils import save_sparse_embeds, load_sparse_embeds
from .utils import create_sparse_embeds_path
from .embed import compute_node_input_embeddings
from .embed import GSNodeInputLayer
from .gs_layer import GSLayerBase
from .gnn_encoder_base import dist_inference
from ..utils import get_rank
from ..dataloading.dataset import prepare_batch_input

from ..config import (GRAPHSTORM_MODEL_ALL_LAYERS,
                      GRAPHSTORM_MODEL_EMBED_LAYER,
                      GRAPHSTORM_MODEL_GNN_LAYER,
                      GRAPHSTORM_MODEL_DECODER_LAYER)

class GSOptimizer():
    """ A combination of optimizers.

    A model might requires multiple optimizers. For example, we need a sparse optimizer
    for sparse embeddings, an optimizer to fine-tune the BERT module and an optimizer for
    the dense modules.

    Parameters
    ----------
    dense_opts : list of th.optim.Optimizer
        Optimizer objects for dense model parameters.
        TODO(xiangsx): we only retrieve the first item of the list as
        we only support one dense optimizer.
    lm_opts: list of th.optim.Optimizer
        Optimizer objects for language model parameters.
        TODO(xiangsx): we only retrieve the first item of the list as
        we only support one language model optimizer.
        Note: by default it is None.
    sparse_opts : list
        A list of optimizer objects for sparse model parameters.
        Note: by default it is None. No sparse optimizer is used.
    """
    def __init__(self, dense_opts=None, lm_opts=None, sparse_opts=None):
        if dense_opts is None:
            # There will be no dense optimizer
            # When doing graph-ware language model finetuning
            dense_opts = []
        if lm_opts is None:
            # If language model is not used, there will be no lm optimizer
            lm_opts = []
        if sparse_opts is None:
            sparse_opts = []
        assert isinstance(dense_opts, list), \
            "dense_opts(dense model) should be a list or None."
        assert isinstance(lm_opts, list), \
            "lm_opts (language model optimizers) should be a list or None."
        assert isinstance(sparse_opts, list), \
            "sparse_opts should be a list or None."
        self.dense_opts = dense_opts
        self.lm_opts = lm_opts
        self.sparse_opts = sparse_opts
        all_opts = dense_opts + lm_opts + sparse_opts
        assert len(all_opts) > 0, "Optimizer list need to be defined"
        for optimizer in all_opts:
            assert optimizer is not None

    def zero_grad(self):
        """ Setting the gradient to zero
        """
        all_opts = self.dense_opts + self.lm_opts + self.sparse_opts
        for optimizer in all_opts:
            optimizer.zero_grad()

    def step(self):
        """ Moving the optimizer
        """
        all_opts = self.dense_opts + self.lm_opts + self.sparse_opts
        for optimizer in all_opts:
            optimizer.step()

    def load_opt_state(self, path, device=None):
        """ Load the optimizer states
        """
        load_opt_state(path, self.dense_opts, self.lm_opts, self.sparse_opts)
        if device is not None:
            self.move_to_device(device)

    def save_opt_state(self, path):
        """ Save the optimizer states.
        """
        if get_rank() == 0:
            save_opt_state(path, self.dense_opts, self.lm_opts, self.sparse_opts)

    def move_to_device(self, device):
        """ Move the optimizer to the specified device.
        """
        # Move the optimizer state to the specified device.
        # We only need to move the states of the dense optimizers.
        # The states of the sparse optimizers should stay in CPU.
        for opt in self.dense_opts:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, th.Tensor):
                        state[k] = v.to(device)

        for opt in self.lm_opts:
            for state in opt.state.values():
                for k, v in state.items():
                    if isinstance(v, th.Tensor):
                        state[k] = v.to(device)

class GSgnnModelBase(nn.Module):
    """ GraphStorm GNN model base class

    Any GNN model trained by GraphStorm should inherit from this class. It contains
    some abstract methods that should be defined in the GNN model classes.
    It also provides some utility methods.
    """

    @abc.abstractmethod
    def restore_model(self, restore_model_path, model_layer_to_load):
        """Load saving checkpoints of a GNN model.

        A user who implement this method should load the parameters of the GNN model.
        This method does not need to load the optimizer state.

        Parameters
        ----------
        restore_model_path : str
            The path where we can restore the model.
        model_layer_to_load: list of str
            list of model layers to load. Supported layers include
            'gnn', 'embed', 'decoder'
        """

    @abc.abstractmethod
    def save_model(self, model_path):
        ''' Save the GNN model.

        When saving a GNN model, we need to save the dense parameters and sparse parameters.

        Parameters
        ----------
        model_path : str
            The path where all model parameters and optimizer states are saved.
        '''

    @abc.abstractmethod
    def create_optimizer(self):
        """Create the optimizer that optimizes the model.

        A user who defines a model should also define the optimizer for this model.
        By using this method, a user can define the optimization algorithm,
        the learning rate as well as any other hyperparameters.

        A model may require multiple optimizers. For example, we should define
        an optimizer for sparse embeddings and an optimizer for the dense parameters
        of a GNN model. In this case, a user should use a GSOptimizer to combine these
        optimizers.

        Example:
        Case 1: if there is only one optimizer:
            def create_optimizer(self):
                # define torch.optim.Optimizer
                return optimizer

        Case 2: if there are both dense and sparse optimizers:
            def create_optimizer(self):
                dense = [dense_opt] # define torch.optim.Optimizer
                sparse = [sparse_opt] # define dgl sparse Optimizer
                optimizer = GSOptimizer(dense_opts=dense,
                                        lm_opts=None,
                                        sparse_opts=sparse)
                return optimizer
        """

    #pylint: disable=unused-argument
    def prepare_input_encoder(self, train_data):
        """ Preparing input layer for training or inference.
            The input layer can pre-compute node features in the preparing step
            if needed. For example pre-compute all BERT embeddings

            Default: do nothing

        Parameters
        ----------
        train_data: GSgnnData
            Graph data
        """

    #pylint: disable=unused-argument
    def freeze_input_encoder(self, train_data):
        """ Freeze input layer for model training.

            Default: do nothing

        Parameters
        ----------
        train_data: GSgnnData
            Graph data
        """

    #pylint: disable=unused-argument
    def unfreeze_input_encoder(self):
        """ Unfreeze input layer for model training

            Default: do nothing
        """

    @property
    def device(self):
        """ The device where the model runs.

        Here we assume that all model parameters are on the same device.
        """
        return next(self.parameters()).device

# This class does not implement the nn.Module's forward abstract method. Because children classes
# implement this, so disable here.
class GSgnnModel(GSgnnModelBase):    # pylint: disable=abstract-method
    """ GraphStorm GNN model

    This class provides a GraphStorm GNN model implementation split into five components:
    node input encoder, edge input encoder, GNN encoder, decoder and loss function.
    These components can be customized by a user.
    """
    def __init__(self):
        super(GSgnnModel, self).__init__()
        self._node_input_encoder = None
        self._edge_input_encoder = None
        self._gnn_encoder = None
        self._decoder = None
        self._loss_fn = None
        self._optimizer = None

    def get_dense_params(self):
        """retrieve the all dense layers' parameters as a parameter list except for
        language model related parameters.

        Returns
        -------
        list of Parameters: the dense parameters
        """
        params = []
        if self.gnn_encoder is not None and isinstance(self.gnn_encoder, nn.Module):
            params += list(self.gnn_encoder.parameters())
        if self.node_input_encoder is not None:
            assert isinstance(self.node_input_encoder, GSNodeInputLayer), \
                "node_input_encoder must be a GSNodeInputLayer"
            params += list(self.node_input_encoder.get_general_dense_parameters())
        # TODO(zhengda) we need to test a model with encoders on edge data.
        if self.edge_input_encoder is not None:
            assert False, "edge_input_encoder not supported"
            params += list(self.edge_input_encoder.get_general_dense_parameters())
        if self.decoder is not None and isinstance(self.decoder, nn.Module):
            params += list(self.decoder.parameters())
        return params

    def get_lm_params(self):
        """ get the language model related parameters

        Returns
        -------
        list of Parameters: the language model parameters.
        """
        params = []
        if self.node_input_encoder is not None:
            assert isinstance(self.node_input_encoder, GSNodeInputLayer), \
                "node_input_encoder must be a GSNodeInputLayer"
            params += list(self.node_input_encoder.get_lm_dense_parameters())
        if self.edge_input_encoder is not None:
            assert False, "edge_input_encoder not supported"
            params += list(self.edge_input_encoder.get_lm_dense_parameters())

        return params

    def get_sparse_params(self):
        """ get the sparse parameters of the model.

        Returns
        -------
        list of Tensors: the sparse embeddings.
        """
        params = []
        if self.node_input_encoder is not None:
            params += self.node_input_encoder.get_sparse_params()
        return params

    def set_node_input_encoder(self, encoder):
        """set the input encoder for nodes.

        Parameters
        ----------
        encoder : GSLayer
            The encoder for node features.
        """
        assert isinstance(encoder, GSLayerBase), \
                'The node input encoder has the class of GSLayerBase.'
        if self.gnn_encoder is not None:
            assert encoder.out_dims == self.gnn_encoder.in_dims, \
                    'The output dimensions of the node input encoder should ' \
                    + 'match the input dimension of the GNN encoder.'
        self._node_input_encoder = encoder

    def set_edge_input_encoder(self, encoder):
        """set the input encoder for edges.

        Parameters
        ----------
        encoder : GSLayer
            The encoder for edge features.
        """
        assert isinstance(encoder, GSLayerBase), \
                'The edge input encoder should be the class of GSLayerBase.'
        # TODO(zhengda) we need to check the dimensions.
        self._edge_input_encoder = encoder

    def set_gnn_encoder(self, encoder):
        """set the GNN layers.

        Parameters
        ----------
        encoder : GSLayer
            The GNN encoder.
        """
        # GNN encoder is not used.
        if encoder is None:
            self._gnn_encoder = None
            if self.node_input_encoder is not None and self.decoder is not None:
                assert self.node_input_encoder.out_dims == self.decoder.in_dims, \
                    'When GNN encoder is not used, the output dimensions of ' \
                    'the node input encoder should match the input dimension of' \
                    'the decoder.'
            return

        assert isinstance(encoder, GSLayerBase), \
                'The GNN encoder should be the class of GSLayerBase.'
        if self.node_input_encoder is not None:
            assert self.node_input_encoder.out_dims == encoder.in_dims, \
                    'The output dimensions of the node input encoder should ' \
                    + 'match the input dimension of the GNN encoder.'
        if self.decoder is not None:
            assert encoder.out_dims == self.decoder.in_dims, \
                    'The output dimensions of the GNN encoder should ' \
                    + 'match the input dimension of the decoder.'
        self._gnn_encoder = encoder

    def set_decoder(self, decoder):
        """set the decoder layer.

        Parameters
        ----------
        decoder : GSLayer
            The decoder.
        """
        assert isinstance(decoder, GSLayerBase), \
                'The decoder should be the class of GSLayerBase.'
        if self.gnn_encoder is not None:
            assert self.gnn_encoder.out_dims == decoder.in_dims, \
                    'The output dimensions of the GNN encoder should ' \
                    + 'match the input dimension of the decoder.'
        self._decoder = decoder

    def set_loss_func(self, loss_fn):
        """set the loss function.

        Parameters
        ----------
        loss_fn : Pytorch nn.Module
            The loss function.
        """
        assert isinstance(loss_fn, nn.Module), \
                'The loss function should be the class of nn.Module.'
        self._loss_fn = loss_fn

    def prepare_input_encoder(self, train_data):
        """ Preparing input layer for training or inference.
        """
        if self._node_input_encoder is not None:
            self._node_input_encoder.prepare(train_data.g)

        if self._edge_input_encoder is not None:
            self._edge_input_encoder.prepare(train_data.g)

    def freeze_input_encoder(self, train_data):
        """ Freeze input layer for model training.
        """
        if self._node_input_encoder is not None:
            self._node_input_encoder.freeze(train_data.g)

        if self._edge_input_encoder is not None:
            self._edge_input_encoder.freeze(train_data.g)

    def unfreeze_input_encoder(self):
        """ Unfreeze input layer for model training
        """
        if self._node_input_encoder is not None:
            self._node_input_encoder.unfreeze()

        if self._edge_input_encoder is not None:
            self._edge_input_encoder.unfreeze()

    def restore_model(self, restore_model_path,
                      model_layer_to_load=None):
        """load saving checkpoints for GNN models.

        Parameters
        ----------
        restore_model_path : str
            The path where we can restore the model.
        model_layer_to_load: list of str
            list of model layers to load. Supported layers include
            'gnn', 'embed', 'decoder'
        """
        # Restore the model weights from a checkpoint saved previously.
        if restore_model_path is not None:
            print('load GNN model from ', restore_model_path)
            # TODO(zhengda) we need to load edge_input_encoder.
            model_layer_to_load = GRAPHSTORM_MODEL_ALL_LAYERS \
                if model_layer_to_load is None else model_layer_to_load
            load_gsgnn_model(restore_model_path,
                self.gnn_encoder \
                    if GRAPHSTORM_MODEL_GNN_LAYER in model_layer_to_load else None,
                self.node_input_encoder \
                    if GRAPHSTORM_MODEL_EMBED_LAYER in model_layer_to_load else None,
                self.decoder \
                    if GRAPHSTORM_MODEL_DECODER_LAYER in model_layer_to_load else None)

            print('Load Sparse embedding from ', restore_model_path)
            load_sparse_embeds(restore_model_path,
                                self.node_input_encoder,
                                get_rank(),
                                th.distributed.get_world_size())
        # We need to make sure that the sparse embedding is completely loaded
        # before all processes use the model.
        th.distributed.barrier()

    def init_optimizer(self, lr, sparse_optimizer_lr, weight_decay, lm_lr=None):
        """initialize the model's optimizers

        Parameters
        ----------
        lr : float
            The learning rate for dense parameters
            The learning rate for general dense parameters
        sparse_optimizer_lr : float
            The learning rate for sparse parameters
        weight_decay : float
            The weight decay for the optimizer.
        lm_lr: float
            Language model fine-tuning learning rate for
            langauge model dense parameters.
        """
        sparse_params = self.get_sparse_params()
        if len(sparse_params) > 0:
            emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=sparse_optimizer_lr)
            sparse_opts = [emb_optimizer]
        else:
            sparse_opts = []

        dense_params = self.get_dense_params()
        if len(dense_params) > 0:
            optimizer = th.optim.Adam(self.get_dense_params(), lr=lr,
                                      weight_decay=weight_decay)
            dense_opts = [optimizer]
        else:
            dense_opts = []

        lm_params = self.get_lm_params()
        if len(lm_params) > 0:
            lm_optimizer = th.optim.Adam(self.get_lm_params(), \
                                         lr=lm_lr if lm_lr is not None else lr,
                                         weight_decay=weight_decay)
            lm_opts = [lm_optimizer]
        else:
            lm_opts = []

        self._optimizer = GSOptimizer(dense_opts=dense_opts,
                                      lm_opts=lm_opts,
                                      sparse_opts=sparse_opts)

    def create_optimizer(self):
        """the optimizer
        """
        return self._optimizer

    def comput_input_embed(self, input_nodes, input_feats):
        """ Compute input encoder embedding on a minibatch

        Parameters
        ----------
        input_nodes : dict of Tensors
            The input nodes.
        input_feats : dict of Tensors
            The input node features.

        Returns
        -------
        dict of Tensors: The GNN embeddings.
        """
        embs = self.node_input_encoder(input_feats, input_nodes)

        return embs

    def compute_embed_step(self, blocks, input_feats):
        """ Compute the GNN embeddings on a mini-batch.

        This function is used for mini-batch inference.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message flow graphs for computing GNN embeddings.
        input_feats : dict of Tensors
            The input node features.

        Returns
        -------
        dict of Tensors: The GNN embeddings.
        """
        device = blocks[0].device
        if self.node_input_encoder is not None:
            input_nodes = {ntype: blocks[0].srcnodes[ntype].data[dgl.NID].cpu() \
                    for ntype in blocks[0].srctypes}
            embs = self.node_input_encoder(input_feats, input_nodes)
            embs = {name: emb.to(device) for name, emb in embs.items()}
        else:
            embs = input_feats
        if self.gnn_encoder is not None:
            gnn_embs = self.gnn_encoder(blocks, embs)
        else:
            gnn_embs = embs
        return gnn_embs

    def save_model(self, model_path):
        ''' Save the GNN model.

        When saving a GNN model, we need to save the dense parameters and sparse parameters.

        Parameters
        ----------
        model_path : str
            The path where all model parameters and optimizer states are saved.
        '''
        start_save_t = time.time()
        # Only rank 0 save dense model parameters
        # We assume the model is written into a shared filesystem accessable
        # to all trainers.
        if get_rank() == 0:
            save_gsgnn_model(model_path, self.gnn_encoder, self.node_input_encoder, self.decoder)

        # Saving sparse embedding is done in a distributed way.
        if get_rank() == 0:
            # Need to create embedding path and chmod to 0o777 first
            create_sparse_embeds_path(model_path, self.node_input_encoder)
        # make sure rank 0 creates the folder and change permission first
        th.distributed.barrier()

        save_sparse_embeds(model_path,
                           self.node_input_encoder,
                           get_rank(),
                           th.distributed.get_world_size())
        if get_rank() == 0:
            print('successfully save the model to ' + model_path)
            print('Time on save model {}'.format(time.time() - start_save_t))

    @property
    def node_input_encoder(self):
        """the input layer's node encoder used in this GNN class
        """
        return self._node_input_encoder

    @property
    def edge_input_encoder(self):
        """the input layer's edge encoder used in this GNN class
        """
        return self._edge_input_encoder

    @property
    def gnn_encoder(self):
        """the gnn encoder used in this GNN class
        """
        return self._gnn_encoder

    @property
    def num_gnn_layers(self):
        """the number of GNN layers.
        """
        return self.gnn_encoder.num_layers if self.gnn_encoder is not None else 0

    @property
    def decoder(self):
        """the decoder layer used in this GNN class
        """
        return self._decoder

    @property
    def loss_func(self):
        """the loss function used in this GNN class
        """
        return self._loss_fn

def do_full_graph_inference(model, data, batch_size=1024, fanout=None, edge_mask=None,
                            task_tracker=None):
    """ Do fullgraph inference

    It may use some of the edges indicated by `edge_mask` to compute GNN embeddings.

    Parameters
    ----------
    model: torch model
        GNN model
    data : GSgnnData
        The GraphStorm dataset
    batch_size : int
        The batch size for inferencing a GNN layer
    fanout: list of int
        The fanout for computing the GNN embeddings in a GNN layer.
    edge_mask : str
        The edge mask that indicates what edges are used to compute GNN embeddings.
    task_tracker: GSTaskTrackerAbc
        Task tracker

    Returns
    -------
    dict of th.Tensor : node embeddings.
    """
    assert isinstance(model, GSgnnModel), "Only GSgnnModel supports full-graph inference."
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    th.distributed.barrier()
    if model.gnn_encoder is None:
        # Only graph aware but not GNN models
        embeddings = compute_node_input_embeddings(data.g,
                                                   batch_size,
                                                   model.node_input_encoder,
                                                   task_tracker=task_tracker,
                                                   feat_field=data.node_feat_field)
    elif model.node_input_encoder.require_cache_embed():
        # If the input encoder has heavy computation, we should compute
        # the embeddings and cache them.
        input_embeds = compute_node_input_embeddings(data.g,
                                                     batch_size,
                                                     model.node_input_encoder,
                                                     task_tracker=task_tracker,
                                                     feat_field=data.node_feat_field)
        model.eval()
        device = model.gnn_encoder.device
        def get_input_embeds(input_nodes):
            if not isinstance(input_nodes, dict):
                assert len(data.g.ntypes) == 1
                input_nodes = {data.g.ntypes[0]: input_nodes}
            return {ntype: input_embeds[ntype][ids].to(device) \
                    for ntype, ids in input_nodes.items()}
        embeddings = dist_inference(data.g, model.gnn_encoder, get_input_embeds,
                                    batch_size, fanout, edge_mask=edge_mask,
                                    task_tracker=task_tracker)
        model.train()
    else:
        model.eval()
        device = model.gnn_encoder.device
        def get_input_embeds(input_nodes):
            if not isinstance(input_nodes, dict):
                assert len(data.g.ntypes) == 1
                input_nodes = {data.g.ntypes[0]: input_nodes}
            feats = prepare_batch_input(data.g, input_nodes, dev=device,
                                        feat_field=data.node_feat_field)
            return model.node_input_encoder(feats, input_nodes)
        embeddings = dist_inference(data.g, model.gnn_encoder, get_input_embeds,
                                    batch_size, fanout, edge_mask=edge_mask,
                                    task_tracker=task_tracker)
        model.train()
    if get_rank() == 0:
        print(f"computing GNN embeddings: {time.time() - t1:.4f} seconds")
    return embeddings
