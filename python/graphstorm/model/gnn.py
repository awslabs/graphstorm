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
import logging
import time
import torch as th
import dgl
from torch import nn

from .utils import load_model as load_gsgnn_model
from .utils import save_model as save_gsgnn_model
from .utils import save_opt_state, load_opt_state
from .utils import save_sparse_embeds, load_sparse_embeds, load_wg_sparse_embeds
from .utils import create_sparse_embeds_path
from .utils import LazyDistTensor
from .utils import get_data_range
from .embed import compute_node_input_embeddings
from .embed import GSNodeInputLayer
from .gs_layer import GSLayerBase
from .gnn_encoder_base import dist_minibatch_inference
from ..utils import (
    get_rank,
    get_world_size,
    barrier
)
from ..wholegraph import is_wholegraph_optimizer, create_wholememory_optimizer, WholeGraphDistTensor

from ..dataloading.dataset import prepare_batch_input

from ..config import (GRAPHSTORM_MODEL_ALL_LAYERS,
                      GRAPHSTORM_MODEL_EMBED_LAYER,
                      GRAPHSTORM_MODEL_GNN_LAYER,
                      GRAPHSTORM_MODEL_DECODER_LAYER,
                      GRAPHSTORM_MODEL_DENSE_EMBED_LAYER,
                      GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER)

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
            # WholeGraph optimizer does not have/need to zero_grad
            # the backward pass does not accum. grad
            if not is_wholegraph_optimizer(optimizer):
                optimizer.zero_grad()

    def step(self):
        """ Moving the optimizer
        """
        all_opts = self.dense_opts + self.lm_opts + self.sparse_opts
        for optimizer in all_opts:
            if is_wholegraph_optimizer(optimizer):
                #TODO(@chang-l): request wholegraph to update
                # their optimizer to align with pytorch conventions
                optimizer.step(optimizer.lr)
            else:
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
    """ GraphStorm GNN model base class.

    Any GNN model trained by GraphStorm should inherit from this class. It contains
    some abstract methods that should be defined by the inherited classes. It also
    provides some utility methods.
    """

    @abc.abstractmethod
    def restore_dense_model(self, restore_model_path,
                            model_layer_to_load=None):
        """ Restore dense models, e.g., GNN Encoders, Decoders, etc.

        All model parameters except for learnable node embeddings, i.e.,
        ``dgl.distributed.DistEmbedding``, are restored by this function.
        This fuction will go though all the model layers and load the corresponding
        parameters from ``restore_model_path``.

        In some cases, users can choose which model layer(s) to load by
        setting ``model_layer_to_load``. ``model_layer_to_load`` is
        designed to indicate the names of model layer(s) that should be restored.

        Example Implementation:
        ------------------------
        The code below provides examplary implementation of this abstract method.

        To restore model parameters for a model with all three layers of a GraphStorm GNN
        model, including an input layer, a GNN layer and a decoder layer:

        .. code:: python

            # suppose we are going to load all three layers.
            input_encoder = self.input_encoder
            gnn_model = self.gnn_model
            decoder = self.decoder

            checkpoint = th.load(os.path.join(model_path, 'model.bin'),
                                 map_location='cpu',
                                 weights_only=True)

            assert 'gnn' in checkpoint
            assert 'input' in checkpoint
            assert 'decoder' in checkpoint

            input_encoder.load_state_dict(checkpoint['input'], strict=False)
            gnn_model.load_state_dict(checkpoint['gnn'])
            decoder.load_state_dict(checkpoint['decoder'])

        Parameters
        ----------
        restore_model_path : str
            The path where the model was stored.
        model_layer_to_load: list of str
            List of model layers to load. This arguement is used to indicate
            which model layer(s) are going to be restored from the model checkpoint.
            Default: None.
        """

    @abc.abstractmethod
    def restore_sparse_model(self, restore_model_path):
        """ Restore sparse models, e.g., learnable node embeddings.

        Learnable node embeddings are restored by this function.

        Example Implementation:
        ------------------------
        The code below provides examplary implementation of this abstract method.

        To load sparse model parameters for a node_input_encoder:

        .. code:: python

            from graphstorm.model.utils import load_sparse_emb

            for ntype, sparse_emb in sparse_embeds.items():
                load_sparse_emb(sparse_emb, os.path.join(model_path, ntype))

        Parameters
        ----------
        restore_model_path : str
            The path where the model was stored.
        """

    @abc.abstractmethod
    def save_dense_model(self, model_path):
        """Save dense models, e.g., GNN Encoders, Decoders, etc.

        All model parameters except for learnable node embeddings, i.e.,
        ``dgl.distributed.DistEmbedding``, are saved by this function.
        This fuction should go though all model layers and save the correspoinding
        parameters under ``model_path``.

        Example Implementation:
        ------------------------
        The code below provides an examplary implementation of this abstract method.

        .. code:: python

            # This function is only called by rank 0
            input_encoder = self.input_encoder
            gnn_model = self.gnn_model
            decoder = self.decoder

            model_states = {}
            model_states['gnn'] = gnn_model.state_dict()
            model_states['input'] = input_encoder.state_dict()
            model_states['decoder'] = decoder.state_dict()

            os.makedirs(model_path, exist_ok=True)
            # mode 767 means rwx-rw-rwx:
            os.chmod(model_path, 0o767)
            th.save(model_states, os.path.join(model_path, 'model.bin'))

        Parameters
        ----------
        model_path : str
            The path where all model parameters and optimizer states will be saved.
        """

    @abc.abstractmethod
    def save_sparse_model(self, model_path):
        """Save sparse models, e.g., learnable node embeddings.

        Learnable node embeddings are saved by this function. Saving learnable
        node embeddings only works when 1) the training task is run on a single machine
        or 2) the training task is running on a distributed environment with a
        shared file system.

        Example Implementation:
        ------------------------
        The code below provides an examplary implementation of this abstract method.

        The implementation of save_sparse_model usually includes two steps:

        Step 1: Create a path to save the learnable node embeddings.

        .. code:: python

            from graphstorm.model.util import create_sparse_emb_path

            for ntype, sparse_emb in sparse_embeds.items():
                create_sparse_emb_path(model_path, ntype)
            # make sure rank 0 creates the folder and change permission first

        Step 2: Save learnable node embeddings.

        .. code:: python

            from graphstorm.model.utils import save_sparse_emb

            for ntype, sparse_emb in sparse_embeds.items():
                save_sparse_emb(model_path, sparse_emb, ntype)

        Parameters
        ----------
        model_path : str
            The path where all model sparse parameters will be saved.
        """

    def normalize_node_embs(self, embs):
        """ Normalize node embeddings when needed.

        Normalize_node_embs should be called in ``forward()`` and ``predict()`` functions.
        By default, this method does no normalize the input embeddings, but simply return them.

        Parameters
        ----------
        embs: dict of Tensors
            A dict of node embeddings to be normalized.

        Returns
        -------
        dict of Tensors:
            Normalized node embeddings.
        """
        # by default, we do nothing
        return embs

    def restore_model(self, restore_model_path, model_layer_to_load=None):
        """Restore saved checkpoints of a GNN model.

        Users who want to overwrite this method should load the parameters of the GNN model.
        This method does not need to load the optimizer state.

        Examples
        --------
        Load a model from "/tmp/checkpoints".

        .. code:: python

            # CustomGSgnnModel is a child class of GSgnnModelBase
            model = CustomGSgnnModel()

            # Restore model parameters from "/tmp/checkpoints"
            model.restore_model("/tmp/checkpoints")

        Parameters
        ----------
        restore_model_path : str
            The path where the model was stored.
        model_layer_to_load: list of str
            list of model layers to load. Supported layers include: "embed", "gnn",
            "decoder".
        """
        start_load_t = time.time()
        # Restore the model weights from a checkpoint saved previously.
        if restore_model_path is not None:
            if get_rank() == 0:
                logging.debug('load model from %s', restore_model_path)
            self.restore_dense_model(restore_model_path, model_layer_to_load)

            # If a user doesn't specify the layer to load,
            # or they specify to load the embed layer or more specifically sparse embed layer.
            if model_layer_to_load is None \
                    or GRAPHSTORM_MODEL_EMBED_LAYER in model_layer_to_load \
                    or GRAPHSTORM_MODEL_SPARSE_EMBED_LAYER in model_layer_to_load:
                if get_rank() == 0:
                    logging.debug('Load Sparse embedding from %s', restore_model_path)
                self.restore_sparse_model(restore_model_path)

        # We need to make sure that the sparse embedding is completely loaded
        # before all processes use the model.
        barrier()

        if get_rank() == 0:
            logging.info('successfully load the model from %s', restore_model_path)
            logging.info('Time on load model: %.3f seconds', time.time() - start_load_t)

    def save_model(self, model_path):
        ''' Save a trained model.

        When saving a model, need to save both the dense parameters and sparse parameters.

        Examples
        --------
        Save a model into "/tmp/checkpoints".

        .. code:: python

            # CustomGSgnnModel is a child class of GSgnnModelBase
            model = CustomGSgnnModel()

            # Model parameters will be saved into "/tmp/checkpoints"
            model.save_model("/tmp/checkpoints")

        Parameters
        ----------
        model_path : str
            The path where all model parameters and optimizer states will be saved.
        '''
        start_save_t = time.time()
        # Only rank 0 save dense model parameters
        if get_rank() == 0:
            self.save_dense_model(model_path)

        # We assume the model is written into a shared filesystem accessable
        # to all trainers. Each trainer will save only part of the sparse embedding.
        self.save_sparse_model(model_path)
        # Make sure each process finishes embedding saving.
        barrier()

        if get_rank() == 0:
            logging.info('successfully save the model to %s', model_path)
            logging.info('Time on save model: %.3f seconds', time.time() - start_save_t)


    @abc.abstractmethod
    def create_optimizer(self):
        """Create the optimizer that optimizes the model.

        Users who want to customize a model should define a optimizer for this model.
        By using this method, users can define their customized optimization algorithm,
        the learning rate as well as any other hyperparameters.

        A model may require multiple optimizers. For example, we should define
        an optimizer for sparse embeddings and an optimizer for the dense parameters
        of a GNN model. In this case, a user should use a GSOptimizer to combine these
        optimizers.

        Example:
        --------

        Case 1: if there is only one optimizer:

        .. code:: python

            def create_optimizer(self):
                # define torch.optim.Optimizer
                return optimizer

        Case 2: if there are both dense and sparse optimizers:

        .. code:: python

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
        if needed, e.g., pre-compute all BERT embeddings.

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
        """ Return the device where the model runs.

        This implementation assumes that all model parameters are on the same device.
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
            params += list(self.edge_input_encoder.get_lm_dense_parameters())

        return params

    def has_sparse_params(self):
        """ Return whether there are sparse parameters (learnable embeddings)
            in the model.

        Return
        ------
        bool: True for there are sparse parameters
        """
        if self._optimizer is not None:
            # Optimizer is not None, it is in training,
            # then check sparse optimizers.
            return len(self._optimizer.sparse_opts) > 0

        # Optimizer is None, it is inference,
        # then check sparse parameters.
        return len(self.get_sparse_params()) > 0

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

    def use_wholegraph_sparse_emb(self):
        """ Whether or not to use WholeGraph to host embeddings for sparse optimizer updates.
        """
        if self.node_input_encoder is not None:
            return self.node_input_encoder.use_wholegraph_sparse_emb
        return False

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
                if isinstance(self.decoder, nn.ModuleDict):
                    for ntype in self.decoder:
                        assert self.node_input_encoder.out_dims == self.decoder[ntype].in_dims, \
                            'When GNN encoder is not used, the output dimensions of ' \
                            'the node input encoder should match the input dimension of' \
                            'the decoder.'
                else:
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
            if isinstance(self.decoder, nn.ModuleDict):
                for ntype in self.decoder:
                    assert encoder.out_dims == self.decoder[ntype].in_dims, \
                        'The output dimensions of the GNN encoder should ' \
                        + 'match the input dimension of the decoder.'
            else:
                assert encoder.out_dims == self.decoder.in_dims, \
                        'The output dimensions of the GNN encoder should ' \
                        + 'match the input dimension of the decoder.'
        self._gnn_encoder = encoder

    def set_decoder(self, decoders):
        """set the decoder layer.

        Parameters
        ----------
        decoders : GSLayer or dict[str, GSLayer]
            The decoder or dictionary of GSLayer.
        """
        if isinstance(decoders, dict):
            self._decoder = nn.ModuleDict()
            for name, decoder in decoders.items():
                assert isinstance(
                    decoder, GSLayerBase
                ), "The decoder should be the class of GSLayerBase."
                if self.gnn_encoder is not None:
                    assert self.gnn_encoder.out_dims == decoder.in_dims, (
                        "The output dimensions of the GNN encoder should "
                        + "match the input dimension of the decoder."
                    )
                self._decoder[name] = decoder
        else:
            decoder=decoders
            assert isinstance(decoder, GSLayerBase), \
                    'The decoder should be the class of GSLayerBase.'
            if self.gnn_encoder is not None:
                assert self.gnn_encoder.out_dims == decoder.in_dims, \
                        'The output dimensions of the GNN encoder should ' \
                        + 'match the input dimension of the decoder.'
            self._decoder = decoder

    def set_loss_func(self, loss_fns):
        """set the loss function.

        Parameters
        ----------
        loss_fns : Pytorch nn.Module or dist[str, Pytorch nn.Module]
            The loss function or dictionary of Pytorch nn.Module.
        """
        if isinstance(loss_fns, dict):
            self._loss_fn = nn.ModuleDict()
            for name, loss_fn in loss_fns.items():
                assert isinstance(
                    loss_fn, nn.Module
                ), "The loss function should be the class of nn.Module."
                self._loss_fn[name] = loss_fn
        else:
            loss_fn = loss_fns
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

    # pylint: disable=signature-differs
    def restore_dense_model(self, restore_model_path,
                            model_layer_to_load=None):
        # TODO(zhengda) we need to load edge_input_encoder.
        model_layer_to_load = GRAPHSTORM_MODEL_ALL_LAYERS \
                if model_layer_to_load is None else model_layer_to_load
        load_dense_input = GRAPHSTORM_MODEL_EMBED_LAYER in model_layer_to_load \
                or GRAPHSTORM_MODEL_DENSE_EMBED_LAYER in model_layer_to_load
        # load dense models for gnn_encoder, node_input_encoder and decoder
        load_gsgnn_model(restore_model_path,
                         self.gnn_encoder \
                            if GRAPHSTORM_MODEL_GNN_LAYER in model_layer_to_load else None,
                         self.node_input_encoder if load_dense_input else None,
                         self.decoder \
                            if GRAPHSTORM_MODEL_DECODER_LAYER in model_layer_to_load else None)

    def restore_sparse_model(self, restore_model_path):
        # restore sparse embeddings for node_input_encoder.
        if self.node_input_encoder.use_wholegraph_sparse_emb:
            # restore sparse embeddings from the output of wholegraph sparse emb/opt
            load_wg_sparse_embeds(restore_model_path, self.node_input_encoder)
        else:
            load_sparse_embeds(restore_model_path, self.node_input_encoder)


    def init_optimizer(self, lr, sparse_optimizer_lr=None, weight_decay=0, lm_lr=None):
        """initialize the model's optimizers

        Parameters
        ----------
        lr : float
            The learning rate for dense parameters
            The learning rate for general dense parameters
        sparse_optimizer_lr : float
            The learning rate for sparse parameters. Default is None and will use the lr value.
        weight_decay : float
            The weight decay for the optimizer. Default is 0.0.
        lm_lr: float
            Language model fine-tuning learning rate for
            langauge model dense parameters.
        """
        sparse_params = self.get_sparse_params()
        # check and set the sparse optimizer learning rate
        if sparse_optimizer_lr is None:
            sparse_optimizer_lr = lr
        if len(sparse_params) > 0:
            if self.use_wholegraph_sparse_emb():
                if sparse_params[0].optimizer is not None:
                    # When sparse embeddings are loaded from files in load_wg_sparse_emb(),
                    # wg_optimizer is required to be created before loading sparse embeddings.
                    # This workaround bypasses the wg_optimizer creation here for the scenarios
                    # where the sparse embedding loading happens before init_optimizer().
                    for param in sparse_params:
                        assert isinstance(param, WholeGraphDistTensor) and param.use_wg_optimizer, \
                            "Please create params (WholeGraph tensor) with use_wg_optimizer=True."
                        assert param.optimizer == sparse_params[0].optimizer, \
                            "Please create all WholeGraph sparse params with the same optimizer."
                    emb_optimizer = sparse_params[0].optimizer
                else:
                    # To use wholegraph sparse optimizer, optimizer needs to be created
                    # before sparse embeddings within attach_wg_optimizer.
                    emb_optimizer = create_wholememory_optimizer("adam", {})
                    for param in sparse_params:
                        assert isinstance(param, WholeGraphDistTensor) and param.use_wg_optimizer, \
                            "Please create params (WholeGraph tensor) with use_wg_optimizer=True."
                        assert param.optimizer is None, \
                            "Please create all WholeGraph sparse params with the same optimizer."
                        param.attach_wg_optimizer(emb_optimizer)

                # TODO(@chang-l): Wrap the wholegraph optimizer in a class to
                # take an extra input argument: lr
                emb_optimizer.lr = sparse_optimizer_lr
            else:
                emb_optimizer = dgl.distributed.optim.SparseAdam(
                    sparse_params, lr=sparse_optimizer_lr
                )
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

    def compute_embed_step(self, blocks, input_feats, input_nodes):
        """ Compute the GNN embeddings on a mini-batch.

        This function is used for mini-batch inference.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graph (MFG) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        input_feats : dict of Tensors
            The input node features.
        input_nodes : dict of Tensors
            The input node IDs.

        Returns
        -------
        dict of Tensors: The GNN embeddings.
        """
        device = blocks[0].device
        if self.node_input_encoder is not None:
            embs = self.node_input_encoder(input_feats, input_nodes)
            embs = {name: emb.to(device) for name, emb in embs.items()}
        else:
            embs = input_feats
        if self.gnn_encoder is not None:
            gnn_embs = self.gnn_encoder(blocks, embs)
        else:
            gnn_embs = embs
        return gnn_embs

    def save_dense_model(self, model_path):
        save_gsgnn_model(model_path, self.gnn_encoder, self.node_input_encoder, self.decoder)

    def save_sparse_model(self, model_path):
        # Saving sparse embedding is done in a distributed way.
        create_sparse_embeds_path(model_path, self.node_input_encoder)
        save_sparse_embeds(model_path,
                           self.node_input_encoder)

    def inplace_normalize_node_embs(self, embs):
        """ Do inplace node embedding normalization.

            This function is called by do_full_graph_inference().

            Parameters
            ----------
            embs: dict of Tensor
                Node embeddings.
        """
        rank = get_rank()
        world_size = get_world_size()
        for key, emb in embs.items():
            if isinstance(emb, (dgl.distributed.DistTensor, LazyDistTensor)):
                # If emb is a distributed tensor, multiple processes are doing
                # embdding normalization concurrently. We need to split
                # the task. (From full_graph_inference)
                start, end = get_data_range(rank, world_size, len(emb))
            else:
                # If emb is just a torch Tensor. do normalization directly.
                # (From mini_batch_inference)
                start, end = 0, len(emb)
            idx = start
            while idx + 1024 < end:
                emb[idx:idx+1024] = \
                    self.normalize_node_embs({key:emb[idx:idx+1024]})[key]
                idx += 1024
            emb[idx:end] = \
                self.normalize_node_embs({key:emb[idx:end]})[key]

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

def do_mini_batch_inference(model, data, batch_size=1024,
                            fanout=None, edge_mask=None, infer_ntypes=None,
                            task_tracker=None):
    """ Do mini batch inference

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
    infer_ntypes: list of str
        Node types that need to compute node embeddings.
    task_tracker: GSTaskTrackerAbc
        Task tracker

    Returns
    -------
    dict of th.Tensor : node embeddings.
    """
    if get_rank() == 0:
        logging.debug("Perform mini-batch inference on the full graph.")
    t1 = time.time() # pylint: disable=invalid-name
    barrier()
    if model.gnn_encoder is None:
        # Only graph aware but not GNN models
        embeddings = compute_node_input_embeddings(data.g,
                                                   batch_size,
                                                   model.node_input_encoder,
                                                   task_tracker=task_tracker,
                                                   feat_field=data.node_feat_field,
                                                   target_ntypes=infer_ntypes)
        model.eval()
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
        embeddings = dist_minibatch_inference(data.g,
                                                model.gnn_encoder,
                                                get_input_embeds,
                                                batch_size, fanout,
                                                edge_mask=edge_mask,
                                                target_ntypes=infer_ntypes,
                                                task_tracker=task_tracker)
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
        embeddings = dist_minibatch_inference(data.g,
                                                model.gnn_encoder,
                                                get_input_embeds,
                                                batch_size, fanout,
                                                edge_mask=edge_mask,
                                                target_ntypes=infer_ntypes,
                                                task_tracker=task_tracker)
    # Called when model.eval()
    # TODO: do_mini_batch_inference is not TRUE mini-batch inference
    #       Need to change the implementation.
    model.inplace_normalize_node_embs(embeddings)
    model.train()
    if get_rank() == 0:
        logging.debug("computing GNN embeddings: %.4f seconds", time.time() - t1)
    return embeddings

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
    if get_rank() == 0:
        logging.debug("Perform full-graph inference with batch size %d and fanout %s.",
                      batch_size, str(fanout))
    assert isinstance(model, GSgnnModel) or type(model).__name__ == 'GLEM',\
        "Only GSgnnModel and GLEM support full-graph inference."
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    barrier()
    if model.gnn_encoder is None:
        # Only graph aware but not GNN models
        embeddings = compute_node_input_embeddings(data.g,
                                                   batch_size,
                                                   model.node_input_encoder,
                                                   task_tracker=task_tracker,
                                                   feat_field=data.node_feat_field)
        model.eval()
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
            res = {}
            # If the input node layer doesn't generate embeddings for a node type,
            # we ignore it. This behavior is the same as reading node features below.
            for ntype, ids in input_nodes.items():
                if ntype in input_embeds:
                    res[ntype] = input_embeds[ntype][ids].to(device)
            return res
        embeddings = model.gnn_encoder.dist_inference(data.g, get_input_embeds,
                                                    batch_size, fanout, edge_mask=edge_mask,
                                                    task_tracker=task_tracker)
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

        embeddings = model.gnn_encoder.dist_inference(data.g, get_input_embeds,
                                                    batch_size, fanout, edge_mask=edge_mask,
                                                    task_tracker=task_tracker)
    # Called when model.eval()
    model.inplace_normalize_node_embs(embeddings)
    model.train()

    if get_rank() == 0:
        logging.debug("computing GNN embeddings: %.4f seconds", time.time() - t1)
    return embeddings
