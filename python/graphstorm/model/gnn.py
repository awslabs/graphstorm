"""GNN model in GraphStorm"""

import abc
import time
import torch as th
import dgl
from torch import nn

from .utils import load_model as load_gsgnn_model
from .utils import save_model as save_gsgnn_model
from .utils import save_opt_state, load_opt_state
from .utils import save_sparse_embeds, load_sparse_embeds
from .embed import compute_node_input_embeddings
from .gs_layer import GSLayerBase
from .gnn_encoder_base import dist_inference
from ..utils import get_rank

class GSOptimizer():
    """ A combination of optimizers.

    A model might requires multiple optimizers. For example, we need a sparse optimizer
    for sparse embeddings, an optimizer to fine-tune the BERT module and an optimizer for
    the dense modules.

    Parameters
    ----------
    dense_opts : list
        A list of optimizer objects for dense model parameters.
    sparse_opts : list
        A list of optimizer objects for sparse model parameters.
    """
    def __init__(self, dense_opts, sparse_opts=None):
        if sparse_opts is None:
            sparse_opts = []
        assert isinstance(dense_opts, list), "dense_opts should be a list."
        assert isinstance(sparse_opts, list), "sparse_opts should be a list."
        self.dense_opts = dense_opts
        self.sparse_opts = sparse_opts
        all_opts = dense_opts + sparse_opts
        assert len(all_opts) > 0, "Optimizer list need to be defined"
        for optimizer in all_opts:
            assert optimizer is not None

    def zero_grad(self):
        """ Setting the gradient to zero
        """
        all_opts = self.dense_opts + self.sparse_opts
        for optimizer in all_opts:
            optimizer.zero_grad()

    def step(self):
        """ Moving the optimizer
        """
        all_opts = self.dense_opts + self.sparse_opts
        for optimizer in all_opts:
            optimizer.step()

    def load_opt_state(self, path, device=None):
        """ Load the optimizer states
        """
        load_opt_state(path, self.dense_opts, self.sparse_opts)
        if device is not None:
            self.move_to_device(device)

    def save_opt_state(self, path):
        """ Save the optimizer states.
        """
        save_opt_state(path, self.dense_opts, self.sparse_opts)

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

class GSgnnModelBase(nn.Module):
    """ GraphStorm GNN model base class

    Any GNN model trained by GraphStorm should inherit from this class. It contains
    some abstract methods that should be defined in the GNN model classes.
    It also provides some utility methods.
    """

    @abc.abstractmethod
    def restore_model(self, restore_model_path):
        """Load saving checkpoints of a GNN model.

        A user who implement this method should load the parameters of the GNN model.
        This method does not need to load the optimizer state.

        Parameters
        ----------
        restore_model_path : str
            The path where we can restore the model.
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
        of a GNN model. In this case, a user can use GSOptimizer to combine these
        optimizers.
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
        """retrieve the all dense layers' parameters as a parameter list.

        TODO(zhengda) we don't need this. We only need to call self.parameters().
        """
        params = []
        if self.gnn_encoder is not None and isinstance(self.gnn_encoder, nn.Module):
            params += list(self.gnn_encoder.parameters())
        if self.node_input_encoder is not None and isinstance(self.node_input_encoder, nn.Module):
            params += list(self.node_input_encoder.parameters())
        # TODO(zhengda) we need to test a model with encoders on edge data.
        if self.edge_input_encoder is not None and isinstance(self.edge_input_encoder, nn.Module):
            params += list(self.edge_input_encoder.parameters())
        if self.decoder is not None and isinstance(self.decoder, nn.Module):
            params += list(self.decoder.parameters())
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

    def restore_model(self, restore_model_path):
        """load saving checkpoints for GNN models.

        Parameters
        ----------
        restore_model_path : str
            The path where we can restore the model.
        """
        # Restore the model weights from a checkpoint saved previously.
        if restore_model_path is not None:
            print('load GNN model from ', restore_model_path)
            # TODO(zhengda) we need to load edge_input_encoder.
            load_gsgnn_model(restore_model_path, self.gnn_encoder,
                             self.node_input_encoder, self.decoder)
            if get_rank() == 0:
                # TODO(zhengda) we should load the sparse embeddings in parallel in the future.
                print('Load Sparse embedding from ', restore_model_path)
                load_sparse_embeds(restore_model_path, self.node_input_encoder)

    def init_optimizer(self, lr, sparse_lr, weight_decay):
        """initialize the model's optimizers

        Parameters
        ----------
        lr : float
            The learning rate for dense parameters
        sparse_lr : float
            The learning rate for sparse parameters
        weight_decay : float
            The weight decay for the optimizer.
        """
        sparse_params = self.get_sparse_params()
        if len(sparse_params) > 0:
            emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=sparse_lr)
        else:
            emb_optimizer = None
        dense_params = self.get_dense_params()
        if len(dense_params) > 0:
            optimizer = th.optim.Adam(self.get_dense_params(), lr=lr,
                                      weight_decay=weight_decay)
        else:
            # this may happen for link prediction with LM if the embedding layer is not specified.
            optimizer = None

        # if not train then the optimizers are not needed.
        dense_opts = [optimizer] if optimizer is not None else []
        sparse_opts = [emb_optimizer] if emb_optimizer is not None else []
        self._optimizer = GSOptimizer(dense_opts=dense_opts, sparse_opts=sparse_opts)

    def create_optimizer(self):
        """the optimizer
        """
        return self._optimizer

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
        save_gsgnn_model(model_path, self.gnn_encoder, self.node_input_encoder, self.decoder)
        save_sparse_embeds(model_path, self.node_input_encoder)
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
        return self.gnn_encoder.n_layers if self.gnn_encoder is not None else 0

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

def do_full_graph_inference(model, data, batch_size=1024, edge_mask=None, task_tracker=None):
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
    edge_mask : str
        The edge mask that indicates what edges are used to compute GNN embeddings.
    task_tracker: GSTaskTrackerAbc
        Task tracker

    Returns
    -------
    dict of th.Tensor : node embeddings.
    """
    assert isinstance(model, GSgnnModel), "Only GSgnnModel supports full-graph inference."
    node_embed = compute_node_input_embeddings(data.g,
                                               batch_size,
                                               model.node_input_encoder,
                                               task_tracker=task_tracker,
                                               feat_field=data.node_feat_field)
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    th.distributed.barrier()
    model.eval()
    embeddings = dist_inference(data.g, model.gnn_encoder, node_embed,
                                batch_size, -1, edge_mask=edge_mask,
                                task_tracker=task_tracker)
    # TODO(zhengda) we should avoid getting rank from the graph.
    if get_rank() == 0:
        print(f"computing GNN embeddings: {time.time() - t1:.4f} seconds")
    model.train()
    return embeddings
