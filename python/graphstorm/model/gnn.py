"""GNN model in GraphStorm"""

import time
import torch as th
import dgl
from torch import nn

from .utils import LazyDistTensor
from .utils import load_model as load_gsgnn_model
from .utils import save_model as save_gsgnn_model
from .utils import save_opt_state, load_opt_state
from .utils import save_sparse_embeds, load_sparse_embeds
from .utils import get_feat_size
from .embed import compute_node_input_embeddings, prepare_batch_input
from .embed import GSNodeInputLayer
from .gs_layer import GSLayerBase
from .rgcn_encoder import RelationalGCNEncoder
from .rgat_encoder import RelationalGATEncoder

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
    def __init__(self, dense_opts, sparse_opts):
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

    def load_opt_state(self, path):
        """ Load the optimizer states
        """
        load_opt_state(path, self.dense_opts, self.sparse_opts)

    def save_opt_state(self, path):
        """ Save the optimizer states.
        """
        save_opt_state(path, self.dense_opts, self.sparse_opts)

# This class does not implement the nn.Module's forward abstract method. Because children classes
# implement this, so disable here.
class GSgnnModel(nn.Module):    # pylint: disable=abstract-method
    """ GraphStorm GNN model

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    """
    def __init__(self, g):
        super(GSgnnModel, self).__init__()
        self._g = g
        self._node_input_encoder = None
        self._edge_input_encoder = None
        self._gnn_encoder = None
        self._decoder = None
        self._loss_fn = None

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
        g = self.g

        # Restore the model weights from a checkpoint saved previously.
        if restore_model_path is not None:
            print('load GNN model from ', restore_model_path)
            # TODO(zhengda) we need to load edge_input_encoder.
            load_gsgnn_model(restore_model_path, self.gnn_encoder,
                             self.node_input_encoder, self.decoder)
            if g.rank() == 0:
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

        Returns
        -------
        Optimier: the optimizer object
        """
        g = self.g
        if g.rank() == 0:
            print("Init optimizer ...")
        sparse_params = self.get_sparse_params()
        if len(sparse_params) > 0:
            emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=sparse_lr)
            if g.rank() == 0:
                print('optimize DGL sparse embedding.')
        else:
            emb_optimizer = None
        if g.rank() == 0:
            print('optimize GNN layers and embedding layers')
        dense_params = self.get_dense_params()
        if len(dense_params) > 0:
            optimizer = th.optim.Adam(self.get_dense_params(), lr=lr,
                                      weight_decay=weight_decay)
        else:
            # this may happen for link prediction with LM if the embedding layer is not specified.
            optimizer = None

        # if not train then the optimizers are not needed.
        return GSOptimizer(dense_opts=[optimizer] if optimizer is not None else [],
                           sparse_opts=[emb_optimizer] if emb_optimizer is not None else [])

    def compute_embed_step(self, blocks, input_feats, input_nodes):
        """ Compute the GNN embeddings on a mini-batch.

        This function is used for mini-batch inference.

        Parameters
        ----------
        blocks : list of DGLBlock
            The message flow graphs for computing GNN embeddings.
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
            embs = self.node_input_encoder(input_feats, input_nodes=input_nodes)
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

    def compute_embeddings(self, g, feat_name, target_nidx,
                           fanout=None, batch_size=None, mini_batch_infer=False,
                           task_tracker=None):
        """ Compute node embeddings in the inference.

        Parameters
        ----------
        g : DGLGraph
            The input graph
        feat_name : str
            The feature name
        target_nidx: dict of tensors
            The idices of nodes to generate embeddings.
        fanout : list of int
            The fanout for sampling neighbors for mini-batch inference.
        batch_size : int
            The mini-batch size
        mini_batch_infer : bool
            whether or not to use mini-batch inference.
        task_tracker : GSTaskTrackerAbc
            The task tracker

        Returns
        -------
        dict of Tensors: the GNN embeddings.
        """
        th.distributed.barrier()
        self.eval()

        # If we need to compute all node embeddings, full-graph inference is much more efficient.
        # Let's use mini-batch inference if we need to compute the embeddings of a subset of nodes.
        if mini_batch_infer and target_nidx is not None:
            if g.rank() == 0:
                print('Compute embeddings with mini-batch inference.')
            assert isinstance(fanout, list), \
                    'The fanout for mini-batch inference should be a list of integers.'
            embeddings = do_mini_batch_inference(g=g,
                                                 model=self,
                                                 target_nidx=target_nidx,
                                                 fanout=fanout,
                                                 batch_size=batch_size,
                                                 task_tracker=task_tracker,
                                                 feat_field=feat_name)
        else:
            if g.rank() == 0:
                print('Compute embeddings with full-graph inference.')
            embeddings = do_full_graph_inference(g=g,
                                                 model=self,
                                                 # We don't need neighbor sampling.
                                                 fanout=-1,
                                                 batch_size=batch_size,
                                                 task_tracker=task_tracker,
                                                 feat_field=feat_name)
            if target_nidx is not None:
                embeddings = {ntype: LazyDistTensor(embeddings[ntype], target_nidx[ntype]) \
                              for ntype in target_nidx.keys()}
        # wait all workers to finish
        self.train()
        th.distributed.barrier()
        return embeddings

    @property
    def g(self):
        """the graph used in this GNN class
        """
        return self._g

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

    @property
    def device(self):
        """ The device where the model runs.

        Here we assume that all model parameters are on the same device.
        """
        return next(self.parameters()).device

def do_full_graph_inference(g, model, fanout=-1,
                            batch_size=None, task_tracker=None,
                            feat_field='feat'):
    """ Do fullgraph inference

        Parameters
        ----------
        g: DistDGLGraph
            DGLGraph
        model: torch model
            GNN model
        fanout: int
            The fanout for computing the embedding of a layer.
        batch_size: int
            The batch size for computing the embeddings of a layer.
        task_tracker: GSTaskTrackerAbc
            Task tracker
        feat_field: str
            Field to extract features

        Returns
        -------
        Node embeddings: dict of str to th.Tensor
    """
    if batch_size is None:
        batch_size = 1024
    node_embed = compute_node_input_embeddings(g,
                                               batch_size,
                                               model.node_input_encoder,
                                               task_tracker=task_tracker,
                                               feat_field=feat_field)
    t1 = time.time() # pylint: disable=invalid-name
    # full graph evaluation
    th.distributed.barrier()
    model.eval()
    embeddings = model.gnn_encoder.dist_inference(g, batch_size, model.device, 0, node_embed,
                                                  fanout, task_tracker=task_tracker)
    if g.rank() == 0:
        print(f"computing GNN embeddings: {time.time() - t1:.4f} seconds")
    model.train()
    return embeddings

# pylint: disable=invalid-name
def do_mini_batch_inference(g, model, target_nidx,
                            fanout, batch_size,
                            task_tracker=None, feat_field='feat'):
    """ Do mini batch inference

        Parameters
        ----------
        g: DistDGLGraph
            The distributed graph.
        model: torch model
            GNN model
        target_nidx: dict of th.Tensor
            Target node idices
        fanout: int
            Inference fanout
        batch_size: int
            The batch size
        task_tracker: GSTaskTrackerAbc
            Task tracker
        feat_field: str
            field to extract features

        Returns
        -------
        Node embeddings: dict of str to th.Tensor
    """
    t0 = time.time()
    # train sampler
    target_idxs_dist = {}
    embeddings = {}
    pb = g.get_partition_book()
    # this mapping will hold the mapping among the rows of
    # the embedding matrix to the original target ids
    for key in target_nidx:
        # Note: The space overhead here, i.e., using a global target_mask,
        # is O(N), N is number of nodes.
        # Use int8 as all_reduce does not work well with bool
        # TODO(zhengda) we need to reduce the memory complexity described above.
        target_mask = th.full((g.num_nodes(key),), 0, dtype=th.int8)
        target_mask[target_nidx[key]] = 1

        # As each trainer may only focus on its own val/test set,
        # i.e., the val/test sets only contain local nodes or edges.
        # we need to get the full node or edge list before node_split
        # Here we use all_reduce to sync the target node/edge mask.
        # TODO(xiangsx): make it work with NCCL
        # TODO(xiangsx): make it more efficient
        th.distributed.all_reduce(target_mask,
            op=th.distributed.ReduceOp.MAX)

        node_trainer_ids=g.nodes[key].data['trainer_id'] \
            if 'trainer_id' in g.nodes[key].data else None
        target_idx_dist = dgl.distributed.node_split(
                target_mask.bool(),
                pb, ntype=key, force_even=False,
                node_trainer_ids=node_trainer_ids)
        target_idxs_dist[key] = target_idx_dist
        # TODO(zhengda) we still create a distributed tensor.
        # This may not be necessary.
        embeddings[key] = dgl.distributed.DistTensor(
            (g.number_of_nodes(key), model.gnn_encoder.out_dims),
            dtype=th.float32, name='output_embeddings',
            part_policy=g.get_node_partition_policy(key),
            persistent=True)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
    loader = dgl.dataloading.DistNodeDataLoader(g, target_idxs_dist, sampler,
                                                batch_size=batch_size,
                                                shuffle=False, num_workers=0)
    th.distributed.barrier()
    if model is not None:
        model.eval()
    th.cuda.empty_cache()
    device = model.device
    with th.no_grad():
        for iter_l, (input_nodes, seeds, blocks) in enumerate(loader):
            if task_tracker is not None:
                task_tracker.keep_alive(iter_l)

            # in the case of a graph with a single node type the returned seeds will not be
            # a dictionary but a tensor of integers this is a possible bug in the DGL code.
            # Otherwise we will select the seeds that correspond to the category node type
            if isinstance(input_nodes, dict) is False:
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}

            if isinstance(seeds, dict) is False:
                # This happens on a homogeneous graph.
                assert len(g.ntypes) == 1
                seeds = {g.ntypes[0]: seeds}

            blocks = [blk.to(device) for blk in blocks]
            inputs = prepare_batch_input(g,
                                         input_nodes,
                                         dev=device,
                                         feat_field=feat_field)

            input_nodes = {ntype: inodes.long().to(device) \
                    for ntype, inodes in input_nodes.items()}
            final_embs = model.compute_embed_step(blocks, inputs, input_nodes)
            for key in seeds:
                # we need to go over the keys in the seed dictionary and not the final_embs.
                # The reason is that our model
                # will always return a zero tensor if there are no nodes of a certain type.
                if len(seeds[key]) > 0:
                    # it might be the case that one key has a zero tensor this will cause a failure.
                    embeddings[key][seeds[key]] = final_embs[key].cpu()

    if model is not None:
        model.train()
    th.distributed.barrier()
    t1 = time.time()

    if g.rank() == 0:
        print(f'Computing GNN embeddings: {(t1 - t0):.4f} seconds')

    if target_nidx is not None:
        embeddings = {ntype: LazyDistTensor(embeddings[ntype], target_nidx[ntype]) \
            for ntype in target_nidx.keys()}
    return embeddings

def set_gnn_encoder(model, g, config, train_task):
    """ Create GNN encoder.

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        Configurations
    train_task : bool
        Whether this model is used for training.
    """
    # Set input layer
    feat_size = get_feat_size(g, config.feat_name)
    encoder = GSNodeInputLayer(g, feat_size, config.n_hidden,
                               dropout=config.dropout,
                               use_node_embeddings=config.use_node_embeddings)
    model.set_node_input_encoder(encoder)

    # Set GNN encoders
    model_encoder_type = config.model_encoder_type
    dropout = config.dropout if train_task else 0
    if model_encoder_type == "rgcn":
        n_bases = config.n_bases
        # we need to set the n_layers -1 because there is an output layer
        # that is hard coded.
        gnn_encoder = RelationalGCNEncoder(g,
                                           config.n_hidden, config.n_hidden,
                                           num_bases=n_bases,
                                           num_hidden_layers=config.n_layers -1,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop)
    elif model_encoder_type == "rgat":
        # we need to set the n_layers -1 because there is an output layer that is hard coded.
        gnn_encoder = RelationalGATEncoder(g,
                                           config.n_hidden,
                                           config.n_hidden,
                                           config.n_heads,
                                           num_hidden_layers=config.n_layers -1,
                                           dropout=dropout,
                                           use_self_loop=config.use_self_loop)
    else:
        assert False, "Unknown gnn model type {}".format(self.model_encoder_type)
    model.set_gnn_encoder(gnn_encoder)
