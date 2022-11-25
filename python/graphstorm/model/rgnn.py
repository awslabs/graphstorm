"""GNN Encoder"""
import abc
import os

import time
import math
import numpy as np
import torch as th
import dgl
import psutil
from torch.nn.parallel import DistributedDataParallel

from .utils import do_fullgraph_infer, do_mini_batch_inference, LazyDistTensor
from .extract_node_embeddings import prepare_batch_input
from .embed import DistGraphEmbed
from .utils import save_model as save_gsgnn_model
from .utils import save_sparse_embeds
from .utils import load_model as load_gsgnn_model
from .utils import load_sparse_embeds
from .utils import save_embeddings as save_gsgnn_embeddings
from .utils import remove_saved_models as remove_gsgnn_models
from .utils import load_opt_state, save_opt_state
from .utils import TopKList
from .rgat_encoder import RelationalGATEncoder
from .rgcn_encoder import RelationalGCNEncoder

from ..config.config import BUILTIN_GNN_ENCODER

class OptimizerCombiner():
    def __init__(self, optimizer_list):
        """

        Parameters
        ----------
        optimizer_list : a list containing optimizer objects. Some of the items in the list may be None, signfying that
        the optimizer is not instatiated.
        """
        self.optimizer_list = optimizer_list
        assert len(self.optimizer_list) > 0, "Optimizer list need to be defined"
        no_optimizer = True
        for optimizer in self.optimizer_list:
            if optimizer is not None:
                no_optimizer = False
        assert not no_optimizer, "At least one optimizer needs to exist"

    def zero_grad(self):
        """
        Setting the gradient to zero
        Returns
        -------

        """
        for optimizer in self.optimizer_list:
            if optimizer is not None:
                optimizer.zero_grad()

    def step(self):
        """
        Moving the optimizer
        Returns
        -------

        """
        for optimizer in self.optimizer_list:
            if optimizer is not None:
                optimizer.step()

class GSgnnBase():
    """ Base RGNN model

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        Configurations
    task_tracker: GSTaskTrackerAbc
        Task tracker used to log task progress
    train_task: bool
        Whether it is a training task
    verbose: bool
        If True, more information is printed
    """
    def __init__(self, g, config, task_tracker=None, train_task=True):
        if config.verbose:
            print(config)

        self._g = g
        self._no_validation = config.no_validation
        self._model_encoder_type = config.model_encoder_type

        if self.model_encoder_type == "rgat":
            self._n_heads = config.n_heads if hasattr(config, 'n_heads') else 4 # default we use 4 heads

        self._n_layers = config.n_layers if self.model_encoder_type in BUILTIN_GNN_ENCODER else 0
        assert self._n_layers >= 0, "number of GNN layers should be >= 0."

        # Model related
        # this parameter specifies whether the emb layer will be used. By default is in True for the gnn models
        self._n_bases = config.n_bases if self.model_encoder_type == "rgcn" else None
        self._use_self_loop = config.use_self_loop if self.model_encoder_type in BUILTIN_GNN_ENCODER else False
        self._self_loop_init = config.self_loop_init if self.model_encoder_type in BUILTIN_GNN_ENCODER else False
        self._n_hidden = config.n_hidden
        # we should set the minibatch to True in the LM case since we do not have any benefit for full graph inference
        self._mini_batch_infer = config.mini_batch_infer if self.model_encoder_type in BUILTIN_GNN_ENCODER else True
        # combining node features with learnable node embeddings.
        self._use_node_embeddings = config.use_node_embeddings
        self._feat_name = config.feat_name

        # computation related
        self._mixed_precision = config.mixed_precision
        self._dropout = 0

        # train specific
        if train_task:
            self._init_train_config(config)

        # evaluation specific
        self._eval_fanout = config.eval_fanout if self.model_encoder_type in BUILTIN_GNN_ENCODER else [0]
        # add a check if all the edge types are specified :
        if isinstance(self.eval_fanout[0], dict):
            for fanout_dic in self.eval_fanout:
                for e in g.etypes:
                    assert e in fanout_dic.keys(), "The edge type {} is not included in the specified eval fanout".format(e)

        # distributed training config
        self._local_rank = config.local_rank
        self._ip_config = config.ip_config
        self._graph_name = config.graph_name
        self._part_config = config.part_config
        self._save_model_per_iters = config.save_model_per_iters
        self._save_model_path = config.save_model_path
        self._restore_model_path = config.restore_model_path
        self._restore_optimizer_path = config.restore_optimizer_path
        self._restore_model_encoder_path = config.restore_model_encoder_path
        self._save_embeds_path = config.save_embeds_path

        self._debug = config.debug
        self._verbose = config.verbose

        self._evaluator = None
        # evaluation
        self._eval_batch_size = config.eval_batch_size

        self.model_conf = None
        self.gnn_encoder = None
        self.embed_layer = None
        self.decoder = None

        # model saving or removing
        self.topk_model_to_save = config.topk_model_to_save

        if self.topk_model_to_save > 0:
            self.topklist = TopKList(self.topk_model_to_save)    # A list to store the top k best
                                                                 # perf epoch+iteration for 
                                                                 # saving/removing models.
        else:
            if self.save_model_per_iters > 0:
                self.topklist = TopKList(math.inf)  # If not specify the top k and need to save 
                                                    # per n iterations, save all models at the 
                                                    # n iterations.
            else:
                self.topklist = TopKList(0)         # During inference, the n_epochs could be 0,
                                                    # so set k to 0 and not allow any insertion.
                                                    # During training, if users do not specify
                                                    # neither top k value or n per iteration, we
                                                    # will not store any models neither.

        # tracker info
        self._task_tracker = task_tracker

        # setup cuda env
        self.setup_cuda(config.local_rank)

    def _init_train_config(self, config):
        # neighbor sample related
        self._fanout = config.fanout if self.model_encoder_type in BUILTIN_GNN_ENCODER else [0]
        # add a check if all the edge types are specified:
        if isinstance(self.fanout[0], dict):
            for fanout_dic in self.fanout:
                for e in self.g.etypes:
                    assert e in fanout_dic.keys(), "The edge type {} is not included in the specified fanout".format(e)

        # training related
        self._batch_size = config.batch_size
        self._dropout = config.dropout
        self._mp_opt_level = config.mp_opt_level
        self._sparse_lr = config.sparse_lr
        self._lr = config.lr
        self._weight_decay = config.wd_l2norm
        self._n_epochs = config.n_epochs
        assert self._batch_size > 0, "batch size must > 0."

    def log_metric(self, metric_name, metric_value, step, report_step=None):
        """ log evaluation metric

        Parameters
        ----------
        metric_name: str
            Evaluation metric name
        metric_value: float
            Value
        step: int
            Current step
        report_step: int
            Deprecated. Will be deleted later
            TODO(xiangsx): delete report_step
        """
        if self.task_tracker is None:
            return

        self.task_tracker.log_metric(metric_name, metric_value, step)

    def keep_alive(self, report_step):
        """ Dummy log, send keep alive message to mlflow server

        Parameters
        ----------
        report_step: int
            Current exec step. Used to decide whether send dummy info
        """
        if self.task_tracker is None:
            return

        self.task_tracker.keep_alive(report_step)

    def log_param(self, param_name, param_value):
        """ Log parameters

        Parameters
        ----------
        param_name: str
            Parameter name
        param_value:
            Parameter value
        """
        if self.task_tracker is None:
            return

        self.task_tracker.log_param(param_name, param_value)

    def log_params(self, param_value):
        """ Log a dict of parameters

        Parameter
        ---------
        param_value: dict
            Key value pairs of parameters to log
        """
        if self.task_tracker is None:
            return

        self.task_tracker.log_params(param_value)

    def log_print_metrics(self, val_score, test_score, dur_eval, total_steps, train_score=None):
        """
        This function prints and logs all the metrics for evaluation

        Parameters
        ----------
        train_score: dict
            Training score
        val_score: dict
            Validation score
        test_score: dict
            Test score
        dur_eval:
            Total evaluation time
        total_steps: int
            The corresponding step/iteration
        """
        if self.task_tracker is None:
            return

        best_val_score = self.evaluator.best_val_score
        best_test_score = self.evaluator.best_test_score
        best_iter_num = self.evaluator.best_iter_num
        self.task_tracker.log_iter_metrics(train_score, val_score, test_score,
            best_val_score, best_test_score, best_iter_num,
            dur_eval, total_steps)

    def setup_cuda(self, local_rank):
        # setup cuda env
        use_cuda = th.cuda.is_available()
        assert use_cuda, "Only support GPU training"
        dev_id = local_rank
        th.cuda.set_device(dev_id)
        self._dev_id = dev_id

    def get_sparse_params(self):
        if self.sparse_embeds is not  None and len(self.sparse_embeds) > 0:
            return list(self.sparse_embeds.values())
        else:
            return []

    def get_dense_params(self):
        params = []
        if self.gnn_encoder is not None:
            params += list(self.gnn_encoder.parameters())
        if self.embed_layer is not None:
            params += list(self.embed_layer.parameters())
        if self.decoder is not None:
            params += list(self.decoder.parameters())
        return params

    def init_emb_layer(self, g, feat_size, dev_id):
        # create embeddings
        embed_layer = DistGraphEmbed(g,
                                     feat_size,
                                     self.n_hidden,
                                     dropout=self.dropout,
                                     self_loop_init=self.self_loop_init,
                                     use_node_embeddings=self.use_node_embeddings)
        self.sparse_embeds = embed_layer.sparse_embeds
        # If there are dense parameters in the embedding layer
        # or we use Pytorch saprse embeddings.
        if len(embed_layer.input_projs) > 0:
            embed_layer = embed_layer.to(dev_id)
            embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        return embed_layer

    def init_gnn_encoder(self, g, dev_id):
        # create rgcn encoder model if the type is gnn model
        if self.model_encoder_type == "rgcn":
            # we need to set the n_layers -1 because there is an output layer
            # that is hard coded.
            gnn_encoder = RelationalGCNEncoder(g,
                                        self.n_hidden, self.n_hidden,
                                        num_bases=self.n_bases,
                                        num_hidden_layers=self.n_layers -1,
                                        dropout=self.dropout,
                                        use_self_loop=self.use_self_loop,
                                        self_loop_init=self.self_loop_init)
        elif self.model_encoder_type == "rgat":
            # we need to set the n_layers -1 because there is an output layer
            # that is hard coded.
            gnn_encoder = RelationalGATEncoder(g,
                                            self.n_hidden,
                                            self.n_hidden,
                                            self._n_heads,
                                            num_hidden_layers=self.n_layers -1,
                                            dropout=self.dropout,
                                            use_self_loop=self.use_self_loop)
        elif self.model_encoder_type == "lm":
            gnn_encoder = None
        else:
            assert False, "Unknown gnn model type {}".format(self.model_encoder_type)

        if gnn_encoder is not None:
            # Create distributed data parallel.
            gnn_encoder = gnn_encoder.to(dev_id)
            gnn_encoder = DistributedDataParallel(gnn_encoder,
                                                  device_ids=[dev_id],
                                                  output_device=dev_id,
                                                  find_unused_parameters=False)

        return gnn_encoder

    def init_dist_encoder(self, train):
        # prepare features
        g = self.g
        dev_id = self.dev_id

        feat_size = {}
        for ntype in g.ntypes:
            feat_field = self._feat_name
            # user can specify the name of the field
            feat_name = None if feat_field is None else \
                feat_field if isinstance(feat_field, str) \
                else feat_field[ntype] if ntype in feat_field else None

            if feat_name is None:
                feat_size[ntype] = 0
            else:
                # We force users to know which node type has node feature
                # This helps avoid unexpected training behavior.
                assert feat_name in g.nodes[ntype].data, \
                    f"Warning. The feat with name {feat_name} " \
                    f"does not exists for the node type {ntype}" \
                    "If not all your ntypes have node feature," \
                    "you can use --feat-name 'ntype0:feat0 ntype1:feat0`" \
                    "to specify feature names for each node type."
                feat_size[ntype] = g.nodes[ntype].data[feat_name].shape[1]

        embed_layer = self.init_emb_layer(g, feat_size, dev_id)
        gnn_encoder = self.init_gnn_encoder(g, dev_id)

        self.gnn_encoder = gnn_encoder
        self.embed_layer = embed_layer

    def init_dist_decoder(self, train):
        pass

    def init_model_optimizers(self, train):
        if train:
            g = self.g
            if g.rank() == 0:
                print("Init optimizer ...")
            sparse_params = self.get_sparse_params()
            if len(sparse_params) > 0:
                emb_optimizer = dgl.distributed.optim.SparseAdam(sparse_params, lr=self.sparse_lr)
                if g.rank() == 0:
                    print('optimize DGL sparse embedding.')
            else:
                emb_optimizer = None
            if g.rank() == 0:
                print('optimize GNN layers and embedding layers')
            dense_params = self.get_dense_params()
            if len(dense_params) > 0:
                optimizer = th.optim.Adam(self.get_dense_params(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                # this may happen for link prediction with LM if the embedding layer is not specified.
                optimizer = None

            # if not train then the optimizers are not needed.
            self.emb_optimizer = emb_optimizer
            self.optimizer = optimizer
        else:
            self.emb_optimizer = None
            self.optimizer = None

    def restoring_gsgnn(self, train):
        g = self.g

        # Restore the model weights and or optimizers to a checkpoint saved previously.
        if self.restore_model_path is not None:
            print('load GNN model from ', self.restore_model_path)
            load_gsgnn_model(self.restore_model_path, self.gnn_encoder, self.embed_layer, self.decoder)
            if g.rank() == 0:
                print('Load Sparse embedding from ', self.restore_model_path)
                load_sparse_embeds(self.restore_model_path, self.embed_layer)

        if self.restore_optimizer_path is not None and train:
            print('load GNN optimizer state from ', self.restore_model_path)
            load_opt_state(self.restore_model_path, self.optimizer, self.emb_optimizer)

        if self.restore_model_encoder_path is not None:
            print('load GNN model encoder from ', self.restore_model_encoder_path)
            load_gsgnn_model(self.restore_model_encoder_path, self.gnn_encoder, self.embed_layer, decoder=None)

    def init_gsgnn_model(self, train=True):
        ''' Initialize the GNN model.

        Argument
        --------
        train : bool
            Indicate whether the model is initialized for training.
        '''
        self.init_dist_encoder(train)
        self.init_dist_decoder(train)
        self.init_model_optimizers(train)
        self.restoring_gsgnn(train)
        self.setup_combine_optimizer(train)

    def setup_combine_optimizer(self, train):
        """
        Instantiated combined optimizer

        Parameters
        ----------
        train : bool
            Indicate whether the model is initialized for training.

        Returns
        -------

        """
        if train:
            self.combine_optimizer = OptimizerCombiner(optimizer_list=[self.optimizer,
                                                                       self.emb_optimizer])
        else:
            self.combine_optimizer = None


    def compute_embeddings(self, g, device, target_nidx=None):
        """
        compute node embeddings

        Parameters
        ----------
        g : DGLGraph
            The input graph
        device: th.device
            Device to run the computation
        target_nidx: dict of tensors
            The idices of nodes to generate embeddings.
        """
        if self.model_encoder_type in BUILTIN_GNN_ENCODER:
            embeddings = self.compute_gnn_embeddings(g, device, target_nidx)
        else:
            embeddings = self.compute_lm_embeddings(g, device, target_nidx)
        return embeddings

    def compute_gnn_embeddings(self, g, device, target_nidx):
        """
        compute node embeddings

        Parameters
        ----------
        g : DGLGraph
            The input graph
        device: th.device
            Device to run the computation
        target_nidx: dict of tensors
            The idices of nodes to generate embeddings.
        """
        assert self.gnn_encoder is not None, "GNN model should be initialized"
        assert self.embed_layer is not None, "Node embedding layer should be initialized"
        th.distributed.barrier()
        self.gnn_encoder.eval()
        self.embed_layer.eval()

        # If we need to compute all node embeddings, full-graph inference is much more efficient.
        # Let's use mini-batch inference if we need to compute the embeddings of a subset of nodes.
        if self.mini_batch_infer and target_nidx is not None:
            pb = g.get_partition_book()
            if g.rank() == 0:
                print('Compute embeddings with mini-batch inference.')
            embeddings = do_mini_batch_inference(model=self.gnn_encoder,
                                                 embed_layer=self.embed_layer,
                                                 device=device,
                                                 target_nidx=target_nidx,
                                                 g=g,
                                                 pb=pb,
                                                 n_hidden=self.n_hidden,
                                                 fanout=self.eval_fanout,
                                                 eval_batch_size=self.eval_batch_size,
                                                 task_tracker=self.task_tracker,
                                                 feat_field=self._feat_name)
        else:
            if g.rank() == 0:
                print('Compute embeddings with full-graph inference.')
            embeddings = do_fullgraph_infer(g=g,
                                            model=self.gnn_encoder,
                                            embed_layer=self.embed_layer,
                                            device=device,
                                            eval_fanout_list=self.eval_fanout,
                                            eval_batch_size=self.eval_batch_size,
                                            task_tracker=self.task_tracker,
                                            feat_field=self._feat_name)
            if target_nidx is not None:
                embeddings = {ntype: LazyDistTensor(embeddings[ntype], target_nidx[ntype]) for ntype in target_nidx.keys()}
        # wait all workers to finish
        self.gnn_encoder.train()
        self.embed_layer.train()
        th.distributed.barrier()
        return embeddings

    def encoder_forward(self, blocks, input_nodes, gnn_forward_time, epoch):
        g = self.g
        device = 'cuda:%d' % self.dev_id
        model = self.gnn_encoder
        embed_layer = self.embed_layer

        blocks = [blk.to(device) for blk in blocks]

        if self.debug:
            th.distributed.barrier()
        t1 = time.time()
        inputs = prepare_batch_input(g,
                                     input_nodes,
                                     dev=device,
                                     verbose=self.verbose,
                                     feat_field=self._feat_name)
        t2 = time.time()

        input_nodes = {ntype: inodes.long().to(device) for ntype, inodes in input_nodes.items()}
        emb = embed_layer(inputs, input_nodes=input_nodes) if embed_layer is not None else inputs
        gnn_embs = model(emb, blocks) if model is not None else {ntype: nemb.to(device)
                for ntype, nemb in emb.items()}

        t3 = time.time()
        gnn_forward_time += (t3 - t2)

        return gnn_embs, gnn_forward_time

    def save_model_embed(self, epoch, i, g):
        '''Save the model and node embeddings for a certain iteration in an epoch..
        '''
        model_conf = self.model_conf
        model = self.gnn_encoder
        embed_layer = self.embed_layer
        decoder = self.decoder
        assert model_conf is not None

        # sync before model saving
        th.distributed.barrier()
        if self.save_model_path is not None and g.rank() == 0:
            start_save_t = time.time()

            save_model_path = self._gen_model_path(self.save_model_path, epoch, i)

            save_gsgnn_model(model_conf, save_model_path, model, embed_layer, decoder)
            save_sparse_embeds(save_model_path, embed_layer)
            save_opt_state(save_model_path, self.optimizer, self.emb_optimizer)
            print('successfully save the model to ' + save_model_path)
            print('Time on save model {}'.format(time.time() - start_save_t))

        if self.save_embeds_path is not None:
            # Generate all the node embeddings
            device = 'cuda:%d' % self.dev_id
            embeddings = self.compute_embeddings(g, device)

            # save embeddings in a distributed way
            save_embeds_path = self._gen_model_path(self.save_embeds_path, epoch, i)

            save_gsgnn_embeddings(save_embeds_path, embeddings, g.rank(), th.distributed.get_world_size())

        # wait for rank0 to save the model and/or embeddings
        th.distributed.barrier()

    def remove_saved_model_embed(self, epoch, i, g_rank):
        """ remove previously saved model, which may not be the best K performed or other reasons.
            This function will remove the entire folder.

        Parameters
        ----------
        epoch: int
            The number of training epoch.
        i: int
            The number of iteration in a training epoch.
        g_rank: int
            The rank of the give graph.
        """
        if self.save_model_path is not None and g_rank == 0:
            # construct model path
            saved_model_path = self._gen_model_path(self.save_model_path, epoch, i)

            # remove the folder that contains saved model files.
            remove_status = remove_gsgnn_models(saved_model_path)
            if remove_status == 0:
                print(f'Successfully removed the saved model files in {saved_model_path}')

    def save_topk_models(self, epoch, i, g, val_score):
        """ Based on the given val_score, decided if save the current model trained in the i_th
            iteration and the epoch_th epoch.

        Parameters
        ----------
        epoch: int
            The number of training epoch.
        i: int
            The number of iteration in a training epoch.
        g: DGLDistGraph
            The distributed graph used in the current training.
        val_score: dict or None
            A dictionary contains scores from evaluator's validation function. It could be None 
            that means there is either no evluator or not do validation. In that case, just set
            the score rank as 1st to save all models or the last k models.
        """

        # compute model validation score rank in evaluator
        if val_score is None:
            score_rank = 1
        else:
            score_rank = self.evaluator.get_val_score_rank(val_score)

        insert_success, (return_epoch, return_i) = self.topklist.insert(score_rank, (epoch, i))

        if insert_success:
            # if success, should always save this epoch and/or iteration models, and remove the
            # previous worst model saved if the return_epoch or return_i is different from current
            # epoch and i
            if return_epoch != epoch or return_i != i:
                # here the return_epoch and return_i are the epoch and iteration number that
                # performan worst in the previous top k list.
                self.remove_saved_model_embed(return_epoch, return_i, g.rank())

            # save this epoch and iteration's model and node embeddings
            self.save_model_embed(epoch, i, g)

    def _gen_model_path(self, base_path, epoch, i):
        """
        Generate the model path for both saving and removing a folder that contains model files.
        """
        model_path = os.path.join(base_path, 'epoch-' + str(epoch))
        if i is not None:
            model_path = model_path + '-iter-' + str(i)

        return model_path

    def print_info(self, epoch, i, num_input_nodes, compute_time):
        ''' Print basic information during training

        Parameters:
        epoch: int
            The epoch number
        i: int
            The current iteration
        num_input_nodes: int
            number of input nodes
        compute_time: tuple of ints
            A tuple of (gnn forward time and backward time)
        '''
        gnn_forward_time, back_time = compute_time
        device = 'cuda:%d' % self.dev_id

        print("Epoch {:05d} | Batch {:03d} | GPU Mem reserved: {:.4f} MB | Peak Mem alloc: {:.4f} MB".
                format(epoch, i,
                    th.cuda.memory_reserved(device) / 1024 / 1024,
                    th.cuda.max_memory_allocated(device) / 1024 /1024))
        print('Epoch {:05d} | Batch {:03d} | RAM memory {} used'.format(epoch, i, psutil.virtual_memory()))
        print('Epoch {:05d} | Batch {:03d} | Avg input nodes per iter {} | GNN forward {:05f} | Backward {:05f}'.format(
            epoch, i,
            num_input_nodes,
            gnn_forward_time,
            back_time))

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

    @property
    def task_tracker(self):
        """ Task tracker to log train/inference progress
        """
        return self._task_tracker

    def register_task_tracker(self, task_tracker):
        """ Set task tracker

        Parameter
        ---------
        task_tracker: GSTaskTrackerAbc
            task tracker
        """
        self._task_tracker = task_tracker

    @property
    def save_embeds_path(self):
        return self._save_embeds_path

    @property
    def g(self):
        return self._g

    @property
    def dev_id(self):
        return self._dev_id

    @property
    def verbose(self):
        return self._verbose

    @property
    def no_validation(self):
        return self._no_validation

    @property
    def local_rank(self):
        return self._local_rank

    @property
    def fanout(self):
        return self._fanout

    @property
    def eval_fanout(self):
        return self._eval_fanout

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def use_self_loop(self):
        return self._use_self_loop

    @property
    def self_loop_init(self):
        return self._self_loop_init

    @property
    def n_hidden(self):
        return self._n_hidden

    @property
    def mini_batch_infer(self):
        return self._mini_batch_infer

    @property
    def n_bases(self):
        return self._n_bases

    @property
    def dropout(self):
        return self._dropout

    @property
    def mixed_precision(self):
        return self._mixed_precision

    @property
    def mp_opt_level(self):
        return self._mp_opt_level

    @property
    def sparse_lr(self):
        return self._sparse_lr

    @property
    def lr(self):
        return self._lr

    @property
    def weight_decay(self):
        return self._weight_decay

    @property
    def debug(self):
        return self._debug

    @property
    def n_epochs(self):
        return self._n_epochs

    @property
    def model_encoder_type(self):
        return self._model_encoder_type

    @property
    def gnn_model_type(self):
        return self.model_encoder_type if self.model_encoder_type in BUILTIN_GNN_ENCODER else "None"

    @property
    def save_model_per_iters(self):
        return self._save_model_per_iters

    @property
    def save_model_path(self):
        return self._save_model_path

    @property
    def restore_model_path(self):
        return self._restore_model_path

    @property
    def restore_model_encoder_path(self):
        return self._restore_model_encoder_path

    @property
    def restore_optimizer_path(self):
        return self._restore_optimizer_path

    def register_evaluator(self, evaluator):
        self._evaluator = evaluator

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def eval_batch_size(self):
        return self._eval_batch_size

    @property
    def use_node_embeddings(self):
        return self._use_node_embeddings
