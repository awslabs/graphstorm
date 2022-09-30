"""GNN Encoder"""
import abc

import time
import numpy as np
import torch as th
import dgl
import psutil
from torch.nn.parallel import DistributedDataParallel

from .utils import do_fullgraph_infer, do_mini_batch_inference, LazyDistTensor, rand_gen_trainmask
from .emb_cache import EmbedCache
from .extract_node_embeddings import extract_bert_embeddings_dist, prepare_batch_input
from ..data.constants import TOKEN_IDX
from .embed import DistGraphEmbed
from .hbert import wrap_bert, get_bert_flops_info
from .utils import save_model as save_gsgnn_model
from .utils import save_sparse_embeds
from .utils import load_model as load_gsgnn_model
from .utils import load_sparse_embeds
from .utils import save_embeddings as save_gsgnn_embeddings
from .utils import load_opt_state, save_opt_state
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

    Note: we assume each node type has a standalone BERT model.

    Parameters
    ----------
    g: DGLGraph
        The graph used in training and testing
    config: GSConfig
        Configurations
    bert_model: dict
        A dict of BERT models in a format of ntype -> bert_model
    train_task: bool
        Whether it is a training task
    verbose: bool
        If True, more information is printed
    """
    def __init__(self, g, config, bert_model, train_task=True, verbose=False):
        if verbose:
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
        self._pretrain_emb_layer = True if self.model_encoder_type in BUILTIN_GNN_ENCODER else config.pretrain_emb_layer
        self._n_bases = config.n_bases if self.model_encoder_type == "rgcn" else None
        self._use_self_loop = config.use_self_loop if self.model_encoder_type in BUILTIN_GNN_ENCODER else False
        self._self_loop_init = config.self_loop_init if self.model_encoder_type in BUILTIN_GNN_ENCODER else False
        self._n_hidden = config.n_hidden
        # we should set the minibatch to True in the LM case since we do not have any benefit for full graph inference
        self._mini_batch_infer = config.mini_batch_infer if self.model_encoder_type in BUILTIN_GNN_ENCODER else True
        # disable combining node embedding with bert embeding
        self._use_node_embeddings = False # config.use_node_embeddings
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
        self._restore_bert_model_path = config.restore_bert_model_path
        self._save_embeds_path = config.save_embeds_path

        self._debug = config.debug
        self._verbose = verbose

        assert isinstance(bert_model, dict), \
            "The input bert_model must be a dict of BERT models " \
            "in the format of ntype -> bert_model"
        self._bert_model = bert_model

        self._evaluator = None
        # evaluation
        self._eval_batch_size = config.eval_batch_size
        self._bert_infer_bs = config.bert_infer_bs

        self.model_conf = None
        self.gnn_encoder = None
        self.embed_layer = None
        self.decoder = None
        self.bert_train = {}
        self.bert_static = {}
        self.bert_hidden_size = {}

        # tracker info
        self.tracker = None
        if g.rank() == 0 and config.mlflow_tracker:
            # only rank 0 report to mlflow
            self._mlflow_exp_name = config.mlflow_exp_name
            self._mlflow_run_name = config.mlflow_run_name
            self._mlflow_report_frequency = config.mlflow_report_frequency
            self.init_tracker() # Init self.tracker
            # log all config parameters
            self.log_params(vars(config))
        else:
            self._mlflow_report_frequency = 0 # dummy, as self.tracker is None, it is not used

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

        self._train_nodes = config.train_nodes
        self._use_bert_cache = config.use_bert_cache
        self._refresh_cache = config.refresh_cache

        # training related
        self._batch_size = config.batch_size
        self._dropout = config.dropout
        self._mp_opt_level = config.mp_opt_level
        self._sparse_lr = config.sparse_lr
        self._lr = config.lr
        self._bert_lr = config.bert_tune_lr
        self._weight_decay = config.wd_l2norm
        self._n_epochs = config.n_epochs
        self._gnn_warmup_epochs = config.gnn_warmup_epochs
        assert self._batch_size > 0, "batch size must > 0."

    def init_tracker(self):
        """Initialize mlflow tracker"""
        # we need this identifier to allow logging for all processes and avoid
        # writting in the same location that is not allowed
        from m5_job_tracker import tracking_config
        tracking_id = ":proc_nbr:"+str(self.g.rank())
        tracking_config.set_tensorboard_config(
            {"log_dir": self._mlflow_exp_name+tracking_id}
        )
        mlflow_config = {"experiment_name": self._mlflow_exp_name+tracking_id,
                         "run_name": self._mlflow_run_name+tracking_id}

        tracking_config.set_mlflow_config(mlflow_config)
        self.tracker = tracking_config.get_or_create_client()
        print("Tracker run info : " + str(self.tracker.get_run_info()))
        # Log host level metadata
        self.tracker.log_current_host_attributes()

    def log_metric(self, metric_name, metric_value, step, report_step=None):
        if self.tracker is not None:
            if report_step is None or (report_step is not None and report_step % self.mlflow_report_frequency == 0):
                self.tracker.log_metric(metric_name, metric_value, step)

    def log_param(self, param_name, param_value, report_step=None):
        if self.tracker is not None:
            if report_step is None or (report_step is not None and report_step % self.mlflow_report_frequency == 0):
                self.tracker.log_param(param_name, param_value)

    def log_params(self, param_value):
        if self.tracker is not None:
            self.tracker.log_params(param_value)

    def log_print_metrics(self, val_score, test_score, dur_eval, total_steps, train_score=None):
        """
        This function prints and logs all the metrics for evaluation

        """
        for metric in self.evaluator.metric:
            train_score_metric = train_score[metric] if train_score is not None else -1
            val_score_metric = val_score[metric]
            test_score_metric = test_score[metric]
            best_val_score_metric = self.evaluator.best_val_score[metric]
            best_test_score_metric = self.evaluator.best_test_score[metric]
            best_iter_metric = self.evaluator.best_iter_num[metric]
            # TODO ivasilei hide the complexity and modularize with a function inside the ClassificationMetric Class
            if isinstance(val_score_metric, dict):
                # this case happens when the results are per class as for example the per_class_f1_score metric
                # in that case the val_score may be { 0:{recall: 0.7, precision: 0.4, f1-score:0.3 },
                # 1:{recall: 0.1, precision: 0.2, f1-score:0.1 }, 2:{recall: 0.2, precision: 0.6, f1-score:0.4 } }
                # we read the dictionary and update the metric correspondingly
                for key in val_score_metric.keys():
                    # the first hierarchy is the label type
                    label_type = key
                    metric_ = "Class_type_" + label_type
                    val_score_metric_ = val_score_metric[label_type]
                    test_score_metric_ = test_score_metric[label_type]
                    best_val_score_metric_ = best_val_score_metric[label_type]
                    best_test_score_metric_ = best_test_score_metric[label_type]
                    if isinstance(val_score_metric_, dict):
                        for key1 in val_score_metric_.keys():
                            # the second hierarchy is the metric type
                            sub_metric = key1
                            metric__ = metric_ + "_sub_metric_" +sub_metric
                            val_score_metric__ = val_score_metric_[sub_metric]
                            test_score_metric__ = test_score_metric_[sub_metric]
                            best_val_score_metric__ = best_val_score_metric_[sub_metric]
                            best_test_score_metric__ = best_test_score_metric_[sub_metric]

                            self.log_print_per_metric(metric__, train_score_metric, val_score_metric__,
                                                      test_score_metric__, dur_eval, total_steps,
                                                      best_val_score_metric__, best_test_score_metric__,
                                                      best_iter_metric, train_score)
                    else:
                            self.log_print_per_metric(metric_, train_score_metric, val_score_metric_,
                                                      test_score_metric_, dur_eval, total_steps, best_val_score_metric_,
                                                      best_test_score_metric_, best_iter_metric, train_score)
            else:
                self.log_print_per_metric(metric, train_score_metric, val_score_metric, test_score_metric, dur_eval,
                                          total_steps, best_val_score_metric, best_test_score_metric, best_iter_metric,
                                          train_score)

    def log_print_per_metric(self, metric, train_score_metric, val_score_metric, test_score_metric, dur_eval,
                             total_steps, best_val_score_metric, best_test_score_metric, best_iter_metric, train_score):
        """
        This functions prints and logs for a specific metric.

        """
        print("Train {}: {:.4f}, Val {}: {:.4f}, Test {}: {:.4f}, Eval time: {:.4f}, Evaluation step: {:.4f}".format(
            metric, train_score_metric, metric, val_score_metric, metric, test_score_metric, dur_eval, total_steps))
        print("Best val {}: {:.4f}, Best test {}: {:.4f}, Best iter: {:.4f}".format(
            metric, best_val_score_metric, metric, best_test_score_metric, best_iter_metric))
        self.log_metric("Best val {}".format(metric), best_val_score_metric, total_steps)
        self.log_metric("Best test {}".format(metric), best_test_score_metric, total_steps)
        self.log_metric("Best iter {}".format(metric), best_iter_metric, total_steps)
        self.log_metric("Val {}".format(metric), val_score_metric, total_steps)
        self.log_metric("Test {}".format(metric), test_score_metric, total_steps)
        if train_score is not None:
            self.log_metric("Train {}".format(metric), train_score_metric, total_steps)

        print()

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

    def init_emb_layer(self, g, feat_size, text_feat_ntypes, dev_id):
        if self.pretrain_emb_layer:
            # create embeddings
            embed_layer = DistGraphEmbed(g,
                                        feat_size,
                                        text_feat_ntypes,
                                        self.n_hidden,
                                        bert_dim=self.bert_hidden_size,
                                        dropout=self.dropout,
                                        self_loop_init=self.self_loop_init,
                                        use_node_embeddings=self.use_node_embeddings)
            self.sparse_embeds = embed_layer.sparse_embeds
            # If there are dense parameters in the embedding layer
            # or we use Pytorch saprse embeddings.
            if len(embed_layer.input_projs) > 0:
                embed_layer = embed_layer.to(dev_id)
                embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        else:
            embed_layer = None
            self.sparse_embeds = None

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

    def init_bert_encoder(self, g, bert_model, dev_id):
        print("Init distributed bert model ...")
        assert not self.mixed_precision
        for ntype in bert_model.keys():
            bert_model[ntype] = DistributedDataParallel(bert_model[ntype].to(dev_id),
                                                        device_ids=[dev_id], output_device=dev_id)
        return bert_model

    def init_dist_encoder(self, train):
        # prepare features
        g = self.g
        bert_model = self.bert_model
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

        text_feat_ntypes = []
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                text_feat_ntypes.append(ntype)
        embed_layer = self.init_emb_layer(g, feat_size, text_feat_ntypes, dev_id)
        gnn_encoder = self.init_gnn_encoder(g, dev_id)
        bert_model = self.init_bert_encoder(g, bert_model, dev_id)

        self.gnn_encoder = gnn_encoder
        self.embed_layer = embed_layer
        self._bert_model = bert_model

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

    def init_bert_model(self, train):
        bert_params = list([])
        bert_model = self.bert_model
        g = self.g
        if len(bert_model) > 0:
            for ntype, bm in bert_model.items():
                params = list(bm.parameters())
                bert_params = bert_params + params
                num_params = np.sum([np.prod(param.shape) for param in params])
                if g.rank() == 0:
                    print('Bert model of {} has {} parameters'.format(ntype, num_params))

            if train:
                fine_tune_opt = th.optim.Adam(bert_params, lr=self.bert_lr)
            bert_train, bert_static = wrap_bert(g,
                                                bert_model,
                                                bert_infer_bs=self.bert_infer_bs,
                                                debug=self.debug)
        else:
            fine_tune_opt = None
            bert_train = {}
            bert_static = {}

        self.bert_train = bert_train
        self.bert_static = bert_static
        if train:
            self.fine_tune_opt = fine_tune_opt
        else:
            self.fine_tune_opt = None

    def restoring_gsgnn(self, train):
        g = self.g

        # Restore the model weights and or optimizers to a checkpoint saved previously.
        if self.restore_model_path is not None:
            print('load GNN model from ', self.restore_model_path)
            load_gsgnn_model(self.restore_model_path, self.gnn_encoder, self.embed_layer, self.bert_model, self.decoder)
            if g.rank() == 0:
                print('Load Sparse embedding from ', self.restore_model_path)
                load_sparse_embeds(self.restore_model_path, self.embed_layer)

        if self.restore_optimizer_path is not None and train:
            print('load GSgnn optimizer state from ', self.restore_model_path)
            load_opt_state(self.restore_model_path, self.optimizer, self.fine_tune_opt, self.emb_optimizer)

        if self.restore_model_encoder_path is not None:
            print('load GNN model encoder from ', self.restore_model_encoder_path)
            load_gsgnn_model(self.restore_model_encoder_path, self.gnn_encoder, self.embed_layer, self.bert_model, decoder=None)

        # Restore the bert model to a checkpoint saved previously.
        if self.restore_bert_model_path is not None:
            print('load BERT model from ', self.restore_bert_model_path)
            load_gsgnn_model(self.restore_bert_model_path, None, None, self.bert_model, None)

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
        self.init_bert_model(train)

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
            self.combine_optimizer = OptimizerCombiner(optimizer_list=[self.optimizer, self.emb_optimizer,
                                                                       self.fine_tune_opt])
        else:
            self.combine_optimizer = None


    def generate_bert_cache(self, g):
        ''' Generate the cache of the BERT embeddings on the nodes of the input graph.

        The cache is specific to the input graph and the embeddings are generated with the model
        in this class.

        Parameters
        ----------
        g : DGLGraph
            The input graph

        Returns
        -------
        dict of EmbedCache
            The embedding caches for each node type.
        '''
        device = 'cuda:%d' % self.dev_id
        bert_emb_cache = {}
        assert isinstance(self.bert_train, dict)
        assert isinstance(self.bert_static, dict)
        assert len(self.bert_static) > 0 # Only bert_static is used in this case
        embs = extract_bert_embeddings_dist(g, self.bert_infer_bs, self.bert_train, self.bert_static,
                                            self.bert_hidden_size, dev=device,
                                            verbose=self.verbose, client=self.tracker,
                                            mlflow_report_frequency=self.mlflow_report_frequency)
        for ntype, emb in embs.items():
            bert_emb_cache[ntype] = EmbedCache(emb)
        return bert_emb_cache

    def compute_embeddings(self, g, device, bert_emb_cache, target_nidx=None):
        """
        compute node embeddings

        Parameters
        ----------
        g : DGLGraph
            The input graph
        device: th.device
            Device to run the computation
        bert_emb_cache : dict of embedding cache
            The embedding cache for the nodes in the input graph.
        target_nidx: dict of tensors
            The idices of nodes to generate embeddings.
        """
        if self.model_encoder_type in BUILTIN_GNN_ENCODER:
            embeddings = self.compute_gnn_embeddings(g, device, bert_emb_cache, target_nidx)
        else:
            embeddings = self.compute_lm_embeddings(g, device, bert_emb_cache, target_nidx)
        return embeddings

    def compute_gnn_embeddings(self, g, device, bert_emb_cache, target_nidx):
        """
        compute node embeddings

        Parameters
        ----------
        g : DGLGraph
            The input graph
        device: th.device
            Device to run the computation
        bert_emb_cache : dict of embedding cache
            The embedding cache for the nodes in the input graph.
        target_nidx: dict of tensors
            The idices of nodes to generate embeddings.
        """
        assert self.gnn_encoder is not None, "GNN model should be initialized"
        assert self.embed_layer is not None, "Node embedding layer should be initialized"
        th.distributed.barrier()
        self.gnn_encoder.eval()
        self.embed_layer.eval()
        for ntype in g.ntypes:
            if len(self.bert_train) > 0 and \
                ntype in self.bert_train.keys():
                self.bert_train[ntype].eval()
                self.bert_static[ntype].eval()

        # If we need to compute all node embeddings, full-graph inference is much more efficient.
        # Let's use mini-batch inference if we need to compute the embeddings of a subset of nodes.
        if self.mini_batch_infer and target_nidx is not None:
            pb = g.get_partition_book()
            if g.rank() == 0:
                print('Compute embeddings with mini-batch inference.')
            embeddings = do_mini_batch_inference(model=self.gnn_encoder,
                                                 embed_layer=self.embed_layer,
                                                 bert_train=self.bert_train,
                                                 bert_static=self.bert_static,
                                                 bert_hidden_size=self.bert_hidden_size,
                                                 device=device,
                                                 bert_emb_cache=bert_emb_cache,
                                                 target_nidx=target_nidx,
                                                 g=g,
                                                 pb=pb,
                                                 n_hidden=self.n_hidden,
                                                 fanout=self.eval_fanout,
                                                 eval_batch_size=self.eval_batch_size,
                                                 use_bert_embeddings_for_validation=False,
                                                 client=self.tracker,
                                                 mlflow_report_frequency=self.mlflow_report_frequency,
                                                 feat_field=self._feat_name)
        else:
            if g.rank() == 0:
                print('Compute embeddings with full-graph inference.')
            embeddings = do_fullgraph_infer(g=g,
                                            model=self.gnn_encoder,
                                            embed_layer=self.embed_layer,
                                            bert_train=self.bert_train,
                                            bert_static=self.bert_static,
                                            bert_hidden_size=self.bert_hidden_size,
                                            device=device,
                                            bert_emb_cache=bert_emb_cache,
                                            bert_infer_bs=self.bert_infer_bs,
                                            eval_fanout_list=self.eval_fanout,
                                            eval_batch_size=self.eval_batch_size,
                                            client=self.tracker,
                                            mlflow_report_frequency=self.mlflow_report_frequency,
                                            feat_field=self._feat_name)
            if target_nidx is not None:
                embeddings = {ntype: LazyDistTensor(embeddings[ntype], target_nidx[ntype]) for ntype in target_nidx.keys()}
        # wait all workers to finish
        self.gnn_encoder.train()
        self.embed_layer.train()
        for ntype in g.ntypes:
            if len(self.bert_train) > 0 and \
                ntype in self.bert_train.keys():
                self.bert_train[ntype].train()
                self.bert_static[ntype].train()
        th.distributed.barrier()
        return embeddings

    def compute_lm_embeddings(self, g, device, emb_cache, target_nidx_per_ntype):
        """
        compute node embeddings for lm

        Parameters
        ----------
        g : DGLGraph
            The input graph
        target_nidx_per_ntype: dictionary of tensors
            The idices of nodes to generate embeddings.
        """
        th.distributed.barrier()
        if self.embed_layer is not None:
            self.embed_layer.eval()
        for ntype in g.ntypes:
            if ntype in self.bert_train.keys():
                self.bert_train[ntype].eval()
                self.bert_static[ntype].eval()
        if g.rank() == 0:
            t = time.time()
            print('Compute lm embeddings with mini-batch inference.')
        if target_nidx_per_ntype is None:
            # if the dictionary is not initialized for full graph inference for example this code will fail.
            target_nidx_per_ntype = {ntype: th.arange(g.number_of_nodes(ntype)) for ntype in g.ntypes}
        embeddings = do_mini_batch_inference(model=self.gnn_encoder,
                                             embed_layer=self.embed_layer,
                                             bert_train=self.bert_train,
                                             bert_static=self.bert_static,
                                             bert_hidden_size=self.bert_hidden_size,
                                             device=device,
                                             bert_emb_cache=emb_cache,
                                             target_nidx=target_nidx_per_ntype,
                                             g=g,
                                             pb=g.get_partition_book(),
                                             n_hidden=self.n_hidden,
                                             fanout=self.eval_fanout,
                                             eval_batch_size=self.eval_batch_size,
                                             use_bert_embeddings_for_validation=False,
                                             client=self.tracker,
                                             mlflow_report_frequency=self.mlflow_report_frequency,
                                             feat_field=self._feat_name)
        if g.rank() == 0:
            print("Compute embeddings with mini-batch inference takes : {:.4f}".format(time.time()-t))
        if self.embed_layer is not None:
            self.embed_layer.train()
        for ntype in g.ntypes:
            if ntype in self.bert_train.keys():
                self.bert_train[ntype].train()
                self.bert_static[ntype].train()
        return embeddings

    def encoder_forward(self, blocks, input_nodes, bert_emb_cache, bert_forward_time, gnn_forward_time, epoch):
        g = self.g
        bert_train = self.bert_train
        bert_static = self.bert_static
        bert_hidden_size = self.bert_hidden_size
        device = 'cuda:%d' % self.dev_id
        model = self.gnn_encoder
        embed_layer = self.embed_layer

        blocks = [blk.to(device) for blk in blocks]

        # Only use training nodes during bert back propagation
        train_node_masks = rand_gen_trainmask(g, input_nodes, self.train_nodes,
                                              disable_training=epoch < self.gnn_warmup_epochs)

        if self.debug:
            th.distributed.barrier()
        t1 = time.time()
        inputs, _ = prepare_batch_input(g,
                                        bert_train,
                                        bert_static,
                                        bert_hidden_size,
                                        input_nodes,
                                        train_mask=train_node_masks,
                                        emb_cache=bert_emb_cache,
                                        dev=device,
                                        verbose=self.verbose,
                                        feat_field=self._feat_name)
        t2 = time.time()
        bert_forward_time += (t2 - t1)

        input_nodes = {ntype: inodes.long().to(device) for ntype, inodes in input_nodes.items()}
        emb = embed_layer(inputs, input_nodes=input_nodes) if embed_layer is not None else inputs
        gnn_embs = model(emb, blocks) if model is not None else {ntype: nemb.to(device)
                for ntype, nemb in emb.items()}

        t3 = time.time()
        gnn_forward_time += (t3 - t2)

        return gnn_embs, bert_forward_time, gnn_forward_time

    def save_model_embed(self, epoch, i, g, bert_emb_cache):
        '''Save the model and node embeddings for a certain iteration in an epoch..
        '''
        model_conf = self.model_conf
        model = self.gnn_encoder
        embed_layer = self.embed_layer
        decoder = self.decoder
        bert_model = self.bert_model
        assert model_conf is not None
        assert bert_model is not None

        # sync before model saving
        th.distributed.barrier()
        if self.save_model_path is not None and g.rank() == 0:
            start_save_t = time.time()
            save_model_path = self.save_model_path + '-' + str(epoch)
            if i is not None:
                save_model_path = save_model_path + '-' + str(i)
            save_gsgnn_model(model_conf, save_model_path, model, embed_layer, bert_model, decoder)
            save_sparse_embeds(save_model_path, embed_layer)
            save_opt_state(save_model_path, self.optimizer, self.fine_tune_opt, self.emb_optimizer)
            print('successfully save the model to ' + save_model_path)
            print('Time on save model {}'.format(time.time() - start_save_t))

        if self.save_embeds_path is not None:
            # Generate all the node embeddings
            device = 'cuda:%d' % self.dev_id
            embeddings = self.compute_embeddings(g, device, bert_emb_cache)

            # save embeddings in a distributed way
            save_embeds_path = self.save_embeds_path + '-' + str(epoch)
            if i is not None:
                save_embeds_path = save_embeds_path + '-' + str(i)
            save_gsgnn_embeddings(save_embeds_path, embeddings, g.rank(), th.distributed.get_world_size())

        # wait for rank0 to save the model and/or embeddings
        th.distributed.barrier()

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
            A tuple of (bert forward time, gnn forward time and backward time)
        '''
        bert_forward_time, gnn_forward_time, back_time = compute_time
        device = 'cuda:%d' % self.dev_id

        if self.debug:
            flops_strs = get_bert_flops_info(self.bert_train, self.bert_static)
            print('Epoch {:05d} | Batch {:03d} | {}'.format(epoch, i, flops_strs))

        print("Epoch {:05d} | Batch {:03d} | GPU Mem reserved: {:.4f} MB | Peak Mem alloc: {:.4f} MB".
                format(epoch, i,
                    th.cuda.memory_reserved(device) / 1024 / 1024,
                    th.cuda.max_memory_allocated(device) / 1024 /1024))
        print('Epoch {:05d} | Batch {:03d} | RAM memory {} used'.format(epoch, i, psutil.virtual_memory()))
        print('Epoch {:05d} | Batch {:03d} | Avg input nodes per iter {} | Bert forward {:05f} | GNN forward {:05f} | Backward {:05f}'.format(
            epoch, i,
            num_input_nodes,
            bert_forward_time,
            gnn_forward_time,
            back_time))

    @abc.abstractmethod
    def fit(self):
        pass

    @abc.abstractmethod
    def eval(self):
        pass

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
    def bert_model(self):
        return self._bert_model

    @property
    def use_self_loop(self):
        return self._use_self_loop

    @property
    def pretrain_emb_layer(self):
        return self._pretrain_emb_layer

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
    def bert_lr(self):
        return self._bert_lr

    @property
    def debug(self):
        return self._debug

    @property
    def bert_infer_bs(self):
        return self._bert_infer_bs

    @property
    def train_nodes(self):
        return self._train_nodes

    @property
    def use_bert_cache(self):
        return self._use_bert_cache

    @property
    def refresh_cache(self):
        return self._refresh_cache

    @property
    def gnn_warmup_epochs(self):
        return self._gnn_warmup_epochs

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
    def restore_bert_model_path(self):
        return self._restore_bert_model_path

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

    @property
    def mlflow_report_frequency(self):
        return self._mlflow_report_frequency
