import math
import os

import dgl
import torch as th

from ..model.utils import TopKList
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.utils import remove_saved_models as remove_gsgnn_models
from ..tracker import get_task_tracker_class

from ..utils import sys_tracker

class GSgnnTrainer():
    """ Generic GSgnn trainer.

    Parameters
    ----------
    config: GSConfig
        Configurations
    """
    def __init__(self, config):
        super(GSgnnTrainer, self).__init__()

        self._no_validation = config.no_validation

        self._mini_batch_infer = config.mini_batch_infer
        self._init_train_config(config)

        # neighbor sample related
        self._fanout = config.fanout
        # evaluation specific
        self._eval_fanout = config.eval_fanout
        # add a check if all the edge types are specified :
        # TODO(zhengda) let's check this somewhere else.
        #if isinstance(self.eval_fanout[0], dict):
        #    for fanout_dic in self.eval_fanout:
        #        for e in g.etypes:
        #            assert e in fanout_dic.keys(), \
        #                    "The edge type {} is not included in the specified eval fanout".format(e)

        # distributed training config
        self._local_rank = config.local_rank
        self._ip_config = config.ip_config
        self._part_config = config.part_config
        self._save_model_per_iters = config.save_model_per_iters
        self._save_model_path = config.save_model_path
        self._restore_model_path = config.restore_model_path
        self._restore_optimizer_path = config.restore_optimizer_path
        self._save_embeds_path = config.save_embeds_path
        self._batch_size = config.batch_size

        self._debug = config.debug
        self._verbose = config.verbose

        self._optimizer = None
        self._evaluator = None
        # evaluation
        self._eval_batch_size = config.eval_batch_size

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
        # setup cuda env
        self.setup_cuda(config.local_rank)

        self.init_dist_context(config.ip_config,
                               config.graph_name,
                               config.part_config,
                               config.backend)

        # Set task tracker
        tracker_class = get_task_tracker_class(config.task_tracker)
        task_tracker = tracker_class(config, self._g.rank())
        self.setup_task_tracker(task_tracker)
        self.log_params(config.__dict__)

    def _init_train_config(self, config):
        # neighbor sample related
        self._fanout = config.fanout
        # add a check if all the edge types are specified:
        # TODO(zhengda) let's check this somewhere else
        #if isinstance(self.fanout[0], dict):
        #    for fanout_dic in self.fanout:
        #        for e in self.g.etypes:
        #            assert e in fanout_dic.keys(), "The edge type {} is not included in the specified fanout".format(e)

        # training related
        self._batch_size = config.batch_size
        self._dropout = config.dropout
        self._sparse_lr = config.sparse_lr
        self._lr = config.lr
        self._weight_decay = config.wd_l2norm
        self._n_epochs = config.n_epochs
        assert self._batch_size > 0, "batch size must > 0."


    def setup_cuda(self, local_rank):
        """ Set up the CUDA device of this trainer.

        The CUDA device is set up based on the local rank.

        Parameters
        ----------
        local_rank : int
            The local rank in the machine.
        """
        # setup cuda env
        use_cuda = th.cuda.is_available()
        assert use_cuda, "Only support GPU training"
        dev_id = local_rank
        th.cuda.set_device(dev_id)
        self._dev_id = dev_id

    def init_dist_context(self, ip_config, graph_name, part_config, backend):
        """ Initialize distributed inference context

        Parameters
        ----------
        ip_config: str
            File path of ip_config file
        graph_name: str
            Name of the graph
        part_config: str
            File path of partition config
        backend: str
            Torch distributed backend
        """

        # We need to use socket for communication in DGL 0.8. The tensorpipe backend has a bug.
        # This problem will be fixed in the future.
        dgl.distributed.initialize(ip_config, net_type='socket')
        self._g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
        sys_tracker.check("load DistDGL")
        th.distributed.init_process_group(backend=backend)

    def setup_task_tracker(self, task_tracker):
        self._task_tracker = task_tracker

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
        self.task_tracker.log_iter_metrics(self.evaluator.metric,
                train_score=train_score, val_score=val_score,
                test_score=test_score, best_val_score=best_val_score,
                best_test_score=best_test_score, best_iter_num=best_iter_num,
                eval_time=dur_eval, total_steps=total_steps)

    def save_model_embed(self, g, model, epoch, i):
        '''Save the model and node embeddings for a certain iteration in an epoch.
        '''
        # TODO(zhengda) we need to separate model saving and embedding computation.
        # sync before model saving
        th.distributed.barrier()
        if self.save_model_path is not None and g.rank() == 0:
            save_model_path = self._gen_model_path(self.save_model_path, epoch, i)
            model.module.save_model(save_model_path)
            self.optimizer.save_opt_state(save_model_path)

        feat_name = self.config.feat_name
        eval_fanout = self.config.eval_fanout
        eval_batch_size = self.config.eval_batch_size
        mini_batch_infer = self.config.mini_batch_infer
        if self.save_embeds_path is not None:
            # Generate all the node embeddings
            embeddings = model.module.compute_embeddings(g, feat_name, None,
                                                         eval_fanout, eval_batch_size,
                                                         mini_batch_infer)

            # save embeddings in a distributed way
            save_embeds_path = self._gen_model_path(self.save_embeds_path, epoch, i)
            save_gsgnn_embeddings(save_embeds_path, embeddings, g.rank(),
                                  th.distributed.get_world_size())

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

    def save_topk_models(self, g, model, epoch, i, val_score):
        """ Based on the given val_score, decided if save the current model trained in the i_th
            iteration and the epoch_th epoch.

        Parameters
        ----------
        g: DGLDistGraph
            The distributed graph used in the current training.
        model : pytorch model
            The GNN model.
        epoch: int
            The number of training epoch.
        i: int
            The number of iteration in a training epoch.
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
            self.save_model_embed(g, model, epoch, i)

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
            A tuple of (forward time and backward time)
        '''
        gnn_forward_time, back_time = compute_time
        device = 'cuda:%d' % self.dev_id

        print("Epoch {:05d} | Batch {:03d} | GPU Mem reserved: {:.4f} MB | Peak Mem alloc: {:.4f} MB".
                format(epoch, i,
                    th.cuda.memory_reserved(device) / 1024 / 1024,
                    th.cuda.max_memory_allocated(device) / 1024 /1024))
        print('Epoch {:05d} | Batch {:03d} | RAM memory {} used'.format(epoch, i, psutil.virtual_memory()))
        print('Epoch {:05d} | Batch {:03d} | Avg input nodes per iter {} | forward {:05f} | Backward {:05f}'.format(
            epoch, i,
            num_input_nodes,
            gnn_forward_time,
            back_time))

    def restore_model(self, model, train):
        model.restore_model(train, self.restore_model_path)
        if self.restore_optimizer_path is not None and train:
            print('load GNN optimizer state from ', self.restore_model_path)
            self.optimizer.load_opt_state(self.restore_model_path)

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
    def batch_size(self):
        return self._batch_size

    @property
    def mini_batch_infer(self):
        return self._mini_batch_infer

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
    def save_model_per_iters(self):
        return self._save_model_per_iters

    @property
    def save_model_path(self):
        return self._save_model_path

    @property
    def save_embeds_path(self):
        return self._save_embeds_path

    @property
    def restore_model_path(self):
        return self._restore_model_path

    @property
    def restore_optimizer_path(self):
        return self._restore_optimizer_path

    def register_evaluator(self, evaluator):
        self._evaluator = evaluator

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def eval_batch_size(self):
        return self._eval_batch_size

    @property
    def task_tracker(self):
        return self._task_tracker

    @property
    def dev_id(self):
        return self._dev_id
