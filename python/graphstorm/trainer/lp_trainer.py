import time

import dgl
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..dataloading import GSgnnLinkPredictionTrainData
from ..dataloading import GSgnnLinkPredictionDataLoader
from ..dataloading import GSgnnLPJointNegDataLoader
from ..dataloading import GSgnnLPLocalUniformNegDataLoader
from ..dataloading import GSgnnAllEtypeLPJointNegDataLoader
from ..dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from ..model import prepare_batch_input
from ..model import create_lp_gnn_model
from ..eval import GSgnnMrrLPEvaluator
from .gsgnn_trainer import GSgnnTrainer

from ..dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from ..dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from ..dataloading import BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER
from ..dataloading import BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER
from ..dataloading import BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER

from ..utils import sys_tracker

def get_eval_class(config):
    return GSgnnMrrLPEvaluator

class GSgnnLinkPredictionTrainer(GSgnnTrainer):
    """ Link prediction trainer.

    This is a highlevel trainer wrapper that can be used directly to train a link prediction model.

    Usage:
    ```
    from graphstorm.config import GSConfig
    from graphstorm.model.huggingface import HuggingfaceBertLoader
    from graphstorm.model import GSgnnLinkPredictionTrainer

    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    trainer = GSgnnLinkPredictionTrainer(config, lm_models)
    trainer.fit()
    ```

    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self, config):
        super(GSgnnLinkPredictionTrainer, self).__init__(config)
        self.config = config

        self.train_etypes = config.train_etype
        self.eval_etypes = config.eval_etype

        # sampling related
        self.negative_sampler = config.negative_sampler
        self.num_negative_edges = config.num_negative_edges
        self.exclude_training_targets = config.exclude_training_targets
        self.reverse_edge_types_map = config.reverse_edge_types_map

    def save(self):
        pass

    def load(self):
        pass

    def register_evaluator(self, evaluator):
        self._evaluator = evaluator

    def create_dataloader(self, g, train_data, num_gnn_layers, device):
        if self.negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            dataloader = GSgnnLinkPredictionDataLoader(g,
                                                       train_data,
                                                       self.fanout,
                                                       num_gnn_layers,
                                                       self.batch_size,
                                                       self.num_negative_edges,
                                                       device,
                                                       self.exclude_training_targets,
                                                       self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
            dataloader = GSgnnLPJointNegDataLoader(g,
                                                   train_data,
                                                   self.fanout,
                                                   num_gnn_layers,
                                                   self.batch_size,
                                                   self.num_negative_edges,
                                                   device,
                                                   self.exclude_training_targets,
                                                   self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER:
            dataloader = GSgnnLPLocalUniformNegDataLoader(g,
                                                          train_data,
                                                          self.fanout,
                                                          num_gnn_layers,
                                                          self.batch_size,
                                                          self.num_negative_edges,
                                                          device,
                                                          self.exclude_training_targets,
                                                          self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER:
            dataloader = GSgnnAllEtypeLinkPredictionDataLoader(g,
                                                       train_data,
                                                       self.fanout,
                                                       num_gnn_layers,
                                                       self.batch_size,
                                                       self.num_negative_edges,
                                                       device,
                                                       self.exclude_training_targets,
                                                       self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER:
            dataloader = GSgnnAllEtypeLPJointNegDataLoader(g,
                                                   train_data,
                                                   self.fanout,
                                                   num_gnn_layers,
                                                   self.batch_size,
                                                   self.num_negative_edges,
                                                   device,
                                                   self.exclude_training_targets,
                                                   self.reverse_edge_types_map)
        else:
            raise Exception('Unknown negative sampler')
        return dataloader

    def fit(self, full_graph_training=False):
        sys_tracker.check('fit start')
        g = self._g
        pb = g.get_partition_book()
        config = self.config
        device = 'cuda:%d' % self.dev_id
        feat_field = config.feat_name

        train_data = GSgnnLinkPredictionTrainData(g, pb, self.train_etypes,
                                                  self.eval_etypes, full_graph_training)
        sys_tracker.check('construct training data')

        if g.rank() == 0:
            print("Use {} negative sampler with exclude training target {}".format(
                self.negative_sampler,
                self.exclude_training_targets))

        eval_class = get_eval_class(config)
        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self._evaluator = eval_class(g, config, train_data)
            self._evaluator.setup_task_tracker(self.task_tracker)
        model = create_lp_gnn_model(g, config, train_task=True)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[self.dev_id],
                                        output_device=self.dev_id)
        self._optimizer = model.module.init_optimizer(self.lr, self.sparse_lr, self.weight_decay)

        dataloader = self.create_dataloader(g, train_data, model.module.num_gnn_layers, device)

        # training loop
        dur = []
        best_epoch = 0
        num_input_nodes = 0
        total_steps = 0
        early_stop = False # used when early stop is True
        forward_time = 0
        back_time = 0

        sys_tracker.check('start training')
        for epoch in range(self.n_epochs):
            model.train()
            t0 = time.time()
            for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
                total_steps += 1

                if type(input_nodes) is not dict:
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}

                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]

                batch_tic = time.time()
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [blk.to(device) for blk in blocks]
                input_feats = prepare_batch_input(g, input_nodes,
                                                  dev=device,
                                                  feat_field=feat_field)

                t2 = time.time()
                loss = model(blocks, pos_graph, neg_graph, input_feats, input_nodes)

                t3 = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                forward_time += (t3 - t2)
                back_time += (time.time() - t3)

                self.log_metric("Train loss", loss.item(), total_steps, report_step=total_steps)

                if i % 20 == 0 and g.rank() == 0:
                    if self.verbose:
                        self.print_info(epoch, i, num_input_nodes / 20,
                                        (forward_time / 20, back_time / 20))
                    print("Epoch {:05d} | Batch {:03d} | Train Loss: {:.4f} | Time: {:.4f}".
                            format(epoch, i, loss.item(), time.time() - batch_tic))
                    num_input_nodes = forward_time = back_time = 0

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):
                    val_score = self.eval(model, g.rank(), total_steps)

                    if self.evaluator.do_early_stop(val_score):
                        early_stop = True

                # Every n iterations, check to save the top k models. If has validation score, will save
                # the best top k. But if no validation, will either save the last k model or all models
                # depends on the setting of top k
                if self.save_model_per_iters > 0 and i % self.save_model_per_iters == 0 and i != 0:
                    self.save_topk_models(g, model, epoch, i, val_score)

                # early_stop, exit current interation.
                if early_stop is True:
                    break

            # ------- end of an epoch -------

            th.distributed.barrier()
            epoch_time = time.time() - t0
            if g.rank() == 0:
                print("Epoch {} take {}".format(epoch, epoch_time))
            dur.append(epoch_time)

            val_score = None
            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(model, g.rank(), total_steps)

                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(g, model, epoch, None, val_score)

            th.distributed.barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if g.rank() == 0:
            output = dict(best_test_mrr=self.evaluator.best_test_score, best_val_mrr=self.evaluator.best_val_score,
                          peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024,
                          best_epoch=best_epoch)
            self.log_params(output)

            if self.verbose:
                # print top k info only when required because sometime the top k is just the last k
                print(f'Top {len(self.topklist.toplist)} ranked models:')
                print([f'Rank {i+1}: epoch-{epoch}-iter-{iter}' \
                        for i, (epoch, iter) in enumerate(self.topklist.toplist)])

    def eval(self, model, rank, total_steps, train_score=None):
        """ do the model evaluation using validiation and test sets

            Parameters
            ----------
            rank: int
                Distributed rank
            embeddings: dict of DistTensor
                node embeddings
            total_steps: int
                Total number of iterations.
            train_score: float
                Training mrr, used in print.

            Returns
            -------
            float: validation mrr
        """
        sys_tracker.check('before eval')
        g = self._g
        feat_name = self.config.feat_name
        eval_fanout = self.config.eval_fanout
        eval_batch_size = self.config.eval_batch_size
        mini_batch_infer = self.config.mini_batch_infer
        device = 'cuda:%d' % self.dev_id
        test_start = time.time()
        # TODO(zhengda) we need to predict differently.
        embeddings = model.module.compute_embeddings(g, feat_name, None,
                                                     eval_fanout, eval_batch_size,
                                                     mini_batch_infer, self.task_tracker)
        sys_tracker.check('compute embeddings')
        decoder = model.module.decoder
        train_mrr = self.evaluator.evaluate_on_train_set(embeddings, decoder, device)
        sys_tracker.check('evaluate training')
        val_mrr, test_mrr = self.evaluator.evaluate(embeddings, decoder, total_steps, device)
        sys_tracker.check('evaluate validation/test')

        if rank == 0:
            self.log_print_metrics(val_score=val_mrr,
                                    test_score=test_mrr,
                                    dur_eval=time.time() - test_start,
                                    total_steps=total_steps,
                                    train_score=train_score)
        return val_mrr
