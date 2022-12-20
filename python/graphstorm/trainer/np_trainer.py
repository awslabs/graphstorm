import time

import dgl
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..dataloading import GSgnnNodeDataLoader
from ..dataloading import GSgnnNodeTrainData
from ..model import prepare_batch_input
from ..model import create_node_gnn_model
from ..eval import GSgnnAccEvaluator
from ..eval import GSgnnRegressionEvaluator
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker

def get_eval_class(config):
    if config.task_type == "node_regression":
        return GSgnnRegressionEvaluator
    elif config.task_type == 'node_classification':
        return GSgnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

class GSgnnNodePredictTrainer(GSgnnTrainer):
    """ A trainer for node prediction

    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self, config):
        super(GSgnnNodePredictTrainer, self).__init__(config)
        self.config = config
        self.predict_ntype = config.predict_ntype

    def save(self):
        pass

    def load(self):
        pass

    def register_evaluator(self, evaluator):
        self._evaluator = evaluator

    def fit(self):
        sys_tracker.check('fit start')
        g = self._g
        pb = g.get_partition_book()
        config = self.config
        device = 'cuda:%d' % self.dev_id
        feat_field = config.feat_name

        train_data = GSgnnNodeTrainData(g, pb, self.predict_ntype, config.label_field)
        sys_tracker.check('construct training data')

        eval_class = get_eval_class(config)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self._evaluator = eval_class(g, config, train_data)
            self.evaluator.setup_task_tracker(self.task_tracker)
        model = create_node_gnn_model(g, self.config, train_task=True)
        model = model.to(self.dev_id)
        model = DistributedDataParallel(model, device_ids=[self.dev_id],
                                        output_device=self.dev_id)
        self._optimizer = model.module.init_optimizer(self.lr, self.sparse_lr,
                                                      self.weight_decay)

        dataloader = GSgnnNodeDataLoader(g,
                                         train_data,
                                         self.fanout,
                                         model.module.num_gnn_layers,
                                         self.batch_size,
                                         device)

        # training loop
        dur = []
        total_steps = 0
        num_input_nodes = 0
        forward_time = 0
        back_time = 0
        early_stop = False # used when early stop is True
        sys_tracker.check('start training')
        for epoch in range(self.n_epochs):
            model.train()
            t0 = time.time()
            for i, (input_nodes, seeds, blocks) in enumerate(dataloader):
                total_steps += 1

                # in the case of a graph with a single node type the returned seeds will not be
                # a dictionary but a tensor of integers this is a possible bug in the DGL code.
                # Otherwise we will select the seeds that correspond to the category node type
                if type(seeds) is dict:
                    seeds = seeds[self.predict_ntype]     # we only predict the nodes with type "category"
                if type(input_nodes) is not dict:
                    input_nodes = {self.predict_ntype: input_nodes}
                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]
                batch_tic = time.time()

                input_feats = prepare_batch_input(g, input_nodes,
                                                  dev=device,
                                                  feat_field=feat_field)
                lbl = train_data.labels[seeds].to(device)
                blocks = [block.to(device) for block in blocks]
                t2 = time.time()
                loss = model(blocks, input_feats, input_nodes, lbl)

                t3 = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                forward_time += (t3 - t2)
                back_time += (time.time() - t3)

                self.log_metric("Train loss", loss.item(), total_steps, report_step=total_steps)

                if i % 20 == 0 and g.rank() == 0:
                    if self.verbose:
                        self.print_info(epoch, i,  num_input_nodes / 20,
                                        (forward_time / 20, back_time / 20))
                    print("Part {} | Epoch {:05d} | Batch {:03d} | Total_Train Loss: {:.4f} | Time: {:.4f}".
                            format(g.rank(), epoch, i,  loss.item(), time.time() - batch_tic))
                    num_input_nodes = gnn_forward_time = back_time = 0

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):
                    val_score = self.eval(model, g.rank(), train_data, total_steps)

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

            # end of an epoch
            th.distributed.barrier()
            epoch_time = time.time() - t0
            if g.rank() == 0:
                print("Epoch {} take {}".format(epoch, epoch_time))
            dur.append(epoch_time)

            val_score = None
            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(model, g.rank(), train_data, total_steps)
                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(g, model, epoch, None, val_score)

            # early_stop, exit training
            if early_stop is True:
                break

        if g.rank() == 0:
            if self.verbose:
                if self.evaluator is not None:
                    self.evaluator.print_history()

        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if g.rank() == 0:
            output = dict(best_test_score=self.evaluator.best_test_score,
                          best_val_score=self.evaluator.best_val_score,
                          peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024)
            self.log_params(output)

            if self.verbose:
                # print top k info only when required because sometime the top k is just the last k
                print(f'Top {len(self.topklist.toplist)} ranked models:')
                print([f'Rank {i+1}: epoch-{epoch}' for i, epoch in enumerate(self.topklist.toplist)])

    def eval(self, model, rank, train_data, total_steps):
        """ do the model evaluation using validiation and test sets

            Parameters
            ----------
            model : Pytorch model
                The GNN model.
            rank: int
                Distributed rank
            train_data: GSgnnNodeTrainData
                Training data
            total_steps: int
                Total number of iterations.

            Returns
            -------
            float: validation score
        """
        sys_tracker.check('before eval')
        g = self._g
        feat_name = self.config.feat_name
        eval_fanout = self.config.eval_fanout
        eval_batch_size = self.config.eval_batch_size
        mini_batch_infer = self.config.mini_batch_infer
        predict_ntype = self.config.predict_ntype
        teval = time.time()
        target_nidx = th.cat([train_data.val_idx, train_data.test_idx])
        pred, _ = model.module.predict(g, feat_name, {predict_ntype: target_nidx}, eval_fanout,
                                       eval_batch_size, mini_batch_infer, self.task_tracker)
        sys_tracker.check('predict')

        val_pred, test_pred = th.split(pred,
                                       [len(train_data.val_idx),
                                       len(train_data.test_idx)])
        val_label = train_data.labels[train_data.val_idx]
        val_label = val_label.to(val_pred.device)
        test_label = train_data.labels[train_data.test_idx]
        test_label = test_label.to(test_pred.device)
        val_score, test_score = self.evaluator.evaluate(val_pred, test_pred,
                                                        val_label, test_label,
                                                        total_steps)
        sys_tracker.check('evaluate')
        if rank == 0:
            self.log_print_metrics(val_score=val_score,
                                    test_score=test_score,
                                    dur_eval=time.time() - teval,
                                    total_steps=total_steps)
        return val_score
