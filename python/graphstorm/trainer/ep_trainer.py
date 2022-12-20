import time

import dgl
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..dataloading import GSgnnEdgePredictionTrainData
from ..dataloading import GSgnnEdgePredictionDataLoader
from ..dataloading.utils import modify_fanout_for_target_etype
from ..model import prepare_batch_input
from ..model import create_edge_gnn_model
from ..eval import GSgnnAccEvaluator
from ..eval import GSgnnRegressionEvaluator
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker

def get_eval_class(config):
    if config.task_type == "edge_regression":
        return GSgnnRegressionEvaluator
    elif config.task_type == 'edge_classification':
        return GSgnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

class GSgnnEdgePredictionTrainer(GSgnnTrainer):
    """ Edge prediction trainer.

    This is a highlevel trainer wrapper that can be used directly to train a edge prediction model.

    Usage:
    ```
    from graphstorm.config import GSConfig
    from graphstorm.model.huggingface import HuggingfaceBertLoader
    from graphstorm.model import GSgnnEdgePredictionTrainer

    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    trainer = GSgnnEdgePredictionTrainer(config, lm_models)
    trainer.fit()
    ```

    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self, config):
        super(GSgnnEdgePredictionTrainer, self).__init__(config)
        self.config = config

        # TODO needs to be extended to multiple
        self.target_etype = config.target_etype

        # sampling related
        self.reverse_edge_types_map = config.reverse_edge_types_map
        self.remove_target_edge = config.remove_target_edge
        self.exclude_training_targets = config.exclude_training_targets

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
        device = 'cuda:%d' % self.dev_id
        config = self.config
        feat_field = config.feat_name

        train_data = GSgnnEdgePredictionTrainData(g, pb, self.target_etype, config.label_field)
        sys_tracker.check('construct training data')

        # adjusting the evaluation fanout if the removal of target edge is requested.
        if config.remove_target_edge:
            target_etype = [etype for etype in self.target_etype]
            reverse_edge_types_map = config.reverse_edge_types_map
            for e in target_etype:
                if e in reverse_edge_types_map and reverse_edge_types_map[e] not in target_etype:
                    target_etype.append(reverse_edge_types_map[e])
            self._eval_fanout = modify_fanout_for_target_etype(
                    g=g, fanout=self.eval_fanout, target_etypes=target_etype)

        eval_class = get_eval_class(config)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self._evaluator = eval_class(g, config, train_data)
            self.evaluator.setup_task_tracker(self.task_tracker)

        model = create_edge_gnn_model(g, config, train_task=True)
        model = model.to(self.dev_id)
        model = DistributedDataParallel(model, device_ids=[self.dev_id],
                                        output_device=self.dev_id)
        self._optimizer = model.module.init_optimizer(self.lr, self.sparse_lr, self.weight_decay)

        # TODO(zhengda) we need to make it work for multiple target etypes later.
        target_etype = self.target_etype[0]
        dataloader = GSgnnEdgePredictionDataLoader(g,
                                                   train_data,
                                                   self.fanout,
                                                   model.module.num_gnn_layers,
                                                   self.batch_size,
                                                   self.reverse_edge_types_map,
                                                   self.remove_target_edge,
                                                   self.exclude_training_targets,
                                                   device)

        # training loop
        print("start training...")
        dur = []
        best_epoch = 0
        num_input_nodes = 0
        forward_time = 0
        back_time = 0
        total_steps = 0
        early_stop = False # used when early stop is True

        for epoch in range(self.n_epochs):
            model.train()
            t0 = time.time()
            for i, (input_nodes, batch_graph, blocks) in enumerate(dataloader):
                total_steps += 1
                if type(input_nodes) is not dict:
                    # This happens on a homogeneous graph.
                    assert len(g.ntypes) == 1
                    input_nodes = {"node": input_nodes}

                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]
                batch_tic = time.time()

                # TODO(zhengda) expand code for multiple edge types
                # retrieving seed edge id from the graph to find labels
                seeds = batch_graph.edges[target_etype].data[dgl.EID]
                lbl = train_data.labels[seeds].to(device)
                blocks = [blk.to(device) for blk in blocks]
                batch_graph = batch_graph.to(device)
                input_feats = prepare_batch_input(g, input_nodes,
                                                  dev=device,
                                                  feat_field=feat_field)

                t2 = time.time()
                loss = model(blocks, batch_graph, input_feats, input_nodes, lbl)

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
                    # Print task specific info.
                    print(
                        "Part {} | Epoch {:05d} | Batch {:03d} | Train Loss: {:.4f} | Time: {:.4f}".
                        format(g.rank(), epoch, i, loss.item(), time.time() - batch_tic))
                    num_input_nodes = forward_time = back_time = 0

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

            # ------- end of an epoch -------

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

            th.distributed.barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if g.rank() == 0:
            output = dict(best_test_score=self.evaluator.best_test_score,
                          best_val_score=self.evaluator.best_val_score,
                          peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024,
                          best_epoch=best_epoch)
            self.log_params(output)

            if self.verbose:
                # print top k info only when required because sometime the top k is just the last k
                print(f'Top {len(self.topklist.toplist)} ranked models:')
                print([f'Rank {i+1}: epoch-{epoch}' \
                        for i, epoch in enumerate(self.topklist.toplist)])

    def eval(self, model, rank, train_data, total_steps):
        """ do the model evaluation using validiation and test sets

            Parameters
            ----------
            model : Pytorch model
                The GNN model.
            rank: int
                Distributed rank
            train_data: GSgnnEdgePredictionTrainData
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
        test_start = time.time()
        # TODO(zhengda) support multiple target etypes
        target_etype = self.target_etype[0]
        val_labels = train_data.labels[train_data.val_idxs[target_etype[1]]]
        test_labels = train_data.labels[train_data.test_idxs[target_etype[1]]]
        val_src_dst_pairs = train_data.val_src_dst_pairs[target_etype[1]]
        test_src_dst_pairs = train_data.test_src_dst_pairs[target_etype[1]]
        src_dst_pairs = (th.cat([val_src_dst_pairs[0], test_src_dst_pairs[0]]),
                         th.cat([val_src_dst_pairs[1], test_src_dst_pairs[1]]))
        pred, _ = model.module.predict(g, feat_name, {target_etype[1]: src_dst_pairs}, eval_fanout,
                                       eval_batch_size, mini_batch_infer, self.task_tracker)
        sys_tracker.check('predict')
        assert len(val_src_dst_pairs[0]) == len(val_labels)
        assert len(test_src_dst_pairs[0]) == len(test_labels)
        val_pred, test_pred = th.split(pred, [len(val_src_dst_pairs[0]),
                                              len(test_src_dst_pairs[0])])

        val_score, test_score = self.evaluator.evaluate(val_pred, test_pred,
                                                        val_labels, test_labels, total_steps)
        sys_tracker.check('evaluate')

        if rank == 0:
            self.log_print_metrics(val_score=val_score,
                                test_score=test_score,
                                dur_eval=time.time() - test_start,
                                total_steps=total_steps)
        return val_score
