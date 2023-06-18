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

    GraphStorm trainer for node prediction.
"""
import time
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..model.node_gnn import node_mini_batch_gnn_predict, node_mini_batch_predict
from ..model.node_gnn import GSgnnNodeModelInterface
from ..model.gnn import do_full_graph_inference, GSgnnModelBase, GSgnnModel
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker
from ..utils import rt_profiler

class GSgnnNodePredictionTrainer(GSgnnTrainer):
    """ A trainer for node prediction

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model for node prediction.
    rank : int
        The rank.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, rank, topk_model_to_save=1):
        super(GSgnnNodePredictionTrainer, self).__init__(model, rank, topk_model_to_save)
        assert isinstance(model, GSgnnNodeModelInterface) and isinstance(model, GSgnnModelBase), \
                "The input model is not a node model. Please implement GSgnnNodeModelBase."

    def fit(self, train_loader, num_epochs,
            val_loader=None,
            test_loader=None,
            use_mini_batch_infer=True,
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            freeze_input_layer_epochs=0):
        """ The fit function for node prediction.

        Parameters
        ----------
        train_loader : GSgnnNodeDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnNodeDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnNodeDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
        freeze_input_layer_epochs: int
            Freeze the input layer for N epochs. This is commonly used when
            the input layer contains language models.
            Default: 0, no freeze.
        """
        # Check the correctness of configurations.
        if self.evaluator is not None:
            assert val_loader is not None, \
                    "The evaluator is provided but validation set is not provided."
        if not use_mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."

        # with freeze_input_layer_epochs is 0, computation graph will not be changed.
        static_graph = freeze_input_layer_epochs == 0
        model = DistributedDataParallel(self._model, device_ids=[self.dev_id],
                                        output_device=self.dev_id,
                                        find_unused_parameters=True,
                                        static_graph=static_graph)
        device = model.device
        data = train_loader.data

        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        if freeze_input_layer_epochs > 0:
            self._model.freeze_input_encoder(data)
        # TODO(xiangsx) Support freezing gnn encoder and decoder

        # training loop
        dur = []
        total_steps = 0
        num_input_nodes = 0
        forward_time = 0
        back_time = 0
        early_stop = False # used when early stop is True
        sys_tracker.check('start training')
        g = data.g
        for epoch in range(num_epochs):
            model.train()
            t0 = time.time()
            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            # TODO(zhengda) the dataloader should return node features and labels directly.
            rt_profiler.start_record()
            for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
                rt_profiler.record('train_sample')
                total_steps += 1
                batch_tic = time.time()

                if not isinstance(input_nodes, dict):
                    assert len(g.ntypes) == 1
                    input_nodes = {g.ntypes[0]: input_nodes}
                input_feats = data.get_node_feats(input_nodes, device)
                lbl = data.get_labels(seeds, device)
                rt_profiler.record('train_node_feats')

                blocks = [block.to(device) for block in blocks]
                for _, feats in input_feats.items():
                    num_input_nodes += feats.shape[0]
                rt_profiler.record('train_graph2GPU')

                t2 = time.time()
                # TODO(zhengda) we don't support edge features for now.
                loss = model(blocks, input_feats, None, lbl, input_nodes)
                rt_profiler.record('train_forward')

                t3 = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                rt_profiler.record('train_backward')
                self.optimizer.step()
                rt_profiler.record('train_step')
                forward_time += (t3 - t2)
                back_time += (time.time() - t3)

                self.log_metric("Train loss", loss.item(), total_steps)

                if i % 20 == 0 and self.rank == 0:
                    rt_profiler.print_stats()
                    print("Part {} | Epoch {:05d} | Batch {:03d} | Loss: {:.4f} | Time: {:.4f}".
                            format(self.rank, epoch, i,  loss.item(), time.time() - batch_tic))
                    num_input_nodes = forward_time = back_time = 0

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False) and \
                    val_loader is not None:
                    val_score = self.eval(model.module, val_loader, test_loader,
                                          use_mini_batch_infer, total_steps, return_proba=False)

                    if self.evaluator.do_early_stop(val_score):
                        early_stop = True

                # Every n iterations, check to save the top k models. If has validation score,
                # will save the best top k. But if no validation, will either save
                # the last k model or all models depends on the setting of top k
                if save_model_frequency > 0 and \
                    total_steps % save_model_frequency == 0 and \
                    total_steps != 0:
                    if self.evaluator is None or val_score is not None:
                        # We will save the best model when
                        # 1. There is no evaluation, we will keep the
                        #    latest K models.
                        # 2. There is evaluaiton, we need to follow the
                        #    guidance of validation score.
                        self.save_topk_models(model, epoch, i, val_score, save_model_path)

                rt_profiler.record('train_eval')
                # early_stop, exit current interation.
                if early_stop is True:
                    break

            # end of an epoch
            th.distributed.barrier()
            epoch_time = time.time() - t0
            if self.rank == 0:
                print("Epoch {} take {}".format(epoch, epoch_time))
            dur.append(epoch_time)

            val_score = None
            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(model.module, val_loader, test_loader,
                                      use_mini_batch_infer, total_steps, return_proba=False)
                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(model, epoch, None, val_score, save_model_path)

            # early_stop, exit training
            if early_stop is True:
                break

        rt_profiler.save_profile()
        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if self.rank == 0 and self.evaluator is not None:
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score': self.evaluator.best_val_score,
                       'peak_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024,
                       'best validation iteration': \
                           self.evaluator.best_iter_num[self.evaluator.metric[0]],
                       'best model path': \
                           self.get_best_model_path() if save_model_path is not None else None}
            self.log_params(output)

            if save_perf_results_path is not None:
                self.save_model_results_to_file(self.evaluator.best_test_score,
                                                save_perf_results_path)

    def eval(self, model, val_loader, test_loader, use_mini_batch_infer, total_steps,
             return_proba=True):
        """ do the model evaluation using validiation and test sets

        Parameters
        ----------
        model : Pytorch model
            The GNN model.
        val_loader: GSNodeDataLoader
            The dataloader for validation data
        test_loader : GSNodeDataLoader
            The dataloader for test data.
        total_steps: int
            Total number of iterations.
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.

        Returns
        -------
        float: validation score
        """
        teval = time.time()
        sys_tracker.check('before prediction')
        if use_mini_batch_infer:
            val_pred, _, val_label = node_mini_batch_gnn_predict(model, val_loader, return_proba,
                                                                 return_label=True)
            sys_tracker.check('after_val_score')
            test_pred, _, test_label = node_mini_batch_gnn_predict(model, test_loader, return_proba,
                                                                   return_label=True)
            sys_tracker.check('after_test_score')
        else:
            emb = do_full_graph_inference(model, val_loader.data, fanout=val_loader.fanout,
                                          task_tracker=self.task_tracker)
            sys_tracker.check('after_full_infer')
            val_pred, val_label = node_mini_batch_predict(model, emb, val_loader, return_proba,
                                                          return_label=True)
            sys_tracker.check('after_val_score')
            test_pred, test_label = node_mini_batch_predict(model, emb, test_loader, return_proba,
                                                            return_label=True)
            sys_tracker.check('after_test_score')
        sys_tracker.check('predict')
        val_score, test_score = self.evaluator.evaluate(val_pred, test_pred,
                                                        val_label, test_label, total_steps)
        sys_tracker.check('evaluate')
        if self.rank == 0:
            self.log_print_metrics(val_score=val_score,
                                    test_score=test_score,
                                    dur_eval=time.time() - teval,
                                    total_steps=total_steps)
        return val_score
