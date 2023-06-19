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

    GraphStorm trainer for link prediction
"""
import time
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..model.lp_gnn import GSgnnLinkPredictionModelInterface
from ..model.lp_gnn import lp_mini_batch_predict
from ..model.gnn import do_full_graph_inference, GSgnnModelBase, GSgnnModel
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker
from ..utils import rt_profiler

class GSgnnLinkPredictionTrainer(GSgnnTrainer):
    """ Link prediction trainer.

    This is a highlevel trainer wrapper that can be used directly to train a link prediction model.

    Parameters
    ----------
    model : GSgnnLinkPredictionModelBase
        The GNN model for link prediction.
    rank : int
        The rank.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, rank, topk_model_to_save):
        super(GSgnnLinkPredictionTrainer, self).__init__(model, rank, topk_model_to_save)
        assert isinstance(model, GSgnnLinkPredictionModelInterface) \
                and isinstance(model, GSgnnModelBase), \
                "The input model is not an edge model. Please implement GSgnnEdgeModelBase."

    def fit(self, train_loader, num_epochs,
            val_loader=None,            # pylint: disable=unused-argument
            test_loader=None,           # pylint: disable=unused-argument
            use_mini_batch_infer=True,      # pylint: disable=unused-argument
            save_model_path=None,
            save_model_frequency=None,
            save_perf_results_path=None,
            edge_mask_for_gnn_embeddings='train_mask',
            freeze_input_layer_epochs=0):
        """ The fit function for link prediction.

        Parameters
        ----------
        train_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
        edge_mask_for_gnn_embeddings : str
            The mask that indicates the edges used for computing GNN embeddings for model
            evaluation. By default, we use the edges in the training graph to compute
            GNN embeddings for evaluation.
        freeze_input_layer_epochs: int
            Freeze input layer model for N epochs. This is commonly used when
            the input layer contains language models.
            Default: 0, no freeze.
        """
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
        num_input_nodes = 0
        total_steps = 0
        early_stop = False # used when early stop is True
        forward_time = 0
        back_time = 0
        sys_tracker.check('start training')
        for epoch in range(num_epochs):
            model.train()
            t0 = time.time()

            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            rt_profiler.start_record()
            for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_loader):
                rt_profiler.record('train_sample')
                total_steps += 1
                batch_tic = time.time()

                if not isinstance(input_nodes, dict):
                    assert len(pos_graph.ntypes) == 1
                    input_nodes = {pos_graph.ntypes[0]: input_nodes}
                input_feats = data.get_node_feats(input_nodes, device)
                rt_profiler.record('train_node_feats')

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [blk.to(device) for blk in blocks]
                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]
                rt_profiler.record('train_graph2GPU')

                t2 = time.time()
                # TODO(zhengda) we don't support edge features for now.
                loss = model(blocks, pos_graph, neg_graph,
                             input_feats, None, input_nodes)
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
                    print("Epoch {:05d} | Batch {:03d} | Train Loss: {:.4f} | Time: {:.4f}".
                            format(epoch, i, loss.item(), time.time() - batch_tic))
                    num_input_nodes = forward_time = back_time = 0

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):
                    val_score = self.eval(model.module, data,
                                          val_loader, test_loader, total_steps,
                                          edge_mask_for_gnn_embeddings)
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

            # ------- end of an epoch -------

            th.distributed.barrier()
            epoch_time = time.time() - t0
            if self.rank == 0:
                print("Epoch {} take {}".format(epoch, epoch_time))
            dur.append(epoch_time)

            val_score = None
            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(model.module, data,
                                      val_loader, test_loader, total_steps,
                                      edge_mask_for_gnn_embeddings)

                if self.evaluator.do_early_stop(val_score):
                    early_stop = True

            # After each epoch, check to save the top k models. If has validation score, will save
            # the best top k. But if no validation, will either save the last k model or all models
            # depends on the setting of top k. To show this is after epoch save, set the iteration
            # to be None, so that we can have a determistic model folder name for testing and debug.
            self.save_topk_models(model, epoch, None, val_score, save_model_path)

            th.distributed.barrier()

            # early_stop, exit training
            if early_stop is True:
                break

        rt_profiler.save_profile()
        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if self.rank == 0 and self.evaluator is not None:
            output = {'best_test_mrr': self.evaluator.best_test_score,
                       'best_val_mrr':self.evaluator.best_val_score,
                       'peak_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024,
                       'best validation iteration': \
                           self.evaluator.best_iter_num[self.evaluator.metric[0]],
                       'best model path': \
                           self.get_best_model_path() if save_model_path is not None else None}
            self.log_params(output)

            if save_perf_results_path is not None:
                self.save_model_results_to_file(self.evaluator.best_test_score,
                                                save_perf_results_path)

    def eval(self, model, data, val_loader, test_loader, total_steps,
             edge_mask_for_gnn_embeddings):
        """ do the model evaluation using validiation and test sets

        Parameters
        ----------
        model : Pytorch model
            The GNN model.
        data : GSgnnEdgeTrainData
            The training dataset
        val_loader: GSNodeDataLoader
            The dataloader for validation data
        test_loader : GSNodeDataLoader
            The dataloader for test data.
        total_steps: int
            Total number of iterations.
        edge_mask_for_gnn_embeddings : str
            The mask that indicates the edges used for computing GNN embeddings.

        Returns
        -------
        float: validation score
        """
        test_start = time.time()
        sys_tracker.check('before prediction')
        model.eval()
        emb = do_full_graph_inference(model, data, fanout=val_loader.fanout,
                                      edge_mask=edge_mask_for_gnn_embeddings,
                                      task_tracker=self.task_tracker)
        sys_tracker.check('compute embeddings')
        device = th.device(f"cuda:{self.dev_id}") \
            if self.dev_id >= 0 else th.device("cpu")
        val_scores = lp_mini_batch_predict(model, emb, val_loader, device) \
            if val_loader is not None else None
        sys_tracker.check('after_val_score')
        test_scores = lp_mini_batch_predict(model, emb, test_loader, device)
        sys_tracker.check('after_test_score')
        val_score, test_score = self.evaluator.evaluate(
            val_scores, test_scores, total_steps)
        sys_tracker.check('evaluate validation/test')
        model.train()

        if self.rank == 0:
            self.log_print_metrics(val_score=val_score,
                                   test_score=test_score,
                                   dur_eval=time.time() - test_start,
                                   total_steps=total_steps)
        return val_score
