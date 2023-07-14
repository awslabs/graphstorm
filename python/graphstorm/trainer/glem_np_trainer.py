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

    GraphStorm GLEM trainer for node prediction.
"""

import time
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..model.node_gnn import GSgnnNodeModelInterface
from ..model.node_glem import GLEM
from ..model.gnn import GSgnnModel
from .np_trainer import GSgnnNodePredictionTrainer

from ..utils import sys_tracker
from ..utils import rt_profiler

class GLEMNodePredictionTrainer(GSgnnNodePredictionTrainer):
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
        super(GLEMNodePredictionTrainer, self).__init__(model, rank, topk_model_to_save)
        assert isinstance(model, GSgnnNodeModelInterface) and isinstance(model, GLEM), \
                "The input model is not a GLEM node model. Please implement GLEM."
        self.early_stop = False # used when early stop is True

    def fit(self, train_loader, num_epochs,
            val_loader=None,
            test_loader=None,
            use_mini_batch_infer=True,
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            freeze_input_layer_epochs=0
            ):
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

        # computation graph will be changed during training.
        model = DistributedDataParallel(self._model, device_ids=[self.dev_id],
                                        output_device=self.dev_id,
                                        find_unused_parameters=True,
                                        static_graph=False)
        device = model.device
        data = train_loader.data

        # turn off pl loss in the epochs when LM is frozen
        no_pl = freeze_input_layer_epochs > 0
        if freeze_input_layer_epochs > 0:
            self._model.lm.freeze_input_encoder(data)

        # training loop
        dur = []
        total_steps = 0

        sys_tracker.check('start training')
        g = data.g # dgl.DistGraph
        for epoch in range(num_epochs):
            t0 = time.time()
            rt_profiler.start_record()

            if freeze_input_layer_epochs <= epoch:
                self._model.lm.unfreeze_input_encoder()
                no_pl = False

            use_gnn = not self._model.em_order_gnn_first
            # `use_gnn`` determines which part to train, if `em_order_gnn_first`
            # 1st round: train GNN, fix LM; 2nd round: train LM fix gnn
            for _ in range(2):
                stage_start_time = time.time()
                part_to_train = 'gnn' if use_gnn else 'lm'
                self._model.toggle(part_to_train)
                self._fit_one_epoch(use_gnn, model, g, data, train_loader, val_loader, test_loader,
                                    device, rt_profiler,
                                    epoch, total_steps, use_mini_batch_infer,
                                    save_model_path, save_model_frequency, no_pl)
                stage_finish_time = time.time()
                if self.rank == 0:
                    print("Epoch {}, {} takes {:.2f} seconds".format(
                        epoch, part_to_train, stage_finish_time-stage_start_time))
                use_gnn = not use_gnn

            # early_stop, exit training
            if self.early_stop is True:
                break

            epoch_time = time.time() - t0
            dur.append(epoch_time)

        rt_profiler.save_profile()
        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if self.rank == 0 and self.evaluator is not None:
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score': self.evaluator.best_val_score,
                       'peak_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024}
            self.log_params(output)

            if save_perf_results_path is not None:
                self.save_model_results_to_file(self.evaluator.best_test_score,
                                                save_perf_results_path)

    def _fit_one_epoch(self, use_gnn, model, g, data, train_loader,
                       val_loader, test_loader, device, profiler,
                       epoch, total_steps,
                       use_mini_batch_infer=True,
                       save_model_path=None,
                       save_model_frequency=-1, no_pl=False):
        """Fit model for one epoch
        """
        profiler.start_record()
        for i, (input_nodes, seeds, blocks) in enumerate(train_loader):
            profiler.record('train_sample')
            total_steps += 1
            batch_tic = time.time()

            if not isinstance(input_nodes, dict):
                assert len(g.ntypes) == 1
                input_nodes = {g.ntypes[0]: input_nodes}
            input_feats = data.get_node_feats(input_nodes, device)
            lbl = data.get_labels(seeds, device)
            profiler.record('train_node_feats')

            blocks = [block.to(device) for block in blocks]
            for _, feats in input_feats.items():
                num_input_nodes += feats.shape[0]
            profiler.record('train_graph2GPU')
            # Run forward function to compute loss:
            loss = model(blocks, input_feats, None, lbl, input_nodes, use_gnn=use_gnn, no_pl=no_pl)
            profiler.record('train_forward')

            self.optimizer.zero_grad()
            loss.backward()
            profiler.record('train_backward')
            self.optimizer.step()
            profiler.record('train_step')

            self.log_metric("Train loss", loss.item(), total_steps)

            if i % 20 == 0 and self.rank == 0:
                rt_profiler.print_stats()
                print("Part {} | Epoch {:05d} | Batch {:03d} | Loss: {:.4f} | Time: {:.4f}".
                        format(self.rank, epoch, i,  loss.item(), time.time() - batch_tic))

            val_score = None
            if self.evaluator is not None and \
                self.evaluator.do_eval(total_steps, epoch_end=False) and \
                val_loader is not None:
                val_score = self.eval(model.module, val_loader, test_loader,
                                        use_mini_batch_infer, total_steps, return_proba=False)

                if self.evaluator.do_early_stop(val_score):
                    self.early_stop = True

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
            if self.early_stop is True:
                break

        # end of an epoch
        th.distributed.barrier()

        val_score = None
        if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
            val_score = self.eval(model.module, val_loader, test_loader,
                                    use_mini_batch_infer, total_steps, return_proba=False)
            if self.evaluator.do_early_stop(val_score):
                self.early_stop = True
        # After each epoch, check to save the top k models. If has validation score, will save
        # the best top k. But if no validation, will either save the last k model or all models
        # depends on the setting of top k. To show this is after epoch save, set the iteration
        # to be None, so that we can have a determistic model folder name for testing and debug.
        self.save_topk_models(model, epoch, None, val_score, save_model_path)
