""" GraphStorm trainer for link prediction """
import time
import torch as th
from torch.nn.parallel import DistributedDataParallel

from ..model.lp_gnn import GSgnnLinkPredictionModelInterface
from ..model.gnn import do_full_graph_inference, GSgnnModelBase, GSgnnModel
from .gsgnn_trainer import GSgnnTrainer

from ..utils import sys_tracker

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

    def fit(self, train_loader, n_epochs,
            val_loader=None,            # pylint: disable=unused-argument
            test_loader=None,           # pylint: disable=unused-argument
            mini_batch_infer=True,      # pylint: disable=unused-argument
            save_model_path=None,
            save_model_per_iters=None,
            save_perf_results_path=None):
        """ The fit function for link prediction.

        Parameters
        ----------
        train_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for training.
        n_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnLinkPredictionDataLoader
            The mini-batch sampler for computing test scores.
        mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_per_iters : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
        """
        if not mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."
        model = DistributedDataParallel(self._model, device_ids=[self.dev_id],
                                        output_device=self.dev_id)
        device = model.device

        # training loop
        dur = []
        best_epoch = 0
        num_input_nodes = 0
        total_steps = 0
        early_stop = False # used when early stop is True
        forward_time = 0
        back_time = 0
        data = train_loader.data
        sys_tracker.check('start training')
        for epoch in range(n_epochs):
            model.train()
            t0 = time.time()
            for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_loader):
                total_steps += 1
                batch_tic = time.time()

                if not isinstance(input_nodes, dict):
                    assert len(pos_graph.ntypes) == 1
                    input_nodes = {pos_graph.ntypes[0]: input_nodes}
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [blk.to(device) for blk in blocks]
                input_feats = data.get_node_feats(input_nodes, device)
                for _, nodes in input_nodes.items():
                    num_input_nodes += nodes.shape[0]

                t2 = time.time()
                # TODO(zhengda) we don't support edge features for now.
                loss = model(blocks, pos_graph, neg_graph, input_feats, None)

                t3 = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                forward_time += (t3 - t2)
                back_time += (time.time() - t3)

                self.log_metric("Train loss", loss.item(), total_steps)

                if i % 20 == 0 and self.rank == 0:
                    print("Epoch {:05d} | Batch {:03d} | Train Loss: {:.4f} | Time: {:.4f}".
                            format(epoch, i, loss.item(), time.time() - batch_tic))
                    num_input_nodes = forward_time = back_time = 0

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):
                    val_score = self.eval(model.module, data, total_steps)

                    if self.evaluator.do_early_stop(val_score):
                        early_stop = True

                # Every n iterations, check to save the top k models. If has validation score,
                # will save the best top k. But if no validation, will either save
                # the last k model or all models depends on the setting of top k
                if save_model_per_iters > 0 and i % save_model_per_iters == 0 and i != 0:
                    self.save_topk_models(model, epoch, i, val_score, save_model_path)

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
                val_score = self.eval(model.module, data, total_steps)

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

        print("Peak Mem alloc: {:.4f} MB".format(th.cuda.max_memory_allocated(device) / 1024 /1024))
        if self.rank == 0 and self.evaluator is not None:
            output = dict(best_test_mrr=self.evaluator.best_test_score,
                          best_val_mrr=self.evaluator.best_val_score,
                          peak_mem_alloc_MB=th.cuda.max_memory_allocated(device) / 1024 / 1024,
                          best_epoch=best_epoch)
            self.log_params(output)

            if save_perf_results_path is not None:
                self.save_model_results_to_file(self.evaluator.best_test_score,
                                                save_perf_results_path)

    def eval(self, model, data, total_steps):
        """ do the model evaluation using validiation and test sets

        Parameters
        ----------
        model : Pytorch model
            The GNN model.
        data : GSgnnEdgeTrainData
            The training dataset
        total_steps: int
            Total number of iterations.

        Returns
        -------
        float: validation score
        """
        test_start = time.time()
        sys_tracker.check('before prediction')
        emb = do_full_graph_inference(model, data, task_tracker=self.task_tracker)
        sys_tracker.check('compute embeddings')
        decoder = model.decoder
        val_score, test_score = self.evaluator.evaluate(emb, decoder, total_steps, model.device)
        sys_tracker.check('evaluate validation/test')

        if self.rank == 0:
            self.log_print_metrics(val_score=val_score,
                                   test_score=test_score,
                                   dur_eval=time.time() - test_start,
                                   total_steps=total_steps)
        return val_score
