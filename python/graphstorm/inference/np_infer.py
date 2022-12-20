""" Infer wrapper for node classification and regression
"""
import time
import os
import torch as th

from ..model import create_node_gnn_model
from ..eval import GSgnnAccEvaluator
from ..eval import GSgnnRegressionEvaluator
from .graphstorm_infer import GSInfer
from ..dataloading import GSgnnNodeInferData
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..utils import sys_tracker

def get_eval_class(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "node_regression":
        return GSgnnRegressionEvaluator
    elif config.task_type == 'node_classification':
        return GSgnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

class GSgnnNodePredictInfer(GSInfer):
    """ Node classification/regression infer.

    This is a highlevel infer wrapper that can be used directly
    to do node classification/regression model inference.

    The infer can do three things:
    1. (Optional) Evaluate the model performance on a test set if given
    2. Generate node embeddings
    3. Comput inference results for nodes with target node type.

    Usage:
    ```
    from graphstorm.config import GSConfig
    from graphstorm.model.huggingface import HuggingfaceBertLoader
    from graphstorm.model import GSgnnNodePredictInfer

    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    infer = GSgnnNodePredictInfer(config, lm_models)
    infer.infer()
    ```

    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self, config):
        super(GSgnnNodePredictInfer, self).__init__(config)
        self.predict_ntype = config.predict_ntype

    def infer(self):
        """ Do inference
        """
        sys_tracker.check('infer start')
        g = self._g
        part_book = g.get_partition_book()
        config = self.config
        device = self.device
        feat_name = self.config.feat_name
        eval_fanout = self.config.eval_fanout
        eval_batch_size = self.config.eval_batch_size
        mini_batch_infer = self.config.mini_batch_infer
        save_embeds_path = self.config.save_embeds_path
        save_predict_path = self.config.save_predict_path
        restore_model_path = self.config.restore_model_path

        infer_data = GSgnnNodeInferData(g, part_book, self.predict_ntype, config.label_field)
        sys_tracker.check('create infer data')

        eval_class = get_eval_class(config)
        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, infer_data)
            self.evaluator.setup_task_tracker(self.task_tracker)
        np_model = create_node_gnn_model(g, config, train_task=False)
        np_model.restore_model(restore_model_path)
        np_model = np_model.to(device)

        sys_tracker.check('start inference')
        # TODO: Make it more efficient
        # We do not need to compute the embedding of all node types
        pred, outputs = np_model.predict(g, feat_name, None,
                                      eval_fanout, eval_batch_size,
                                      mini_batch_infer, self.task_tracker)
        sys_tracker.check('predict')

        embeddings = outputs[self.predict_ntype]
        if save_embeds_path is not None:
            save_gsgnn_embeddings(save_embeds_path,
                embeddings, g.rank(), th.distributed.get_world_size())
            th.distributed.barrier()
            sys_tracker.check('save GNN embeddings')

        if save_predict_path is not None and g.rank() == 0:
            os.makedirs(save_predict_path, exist_ok=True)
            th.save(pred, os.path.join(save_predict_path, "predict.pt"))

        th.distributed.barrier()

        # do evaluation if any
        if self.evaluator is not None and \
            self.evaluator.do_eval(0, epoch_end=True):
            test_start = time.time()
            pred = pred[infer_data.test_idx]
            labels = infer_data.labels[infer_data.test_idx]
            pred = pred.to(device)
            labels = labels.to(device)

            val_score, test_score = self.evaluator.evaluate(
                pred, pred,
                labels, labels,
                0)
            if g.rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                        test_score=test_score,
                                        dur_eval=time.time() - test_start,
                                        total_steps=0)
