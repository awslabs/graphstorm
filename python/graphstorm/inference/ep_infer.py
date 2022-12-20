""" Infer wrapper for edge classification and regression
"""
import time
import torch as th

from ..model import create_edge_gnn_model
from ..eval import GSgnnAccEvaluator
from ..eval import GSgnnRegressionEvaluator
from .graphstorm_infer import GSInfer
from ..dataloading import GSgnnEdgePredictionInferData
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..utils import sys_tracker

def get_eval_class(config):
    """ Get evaluation class
    """
    if config.task_type == "edge_regression":
        return GSgnnRegressionEvaluator
    elif config.task_type == 'edge_classification':
        return GSgnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

class GSgnnEdgePredictInfer(GSInfer):
    """ Edge classification/regression infer.

    This is a highlevel infer wrapper that can be used directly
    to do edge classification/regression model inference.

    The infer can do three things:
    1. (Optional) Evaluate the model performance on a test set if given
    2. Generate node embeddings

    Usage:
    ```
    from graphstorm.config import GSConfig
    from graphstorm.model.huggingface import HuggingfaceBertLoader
    from graphstorm.model import GSgnnEdgePredictInfer

    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    infer = GSgnnEdgePredictInfer(config, lm_models)
    infer.infer()
    ```

    Parameters
    ----------
    config: GSConfig
        Task configuration
    """
    def __init__(self, config):
        super(GSgnnEdgePredictInfer, self).__init__(config)
        self.config = config
        self.infer_etype = config.target_etype

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
        restore_model_path = self.config.restore_model_path

        infer_data = GSgnnEdgePredictionInferData(g,
            part_book, self.infer_etype, config.label_field)
        sys_tracker.check('create infer data')
        eval_class = get_eval_class(config)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, infer_data)
            self.evaluator.setup_task_tracker(self.task_tracker)

        ep_model = create_edge_gnn_model(g, config, train_task=False)
        ep_model.restore_model(restore_model_path)
        ep_model = ep_model.to(device)

        sys_tracker.check('start inference')
        if self.evaluator is not None and \
            self.evaluator.do_eval(0, epoch_end=True):
            test_start = time.time()

            # Do evaluation
            target_etypes = infer_data.target_etypes
            assert len(target_etypes) == 1, \
                "Only can do edge classification for one edge type"
            target_etype = target_etypes[0][1]
            test_src_dst_pairs = infer_data.test_src_dst_pairs
            test_labels = infer_data.labels[infer_data.test_idxs[target_etype]]

            # TODO: Make it more efficient
            # We do not need to compute the embedding of all node types
            test_preds, embeddings = ep_model.predict(g, feat_name, test_src_dst_pairs,
                                                      eval_fanout, eval_batch_size,
                                                      mini_batch_infer, self.task_tracker)
            sys_tracker.check('predict')

            val_score, test_score = self.evaluator.evaluate(
                test_preds, test_preds,
                test_labels, test_labels,
                0)

            if g.rank() == 0:
                self.log_print_metrics(val_score=val_score,
                                       test_score=test_score,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)
        else:
            # compute node embeddings
            embeddings = ep_model.compute_embeddings(g, feat_name, None,
                                                     eval_fanout, eval_batch_size,
                                                     mini_batch_infer, self.task_tracker)
            sys_tracker.check('compute GNN embeddings')

        target_ntypes = infer_data.target_ntypes
        embeddings = {ntype: embeddings[ntype] for ntype in target_ntypes}

        # If save_embeds_path is set to None.
        # A user does not want to save the node embedding
        if save_embeds_path is not None:
            # Save node embedding
            save_gsgnn_embeddings(save_embeds_path,
                embeddings, g.rank(), th.distributed.get_world_size())
            sys_tracker.check('save GNN embeddings')
            th.distributed.barrier()
