""" Infer wrapper for link predicion
"""
import time
import torch as th

from ..eval import GSgnnMrrLPEvaluator
from ..dataloading import GSgnnLinkPredictionInferData
from .graphstorm_infer import GSInfer
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.utils import save_relation_embeddings
from ..model.edge_decoder import LinkPredictDistMultDecoder
from ..model import create_lp_gnn_model

def get_eval_class(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    return GSgnnMrrLPEvaluator

class GSgnnLinkPredictionInfer(GSInfer):
    """ Link prediction infer.

    This is a highlevel infer wrapper that can be used directly
    to do link prediction model inference.

    The infer can do two things:
    1. (Optional) Evaluate the model performance on a test set if given
    2. Generate node embeddings

    Usage:
    ```
    from graphstorm.config import GSConfig
    from graphstorm.model.huggingface import HuggingfaceBertLoader
    from graphstorm.model import GSgnnLinkPredictionInfer

    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    infer = GSgnnLinkPredictionInfer(config, lm_models)
    infer.infer()
    ```

    Parameters
    ----------
    config: GSConfig
        Task configuration
    """

    def infer(self):
        """ Do inference
        """
        g = self._g
        part_book = g.get_partition_book()
        config = self.config
        device = self.device
        feat_name = self.config.feat_name
        eval_fanout = self.config.eval_fanout
        eval_batch_size = self.config.eval_batch_size
        mini_batch_infer = self.config.mini_batch_infer
        eval_etype = config.eval_etype
        save_embeds_path = self.config.save_embeds_path
        restore_model_path = self.config.restore_model_path

        infer_data = GSgnnLinkPredictionInferData(g, part_book, eval_etype)
        eval_class = get_eval_class(config)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, infer_data)
            self.evaluator.setup_task_tracker(self.task_tracker)

        lp_model = create_lp_gnn_model(g, config, train_task=False)
        lp_model.restore_model(restore_model_path)
        lp_model = lp_model.to(device)

        print("start inference ...")
        test_start = time.time()
        embeddings = lp_model.compute_embeddings(g, feat_name, None,
                                              eval_fanout, eval_batch_size,
                                              mini_batch_infer, self.task_tracker)
        if self.evaluator is not None and self.evaluator.do_eval(0, epoch_end=True):
            val_mrr, test_mrr = self.evaluator.evaluate(embeddings, lp_model.decoder, 0, device)
            if g.rank() == 0:
                self.log_print_metrics(val_score=val_mrr,
                                       test_score=test_mrr,
                                       dur_eval=time.time() - test_start,
                                       total_steps=0)

        assert save_embeds_path is not None
        # save node embedding
        save_gsgnn_embeddings(save_embeds_path, embeddings,
                              g.rank(), th.distributed.get_world_size())
        th.distributed.barrier()
        # save relation embedding if any
        if g.rank() == 0:
            decoder = lp_model.decoder
            if isinstance(decoder, LinkPredictDistMultDecoder):
                save_relation_embeddings(save_embeds_path, decoder)
