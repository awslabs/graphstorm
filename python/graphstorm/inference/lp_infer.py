""" Infer wrapper for link predicion
"""
from ..model import GSgnnLinkPredictionModel
from ..model import GSgnnMrrLPEvaluator
from ..model.dataloading import GSgnnLinkPredictionInferData
from .graphstorm_infer import GSInfer
from ..tracker import get_task_tracker_class

def get_model_class(config): # pylint: disable=unused-argument
    """ Get model class
    """
    return GSgnnLinkPredictionModel, GSgnnMrrLPEvaluator

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
    bert_model: dict
        A dict of BERT models in the format of node-type -> BERT model
    """
    def __init__(self, config, bert_model):
        super(GSgnnLinkPredictionInfer, self).__init__()
        assert isinstance(bert_model, dict)
        self.bert_model = bert_model
        self.config = config

        self.eval_etypes = None if config.eval_etype is None else \
            [tuple(eval_etype.split(',')) for eval_etype in config.eval_etype]

        # neighbor sample related
        # TODO(xiangsx): Make the following code more flexible to new encoder types.
        #                Turn ["rgat", "rgcn"] to a CONSTANT.
        self.eval_fanout = config.eval_fanout \
            if config.model_encoder_type in ["rgat", "rgcn"] else [0]
        self.n_layers = config.n_layers \
            if config.model_encoder_type in ["rgat", "rgcn"] else 1

        self.eval_batch_size = config.eval_batch_size
        self.device = f'cuda:{int(config.local_rank)}'
        self.init_dist_context(config.ip_config,
                               config.graph_name,
                               config.part_config,
                               config.backend)
        self.evaluator = None

    def infer(self):
        """ Do inference
        """
        g = self._g
        part_book = g.get_partition_book()
        config = self.config

        infer_data = GSgnnLinkPredictionInferData(g, part_book, self.eval_etypes)

        model_class, eval_class = get_model_class(config)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, infer_data)
            eval_metrics = self.evaluator.metric
        else:
            eval_metrics = [] # empty list, no evaluator no evaluation metrics
        tracker_class = get_task_tracker_class(config.task_tracker)
        task_tracker = tracker_class(config, g.rank(), eval_metrics)

        lp_model = model_class(g, config, self.bert_model,
            task_tracker, train_task=False)
        lp_model.init_gsgnn_model(train=False)

        lp_model.register_evaluator(self.evaluator)
        if lp_model.task_tracker is not None:
            self.evaluator.setup_task_tracker(lp_model.task_tracker)

        lp_model.infer()

    @property
    def g(self):
        """ Get the graph
        """
        return self._g
