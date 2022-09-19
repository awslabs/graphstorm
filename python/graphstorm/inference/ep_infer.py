""" Infer wrapper for edge classification and regression
"""
from ..model import GSgnnEdgeClassificationModel
from ..model import GSgnnAccEvaluator
from ..model import GSgnnEdgeRegressModel
from ..model import GSgnnRegressionEvaluator
from .graphstorm_infer import GSInfer
from ..model.dataloading import GSgnnEdgePredictionInferData

def get_model_class(config):
    """ Get model class
    """
    if config.task_type == "edge_regression":
        return GSgnnEdgeRegressModel, GSgnnRegressionEvaluator
    elif config.task_type == 'edge_classification':
        return GSgnnEdgeClassificationModel, GSgnnAccEvaluator
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
    bert_model: dict
        A dict of BERT models in the format of node-type -> BERT model
    """
    def __init__(self, config, bert_model):
        super(GSgnnEdgePredictInfer, self).__init__()
        assert isinstance(bert_model, dict)
        self.bert_model = bert_model
        self.config = config

        self.infer_etype = [tuple(target_etype.split(',')) for target_etype in config.target_etype]

        # neighbor sample related
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

        infer_data = GSgnnEdgePredictionInferData(g,
            part_book, self.infer_etype, config.label_field)
        model_class, eval_class = get_model_class(config)
        ep_model = model_class(g, config, self.bert_model, train_task=False)
        ep_model.init_gsgnn_model(train=False)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, infer_data)

        ep_model.register_evaluator(self.evaluator)
        if ep_model.tracker is not None:
            self.evaluator.setup_tracker(ep_model.tracker)
        ep_model.infer(infer_data)

    @property
    def g(self):
        """ Get the graph
        """
        return self._g
