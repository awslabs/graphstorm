""" Infer wrapper for link predicion
"""
from ..model import M5GNNLinkPredictionModel
from ..model import M5gnnMrrLPEvaluator
from ..model.dataloading import M5gnnLinkPredictionInferData
from .graphstorm_infer import GSInfer

def get_model_class(config): # pylint: disable=unused-argument
    """ Get model class
    """
    return M5GNNLinkPredictionModel, M5gnnMrrLPEvaluator

class M5gnnLinkPredictionInfer(GSInfer):
    """ Link prediction infer.

    This is a highlevel infer wrapper that can be used directly
    to do link prediction model inference.

    The infer can do two things:
    1. (Optional) Evaluate the model performance on a test set if given
    2. Generate node embeddings

    Usage:
    ```
    from m5gnn.config import M5GNNConfig
    from m5gnn.model import M5BertLoader
    from m5gnn.model import M5gnnLinkPredictionInfer

    config = M5GNNConfig(args)
    bert_config = config.bert_config
    m5_models = M5BertLoader(bert_config).load()

    infer = M5gnnLinkPredictionInfer(config, m5_models)
    infer.infer()
    ```

    Parameters
    ----------
    config: M5GNNConfig
        Task configuration
    bert_model: dict
        A dict of BERT models in the format of node-type -> M5 BERT model
    """
    def __init__(self, config, bert_model):
        super(M5gnnLinkPredictionInfer, self).__init__()
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

        infer_data = M5gnnLinkPredictionInferData(g, part_book, self.eval_etypes)

        model_class, eval_class = get_model_class(config)
        lp_model = model_class(g, config, self.bert_model, train_task=False)
        lp_model.init_m5gnn_model(train=False)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, infer_data)

        lp_model.register_evaluator(self.evaluator)
        if lp_model.tracker is not None:
            self.evaluator.setup_tracker(lp_model.tracker)

        lp_model.infer()

    @property
    def g(self):
        """ Get the graph
        """
        return self._g
