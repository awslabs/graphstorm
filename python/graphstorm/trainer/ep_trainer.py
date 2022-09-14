from ..model import M5gnnEdgePredictionTrainData
from ..model import M5gnnEdgePredictionDataLoader
from ..model import M5GNNEdgeClassificationModel
from ..model import M5gnnAccEvaluator
from ..model import M5GNNEdgeRegressModel
from ..model import M5gnnRegressionEvaluator
from .m5gnn_trainer import M5gnnTrainer

def get_model_class(config):
    if config.task_type == "edge_regression":
        return M5GNNEdgeRegressModel, M5gnnRegressionEvaluator
    elif config.task_type == 'edge_classification':
        return M5GNNEdgeClassificationModel, M5gnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

class M5gnnEdgePredictionTrainer(M5gnnTrainer):
    """ Edge prediction trainer.

    This is a highlevel trainer wrapper that can be used directly to train a edge prediction model.

    Usage:
    ```
    from graphstorm.config import M5GNNConfig
    from graphstorm.model import M5BertLoader
    from graphstorm.model import M5gnnEdgePredictionTrainer

    config = M5GNNConfig(args)
    bert_config = config.bert_config
    m5_models = M5BertLoader(bert_config).load()

    trainer = M5gnnEdgePredictionTrainer(config, m5_models)
    trainer.fit()
    ```

    Parameters
    ----------
    config: M5GNNConfig
        Task configuration
    bert_model: dict
        A dict of BERT models in the format of node-type -> M5 BERT model
    """
    def __init__(self, config, bert_model):
        super(M5gnnEdgePredictionTrainer, self).__init__()
        assert isinstance(bert_model, dict)
        self.bert_model = bert_model
        self.config = config

        self.target_etypes = [tuple(target_etype.split(',')) for target_etype in config.target_etype]

        # neighbor sample related
        self.fanout = config.fanout if config.model_encoder_type in ["rgat", "rgcn"] else [0]
        self.n_layers = config.n_layers if config.model_encoder_type in ["rgat", "rgcn"] else 1
        self.batch_size = config.batch_size
        # sampling related
        self.reverse_edge_types_map = config.reverse_edge_types_map
        self.remove_target_edge = config.remove_target_edge if config.model_encoder_type in ["rgat", "rgcn"] else False
        self.exclude_training_targets = config.exclude_training_targets

        self.evaluator = None
        self.device = 'cuda:%d' % config.local_rank
        self.init_dist_context(config.ip_config,
                               config.graph_name,
                               config.part_config,
                               config.backend)

    def save(self):
        pass

    def load(self):
        pass

    def register_evaluator(self, evaluator):
        self.evaluator = evaluator

    def fit(self):
        g = self._g
        pb = g.get_partition_book()
        config = self.config

        train_data = M5gnnEdgePredictionTrainData(g, pb, self.target_etypes, config.label_field)

        dataloader = M5gnnEdgePredictionDataLoader(g,
                                                   train_data,
                                                   self.fanout,
                                                   self.n_layers,
                                                   self.batch_size,
                                                   self.reverse_edge_types_map,
                                                   self.remove_target_edge,
                                                   self.exclude_training_targets,
                                                   self.device)

        model_class, eval_class = get_model_class(config)
        ep_model = model_class(g, config, self.bert_model)
        ep_model.init_m5gnn_model()

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, train_data)

        ep_model.register_evaluator(self.evaluator)
        if ep_model.tracker is not None:
            self.evaluator.setup_tracker(ep_model.tracker)

        ep_model.fit(dataloader, train_data)

    @property
    def g(self):
        return self._g
