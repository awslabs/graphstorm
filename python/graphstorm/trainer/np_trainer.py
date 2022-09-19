import dgl
from importlib import import_module

import torch as th
from ..model import GSgnnNodeDataLoader
from ..model import GSgnnNodeTrainData
from ..model import GSgnnNodeClassModel
from ..model import GSgnnAccEvaluator
from ..model import GSgnnNodeRegressModel
from ..model import GSgnnRegressionEvaluator
from .gsgnn_trainer import GSgnnTrainer

def get_model_class(config):
    if config.task_type == "node_regression":
        return GSgnnNodeRegressModel, GSgnnRegressionEvaluator
    elif config.task_type == 'node_classification':
        return GSgnnNodeClassModel, GSgnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

class GSgnnNodePredictTrainer(GSgnnTrainer):
    """ A trainer for node prediction

    Parameters
    ----------
    config: GSConfig
        Task configuration
    bert_model: dict
        A dict of BERT models in the format of node-type -> BERT model
    """
    def __init__(self, config, bert_model):
        super(GSgnnNodePredictTrainer, self).__init__()
        assert isinstance(bert_model, dict)
        self.bert_model = bert_model
        self.config = config

        self.predict_ntype = config.predict_ntype

        # neighbor sample related
        self.fanout = config.fanout if config.model_encoder_type in ["rgat", "rgcn"] else [0]
        self.n_layers = config.n_layers if config.model_encoder_type in ["rgat", "rgcn"] else 1
        self.batch_size = config.batch_size
        self.evaluator = None
        self.device = 'cuda:%d' % config.local_rank

        self.init_dist_context(config.ip_config,
                               config.graph_name,
                               config.part_config,
                               config.backend)

        for ntype in self.bert_model:
            assert ntype in self._g.ntypes, \
                    'A bert model is created for node type {}, but the node type does not exist.'


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

        train_data = GSgnnNodeTrainData(g, pb, self.predict_ntype, config.label_field)
        dataloader = GSgnnNodeDataLoader(g,
                                         train_data,
                                         self.fanout,
                                         self.n_layers,
                                         self.batch_size,
                                         self.device)

        model_cls, eval_class = get_model_class(config)
        nc_model = model_cls(g, self.config, self.bert_model)
        nc_model.init_gsgnn_model(True)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, train_data)

        nc_model.register_evaluator(self.evaluator)
        if nc_model.tracker is not None:
            self.evaluator.setup_tracker(nc_model.tracker)
        nc_model.fit(dataloader, train_data)
