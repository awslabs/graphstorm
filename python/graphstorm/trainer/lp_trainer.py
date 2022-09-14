from ..model import M5gnnLinkPredictionTrainData
from ..model import M5gnnLinkPredictionDataLoader
from ..model import M5gnnLPJointNegDataLoader
from ..model import M5gnnLPLocalUniformNegDataLoader
from ..model import M5GNNLinkPredictionModel
from ..model import M5gnnMrrLPEvaluator
from .m5gnn_trainer import M5gnnTrainer

from ..model.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from ..model.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from ..model.dataloading import BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER

def get_model_class(config):
    return M5GNNLinkPredictionModel, M5gnnMrrLPEvaluator

class M5gnnLinkPredictionTrainer(M5gnnTrainer):
    """ Link prediction trainer.

    This is a highlevel trainer wrapper that can be used directly to train a link prediction model.

    Usage:
    ```
    from graphstorm.config import M5GNNConfig
    from graphstorm.model import M5BertLoader
    from graphstorm.model import M5gnnLinkPredictionTrainer

    config = M5GNNConfig(args)
    bert_config = config.bert_config
    m5_models = M5BertLoader(bert_config).load()

    trainer = M5gnnLinkPredictionTrainer(config, m5_models)
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
        super(M5gnnLinkPredictionTrainer, self).__init__()
        assert isinstance(bert_model, dict)
        self.bert_model = bert_model
        self.config = config

        self.train_etypes = [tuple(train_etype.split(',')) for train_etype in config.train_etype]
        self.eval_etypes = [tuple(eval_etype.split(',')) for eval_etype in config.eval_etype]

        # neighbor sample related
        self.fanout = config.fanout if config.model_encoder_type in ["rgat", "rgcn"] else [0]
        self.n_layers = config.n_layers if config.model_encoder_type in ["rgat", "rgcn"] else 1
        self.batch_size = config.batch_size
        # sampling related
        self.negative_sampler = config.negative_sampler
        self.num_negative_edges = config.num_negative_edges
        self.exclude_training_targets = config.exclude_training_targets
        self.reverse_edge_types_map = config.reverse_edge_types_map
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

    def fit(self, full_graph_training=False):
        g = self._g
        pb = g.get_partition_book()
        config = self.config

        train_data = M5gnnLinkPredictionTrainData(g, pb, self.train_etypes, self.eval_etypes, full_graph_training)

        if g.rank() == 0:
            print("Use {} negative sampler with exclude training target {}".format(
                self.negative_sampler,
                self.exclude_training_targets))
        if self.negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            dataloader = M5gnnLinkPredictionDataLoader(g,
                                                       train_data,
                                                       self.fanout,
                                                       self.n_layers,
                                                       self.batch_size,
                                                       self.num_negative_edges,
                                                       self.device,
                                                       self.exclude_training_targets,
                                                       self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
            dataloader = M5gnnLPJointNegDataLoader(g,
                                                   train_data,
                                                   self.fanout,
                                                   self.n_layers,
                                                   self.batch_size,
                                                   self.num_negative_edges,
                                                   self.device,
                                                   self.exclude_training_targets,
                                                   self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER:
            dataloader = M5gnnLPLocalUniformNegDataLoader(g,
                                                          train_data,
                                                          self.fanout,
                                                          self.n_layers,
                                                          self.batch_size,
                                                          self.num_negative_edges,
                                                          self.device,
                                                          self.exclude_training_targets,
                                                          self.reverse_edge_types_map)
        else:
            raise Exception('Unknown negative sampler')

        model_class, eval_class = get_model_class(config)
        lp_model = model_class(g, config, self.bert_model)
        lp_model.init_m5gnn_model(True)

        # if no evalutor is registered, use the default one.
        if self.evaluator is None:
            self.evaluator = eval_class(g, config, train_data)

        lp_model.register_evaluator(self.evaluator)
        if lp_model.tracker is not None:
            self.evaluator.setup_tracker(lp_model.tracker)
        lp_model.fit(dataloader)

    @property
    def g(self):
        return self._g
