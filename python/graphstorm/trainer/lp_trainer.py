from ..model import GSgnnLinkPredictionTrainData
from ..model import GSgnnLinkPredictionDataLoader
from ..model import GSgnnLPJointNegDataLoader
from ..model import GSgnnLPLocalUniformNegDataLoader
from ..model import GSgnnLinkPredictionModel
from ..model import GSgnnMrrLPEvaluator
from .gsgnn_trainer import GSgnnTrainer

from ..model.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from ..model.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from ..model.dataloading import BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER

def get_model_class(config):
    return GSgnnLinkPredictionModel, GSgnnMrrLPEvaluator

class GSgnnLinkPredictionTrainer(GSgnnTrainer):
    """ Link prediction trainer.

    This is a highlevel trainer wrapper that can be used directly to train a link prediction model.

    Usage:
    ```
    from graphstorm.config import GSConfig
    from graphstorm.model.huggingface import HuggingfaceBertLoader
    from graphstorm.model import GSgnnLinkPredictionTrainer

    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    trainer = GSgnnLinkPredictionTrainer(config, lm_models)
    trainer.fit()
    ```

    Parameters
    ----------
    config: GSConfig
        Task configuration
    bert_model: dict
        A dict of BERT models in the format of node-type -> BERT model
    """
    def __init__(self, config, bert_model):
        super(GSgnnLinkPredictionTrainer, self).__init__()
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

        train_data = GSgnnLinkPredictionTrainData(g, pb, self.train_etypes, self.eval_etypes, full_graph_training)

        if g.rank() == 0:
            print("Use {} negative sampler with exclude training target {}".format(
                self.negative_sampler,
                self.exclude_training_targets))
        if self.negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            dataloader = GSgnnLinkPredictionDataLoader(g,
                                                       train_data,
                                                       self.fanout,
                                                       self.n_layers,
                                                       self.batch_size,
                                                       self.num_negative_edges,
                                                       self.device,
                                                       self.exclude_training_targets,
                                                       self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
            dataloader = GSgnnLPJointNegDataLoader(g,
                                                   train_data,
                                                   self.fanout,
                                                   self.n_layers,
                                                   self.batch_size,
                                                   self.num_negative_edges,
                                                   self.device,
                                                   self.exclude_training_targets,
                                                   self.reverse_edge_types_map)
        elif self.negative_sampler == BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER:
            dataloader = GSgnnLPLocalUniformNegDataLoader(g,
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
        lp_model.init_gsgnn_model(True)

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
