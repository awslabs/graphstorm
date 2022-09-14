from ..model import M5gnnMLMTrainData
from ..model import LanguageModelMLM
from .m5gnn_trainer import M5gnnTrainer

class M5gnnLanguageModelMLMTrainer(M5gnnTrainer):
    """ Language model node classification trainer

    Parameters
    ----------
    config: M5GNNConfig
        Task configuration
    bert_model: dict
        A dict of BERT models in the format of node-type -> M5 BERT model
    """
    def __init__(self, config, bert_model, tokenizer):
        super(M5gnnLanguageModelMLMTrainer, self).__init__()
        assert isinstance(bert_model, dict)
        assert len(bert_model) == 1, "We can only finetune one bert_model at once"
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.config = config
        self.tune_ntype = list(bert_model.keys())[0]

        self.evaluator = None

        self.init_dist_context(config.ip_config,
                              config.graph_name,
                              config.part_config,
                              config.backend)
        assert self.tune_ntype in self._g.ntypes, \
                'A bert model is created for node type {}, but the node type does not exist.'



    def save(self):
        pass

    def load(self):
        pass

    def fit(self):
        g = self._g
        pb = g.get_partition_book()
        config = self.config

        train_data = M5gnnMLMTrainData(g, pb, self.tune_ntype)

        mlm_model = LanguageModelMLM(g, self.config, self.bert_model, self.tokenizer)
        mlm_model.init_m5gnn_model(True)

        mlm_model.fit(config.batch_size, train_data)
