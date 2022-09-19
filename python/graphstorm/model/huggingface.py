from .hbert import init_bert
from .utils import load_model
from transformers import AutoTokenizer

class HuggingfaceBertTokenizer():
    """ Huggingface BERT tokenizer loader

    HuggingfaceBertTokenizer will load the huggingface BERT tokenizer from disk.

    Parameters
    -----------
    bert_config: dict
        BERT model configurations. In the format of node-type -> BERT config
    config: GSConfig
        Task configuration
    """
    def __init__(self, bert_configs):
        configs = {}
        for bert_config in bert_configs:
            configs[bert_config['node_type']] = {
                "model_name": bert_config['model_name'],
            }
        self._bert_config = configs

    def load(self):
        tokernizers = {}
        for ntype, m_conf in self._bert_config.items():
            tokernizers[ntype] = AutoTokenizer.from_pretrained(m_conf["model_name"])

        return tokernizers

class HuggingfaceMLMBertLoader():
    """ Huggingface BERT model loader

    HuggingfaceBertLoader will load the huggingface BERT model from disk.

    Parameters
    -----------
    bert_config: dict
        BERT model configurations. In the format of node-type -> BERT config
    config: GSConfig
        Task configuration
    """
    def __init__(self, bert_configs):
        configs = {}
        for bert_config in bert_configs:
            configs[bert_config['node_type']] = {
                "model_name": bert_config['model_name'],
                "gradient_checkpoint": bert_config['gradient_checkpoint'],

            }
        self._bert_config = configs

    def load(self):
        bert_model = {}
        for ntype, m_conf in self._bert_config.items():
            # load bert model with LML loss enabled
            b_model = init_bert(bert_model_name=m_conf["model_name"],
                                gradient_checkpointing=m_conf['gradient_checkpoint'],
                                use_bert_loss=True)

            bert_model[ntype] = b_model

        return bert_model

class HuggingfaceBertLoader():
    """ Huggingface BERT model loader

    HuggingfaceBertLoader will load the huggingface BERT model from disk.

    Parameters
    -----------
    bert_config: dict
        BERT model configurations. In the format of node-type -> BERT config
    config: GSConfig
        Task configuration
    """
    def __init__(self, bert_configs):
        configs = {}
        for bert_config in bert_configs:
            configs[bert_config['node_type']] = {
                "model_name": bert_config['model_name'],
                "gradient_checkpoint": bert_config['gradient_checkpoint'],

            }
            if 'finetuned_model_path' in bert_config:
                configs[bert_config['node_type']]['finetuned_model_path'] = bert_config['finetuned_model_path']
                print("Loading pre finetuned model for node with type " + bert_config['node_type'])
        self._bert_config = configs

    def load(self):
        bert_model = {}
        for ntype, m_conf in self._bert_config.items():
            b_model = init_bert(bert_model_name=m_conf["model_name"],
                                gradient_checkpointing=m_conf['gradient_checkpoint'])

            if 'finetuned_model_path' in m_conf:
                model_to_load = {ntype: b_model}
                load_model(model_path=m_conf['finetuned_model_path'], bert_model=model_to_load)
                b_model = model_to_load[ntype]

            bert_model[ntype] = b_model

        return bert_model