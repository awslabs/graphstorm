""" GSgnn semantic match training example
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.model.huggingface import HuggingfaceBertLoader
from graphstorm.trainer import GSgnnLinkPredictionTrainer

def main(args):
    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    trainer = GSgnnLinkPredictionTrainer(config, lm_models)
    trainer.fit()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
