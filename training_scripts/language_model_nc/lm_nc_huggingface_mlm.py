""" GSgnn semantic match training example
"""
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.model.huggingface import HuggingfaceMLMBertLoader, HuggingfaceBertTokenizer
from graphstorm.trainer import GSgnnLanguageModelMLMTrainer


def main(args):
    config = GSConfig(args)

    bert_config = config.bert_config

    lm_models = HuggingfaceMLMBertLoader(bert_config).load()
    tokenizer = HuggingfaceBertTokenizer(bert_config).load()

    trainer = GSgnnLanguageModelMLMTrainer(config, lm_models, tokenizer)

    trainer.fit()
    print("Training completed")

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser = generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)