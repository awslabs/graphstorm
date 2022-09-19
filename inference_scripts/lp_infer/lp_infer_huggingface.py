""" Inference script for link prediction tasks with BERT model from huggingface
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.model.huggingface import HuggingfaceBertLoader
from graphstorm.inference import GSgnnLinkPredictionInfer

def main(args):
    config = GSConfig(args)
    bert_config = config.bert_config
    lm_models = HuggingfaceBertLoader(bert_config).load()

    infer = GSgnnLinkPredictionInfer(config, lm_models)
    infer.infer()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
