""" Inference script for edge classification/regression tasks with GNN only
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import M5GNNConfig
from graphstorm.model.huggingface import HuggingfaceBertLoader
from graphstorm.inference import M5gnnEdgePredictInfer

def main(args):
    config = M5GNNConfig(args)
    bert_config = config.bert_config
    m5_models = HuggingfaceBertLoader(bert_config).load()

    infer = M5gnnEdgePredictInfer(config, m5_models)
    infer.infer()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
