""" M5GNN semantic match training example
"""
from graphstorm.config import get_argument_parser
from graphstorm.config import M5GNNConfig
from graphstorm.model.huggingface import HuggingfaceBertLoader
from graphstorm.trainer import M5gnnNodePredictTrainer


def main(args):
    config = M5GNNConfig(args)

    bert_config = config.bert_config

    m5_models = HuggingfaceBertLoader(bert_config).load()

    trainer = M5gnnNodePredictTrainer(config, m5_models)

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