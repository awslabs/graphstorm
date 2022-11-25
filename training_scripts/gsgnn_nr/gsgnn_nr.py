""" GSgnn semantic match training example
"""
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnNodePredictTrainer

def main(args):
    config = GSConfig(args)
    bert_config = config.bert_config
    trainer = GSgnnNodePredictTrainer(config)
    trainer.fit()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
