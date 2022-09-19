""" GSgnn pure gpu link prediction
"""

import argparse
import dgl

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnLinkPredictionTrainer

def main(args):
    config = GSConfig(args)
    lm_models = {}

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
