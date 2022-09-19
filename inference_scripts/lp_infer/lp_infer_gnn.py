""" Inference script for link prediction tasks with GNN only
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInfer

def main(args):
    config = GSConfig(args)
    bert_models = {}

    infer = GSgnnLinkPredictionInfer(config, bert_models)
    infer.infer()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
