""" Inference script for node classification/regression tasks with GNN only
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnNodePredictInfer

def main(args):
    config = GSConfig(args)
    infer = GSgnnNodePredictInfer(config)
    infer.infer()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
