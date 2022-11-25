""" Inference script for edge classification/regression tasks with GNN only
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnEdgePredictInfer

def main(args):
    config = GSConfig(args)
    infer = GSgnnEdgePredictInfer(config)
    infer.infer()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
