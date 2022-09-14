""" Inference script for node classification/regression tasks with GNN only
"""

from graphstorm.config import get_argument_parser
from graphstorm.config import M5GNNConfig
from graphstorm.inference import M5gnnNodePredictInfer

def main(args):
    config = M5GNNConfig(args)
    m5_models = {}

    infer = M5gnnNodePredictInfer(config, m5_models)
    infer.infer()

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
