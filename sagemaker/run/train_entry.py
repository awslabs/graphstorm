import argparse
import os

from graphstorm.sagemaker.sagemaker_train import parse_train_args as parse_gsf_train_args
from graphstorm.sagemaker.sagemaker_train import run_train

def parse_train_args():
    """  Add arguments for model training
    """
    parser = parse_gsf_train_args()

    # following arguments are required to launch a distributed GraphStorm training task
    parser.add_argument('--data-path', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=str, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--sm-dist-env', type=str, default=os.environ['SM_TRAINING_ENV'])
    parser.add_argument('--master-addr', type=str, default=os.environ['MASTER_ADDR'])
    parser.add_argument('--region', type=str, default=os.environ['AWS_REGION'])

    # Add your args if any

    return parser

if __name__ =='__main__':
    parser = parse_train_args()
    args, unknownargs = parser.parse_known_args()

    run_train(args, unknownargs)
