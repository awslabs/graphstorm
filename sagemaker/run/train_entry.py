"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    SageMaker training entry point.
"""
import argparse
import os
import subprocess

from graphstorm.config import SUPPORTED_TASKS
from graphstorm.sagemaker.sagemaker_train import run_train

def parse_train_args():
    """  Add arguments for model training
    """
    parser = argparse.ArgumentParser(description='gs sagemaker train pipeline')

    parser.add_argument("--task-type", type=str,
        help=f"task type, builtin task type includes: {SUPPORTED_TASKS}")

    # distributed training
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--train-yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--model-artifact-s3", type=str,
        help="S3 location to store the model artifacts.")
    parser.add_argument("--model-checkpoint-to-load", type=str, default=None,
        help="S3 path to a model checkpoint from a previous training task "
             "that is going to be resumed.")
    parser.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by a customer to run customer training logic. \
            Please provide the path of the script within the docker image")
    parser.add_argument('--log-level', default='INFO',
        type=str, choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'FATAL'])

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

    subprocess.run(["df", "-h"], check=True)
    run_train(args, unknownargs)
