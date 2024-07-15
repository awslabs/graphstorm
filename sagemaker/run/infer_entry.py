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

    SageMaker inference entry point
"""
import argparse
import os
import subprocess

from graphstorm.config import SUPPORTED_TASKS
from graphstorm.sagemaker.sagemaker_infer import run_infer

def parse_inference_args():
    """  Add arguments for model inference
    """
    parser = argparse.ArgumentParser(description='gs sagemaker inference pipeline')

    parser.add_argument("--task-type", type=str,
        help=f"task type, builtin task type includes: {SUPPORTED_TASKS}")

    # disrributed training
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--infer-yaml-s3", type=str,
        help="S3 location of inference yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--output-emb-s3", type=str,
        help="S3 location to store GraphStorm generated node embeddings.",
        default=None)
    parser.add_argument("--output-prediction-s3", type=str,
        help="S3 location to store prediction results. " \
             "(Only works with node classification/regression " \
             "and edge classification/regression tasks)",
        default=None)
    parser.add_argument("--model-artifact-s3", type=str,
        help="S3 bucket to load the saved model artifacts")
    parser.add_argument("--raw-node-mappings-s3", type=str, required=False,
        default=None, help="S3 location where the original (str to int) node mappings exist.")
    parser.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by a customer to run customer training logic. \
            Please provide the path of the script within the docker image")
    parser.add_argument("--output-chunk-size", type=int, default=100000,
        help="Number of rows per chunked prediction result or node embedding file.")
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
    parser = parse_inference_args()
    args, unknownargs = parser.parse_known_args()

    subprocess.run(["df", "-h"], check=True)
    run_infer(args, unknownargs)
