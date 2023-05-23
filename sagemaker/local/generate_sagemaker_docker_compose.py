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

    A script that can be used to generate Docker compose files
    that can emulate running SageMaker training jobs locally.
"""
from typing import Dict
import argparse
import sys
import yaml

from graphstorm.config.config import SUPPORTED_TASKS

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, type=str,
        help="GraphStorm SageMaker image name.")
    parser.add_argument("--num-instances", required=True, type=int,
        help="Number of instances to simulate")
    parser.add_argument("--task-type", required=True, type=str,
        help=f"task type, builtin task type includes: {SUPPORTED_TASKS}")
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", required=True,
        help="S3 location of input training graph")
    parser.add_argument("--yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--model-artifact-s3", type=str,
        help="S3 location to store the model artifacts")
    parser.add_argument("--output-emb-s3", type=str,
        help="S3 location to store GraphStorm generated node embeddings.",
        default=None)
    parser.add_argument("--output-prediction-s3", type=str,
        help="S3 location to store prediction results. " \
             "(Only works with node classification/regression " \
             "and edge classification/regression tasks)",
        default=None)
    parser.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by a customer to run customer training logic. \
            Please provide the path of the script within the docker image")
    parser.add_argument("--region", type=str,
        default="us-east-1",
        help="Region")
    parser.add_argument("--aws-access-key-id", type=str,
        help="AWS access key")
    parser.add_argument("--aws-secret-access-key", type=str,
        help="AWS secret access key")
    parser.add_argument("--aws-session-token", type=str,
        help="AWS session token")
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Inidcate that it is an inference task. \
              Used with built-in training/inference scripts"
    )

    return parser

if __name__ == "__main__":
    parser = parse_args()
    args, unknownargs = parser.parse_known_args()

    compose_dict = dict()

    compose_dict['version'] = '3.7'
    compose_dict['networks'] = {'gfs': {
        'external': {'name': 'gfs-network'}}}

    def gen_train_cmd(custom_script):
        entry_point = 'train_entry.py'

        cmd = f'python3  {entry_point} ' \
            f'--task-type {args.task_type} ' \
            f'--graph-data-s3 {args.graph_data_s3} ' \
            f'--graph-name {args.graph_name} ' \
            f'--train-yaml-s3 {args.yaml_s3} ' \
            f'--model-artifact-s3 {args.model_artifact_s3} ' \
            f'{custom_script} ' + ' '.join(unknownargs)
        return cmd

    def gen_infer_cmd(custom_script):
        entry_point = 'infer_entry.py'
        output_emb_s3 = f'--output-emb-s3 {args.output_emb_s3} ' \
                        if args.output_emb_s3 is not None else ''
        output_prediction_s3 = f'--output-prediction-s3 {args.output_prediction_s3} ' \
                        if args.output_prediction_s3 is not None else ''

        cmd = f'python3  {entry_point} ' \
            f'--task-type {args.task_type} ' \
            f'--graph-data-s3 {args.graph_data_s3} ' \
            f'--graph-name {args.graph_name} ' \
            f'--infer-yaml-s3 {args.yaml_s3} ' \
            f'--model-artifact-s3 {args.model_artifact_s3} ' \
            f'{output_emb_s3} {output_prediction_s3} {custom_script} ' + \
            ' '.join(unknownargs)
        return cmd

    def generate_instance_entry(instance_idx, world_size):
        inner_host_list = [f'algo-{i}' for i in range(1, world_size+1)]
        quoted_host_list = ', '.join(f'"{host}"' for host in inner_host_list)
        host_list = f'[{quoted_host_list}]'

        custom_script = '' if args.custom_script is None \
            else f'custom-script {args.custom_script}'

        cmd = gen_infer_cmd(custom_script) if args.inference else gen_train_cmd(custom_script)
        return {
                'image': args.image,
                'container_name': f'algo-{instance_idx}',
                'hostname': f'algo-{instance_idx}',
                'networks': ['gfs'],
                'volumes': [
                    {
                        'type': 'tmpfs',
                        'target': '/dev/shm',
                        'tmpfs': {
                            'size': 16000000000, # ~16gb
                        }
                    },
                ],
                'command': cmd,
                'environment':
                    {
                    'SM_NUM_GPUS': 1,
                    'SM_TRAINING_ENV': \
                        f'{{"hosts": {host_list}, "current_host": "algo-{instance_idx}"}}',
                    'RANK': instance_idx,
                    'WORLD_SIZE': world_size,
                    'MASTER_ADDR': 'algo-1',
                    'AWS_REGION': args.region,
                    'SM_CHANNEL_TRAIN': '/opt/ml/',
                    'AWS_ACCESS_KEY_ID': args.aws_access_key_id,
                    'AWS_SECRET_ACCESS_KEY': args.aws_secret_access_key,
                    'AWS_SESSION_TOKEN': args.aws_session_token,
                    },
                'ports': [22],
                'working_dir': '/opt/ml/code/',
                'deploy': {
                    'resources':
                    {
                        'reservations': {
                            'devices': [{
                                'driver': 'nvidia',
                                'device_ids': [f'{instance_idx-1}'],
                                'capabilities': ['gpu']
                            }]
                        }
                    }
                }
            }

    service_dicts = {f"algo-{i}": generate_instance_entry(i, args.num_instances) for i in range(1, args.num_instances+1)}

    compose_dict['services'] = service_dicts

    filename = \
        (f'docker-compose-{args.task_type}-{args.num_instances}-infer.yaml') \
        if args.inference else \
        (f'docker-compose-{args.task_type}-{args.num_instances}-train.yaml')

    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(compose_dict, f)
