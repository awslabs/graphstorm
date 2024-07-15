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
import logging
from typing import Dict
import argparse
import yaml

SUPPORTED_TASKS = {
    "node_classification",
    "node_regression",
    "edge_classification",
    "edge_regression",
    "link_prediction"
}

def get_parser():
    parent_parser = argparse.ArgumentParser(
        prog="DockerComposeGenerator",
        add_help=False)
    parser = argparse.ArgumentParser(
        prog="DockerComposeGenerator",
        allow_abbrev=False)

    # Common required arguments
    common_req_args = parent_parser.add_argument_group('Common required arguments')

    common_req_args.add_argument("--graph-name", required=True,
        type=str, help="Graph name")
    common_req_args.add_argument("--num-instances", required=True, type=int,
        help="Number of instances to simulate")
    common_req_args.add_argument("--graph-data-s3", required=True,
        help="S3 location of input graph data")
    common_req_args.add_argument("--region", type=str,
        required=True,
        help="Region of the S3 data.")
    # Common optional arguments
    common_opt_args = parent_parser.add_argument_group('Common optional arguments')
    common_opt_args.add_argument("--image", required=False, type=str,
        default='graphstorm:sm',
        help="GraphStorm SageMaker image name.")
    common_opt_args.add_argument("--log-level", required=False, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'FATAL'])
    common_opt_args.add_argument("--aws-access-key-id", type=str,
        help="AWS access key. Also need to provide --aws-secret-access-key and --aws-session-token",
        required=False)
    common_opt_args.add_argument("--aws-secret-access-key", type=str,
        help="AWS secret access key. Also need to provide --aws-access-key-id and --aws-session-token",
        required=False)
    common_opt_args.add_argument("--aws-session-token", type=str,
        help="AWS session token. Also need to provide --aws-access-key-id and --aws-secret-access-key",
        required=False)

    # Subparsers for each action
    subparsers = parser.add_subparsers(dest='action',
        help="Action to perform. Choose between 'partitioning', 'training', and 'inference', "
        "and then follow up with the action-specific arguments.",
        required=True)
    partition_parser = subparsers.add_parser('partitioning',
        help="Generate compose configuration for partitioning. "
        "For available args run: `python generate_sagemaker_docker_compose.py partitioning --help`",
        parents=[parent_parser])
    train_parser = subparsers.add_parser('training',
        help="Generate compose configuration for training "
        "For available args run: `python generate_sagemaker_docker_compose.py training --help`",
        parents=[parent_parser])
    inference_parser = subparsers.add_parser('inference',
        help="Generate compose configuration for inference "
        "For available args run: `python generate_sagemaker_docker_compose.py inference--help`",
        parents=[parent_parser])


    # Partition arguments
    partition_parser.add_argument("--num-parts", required=True,
        help="Number of partitions to generate")
    partition_parser.add_argument("--output-data-s3", required=True,
        help="S3 location to store partitioned graph data")
    partition_parser.add_argument("--skip-partitioning", required=False, action='store_true',
        help="Skip partitioning step and only do GSL object creation. Partition assignments "
             "need to exist under the <output-data-s3>/partitions location.")
    partition_parser.add_argument("--partition-algorithm", required=False,
        default='random', choices=['random'],
        help="Partition algorithm to use.")
    partition_parser.add_argument("--metadata-filename", required=False,
        default="metadata.json", help="Metadata file that describes the files "
        "in the DGL chunked format.")

    # Training arguments
    train_parser.add_argument("--model-artifact-s3", type=str,
        required=True,
        help="S3 prefix to save the model artifacts to")
    train_parser.add_argument("--train-yaml-s3", type=str, required=True,
        help="S3 URI location of training yaml file. "
             "Do not store it with partitioned graph")
    train_parser.add_argument("--task-type", required=True, choices=SUPPORTED_TASKS)
    train_parser.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by the user to run custom training logic. "
             "Please provide the path of the script within the docker image")


    # Inference arguments
    inference_parser.add_argument("--task-type", required=True, choices=SUPPORTED_TASKS)
    inference_parser.add_argument("--output-emb-s3", type=str,
        help="S3 location to store GraphStorm generated node embeddings.",
        default=None)
    inference_parser.add_argument("--raw-node-mappings-s3", type=str,
        required=True,
        help="S3 prefix for original node mappings (str to int)")
    inference_parser.add_argument("--output-prediction-s3", type=str,
        help="S3 location to store prediction results. "
             "(Only works with node classification/regression "
             "and edge classification/regression tasks)",
        default=None)
    inference_parser.add_argument("--custom-script", type=str, default=None,
        help="Custom inference script provided by the user to run custom inference logic. "
             "Please provide the path of the script within the docker image")
    inference_parser.add_argument("--model-artifact-s3", type=str,
        help="S3 prefix to load the saved model artifacts from")
    inference_parser.add_argument("--infer-yaml-s3", type=str, required=True,
        help="S3 URI location of inference yaml file. ")

    return parser

def generate_command(
        known_args: argparse.Namespace,
        unknown_args: str) -> str:
    """
    Will generate the appropriate command for the container given the provided arguments.
    The 'training' and 'inference' actions pass the unknown_args along to GraphStorm.
    """
    if known_args.action == "partitioning":
        skip_partitioning = 'true' if known_args.skip_partitioning else 'false'
        command_str = ' '.join(
            [
                'python3', '-u', 'partition_entry.py',
                '--graph-data-s3', known_args.graph_data_s3,
                '--num-parts', known_args.num_parts,
                '--output-data-s3', known_args.output_data_s3,
                '--metadata-filename', known_args.metadata_filename,
                '--partition-algorithm', known_args.partition_algorithm,
                '--skip-partitioning', skip_partitioning,
                '--log-level' , known_args.log_level,
            ]
        )

    elif known_args.action == "training":
        command_str = ' '.join(
            [
                'python3', '-u', 'train_entry.py',
                '--task-type', known_args.task_type,
                '--graph-name', known_args.graph_name,
                '--graph-data-s3', known_args.graph_data_s3,
                '--model-artifact-s3', known_args.model_artifact_s3,
                '--train-yaml-s3', known_args.train_yaml_s3,
            ]
        )
        if known_args.custom_script:
            command_str += f" --custom-script {known_args.custom_script}"
        command_str +=  ' ' + unknown_args
    elif known_args.action == "inference":
        command_str = ' '.join(
            [
                'python3', '-u', 'infer_entry.py',
                '--model-artifact-s3', known_args.model_artifact_s3,
                '--raw-node-mappings-s3', known_args.raw_node_mappings_s3,
                '--task-type', known_args.task_type,
                '--graph-name', known_args.graph_name,
                '--graph-data-s3', known_args.graph_data_s3,
                '--infer-yaml-s3', known_args.infer_yaml_s3,
            ]
        )
        if known_args.custom_script:
            command_str += f" --custom-script {known_args.custom_script}"
        if known_args.output_prediction_s3:
            if known_args.task_type != "link_prediction":
                command_str += f" --output-prediction-s3 {known_args.output_prediction_s3}"
            else:
                # LP can't save preds
                logging.warning("Cannot save predictions for link prediction task, "
                                "double-check input arguments.")
        if known_args.output_emb_s3:
            command_str += f" --output-emb-s3 {known_args.output_emb_s3}"
        command_str +=  ' ' + unknown_args
    else:
        raise RuntimeError(f"Unknown action: {known_args.action}")

    return command_str

GPU_CAPABILITY = {
    'resources': {
        'reservations': {
            'devices': [{
                'driver': 'nvidia',
                'count': 1,
                'capabilities': ['gpu']
            }]
        }
    }
}

if __name__ == "__main__":
    parser = get_parser()
    args, unknown_args = parser.parse_known_args()
    unknownargs_str = ' '.join(unknown_args)

    compose_dict = {}  # type: Dict

    compose_dict['version'] = '3.7'
    compose_dict['networks'] = {'gsf': {'name': 'gsf-network'}}
    action_needs_gpu = args.action != 'partitioning'

    # Check that all AWS credentials are provided.
    one_cred_exists = False
    all_creds_exist = True
    for aws_cred in [args.aws_access_key_id, args.aws_secret_access_key, args.aws_session_token]:
        if aws_cred:
            one_cred_exists = True
        else:
            all_creds_exist = False

    if one_cred_exists and not all_creds_exist:
        raise RuntimeError("Please provide all AWS credentials.")

    def generate_instance_entry(instance_idx: int, world_size: int, needs_gpu: bool) -> Dict[str, str]:
        """
        Generates one service entry per instance requested. See
        https://docs.docker.com/compose/compose-file/05-services/
        for docker-compose file syntax.
        """
        inner_host_list = [f'algo-{i}' for i in range(1, world_size+1)]
        quoted_host_list = ', '.join(f'"{host}"' for host in inner_host_list)
        host_list = f'[{quoted_host_list}]'

        cmd = generate_command(args, unknownargs_str)
        instance_entry = {
                'image': args.image,
                'container_name': f'algo-{instance_idx}',
                'hostname': f'algo-{instance_idx}',
                'networks': ['gsf'],
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
                    'SM_NUM_GPUS': 1 if needs_gpu else 0,
                    'SM_TRAINING_ENV': \
                        f'{{"hosts": {host_list}, "current_host": "algo-{instance_idx}"}}',
                    'RANK': instance_idx - 1,
                    'WORLD_SIZE': world_size,
                    'MASTER_ADDR': 'algo-1',
                    'AWS_REGION': args.region,
                    'SM_CHANNEL_TRAIN': '/opt/ml/',
                    },
                'ports': [22],
                'working_dir': '/opt/ml/code/',
                'shm_size': '8gb' # See https://github.com/pytorch/pytorch/issues/2244#issuecomment-318864552
            }

        if needs_gpu:
            instance_entry['deploy'] = GPU_CAPABILITY
        if all_creds_exist:
            instance_entry['environment']['AWS_ACCESS_KEY_ID'] = args.aws_access_key_id
            instance_entry['environment']['AWS_SECRET_ACCESS_KEY'] = args.aws_secret_access_key
            instance_entry['environment']['AWS_SESSION_TOKEN'] = args.aws_session_token

        return instance_entry

    service_dicts = {
        f"algo-{i}": generate_instance_entry(
            i, int(args.num_instances), action_needs_gpu) for i in range(1, int(args.num_instances)+1)
    }

    compose_dict['services'] = service_dicts

    filename_prefix = (
        f'docker-compose-{args.action}-{args.graph_name}-'
        f'{args.num_instances}workers')

    if args.action == 'partitioning':
        filename = f'{filename_prefix}-{args.partition_algorithm}-{args.num_parts}parts.yaml'
    elif args.action == 'training':
        filename = f'{filename_prefix}-{args.task_type}.yaml'
    elif args.action == 'inference':
        filename = f'{filename_prefix}-{args.task_type}.yaml'
    else:
        raise RuntimeError(f"Unknown action: {args.action}")

    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(compose_dict, f)
