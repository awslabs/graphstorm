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

    Launch SageMaker graph construction task
"""
import os

import boto3 # pylint: disable=import-error
from sagemaker.processing import (ScriptProcessor,
                                  ProcessingInput,
                                  ProcessingOutput)
import sagemaker

from common_parser import get_common_parser, parse_estimator_kwargs

INSTANCE_TYPE = "ml.m5.12xlarge"

def run_job(input_args, image, unknownargs):
    """ Run job using SageMaker ScriptProcessor

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    unknownargs: dict
        GraphStorm graph construction parameters
    """
    sm_task_name = input_args.task_name # SageMaker task name
    role = input_args.role # SageMaker ARN role
    instance_type = input_args.instance_type # SageMaker instance type
    region = input_args.region # AWS region
    entry_point = input_args.entry_point # GraphStorm gconstruct entry_point
    input_graph_s3 = input_args.input_graph_s3
    output_graph_s3 = input_args.output_graph_s3
    graph_name = input_args.graph_name # Inference graph name
    graph_config_file = input_args.graph_config_file # graph config file

    boto_session = boto3.session.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    sess = sagemaker.session.Session(boto_session=boto_session,
        sagemaker_client=sagemaker_client)

    input_path = '/opt/ml/processing/input'
    config_path = os.path.join(input_path, graph_config_file)
    output_path = '/opt/ml/processing/output'
    command=['python3']

    estimator_kwargs = parse_estimator_kwargs(input_args.sm_estimator_parameters)

    script_processor = ScriptProcessor(
        image_uri=image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        command=command,
        base_job_name=f"gs-gconstruct-{graph_name}",
        sagemaker_session=sess,
        tags=[{"Key":"GraphStorm", "Value":"beta"},
              {"Key":"GraphStorm_Task", "Value":"Processing"}],
        **estimator_kwargs
    )

    script_processor.run(
        code=entry_point,
        arguments=['--graph-config-path', config_path,
                   '--input-path', input_path,
                   '--output-path', output_path,
                   '--graph-name', graph_name] + unknownargs,
        inputs=[
            ProcessingInput(
                source=input_graph_s3,
                destination=input_path
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=output_path,
                destination=output_graph_s3,
                output_name=graph_name,
            ),
        ],
        wait=not input_args.async_execution
    )


def get_gconstruct_parser():
    """
    Get GraphStorm GConstruct task parser.
    """
    parser = get_common_parser()

    gconstruct_arguments = parser.add_argument_group("GraphStorm Gconstruct arguments")

    gconstruct_arguments.add_argument("--input-graph-s3", type=str,
        required=True, help="S3 location of the input graph data")
    gconstruct_arguments.add_argument("--output-graph-s3", type=str,
        required=True, help="S3 location to store the constructed graph")
    gconstruct_arguments.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/run/gconstruct_entry.py",
        help="PATH-TO graphstorm/sagemaker/run/gconstruct_entry.py")

    gconstruct_arguments.add_argument("--graph-name", type=str,
        required=True, help="Graph name")
    gconstruct_arguments.add_argument("--graph-config-file", type=str,
        required=True, help="Graph configuration file. It must be a relative "
        "S3 path to the --input-graph-s3 prefix.")

    return parser

if __name__ == "__main__":
    arg_parser = get_gconstruct_parser()
    args, unknownargs = arg_parser.parse_known_args()
    print(args)

    gconstruct_image = args.image_url

    if not args.instance_type:
        args.instance_type = INSTANCE_TYPE

    run_job(args, gconstruct_image, unknownargs)