"""
    Copyright Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Launch a GraphStorm SageMaker endpoint for realtime inference
"""

import argparse
import json
import logging
import os
import shutil
from time import gmtime, strftime
from uuid import uuid4

import boto3  # pylint: disable=import-error
import sagemaker as sm
from botocore.exceptions import WaiterError
from launch_utils import (check_name_format, extract_ecr_region,
                          upload_data_to_s3, wrap_model_artifacts,
                          has_tokenize_transformation)

# TODO: When adding new realtime inference tasks, modify this list
SUPPORTED_REALTIME_INFER_NC_TASK = 'node_classification'
SUPPORTED_REALTIME_INFER_NR_TASK = 'node_regression'
SUPPORTED_REALTIME_INFER_TASKS = [SUPPORTED_REALTIME_INFER_NC_TASK,
                                  SUPPORTED_REALTIME_INFER_NR_TASK]

# Constants for SageMaker endpoints
_ROOT = os.path.abspath(os.path.dirname(__file__))
ENTRY_FOLDER_NAME = os.path.join(_ROOT, '../run/realtime_entry_points')

# TODO: When add new realtime inference tasks, modify this dict
ENTRY_FILE_NAMES = {
    SUPPORTED_REALTIME_INFER_NC_TASK: 'node_prediction_entry.py'
    }
DEFAULT_GS_MODEL_FILE_NAME = 'model.bin'


def deploy_endpoint(input_args):
    """ Deploys a SageMaker real-time inference endpoint

    SageMaker's documentation for deploying model for real-time inference is in
    https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html.

    The steps of a model deployment are:

    1. prepare model artifacts;
    2. create a deployable "SageMaker Model";
    3. create an "Endpoint Configuration" based on the "SageMaker Model";
    4. create an "Endpoint" based on the "Endpoint Configuration".

    This job follows these steps, and only work on GraphStorm's model training and inference
    pipeline.

    For custom model deployment, we provide a tool as a courtesy under graphstorm/tools folder.

    Parameters:
    -----------
    region: str
        The AWS region where the SageMaker endpoint will be deployed.
    image_uri: str
        The URI of a GraphStorm SageMaker real-time inference Docker image that is located at
        Amazon ECR in the same region specified in the `region` argument.
    role: str
        The ARN string of an AWS account ARN that has SageMaker execution and model registry full
        access role.
    instance_type: str
        The string type of a SageMaker instance type. The default value is \"ml.c6i.xlarge\".
    instance_count: int
        The number of SageMaker instances to be deployed for the endpoint. The default value is 1.
    custom_production_variant: dict
        The dictionary that inludes custom configuration of the SageMaker ProductionVarient
        for identifying a model to host and the resources chosen to deploy for hosting it, as
        documented at
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html.
    async_execution: str
        The string boolean value. Determine if using asynchronous execution mode to creating
        endpoint. Options include "True" and "true" for True, "False" and "false" for False.
        Default is True.
    restore_model_path: str
        The local folder path where GraphStorm trained model artifacts were stored, e.g.,
        \"save_model/epoch-1/\".
    model_yaml_config_file: str
        The YAML file path. This YAML file is the one that was stored by GraphStorm model training
        pipeline during model training. It is NOT the one used as the input arguments
        for GraphStorm model training.
    graph_json_config_file: str
        Path to modified GConstruct/GSProcessing JSON config. This should be available under the
        **output** of GConstruct/GSProcessing.
    upload_tarfile_s3: str
        The S3 location to upload the packed model tar file. This location should be in the same
        region specified in the `region` argument.
    infer_task_type: str
        The name of real time inference task. Options include \"node_classification\".
    model_name: str
        A string to define the name of a SageMaker inference model. This name string must follow
        SageMaker's naming format, which is ^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$. The default
        value is \"GSF-Model4Realtime\".
    log_level: str
        The level of log. Possible values are 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        Default is 'INFO'."
    vpc_subnet_ids: list[str], optional
        List of VPC subnet IDs for the endpoint model. Required if deploying within a VPC.
        When connecting to a Neptune database in a VPC, these should be private subnets
        with optional NAT Gateway access for internet connectivity. VPC configuration is applied
        at the model level to avoid conflicts with endpoint configuration.
    vpc_security_group_ids: list[str], optional
        List of security group IDs for the endpoint model. Required if deploying within a VPC.
        When connecting to a Neptune database, the security group should allow:
        - Outbound access to Neptune DB on port 8182
        - Outbound access to S3 VPC endpoint for model loading
        The Neptune database's security group must allow inbound access from this security group.
        VPC configuration is applied at the model level to avoid conflicts with endpoint configuration.
    """
    # set the logging level
    log_level = input_args.log_level \
                if hasattr(input_args, "log_level") else logging.INFO
    logging.basicConfig(level=log_level, force=True)

    # ================= prepare model artifacts ================= #
    # prepare sessions
    boto_session = boto3.Session(region_name=input_args.region)
    sm_session = sm.Session(boto_session=boto_session)

    # prepare task dependent variables
    current_folder = os.path.dirname(__file__)
    entry_point_dir = os.path.join(current_folder, ENTRY_FOLDER_NAME)

    # TODO: When adding new realtime inference tasks, add new elif here to support them
    if input_args.infer_task_type == SUPPORTED_REALTIME_INFER_NC_TASK:
        entrypoint_file_name = ENTRY_FILE_NAMES[SUPPORTED_REALTIME_INFER_NC_TASK]
        path_to_entry = os.path.join(entry_point_dir, entrypoint_file_name)
    else:
        raise NotImplementedError(f'The given real-time inference task \
                                    {input_args.infer_task_type} is not supported.')

    path_to_model = os.path.join(input_args.restore_model_path, DEFAULT_GS_MODEL_FILE_NAME)
    path_to_model_yaml = input_args.model_yaml_config_file
    path_to_graph_json = input_args.graph_json_config_file
    model_name = input_args.model_name

    # use a temporary folder in the current folder as output folder
    tmp_output_folder = os.path.join(current_folder, f'{uuid4().hex[:8]}')
    model_tarfile_path = wrap_model_artifacts(path_to_model, path_to_model_yaml,
                                              path_to_graph_json, path_to_entry,
                                              tmp_output_folder,
                                              output_tarfile_name=model_name)
    logging.debug('Packed and compressed model artifacts into %s.', model_tarfile_path)

    # upload the model tar file to the given S3 bucket
    model_url_s3 = upload_data_to_s3(input_args.upload_tarfile_s3, model_tarfile_path,
                                    sm_session)
    logging.debug('Uploaded the model tar file to %s.', model_url_s3)

    # clean up the temporary folder after model uploading
    shutil.rmtree(tmp_output_folder)

    # ================= create deployable model ================= #
    image_uri = input_args.image_uri
    role = input_args.role
    sm_client = boto3.client(service_name="sagemaker", region_name=input_args.region)

    container = {
        "Image": image_uri,
        "ModelDataUrl": model_url_s3,
        "Environment": {"SAGEMAKER_PROGRAM": entrypoint_file_name}
    }

    sm_model_name = model_name + '-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    sm_model_name = sm_model_name[:62]

    # Create model with VPC configuration (if provided)
    create_model_kwargs = {
        "ModelName": sm_model_name,
        "ExecutionRoleArn": role,
        "Containers": [container]
    }
    if input_args.vpc_subnet_ids and input_args.vpc_security_group_ids:
        create_model_kwargs["VpcConfig"] = {
            "Subnets": input_args.vpc_subnet_ids,
            "SecurityGroupIds": input_args.vpc_security_group_ids
        }

    create_model_response = sm_client.create_model(**create_model_kwargs)
    logging.debug('Model ARN: %s', create_model_response['ModelArn'])

    # ================= create an endpoint configuration ================= #
    sm_ep_config_name = model_name + "-EndpointConfig-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    sm_ep_config_name = sm_ep_config_name[:62]

    # Create production variant configuration
    default_production_variant = {
        "InstanceType": input_args.instance_type,
        "InitialInstanceCount": input_args.instance_count,
        "InitialVariantWeight": 1,
        "ModelName": sm_model_name,
        "VariantName": "AllTraffic",
    }
    if input_args.custom_production_variant:
        # merge custom ProductionVariant to the default key arguments, and overwrite same keys
        production_variant = {**default_production_variant, **input_args.custom_production_variant}
    else:
        production_variant = default_production_variant

    # Create endpoint config (VPC config is handled at model level)
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=sm_ep_config_name,  # SageMaker has 63 char limit
        ProductionVariants=[production_variant]
    )
    endpoint_arn = create_endpoint_config_response["EndpointConfigArn"]
    logging.debug("Endpoint config Arn: %s", endpoint_arn)

    # ================= create an endpoint ================= #
    sm_ep_name = model_name + "-Endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    sm_ep_name = sm_ep_name[:62] # SageMaker has 63 char limit

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=sm_ep_name, EndpointConfigName=sm_ep_config_name
    )
    logging.debug("Endpoint Arn: %s", create_endpoint_response["EndpointArn"])

    if input_args.async_execution.lower() == 'true':
        resp = sm_client.describe_endpoint(EndpointName=sm_ep_name)
        status = resp["EndpointStatus"]
        print(
            f"Creating endpoint name: '{sm_ep_name}' in "
            f"{input_args.region} region, current status: {status}")
    else:
        print(
            f"Waiting for endpoint '{sm_ep_name}' to "
            f"be in service in {input_args.region} region...")
        waiter = sm_client.get_waiter("endpoint_in_service")

        try:
            waiter.wait(EndpointName=sm_ep_name,
                        WaiterConfig={
                            'Delay': 30,        # seconds between querying
                            'MaxAttempts': 60   # max retries (~30 minutes here)
                        })
            print(
                (f"Endpoint named '{sm_ep_name}' has been successfully created, and ready to be "
                 "invoked!"), )
        except WaiterError as e:
            logging.error("Timed out while creating  endpoint '%s'  "
                          "or endpoint creation failed with reason: %s", sm_ep_name, e)

    return sm_ep_name

def get_realtime_infer_parser():
    """
    Get GraphStorm realtime task parser.
    """
    realtime_infer_parser = argparse.ArgumentParser("GraphStorm Inference Args")

    # SageMaker specific arguments
    realtime_infer_parser.add_argument("--region", type=str, required=True,
        help=("AWS region to launch jobs in. Make sure this region is where the inference "
             "image, and model tar file are located!"))
    realtime_infer_parser.add_argument("--image-uri", type=str, required=True,
        help="GraphStorm SageMaker Inference docker image URI")
    realtime_infer_parser.add_argument("--role", type=str, required=True,
        help="SageMaker execution role to use for the endpoint")
    realtime_infer_parser.add_argument("--instance-type", type=str, default="ml.c6i.xlarge",
        help="instance type for the SageMaker Inference Endpoint")
    realtime_infer_parser.add_argument("--instance-count", type=int, default=1,
        help="Number of inference endpoint instances.")
    realtime_infer_parser.add_argument("--custom-production-variant", type=json.loads,
        help=("A dictionary string that includes custom configurations of the SageMaker "
             "ProductionVariant. Used to identify which model to host and the resources "
             "chosen to deploy for hosting it. See documentation at "
        "https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html"))
    realtime_infer_parser.add_argument("--async-execution", type=str, default='false',
        choices=['True', 'true', 'False', 'false'],
        help=("Set to 'true' to create the endpoint asynchronously, 'false' to wait for creation. "
            "Default: 'false', will try to wait for endpoint to deploy before exiting."))

    # real-time task specific arguments
    realtime_infer_parser.add_argument("--restore-model-path", type=str, required=True,
        help="Path to GraphStorm trained model artifacts we'll use to create the endpoint.")
    realtime_infer_parser.add_argument("--model-yaml-config-file", type=str, required=True,
        help=("The file path to the new YAML configuration file generated in GraphStorm "
             "model training."))
    realtime_infer_parser.add_argument("--graph-json-config-file", type=str, required=True,
        help=("The file path to the updated JSON configuration file created in GraphStorm "
             "graph construction process. This should be available under the output of "
             "GConstruct/GSProcessing"))
    realtime_infer_parser.add_argument("--upload-tarfile-s3", type=str, required=True,
        help="The S3 location for uploading the packed and compressed model artifacts tar file.")
    realtime_infer_parser.add_argument("--infer-task-type", type=str, required=True,
        choices=SUPPORTED_REALTIME_INFER_TASKS,
        help=("The name of real time inference task. Accepted values are: "
               f"{SUPPORTED_REALTIME_INFER_TASKS}"))
    realtime_infer_parser.add_argument("--model-name", type=check_name_format,
        default='GSF-Model4Realtime',
        help=("The name for the to-be created SageMaker objects. The name should follow "
              r"a regular expression pattern: ^[a-zA-Z0-9]([\\-a-zA-Z0-9]*[a-zA-Z0-9])$. "
              "Default is \"GSF-Model4Realtime\". Note: Names will be truncated to 62 chars "
              "to comply with SageMaker's 63 character limit."))

    # Add optional VPC configuration arguments (applied at model level)
    realtime_infer_parser.add_argument("--vpc-subnet-ids", type=str, nargs='+',
        help=("Optional: List of VPC subnet IDs for the endpoint model. Required if deploying within "
              "a VPC. VPC configuration is applied at the model level to avoid endpoint conflicts."))
    realtime_infer_parser.add_argument("--vpc-security-group-ids", type=str, nargs='+',
        help=("Optional: List of security group IDs for the endpoint model. Required if deploying within "
              "a VPC. VPC configuration is applied at the model level to avoid endpoint conflicts."))

    # common arguments
    levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    realtime_infer_parser.add_argument('--log-level', default="INFO", choices=levels,
        help=(f"Change the log level. Optional values are {levels}. Default is 'INFO'."))

    return realtime_infer_parser

def sanity_check_realtime_infer_inputs(input_args):
    """ Verify the user-provided inputs for real-time endpoint deployment


    1. The endpoint should be deployed in the same region as the ECR Docker image.
    2. If VPC configuration is provided, both subnet IDs and security group IDs must be provided.

    """
    ecr_region = extract_ecr_region(input_args.image_uri)
    if ecr_region != input_args.region:
        raise ValueError(
            f'The given Docker image {input_args.image_uri} '
            f'is in the region {ecr_region}, but is different from the --region argument: '
            f'{input_args.region}. The endpoint should be deployed at the same region as the image.'
        )

    # Validate VPC configuration
    if bool(input_args.vpc_subnet_ids) != bool(input_args.vpc_security_group_ids):
        raise ValueError(
            "Both --vpc-subnet-ids and --vpc-security-group-ids must be provided "
            "together when configuring VPC access."
        )

    # check if graph JSON has tokenize_hf transformation. if yes, return with not support error
    with open(input_args.graph_json_config_file, 'r') as f:
        graph_json = json.load(f)

    # TODO: remove this assertion after support tokenize in language models in later release
    # assert not has_tokenize_transformation(graph_json), (f'tokenize_hf transformation '
    #                                                      'and trained language model are not '
    #                                                      'supported on real-time inference'
    #                                                      'endpoints. Please use bert_hf '
    #                                                      'transformation to embed text attributes '
    #                                                      'first, and use the embeddings as one '
    #                                                      'type of feature in model training and '
    #                                                      'inference.')

if __name__ == "__main__":
    arg_parser = get_realtime_infer_parser()
    args = arg_parser.parse_args()

    sanity_check_realtime_infer_inputs(args)

    deploy_endpoint(args)
