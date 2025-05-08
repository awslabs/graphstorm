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

    Launch a SageMaker endpoint for realtime inference
"""

import os
import argparse
import logging
import tempfile
import boto3 # pylint: disable=import-error
import sagemaker as sm
from time import gmtime, strftime
from botocore.exceptions import WaiterError

from launch_utils import (wrap_model_artifacts,
                          check_tarfile_s3_object,
                          upload_data_to_s3,
                          check_name_format)


# TODO: When adding new realtime inference tasks, modify this list
SUPPORTED_REALTIME_INFER_TASKS = ['node_classification']

# Constants for SageMaker endpoints
ENTRY_FOLDER_NAME = 'realtime_entry_points'
# TODO: When add new realtime inference tasks, modify this list
ENTRY_FILE_NAMES = [
    'node_classification_entry.py',
    ]
DEFAULT_GS_MODLE_FILE_NAME = 'model.pt'


def run_job(input_args):
    """ The procedure of deploying a SageMaker real-time inference endpoint
    
    Following SageMaker's document for deploying model for real-time inference in
    https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html,
    the general steps of the deployment include:
    1. prepare model artifacts;
    2. create a deployable model;
    3. create an endpoint configuration based on the deployable model;
    4. create an endpoint based on the endpoint configuration.

    This job follows these steps.
    """
    # ================= prepare model artifacts ================= #
    # prepare sessions
    b3_session = boto3.Session(region_name=input_args.region)
    sm_session = sm.Session(boto_session=b3_session)
    
    # If users provide the S3 tar file path already, directly use it for model creation
    if input_args.model_tarfile_s3:
        entrypoint_file_name = input_args.entrypoint_file_name

        model_url_s3 = input_args.model_tarfile_s3
        assert check_tarfile_s3_object(args.model_tarfile_s3) == True, \
            (f'Not find a tar file in the given S3 URL: {args.model_tarfile_s3}...')

        # model_name is either directly given by users, or had been assigned by the argument
        # validation function
        model_name = input_args.model_name
    else:
        # prepare task dependent variables
        entry_point_dir = os.path.join(os.path.dirname(__file__), ENTRY_FOLDER_NAME)

        # TODO: When adding new realtime inference tasks, add new elif here to support them
        if input_args.infer_task_type == SUPPORTED_REALTIME_INFER_TASKS[0]: # node_classification
            entrypoint_file_name = list(ENTRY_FILE_NAMES)[0]
            path_to_entry = os.path.join(entry_point_dir, entrypoint_file_name)
        else:
            raise NotImplementedError(f'The given real-time inference task \
                                        {input_args.infer_task_type} is not supported.')

        path_to_model = os.path.join(input_args.restore_model_path, DEFAULT_GS_MODLE_FILE_NAME)
        path_to_yaml = input_args.model_yaml_config_file
        path_to_json = input_args.graph_json_config_file
        model_name = input_args.model_name

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_tarfile_path = wrap_model_artifacts(path_to_model, path_to_yaml, path_to_json,
                                                    path_to_entry, tmpdirname,
                                                    output_tarfile_name=model_name)
            logging.info('Packed and compressed model artifacts into {}.'.format(model_tarfile_path))

            # upload the model tar file to the given S3 bucket
            model_url_s3 = upload_data_to_s3(input_args.upload_tarfile_s3, model_tarfile_path,
                                            sm_session)
            logging.info('Uploaded the model tar file to {}'.format(model_url_s3))

    # ================= create deployable model ================= #
    image_url = input_args.image_url
    role = input_args.role
    sm_client = boto3.client(service_name="sagemaker", region_name=input_args.region)

    container = {
        "Image": image_url,
        "ModelDataUrl": model_url_s3,
        "Environment": {"SAGEMAKER_PROGRAM": entrypoint_file_name}
    }

    sm_model_name = model_name + '-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    create_model_response = sm_client.create_model(ModelName=sm_model_name,
                                                   ExecutionRoleArn=role, Containers=[container])
    logging.info('Model ARN: {}'.format(create_model_response['ModelArn']))

    # ================= create an endpoint configuration ================= #
    sm_ep_config_name = model_name + "-EndpointConfig-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=sm_ep_config_name,
            ProductionVariants=[
                {
                    "InstanceType": input_args.instance_type,
                    "InitialInstanceCount": input_args.instance_count,
                    "InitialVariantWeight": 1,
                    "ModelName": sm_model_name,
                    "VariantName": "AllTraffic",
                }
            ],
        )

    logging.info("Endpoint config Arn: {}".format(create_endpoint_config_response["EndpointConfigArn"]))

    # ================= create an endpoint ================= #
    sm_ep_name = model_name + "-Endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=sm_ep_name, EndpointConfigName=sm_ep_config_name
    )
    logging.info("Endpoint Arn: {}".format(create_endpoint_response["EndpointArn"]))

    resp = sm_client.describe_endpoint(EndpointName=sm_ep_name)
    status = resp["EndpointStatus"]
    logging.info("Endpoint Status: " + status)

    logging.info("Waiting for {} endpoint to be in service...".format(sm_ep_name))
    waiter = sm_client.get_waiter("endpoint_in_service")
    
    try:
        waiter.wait(EndpointName=sm_ep_name,
                    WaiterConfig={
                        'Delay': 30,        # seconds between querying
                        'MaxAttempts': 60   # max retries (~30 minutes here)
                    })
        logging.info('{} endpoint has been successfully created, and ready to be \
                 invoked!'.format(sm_ep_name))
    except WaiterError as e:
        logging.error("Waiter timed out or endpoint creation failed:", e)


def get_realtime_infer_parser():
    """
    Get GraphStorm realtime  task parser.
    """
    realtime_infer_parser = argparse.ArgumentParser("GraphStorm Inference Args")

    # SageMaker specific arguments
    realtime_infer_parser.add_argument("--image-url", type=str, required=True,
        help="GraphStorm SageMaker docker image URI")
    realtime_infer_parser.add_argument("--role", type=str, required=True,
        help="SageMaker execution role")
    realtime_infer_parser.add_argument("--region", type=str, required=True,
        help="AWS region to launch jobs in. Make sure this region is where the inference image, \
             and model tar file are located!")
    realtime_infer_parser.add_argument("--instance-type", type=str, default="ml.c6i.xlarge",
        help="instance type for the SageMaker job")
    realtime_infer_parser.add_argument("--instance-count", type=int, default=1,
        help="number of instances")

    # real-time task specific arguments
    realtime_infer_parser.add_argument("--restore-model-path", type=str,
        help="The folder path where trained GraphStorm model parameters were saved.")
    realtime_infer_parser.add_argument("--model-yaml-config-file", type=str,
        help="The file path to the new YAML configuration file generated in GraphStorm \
              model training.")
    realtime_infer_parser.add_argument("--graph-json-config-file", type=str,
        help="The file path to the updated JSON configuration file created in GraphStorm \
              graph construction process.")
    realtime_infer_parser.add_argument("--upload-tarfile-s3", type=str,
        help="The S3 location for uploading the packed model artifacts tar file.")
    realtime_infer_parser.add_argument("--infer-task-type", type=str,
        choices=SUPPORTED_REALTIME_INFER_TASKS,
        help=(f"The name of real time inference task. Options \
               include {SUPPORTED_REALTIME_INFER_TASKS}"))
    realtime_infer_parser.add_argument("--model-name", type=check_name_format,
        default='GSF-Model4Realtime',
        help="The name for the to-be created SageMaker objects. The name should follow \
              a regular expression pattern: ^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$. Default \
              is \"GSF-Model4Realtime\".")

    # customized model specific arguments
    realtime_infer_parser.add_argument("--model-tarfile-s3", type=str,
        help="The S3 location of the compressed model tar file. If provided, it will be used \
              to create a SageMaker Model.")
    realtime_infer_parser.add_argument("--entrypoint-file-name", type=str,
        help="The name of the model entry point file. This file name will be used when you \
              sepcify the --model-tarfile-s3 argument to use a pre-uploaded model tar file.")

    return realtime_infer_parser

def validate_realtime_infer_arguments(args):
    """ Validate the input arguments with the designed launch logic
    
    The real-time inference endpoint lauch has the logic as the followings.
    
    1. If users provide the --model-tarfile-s3, it means it is a custom model. Then users must
       provide the --entrypoint-file-name too, which means users need to implement their own
       SageMaker entry point file, pack it with other model artifacts in to a compressed
       tar file, and upload to the given S3 url.
    2. If users want to use the default launch functions, they must provide five arguments:
       --restore-model-path
       --model-yaml-config-file
       --graph-json-config-file
       --infer-task-type
       --upload-tarfile-s3
    3. If users also provide a model name in the --model-name argument, the script will use it and
       check if it follows the SageMaker naming fomat.
       
       Note: Checking model_name format actually happens in the argument parsing stage. So there is no
              such check in this function.
    4. TODO(James): Will do sanity check of the contents of the given YAML, and JSON files when they
                    become available.

    """
    if args.model_tarfile_s3:
        assert args.entrypoint_file_name, ('To use your own model tar file, please set the \
            --entrypoint-file-name argument, and place the actual file into a subfolder, \
            named \'/code\', inside the tar file, as requested in the SageMaker document: \
            https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#model-directory-structure.')
    else:
        assert args.restore_model_path, f'To use GraphStorm default real-time inference ' + \
                     'endpoint launch script, please set --restore-model-path argument.'
        assert args.model_yaml_config_file, f'To use GraphStorm default real-time ' + \
                     'inference endpoint launch script, please set --model-yaml-config-file ' + \
                     'argument.'
        assert args.graph_json_config_file, f'To use GraphStorm default real-time ' + \
                     'inference endpoint launch script, please set --graph-json-config-file ' + \
                     'argument.'
        assert args.upload_tarfile_s3, f'To use GraphStorm default real-time ' + \
                     'inference endpoint launch script, please set --upload-tarfile-s3 ' + \
                     'argument.'
        assert args.infer_task_type, f'To use GraphStorm default real-time ' + \
                     'inference endpoint launch script, please set --infer-task-type ' + \
                     'argument.'


if __name__ == "__main__":
    arg_parser = get_realtime_infer_parser()
    args = arg_parser.parse_args()
    print(f"Real-time endpoint launch args: '{args}'")

    validate_realtime_infer_arguments(args)

    run_job(args)
