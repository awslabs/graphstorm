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

    Launch a custom model SageMaker endpoint for realtime inference

    This tool is provided as a coutesy to help users deploy a SageMaker endpoint for real-time
    inference with your own GraphStorm-based models.
    It is users responsibility to make sure that model artifacts and related entry
    point file follow SageMaker's specifications, e.g., the deployment document at
    https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html.

"""

import os
import argparse
import logging
import json
from time import gmtime, strftime
from uuid import uuid4

import boto3 # pylint: disable=import-error
from botocore.exceptions import WaiterError


def run_job(input_args):
    """ This job works on custom model only

    Users need to read SageMaker's document in
    https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html
    to deploy custom model for real-time inference. And prepare model artifacts, including the
    entry point file, and compress and zip them into a tar file. To create an endpoint, this
    tar file should be saved into an Amazon S3 location. The S3 URL of the location and the
    entry point file name are required to run this endpoint deployment job.

    Parameters:
    -----------
    region: str
        The AWS region where the SageMaker endpoint will be deployed.
    image_url: str
        The URL of a GraphStorm SageMaker real-time inference Docker image that is located at
        Amazon ECR in the same region specified in the `region` argument.
    role: str
        The ARN string of an AWS account ARN that has SageMaker execution and model registry full
        access role.
    instance_type: str
        The string type of a SageMaker instance type. The default value is \"ml.c6i.xlarge\".
    instance_count: int
        The number of SageMaker instances to be deployed for the endpoint.
    custom_production_variant: dict
        The dictionary that inludes custom configuration of the SageMaker ProductionVarient
        for identifying a model to host and the resources chosen to deploy for hosting it, as
        documented at
        https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html.
    async_execution: str
        The string boolean value, determining if using asynchronous execution mode to creating
        endpoint. Options include "True" and "true" for True, "False" and "false" for False.
        Default is True.
    model_name: str
        A string to define the name of a SageMaker inference model. This name string must follow
        SageMaker's naming format, which is ^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$. The default
        value is \"GSF-Model4Realtime\".
    model_tarfile_s3: str
        The S3 location where a custom model tar file is stored. If provided, this script will
        use this model tarfile to deploy a SageMaker endpoint. This S3 location should be in the
        same region specified in the `region` argument.
    entrypoint_file_name: str
        The name of a Python file. SageMaker will use this file as the entry point of an endpoint.
        This is name is required if the `model_tarfile_s3` is provided. If not, this script will
        GraphStorm's default entry point file according to the `infer_task_type`.
    
    """
    # ================= prepare model artifacts ================= #
    entrypoint_file_name = input_args.entrypoint_file_name
    model_url_s3 = input_args.model_tarfile_s3
    # if not provide model_name, will create a random name
    if input_args.model_name:
        model_name = input_args.model_name
    else:
        model_name = uuid4().hex[:8]

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
    logging.debug('Model ARN: %s', create_model_response['ModelArn'])

    # ================= create an endpoint configuration ================= #
    sm_ep_config_name = model_name + "-EndpointConfig-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    default_product_variant = {
                    "InstanceType": input_args.instance_type,
                    "InitialInstanceCount": input_args.instance_count,
                    "InitialVariantWeight": 1,
                    "ModelName": sm_model_name,
                    "VariantName": "AllTraffic",
                }
    if input_args.custom_production_variant:
        # merge custom ProductionVariant to the default key arguments, and overwrite same keys
        product_variant = {**default_product_variant, **input_args.custom_production_variant}
    else:
        product_variant = default_product_variant

    create_endpoint_config_response = sm_client.create_endpoint_config(
            EndpointConfigName=sm_ep_config_name,
            ProductionVariants=[product_variant],
        )
    endpoint_arn = create_endpoint_config_response["EndpointConfigArn"]
    logging.debug("Endpoint config Arn: %s", endpoint_arn)

    # ================= create an endpoint ================= #
    sm_ep_name = model_name + "-Endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=sm_ep_name, EndpointConfigName=sm_ep_config_name
    )
    logging.debug("Endpoint Arn: %s", create_endpoint_response["EndpointArn"])

    if input_args.async_execution.lower() == 'true':
        resp = sm_client.describe_endpoint(EndpointName=sm_ep_name)
        status = resp["EndpointStatus"]
        logging.info("Creating endpoint name: %s, current status: %s", sm_ep_name, status)
    else:
        logging.info("Waiting for %s endpoint to be in service...", sm_ep_name)
        waiter = sm_client.get_waiter("endpoint_in_service")

        try:
            waiter.wait(EndpointName=sm_ep_name,
                        WaiterConfig={
                            'Delay': 30,        # seconds between querying
                            'MaxAttempts': 60   # max retries (~30 minutes here)
                        })
            logging.info('%s endpoint has been successfully created, and ready to be \
                    invoked!', sm_ep_name)
        except WaiterError as e:
            logging.error("Waiter timed out or endpoint creation failed: %s", e)


def get_realtime_infer_parser():
    """
    Get GraphStorm real-time inference task parser.
    """
    realtime_infer_parser = argparse.ArgumentParser("GraphStorm Custom Model Real-time " + \
                                                     "Inference Args")

    # SageMaker specific arguments
    realtime_infer_parser.add_argument("--region", type=str, required=True,
        help="AWS region to launch jobs in. Make sure this region is where the inference image, \
             and model tar file are located!")
    realtime_infer_parser.add_argument("--image-url", type=str, required=True,
        help="GraphStorm SageMaker docker image URI")
    realtime_infer_parser.add_argument("--role", type=str, required=True,
        help="SageMaker execution role")
    realtime_infer_parser.add_argument("--instance-type", type=str, default="ml.c6i.xlarge",
        help="instance type for the SageMaker job")
    realtime_infer_parser.add_argument("--instance-count", type=int, default=1,
        help="number of instances")
    realtime_infer_parser.add_argument("--custom-production-variant", type=json.loads,
        help="A dictionary string that inludes custom configurations of the SageMaker " + \
             "ProductionVarient for identifying a model to host and the resources " + \
             "chosen to deploy for hosting it as documented at " + \
        "https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariant.html")
    realtime_infer_parser.add_argument("--async-execution", type=str, default='true', 
        choices=['True', 'true', 'False', 'false'],
        help="Determine if f using asynchronous execution mode to creating endpoint. Options" + \
             "include \"True\" and \"true\" for True, \"False\" and \"false\" for False.")
    realtime_infer_parser.add_argument("--model-name", type=str,
        help=r"The name for the to-be created SageMaker objects. The name should follow \
              a regular expression pattern: ^[a-zA-Z0-9]([\-a-zA-Z0-9]*[a-zA-Z0-9])$. Default \
              is \"GSF-Model4Realtime\".")

    # customized model specific arguments
    realtime_infer_parser.add_argument("--model-tarfile-s3", type=str, required=True,
        help="The S3 location of the compressed model tar file to be used to create a SageMaker \
              Model.")
    realtime_infer_parser.add_argument("--entrypoint-file-name", type=str, required=True,
        help="The name of the model entry point file. This file name will be used when you \
              specify the --model-tarfile-s3 argument to use a pre-uploaded model tar file.")

    return realtime_infer_parser


if __name__ == "__main__":
    arg_parser = get_realtime_infer_parser()
    args = arg_parser.parse_args()
    print(f"Custom model real-time endpoint launch args: '{args}'")

    run_job(args)
