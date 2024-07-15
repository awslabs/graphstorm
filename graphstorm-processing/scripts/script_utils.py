"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from ast import literal_eval
from typing import Any, Dict, Sequence, Optional

from botocore.config import Config
import boto3


def check_if_instances_available(instance_type: str, instance_count: int, region: str) -> None:
    """Checks if we have enough of a quota for the requested instance type and count in a region.

    Parameters
    ----------
    instance_type : str
        The type of the instance, e.g. m5.xlarge
    instance_count : int
        The number of instances we want to launch
    region : str
        The region we'll check the quota for, e.g. 'us-west-2'.

    Raises
    ------
    RuntimeError
        Raises a runtime error if the requested capacity exceeds the current quota.
    """
    quota_name = f"{instance_type} for processing job usage"
    quota_value = get_quota_value("sagemaker", quota_name, region)

    if int(quota_value) < instance_count:
        raise RuntimeError(
            f"Requested {instance_count=} goes"
            f" above current ({quota_value=}) for {instance_type=}"
        )

    # TODO: Use sagemaker_client.list_processing_jobs and
    # describe_processing_job to get current utilization


def get_quota_value(service_code: str, quota_name: str, region: str) -> float:
    """For a given service code, quota name, and region, return the quota value.

    Parameters
    ----------
    service_code : str
        The service code for the service we will get quotas value for.
        To find the service code value for an Amazon Web Services service,
        use the ListServices operation.
    quota_name : str
        The descriptive name of the quota, e.g. 'Number of instances for a processing job'
        See https://docs.aws.amazon.com/general/latest/gr/sagemaker.html#limits_sagemaker
        for a full list of SageMaker quotas.
    region : str
        _description_

    Returns
    -------
    float
        The region we'll check the quota for, e.g. 'us-west-2'.

    Raises
    ------
    RuntimeError
        If it wasn't possible to retrieve the service value.
    """

    def get_value_for_quota_name(
        quotas: Sequence[Dict[str, Any]], quota_name: str
    ) -> Optional[float]:
        for quota in quotas:
            if quota["QuotaName"] == quota_name:
                return quota["Value"]
        return None

    quota_client = boto3.client(
        "service-quotas",
        config=Config(
            region_name=region, connect_timeout=5, read_timeout=60, retries={"max_attempts": 20}
        ),
    )
    quota_paginator_client = quota_client.get_paginator("list_service_quotas")

    quota_response = quota_paginator_client.paginate(
        ServiceCode=service_code,
    )

    for quota_set in quota_response:
        quota_value = get_value_for_quota_name(quota_set["Quotas"], quota_name)
        if quota_value:
            break

    if quota_value is None:
        raise RuntimeError(
            f"Could not get quota value for {service_code=}, {quota_name=}, and {region=}"
        )
    return quota_value


def get_max_volume_size_for_processing(region: str) -> int:
    """Get the maximum allowed EBS volume size for a processing instance for the region, in GB.

    Parameters
    ----------
    region : str
        The region we'll check the quota for, e.g. 'us-west-2'.

    Returns
    -------
    int
        The maximum allowed EBS volume size for a processing instance for the region, in GB.
    """
    quota_name = "Size of EBS volume for a processing job instance"
    try:
        quota_value = int(get_quota_value("sagemaker", quota_name, region))
    except Exception:  # pylint: disable=broad-except
        # If we can't get the quota, just use the default value.
        quota_value = 1024
    return quota_value


def determine_byte_size_on_s3(bucket: str, prefix: str, s3_boto_client=None) -> int:
    """Returns the total byte size under all files under a S3 common prefix.

    Parameters
    ----------
    bucket : str
        The bucket under which we are checking.
    prefix : str
        The prefix key under which we will sum the object size.
    s3_boto_client : S3.Client, optional
        Optional boto S3 client, by default None

    Returns
    -------
    int
        Total size of all objects under the prefix, in bytes.
    """
    s3_boto_client = (
        boto3.client("s3", region_name=get_bucket_region(bucket))
        if s3_boto_client is None
        else s3_boto_client
    )
    paginator = s3_boto_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    total_object_size_in_bytes = 0
    for page in pages:
        if "Contents" in page.keys():
            for obj in page["Contents"]:
                total_object_size_in_bytes += int(obj["Size"])

    return total_object_size_in_bytes


def get_bucket_region(bucket: str, s3_boto_client=None) -> str:
    """
    Returns the region of the provided S3 bucket.
    """
    s3_boto_client = boto3.client("s3") if s3_boto_client is None else s3_boto_client
    response = s3_boto_client.head_bucket(Bucket=bucket)
    return response["ResponseMetadata"]["HTTPHeaders"]["x-amz-bucket-region"]


def get_common_parser() -> argparse.ArgumentParser:
    """
    Returns a common argument parser for all launch scripts.
    """
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(
        "--s3-input-prefix",
        type=str,
        help="S3 path to where the input data. E.g. " "'s3://my-bucket/my-gs-data'",
        required=True,
    )
    parser.add_argument(
        "--config-filename",
        type=str,
        help="Name of the config file under the input S3 path",
        required=True,
    )
    parser.add_argument(
        "--image-uri", type=str, help="GSProcessing image URI to use for processing", required=True
    )
    parser.add_argument(
        "--role",
        type=str,
        help="SageMaker Execution role to use for processing",
        required=True,
    )

    # Optional arguments
    parser.add_argument("--region", type=str, default=None)
    parser.add_argument("--instance-type", type=str, default="ml.r5.4xlarge")
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument(
        "--wait-for-job", action="store_true", help="Wait for the job to complete before exiting."
    )
    parser.add_argument(
        "--container-log-level",
        help="The log level the container will use during SageMaker execution",
        type=str,
        required=False,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--host-log-level",
        help="The host (local) log level for this script.",
        type=str,
        required=False,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--sm-processor-parameters",
        type=str,
        default=None,
        help=(
            "Parameters to be passed to the SageMaker Processor as kwargs, "
            "in <key>=<val> format, separated by spaces. "
            "Do not include spaces in the values themselves, e.g.: "
            "--sm-estimator-parameters \"volume_size=100 subnets=['subnet-123','subnet-345'] "
            "security_group_ids=['sg-1234','sg-3456']\". "
            "See https://sagemaker.readthedocs.io/en/stable/api/training/processing.html "
            "for a full list."
        ),
    )

    return parser


def parse_processor_kwargs(arg_string: Optional[str]) -> Dict[str, Any]:
    """
    Parses Processor arguments for SageMaker tasks.
    See
    https://sagemaker.readthedocs.io/en/stable/api/training/processing.html
    for a complete list of available arguments.
    Argument values are evaluated as Python
    literals using ast.literal_eval.

    :param arg_string: String of arguments in the form of
        <key>=<val> separated by spaces.
    :return: Dictionary of parsed arguments.
    """
    if arg_string is None:
        return {}
    typed_args_dict = {}
    for param in arg_string.split(" "):
        k, v = param.split("=")
        typed_args_dict[k] = literal_eval(v)

    return typed_args_dict
