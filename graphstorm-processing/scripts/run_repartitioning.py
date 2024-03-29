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

Run repartitioning job on processed data on SageMaker.

This script allows us to launch a SageMaker Processing job to
repartition the data produced by the graph processing job.

Usage:
    run_repartitioning.py --s3-input-prefix <s3--prefix> \\
        --config-file metadata.json --image-uri <image-uri> --role <role>

Script Parameters
----------
Required:

--s3-input-prefix: str
    S3 path prefix to the input data.
--config-filename: str
    Distributed partitioning metadata JSON file. Created by the processing job,
    the default name should be `metadata.json`.
--image-uri: str
    ECR image URI for the GSProcessing container. Should be in the form:
    ``ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/graphstorm-processing:VERSION``
--role: str
    ARN of the SageMaker execution IAM role to use for the processing job.
    Should be in the form: `a`rn:aws:iam::ACCOUNT_ID:role/ROLE_NAME``

Optional:
--region: str
    AWS region for the processing job. If not provided will use the input bucket's region.
--instance-type: str
    EC2 instance type for the processing job. (Default: 'ml.r5.4xlarge')
--job-name: str
    Prefix name for the processing job. (Default: 'gsprocessing-repartitioning')
--container-log-level: str
    Log level for the container. (Default: INFO)
--host-log-level: str
    Log level for the host. (Default: INFO)
--sm-processor-parameters
    Parameters to be passed to the SageMaker Estimator as kwargs,
    in <key>=<val> format, separated by spaces.
    When passing lists or other objects, do not include spaces in the <val>, e.g.
    ``--sm-estimator-parameters \"volume_size=100 subnets=['subnet-123','subnet-345']
    security_group_ids=['sg-1234','sg-3456']\"``

"""

import argparse
import logging
from time import strftime, gmtime

import boto3
from botocore.config import Config
import sagemaker
from sagemaker.processing import Processor

import script_utils  # pylint: disable=wrong-import-order


def parse_args() -> argparse.Namespace:
    """Parse repartitioning args"""
    parser = script_utils.get_common_parser()  # type: argparse.ArgumentParser

    parser.add_argument(
        "--streaming-repartitioning",
        type=lambda x: (str(x).lower() in ["true", "1"]),
        default=False,
        help="When True will use low-memory file-streaming repartitioning. "
        "Note that this option is much slower than the in-memory default.",
        choices=["True", "False", "1", "0"],
    )
    parser.add_argument(
        "--updated-metadata-file-name",
        type=str,
        help="The name for the updated metadata file.",
        default="updated_row_counts_metadata.json",
    )

    return parser.parse_args()


def main():
    """Main entry point for re-partitioning."""
    args = parse_args()
    logging.basicConfig(level=args.host_log_level)
    # Remove trailing slash from s3_input_prefix
    raw_prefix = args.s3_input_prefix
    s3_input_prefix = raw_prefix[:-1] if raw_prefix[-1] == "/" else raw_prefix
    bucket = s3_input_prefix.split("/")[2]
    key_prefix = s3_input_prefix.split("/", maxsplit=3)[3]
    region = script_utils.get_bucket_region(bucket) if not args.region else args.region

    container_args = [
        "--input-prefix",
        s3_input_prefix,
        "--streaming-repartitioning",
        "True" if args.streaming_repartitioning else "False",
        "--input-metadata-file-name",
        args.config_filename,
        "--log-level",
        args.container_log_level,
        "--updated-metadata-file-name",
        args.updated_metadata_file_name,
    ]

    if args.job_name is None:
        args.job_name = "gsprocessing-repartitioning"

    timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    processing_job_name = f"{args.job_name}-{timestamp}"
    if len(processing_job_name) > 63:
        # Timestamp will be 19 chars, so we take 43 chars from the prefix plus hyphen
        processing_job_name = f"{args.job_name[:43]}-{timestamp}"

    logging.info("Creating repartitioning job for processed data under %s", s3_input_prefix)

    # Create SageMaker and S3 clients with high retry count to avoid thresholding errors
    if not region:
        region = boto3.session.Session().region_name
    sm_boto, s3_boto = [
        boto3.client(
            service,
            config=Config(
                region_name=region, connect_timeout=5, read_timeout=60, retries={"max_attempts": 20}
            ),
            region_name=region,
        )
        for service in ["sagemaker", "s3"]
    ]
    sagemaker_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=region), sagemaker_client=sm_boto
    )

    processor_kwargs = script_utils.parse_processor_kwargs(args.sm_processor_parameters)

    # Instance configuration
    byte_size_on_s3 = script_utils.determine_byte_size_on_s3(bucket, key_prefix, s3_boto)
    input_total_size_in_gb = max(1, byte_size_on_s3 // (1024 * 1024 * 1024))
    max_allowed_volume_size = script_utils.get_max_volume_size_for_processing(region)
    # Heuristic, we assume data_size x 2 should suffice for processing
    if "volume_size_in_gb" not in processor_kwargs:
        desired_volume_size = 2 * input_total_size_in_gb
    else:
        desired_volume_size = processor_kwargs["volume_size_in_gb"]
        processor_kwargs.pop("volume_size_in_gb")
    if desired_volume_size > max_allowed_volume_size:
        logging.warning(
            "Desired volume size (%d GB) is larger than max required, assigning the max: %d GB",
            desired_volume_size,
            max_allowed_volume_size,
        )
    # Ensure we don't request volume larger than max available
    capped_volume_size = min(desired_volume_size, max_allowed_volume_size)
    # Ensure we request at least 30GB for volume
    requested_volume_size_gb = max(capped_volume_size, 30)
    logging.info(
        "Total data size <= %d GB, assigning %d GB storage",
        input_total_size_in_gb,
        requested_volume_size_gb,
    )

    script_utils.check_if_instances_available(args.instance_type, instance_count=1, region=region)

    script_processor = Processor(
        entrypoint=["gs-repartition"],
        role=args.role,
        instance_type=args.instance_type,
        instance_count=1,
        image_uri=args.image_uri,
        sagemaker_session=sagemaker_session,
        volume_size_in_gb=requested_volume_size_gb,
        **processor_kwargs,
    )

    script_processor.run(
        arguments=container_args,
        job_name=processing_job_name,
        wait=args.wait_for_job,
        logs=args.wait_for_job,
    )


if __name__ == "__main__":
    main()
