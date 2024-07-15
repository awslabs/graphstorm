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

Run graph processing job on SageMaker.

Usage:
    run_distributed_processing.py --s3-input-prefix <s3-prefix> \\
        --s3-output-prefix <s3-prefix> --config-file <config-file> \\
        --image-uri <image-uri> --role <role>

Script Parameters
----------
Required:

--s3-input-prefix: str
    S3 or prefix to the input data.
--s3-output-prefix: str
    S3 or prefix to the input data.
--config-filename: str
    GSProcessing configuration filename. Needs to be present under the
    s3-input-prefix.
--image-uri: str
    ECR image URI for the GSProcessing container. Should be in the form:
    ``ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/graphstorm-processing:VERSION``
--role: str
    ARN of the SageMaker execution IAM role to use for the processing job.
    Should be in the form: ``arn:aws:iam::ACCOUNT_ID:role/ROLE_NAME``

Optional:
--region: str
    AWS region for the processing job. If not provided will use the output bucket's region.
--instance-count: int
    Number of EC2 instances to use for the processing job. (Default: 2)
--instance-type: str
    EC2 instance type for the processing job. (Default: 'ml.r5.4xlarge')
--add-reverse-edges: str
    When set to "True", will create reverse edges for every edge type. (Default: "True")
--do-repartition: str
    When set to "True", will repartition the graph files on the leader node if needed.
    (Default: "True")
--num-output-files: int
    Number of output files to generate. If not specified, the number of output files
    will be determined by Spark.
--job-name: str
    Prefix name for the processing job. (Default: 'gs-processing')
--code-path: str
    Path to the code directory under which distributed_executor.py exists.
--container-log-level: str
    Log level for the container. (Default: INFO)
--host-log-level: str
    Log level for the host. (Default: INFO)
--sm-processor-parameters
    Parameters to be passed to the SageMaker Estimator as kwargs,
    in <key>=<val> format, separated by spaces. Do not include spaces in the values themselves,
    e.g.:
    --sm-estimator-parameters \"volume_size=100 subnets=['subnet-123','subnet-345']
    security_group_ids=['sg-1234','sg-3456']\"
"""

import argparse
import logging
from pathlib import Path
import os
from time import strftime, gmtime

import boto3
from botocore.config import Config
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.spark.processing import PySparkProcessor

import script_utils  # pylint: disable=wrong-import-order

_ROOT = os.path.abspath(os.path.dirname(__file__))


def parse_args() -> argparse.Namespace:
    """Parse dist processing args"""
    parser = script_utils.get_common_parser()  # type: argparse.ArgumentParser

    # Required arguments
    parser.add_argument(
        "--s3-output-prefix",
        type=str,
        help="Prefix to S3 path under which the output will be created.",
        required=True,
    )

    # Optional arguments
    parser.add_argument("--instance-count", type=int, default=2)
    parser.add_argument(
        "--add-reverse-edges",
        type=lambda x: (str(x).lower() in ["true", "1"]),
        default=True,
        help="When set to True, will create reverse edges for every edge type.",
    )
    parser.add_argument(
        "--do-repartition",
        type=lambda x: (str(x).lower() in ["true", "1"]),
        default=False,
        help="When set to True, will re-partition the graph data files on the Spark leader.",
    )
    parser.add_argument(
        "--num-output-files",
        type=str,
        default=None,
        help="Number of output files to generate. If not specified, the number of output files "
        "will be determined by Spark. NOTE: This setting affects the max parallelism available "
        "and can negatively affect the performance if set too low.",
    )
    parser.add_argument(
        "--code-path",
        default=None,
        help="Path to the code directory under which distributed_executor.py exists.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=args.host_log_level)
    # Remove trailing slash from S3 paths
    raw_input = args.s3_input_prefix
    raw_output = args.s3_output_prefix
    s3_input_prefix = raw_input[:-1] if raw_input[-1] == "/" else raw_input
    s3_output_prefix = raw_output[:-1] if raw_output[-1] == "/" else raw_output
    s3_train_config = f"{s3_input_prefix}/{args.config_filename}"

    code_path = args.code_path if args.code_path else Path(_ROOT).parent / "graphstorm_processing"

    if args.job_name is None:
        args.job_name = "gs-processing"

    instance_count = args.instance_count
    timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    processing_job_name = f"{args.job_name}-pyspark-{timestamp}"
    if len(processing_job_name) > 63:
        # Timestamp will be 19 chars, so we take 43 chars from the prefix plus hyphen
        processing_job_name = f"{args.job_name[:43]}-{timestamp}"

    s3_train_config = (
        f"{s3_input_prefix}{args.config_filename}"
        if s3_input_prefix.endswith("/")
        else f"{s3_input_prefix}/{args.config_filename}"
    )
    s3_input_bucket = s3_input_prefix.split("/")[2]
    s3_input_key = s3_input_prefix.split("/", maxsplit=3)[3]

    if not s3_output_prefix:
        s3_output_prefix = f"s3://{s3_input_bucket}/gs-processing/output/{processing_job_name}"

    s3_output_bucket = s3_output_prefix.split("/")[2]

    print(
        "Creating SageMaker Processing job for train"
        f" config in {s3_train_config} and outputs to {s3_output_prefix}"
    )

    # Create SageMaker session with high retry count to avoid thresholding errors
    args.region = (
        script_utils.get_bucket_region(s3_output_bucket) if not args.region else args.region
    )
    sm_boto, s3_boto = [
        boto3.client(
            service,
            config=Config(
                region_name=args.region,
                connect_timeout=5,
                read_timeout=60,
                retries={"max_attempts": 20},
            ),
            region_name=args.region,
        )
        for service in ["sagemaker", "s3"]
    ]
    sagemaker_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=args.region), sagemaker_client=sm_boto
    )

    processor_kwargs = script_utils.parse_processor_kwargs(args.sm_processor_parameters)

    max_allowed_volume_size = script_utils.get_max_volume_size_for_processing(args.region)
    if "volume_size_in_gb" not in processor_kwargs:
        byte_size_on_s3 = script_utils.determine_byte_size_on_s3(
            s3_input_bucket, s3_input_key, s3_boto
        ) // (1024 * 1024 * 1024)
        input_total_size_in_gb = max(1, byte_size_on_s3 // (1024 * 1024 * 1024))
        logging.info("Total data size: <= %d GB", input_total_size_in_gb)
        # Heuristic: total storage of 6+ times the total input size should be
        # sufficient for Spark (assuming CSV input)
        # Using anything less than 2*input_size/instance_count led to failures for large datasets
        desired_volume_size = 6 * (input_total_size_in_gb // instance_count)
    else:
        desired_volume_size = processor_kwargs["volume_size_in_gb"]
        processor_kwargs.pop("volume_size_in_gb")

    if desired_volume_size > max_allowed_volume_size:
        logging.warning(
            "Desired volume size (%d GB) is larger than max required, assigning the max: %d GB",
            desired_volume_size,
            max_allowed_volume_size,
        )
    # Ensure we don't request volume larger than max allowed
    capped_volume_size = min(desired_volume_size, max_allowed_volume_size)
    # Ensure we request at least 30GB for volume
    requested_gb_per_instance = max(capped_volume_size, 30)
    logging.info(
        "Assigning %d GB storage per instance (total storage: %d GB).",
        requested_gb_per_instance,
        instance_count * requested_gb_per_instance,
    )

    pyspark_processor = PySparkProcessor(
        role=args.role,
        instance_type=args.instance_type,
        instance_count=instance_count,
        image_uri=args.image_uri,
        sagemaker_session=sagemaker_session,
        volume_size_in_gb=requested_gb_per_instance,
        **processor_kwargs,
    )

    processing_inputs = [
        ProcessingInput(
            input_name="train_config",
            source=s3_train_config,
            destination="/opt/ml/processing/input/data",
        )
    ]
    processing_output = [
        ProcessingOutput(
            output_name="metadata", destination=s3_output_prefix, source="/opt/ml/processing/output"
        )
    ]

    # Arguments for distributed_executor.py
    container_args = [
        "--config-filename",
        args.config_filename,
        "--input-prefix",
        s3_input_prefix,
        "--output-prefix",
        s3_output_prefix,
        "--num-output-files",
        str(args.num_output_files),
        "--add-reverse-edges",
        "True" if args.add_reverse_edges else "False",
        "--do-repartition",
        "True" if args.do_repartition else "False",
        "--log-level",
        args.container_log_level,
    ]

    print(f"Starting PySpark Processing job named {processing_job_name}")
    pyspark_processor.run(
        submit_app=os.path.join(code_path, "distributed_executor.py"),
        inputs=processing_inputs,
        outputs=processing_output,
        arguments=container_args,
        job_name=processing_job_name,
        wait=args.wait_for_job,
        spark_event_logs_s3_uri=f"{s3_output_prefix}/spark-events",
    )
