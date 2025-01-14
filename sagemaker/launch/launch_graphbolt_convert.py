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

    Launch SageMaker GraphBolt conversion step
"""

import logging
import os
from pprint import pformat
from time import strftime, gmtime

import boto3  # pylint: disable=import-error
import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingInput

from common_parser import (  # pylint: disable=wrong-import-order
    get_common_parser,
    parse_estimator_kwargs,
)

INSTANCE_TYPE = "ml.m5.12xlarge"

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def run_gbconvert_job(input_args, image):
    """Run job using SageMaker estimator.PyTorch

        TODO: We may need to simplify the argument list. We can use a config object.

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    """
    # SageMaker base job name
    sm_task_name = input_args.task_name if input_args.task_name else "gs-gb-convert"
    role = input_args.role  # SageMaker ARN role
    instance_type = input_args.instance_type  # SageMaker instance type
    region = input_args.region  # AWS region
    graph_data_s3 = (
        input_args.graph_data_s3
    )  # S3 location storing input graph data (unpartitioned)
    graph_data_s3 = (
        graph_data_s3[:-1] if graph_data_s3[-1] == "/" else graph_data_s3
    )  # The input will be an S3 prefix without trailing /

    sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))

    logging.info("Parameters %s", pformat(input_args))
    if input_args.sm_estimator_parameters:
        logging.info("SageMaker Estimator parameters: '%s'",
                     input_args.sm_estimator_parameters)

    estimator_kwargs = parse_estimator_kwargs(input_args.sm_estimator_parameters)

    gb_convert_processor = ScriptProcessor(
        image_uri=image,
        role=role,
        instance_count=1,
        instance_type=instance_type,
        command=["python3"],
        base_job_name=sm_task_name,
        sagemaker_session=sagemaker_session,
        tags=[
            {"Key": "GraphStorm", "Value": "beta"},
            {"Key": "GraphStorm_Task", "Value": "GraphBoltConvert"},
        ],
        env={"AWS_REGION": region},
        **estimator_kwargs,
    )

    # The partition data will be download to this path on the instance
    local_dist_part_path = "/opt/ml/processing/dist_graph/"

    logging.info("Using source data %s", graph_data_s3)

    # TODO: We'd like to use FastFile here, but DGL makes the data conversion
    # in-place and we can't write to the location where S3 files are loaded
    gb_convert_processor.run(
        code=input_args.entry_point,
        inputs=[
            ProcessingInput(
                input_name="dist_graph_s3_input",
                destination=local_dist_part_path,
                source=graph_data_s3,
                s3_input_mode="File",
            )
        ],
        arguments=[
            "--njobs",
            str(input_args.gb_convert_njobs),
            "--metadata-filename",
            input_args.metadata_filename,
        ],
        wait=not input_args.async_execution,
    )


def get_partition_parser():
    """
    Get GraphStorm partition task parser.
    """
    parser = get_common_parser()

    partition_args = parser.add_argument_group("GraphStorm Partition Arguments")

    partition_args.add_argument(
        "--entry-point",
        type=str,
        default="run/gb_convert_entry.py",
        help="PATH-TO graphstorm/sagemaker/run/gb_convert_entry.py",
    )

    partition_args.add_argument(
        "--gb-convert-njobs",
        type=int,
        default=1,
        help="Number of parallel processes to use for GraphBolt partition conversion.",
    )

    partition_args.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata.json",
        help="Partition metadata file name. Default: 'metadata.json'. If GConstruct was "
        "used should be set to <graph-name>.json",
    )

    partition_args.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "CRITICAL", "FATAL"],
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_partition_parser()
    print(args)

    # NOTE: Ensure no logging has been done before setting logging configuration
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    partition_image = args.image_url
    if not args.instance_type:
        args.instance_type = INSTANCE_TYPE

    run_gbconvert_job(args, partition_image)
