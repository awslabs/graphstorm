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

Executes a graph processing task using PySpark.

This script allows us to run a PySpark job to perform the
graph processing task, preparing the data for distributed
partitioning. It is used as an entry point script with the
`gs-processing` command.

Usage:
    distributed_executor.py --input-prefix <s3-or-local-prefix> \\
        --output-prefix <s3-or-local-prefix> \\
        --config-file <config-file>

Script Parameters
-----------------
--input-prefix: str
    S3 or local path prefix to the input data.
--output-prefix: str
    S3 or local path prefix to where the output data will be generated.
--config-filename: str
    GSProcessing configuration filename, needs to be under --input-prefix.
--num-output-files: int
    Number of output files Spark will create. Note that this affects
    the available parallelism for the graph processing task, so do not
    set this too low to ensure good performance. A good rule of thumb
    is to set this to 2*num_of_cores_in_cluster. If not provided we
    let Spark decide the parallelism level.
--log-level: str
    Log level for the script.
--add-reverse-edges: str
    When set to true (default), we add reverse edges for each edge type.

"""
import dataclasses
import argparse
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict
import tempfile

import boto3

from graphstorm_processing.graph_loaders.dist_heterogeneous_loader import (
    DistHeterogeneousGraphLoader,
)
from graphstorm_processing.config.config_parser import create_config_objects
from graphstorm_processing.config.config_conversion import GConstructConfigConverter
from graphstorm_processing.data_transformations import spark_utils


@dataclasses.dataclass
class ExecutorConfig:
    """Configuration  for the DistributedExecutor

    Parameters
    ----------
    local_config_path : str
        Local path to the config file.
    local_output_path : str
        Local path for output metadata files.
    input_prefix : str
        Prefix for input data. Can be S3 URI or local path.
    output_prefix : str
        Prefix for output data. Can be S3 URI or local path.
    num_output_files : int
        The number of output files Spark will try to create.
    sm_execution : bool
        Whether the execution context is a SageMaker container.
    config_filename : str
        The filename for the configuration file.
    filesystem_type : str
        The filesystem type, can be 'local' or 's3'.
    add_reverse_edges : bool
        Whether to create reverse edges for each edge type.
    """

    local_config_path: str
    local_output_path: str
    input_prefix: str
    output_prefix: str
    num_output_files: int
    sm_execution: bool
    config_filename: str
    filesystem_type: str
    add_reverse_edges: bool
    graph_name: str


@dataclasses.dataclass
class GSProcessingArguments:
    """Arguments for the graph processing task"""

    input_prefix: str
    output_prefix: str
    config_filename: str
    num_output_files: int
    add_reverse_edges: bool
    log_level: str
    graph_name: str


class DistributedExecutor:
    """
    Used as the entry point for distributed execution.

    Parameters
    ----------
    executor_config : ExecutorConfig
        Executor configuration object.
    """

    def __init__(
        self,
        executor_config: ExecutorConfig,
    ):
        self.local_config_path = executor_config.local_config_path
        self.local_output_path = executor_config.local_output_path
        self.input_prefix = executor_config.input_prefix
        self.output_prefix = executor_config.output_prefix
        self.num_output_files = executor_config.num_output_files
        self.config_filename = executor_config.config_filename
        self.filesystem_type = executor_config.filesystem_type
        self.sm_execution = executor_config.sm_execution
        self.add_reverse_edges = executor_config.add_reverse_edges
        self.graph_name = executor_config.graph_name

        # Ensure we have write access to the output path
        if self.filesystem_type == "local":
            if not os.path.exists(self.output_prefix):
                try:
                    os.makedirs(self.output_prefix, exist_ok=True)
                except OSError as e:
                    logging.error("Unable to create output path: %s", e)
                    raise e
        else:
            # Ensure we can read and write files from/to the S3 prefix
            s3 = boto3.resource("s3")
            bucket_name = self.output_prefix.split("/")[2]
            prefix = self.output_prefix.split("/", 3)[3]
            head_bucket_response = s3.meta.client.head_bucket(Bucket=bucket_name)
            assert head_bucket_response["ResponseMetadata"]["HTTPStatusCode"] == 200
            bucket_resouce = s3.Bucket(bucket_name)
            bucket_resouce.put_object(Key=f"{prefix}/test_file.txt", Body=b"test")
            response = bucket_resouce.delete_objects(
                Delete={"Objects": [{"Key": f"{prefix}/test_file.txt"}], "Quiet": True}
            )
            assert "Deleted" in response

        graph_conf = os.path.join(self.local_config_path, self.config_filename)
        with open(graph_conf, "r", encoding="utf-8") as f:
            dataset_config_dict: Dict[str, Any] = json.load(f)

        if "version" in dataset_config_dict:
            self.config_version = dataset_config_dict["version"]
            if self.config_version != "gsprocessing-v1.0":
                logging.warning("Unrecognized version name: %s", self.config_version)
            self.graph_config_dict = dataset_config_dict["graph"]
        else:
            # TODO: Change once GConstruct adds a version to their config spec
            self.config_version = "gconstruct"
            converter = GConstructConfigConverter()
            self.graph_config_dict = converter.convert_to_gsprocessing(dataset_config_dict)["graph"]

        # Create the Spark session for execution
        self.spark = spark_utils.create_spark_session(self.sm_execution, self.filesystem_type)

    def run(self) -> None:
        """
        Executes the Spark processing job.
        """
        logging.info("Performing data processing with PySpark...")
        data_configs = create_config_objects(self.graph_config_dict)

        t0 = time.time()
        logging.info("Constructing DGLGraph for Heterogeneous Graph")
        # Prefer explicit arguments for clarity
        loader = DistHeterogeneousGraphLoader(
            spark=self.spark,
            local_input_path=self.local_config_path,
            local_output_path=self.local_output_path,
            data_configs=data_configs,
            input_prefix=self.input_prefix,
            output_prefix=self.output_prefix,
            num_output_files=self.num_output_files,
            add_reverse_edges=self.add_reverse_edges,
            enable_assertions=False,
            graph_name=self.graph_name,
        )
        loader.load()

        # This is used to upload the output JSON files to S3 on local runs,
        # since we can't rely on SageMaker to do it
        if not self.sm_execution and self.filesystem_type == "s3":
            bucket = self.output_prefix.split("/")[2]
            s3_prefix = self.output_prefix.split("/", 3)[3]
            s3 = boto3.resource("s3")

            output_files = os.listdir(loader.output_path)
            for output_file in output_files:
                s3.meta.client.upload_file(
                    f"{os.path.join(loader.output_path, output_file)}",
                    bucket,
                    f"{s3_prefix}/{output_file}",
                )

        t1 = time.time()
        logging.info("[Prof-info] graph loading time %s", t1 - t0)
        self.spark.stop()


def parse_args() -> argparse.Namespace:
    """Parse the arguments for the execution."""
    parser = argparse.ArgumentParser(description="GraphStorm Processing Entry Point")
    parser.add_argument(
        "--input-prefix",
        type=str,
        help="Path under which input data files are expected. "
        "Can be a local path or S3 prefix (starting with 's3://').",
        required=True,
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        help="Path under which output will be created. "
        "Can be a local path or S3 prefix (starting with 's3://').",
        required=True,
    )
    parser.add_argument(
        "--config-filename",
        type=str,
        help="GSProcessing data configuration filename.",
        required=True,
    )
    parser.add_argument(
        "--num-output-files",
        type=int,
        default=None,
        help=(
            "The desired number of output files to be crea ted per type. "
            "If set to '', None, or '-1', we let Spark decide."
        ),
    )
    parser.add_argument(
        "--add-reverse-edges",
        type=lambda x: (str(x).lower() in ["true", "1"]),
        default=True,
        help="When set to True, will create reverse edges for every edge type.",
    )
    parser.add_argument(
        "--graph-name",
        type=str,
        help="Name for the graph being processed.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level, default is INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser.parse_args()


def main():
    """Main entry point for GSProcessing"""
    # Allows us to get typed arguments from the command line
    gsprocessing_args = GSProcessingArguments(**vars(parse_args()))
    logging.basicConfig(level=gsprocessing_args.log_level)

    # Determine if we're running within a SageMaker container
    is_sagemaker_execution = os.path.exists("/opt/ml/config/processingjobconfig.json")

    if gsprocessing_args.input_prefix.startswith("s3://"):
        assert gsprocessing_args.output_prefix.startswith(
            "s3://"
        ), "When providing S3 input and output prefixes, they must both be S3."
        filesystem_type = "s3"
    else:
        # Ensure input and output prefixes exist and convert to absolute paths
        gsprocessing_args.input_prefix = str(
            Path(gsprocessing_args.input_prefix).resolve(strict=True)
        )
        if not Path(gsprocessing_args.output_prefix).absolute().exists():
            os.makedirs(gsprocessing_args.output_prefix)
        gsprocessing_args.output_prefix = str(
            Path(gsprocessing_args.output_prefix).resolve(strict=True)
        )
        filesystem_type = "local"

    # local input location for config file and execution script
    if is_sagemaker_execution:
        local_config_path = "/opt/ml/processing/input/data"
    else:
        # If not on SageMaker, assume that we are running in a
        # native env with local input or Docker execution with S3 input
        if filesystem_type == "local":
            local_config_path = gsprocessing_args.input_prefix
        else:
            tempdir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
            input_bucket = gsprocessing_args.input_prefix.split("/")[2]
            input_s3_prefix = gsprocessing_args.input_prefix.split("/", 3)[3]
            s3 = boto3.client("s3")
            s3.download_file(
                input_bucket,
                f"{input_s3_prefix}/{gsprocessing_args.config_filename}",
                os.path.join(tempdir.name, gsprocessing_args.config_filename),
            )
            local_config_path = tempdir.name

    # local output location for metadata files
    if is_sagemaker_execution:
        local_output_path = "/opt/ml/processing/output"
    else:
        if filesystem_type == "local":
            local_output_path = gsprocessing_args.output_prefix
        else:
            # Only needed for local execution with S3 data
            local_output_path = tempdir.name

    if gsprocessing_args.num_output_files == "" or gsprocessing_args.num_output_files is None:
        gsprocessing_args.num_output_files = -1

    executor_configuration = ExecutorConfig(
        local_config_path=local_config_path,
        local_output_path=local_output_path,
        input_prefix=gsprocessing_args.input_prefix,
        output_prefix=gsprocessing_args.output_prefix,
        num_output_files=gsprocessing_args.num_output_files,
        config_filename=gsprocessing_args.config_filename,
        sm_execution=is_sagemaker_execution,
        filesystem_type=filesystem_type,
        add_reverse_edges=gsprocessing_args.add_reverse_edges,
        graph_name=gsprocessing_args.graph_name,
    )

    dist_executor = DistributedExecutor(executor_configuration)

    dist_executor.run()

    # Save arguments to file for posterity
    with open(os.path.join(local_output_path, "launch_arguments.json"), "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(gsprocessing_args), f, indent=4)

    # In SageMaker execution, all files under `local_output_path` get automatically
    # uploaded to S3 at the end of the job. For local execution with S3 data, we
    # need to upload all output files manually.
    if not is_sagemaker_execution and filesystem_type == "s3":
        output_bucket = gsprocessing_args.output_prefix.split("/")[2]
        output_s3_prefix = gsprocessing_args.output_prefix.split("/", 3)[3]
        s3 = boto3.resource("s3")
        s3.meta.client.upload_file(
            os.path.join(local_output_path, "launch_arguments.json"),
            output_bucket,
            f"{output_s3_prefix}/launch_arguments.json",
        )


if __name__ == "__main__":
    main()
