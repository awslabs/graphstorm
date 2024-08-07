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
--do-repartition: str
    When set to true, we try to perform the row count alignment step on
    the Spark leader.
"""

import argparse
import copy
import dataclasses
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from collections.abc import Mapping
from typing import Any, Dict

import boto3
import botocore

from graphstorm_processing.graph_loaders.dist_heterogeneous_loader import (
    DistHeterogeneousGraphLoader,
    HeterogeneousLoaderConfig,
    ProcessedGraphRepresentation,
)
from graphstorm_processing.config.config_parser import create_config_objects
from graphstorm_processing.config.config_conversion import GConstructConfigConverter
from graphstorm_processing.constants import TRANSFORMATIONS_FILENAME
from graphstorm_processing.data_transformations import spark_utils, s3_utils
from graphstorm_processing.repartition_files import (
    repartition_files,
    modify_flat_array_metadata,
    ParquetRepartitioner,
)
from graphstorm_processing.graph_loaders.row_count_utils import verify_metadata_match
from graphstorm_processing.constants import ExecutionEnv, FilesystemType


@dataclasses.dataclass
class ExecutorConfig:
    """Configuration  for the DistributedExecutor

    Parameters
    ----------
    local_config_path : str
        Local path to the input config file.
    local_metadata_output_path : str
        Local path for output metadata files.
    input_prefix : str
        Prefix for input data. Can be S3 URI or local path.
    output_prefix : str
        Prefix for output data. Can be S3 URI or local path.
    num_output_files : int
        The number of output files Spark will try to create.
    execution_env : ExecutionEnv
        The kind of execution environment the job will run on.
    config_filename : str
        The filename for the configuration file.
    filesystem_type : FilesystemType
        The filesystem type, can be LOCAL or S3
    add_reverse_edges : bool
        Whether to create reverse edges for each edge type.
    graph_name: str
        The name of the graph being processed
    do_repartition: bool
        Whether to apply repartitioning to the graph on the Spark leader.
    """

    local_config_path: str
    local_metadata_output_path: str
    input_prefix: str
    output_prefix: str
    num_output_files: int
    execution_env: ExecutionEnv
    config_filename: str
    filesystem_type: FilesystemType
    add_reverse_edges: bool
    graph_name: str
    do_repartition: bool


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
    do_repartition: bool


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
        self.local_metadata_output_path = executor_config.local_metadata_output_path
        self.input_prefix = executor_config.input_prefix
        self.output_prefix = executor_config.output_prefix
        self.num_output_files = executor_config.num_output_files
        self.config_filename = executor_config.config_filename
        self.filesystem_type = executor_config.filesystem_type
        self.execution_env = executor_config.execution_env
        self.add_reverse_edges = executor_config.add_reverse_edges
        self.graph_name = executor_config.graph_name
        self.repartition_on_leader = executor_config.do_repartition
        # Input config dict using GSProcessing schema
        self.gsp_config_dict = {}

        # Ensure we have write access to the output path
        if self.filesystem_type == FilesystemType.LOCAL:
            if not os.path.exists(self.output_prefix):
                try:
                    os.makedirs(self.output_prefix, exist_ok=True)
                except OSError as e:
                    logging.error("Unable to create output path: %s", e)
                    raise e
        else:
            # Ensure we can read and write files from/to the S3 prefix
            s3 = boto3.resource("s3")
            bucket_name, prefix = s3_utils.extract_bucket_and_key(self.output_prefix)
            head_bucket_response = s3.meta.client.head_bucket(Bucket=bucket_name)
            assert head_bucket_response["ResponseMetadata"]["HTTPStatusCode"] == 200, (
                f"Could not read objects at S3 output prefix: {self.output_prefix} "
                "Check permissions for execution role."
            )
            bucket_resouce = s3.Bucket(bucket_name)
            bucket_resouce.put_object(Key=f"{prefix}/test_file.txt", Body=b"test")
            response = bucket_resouce.delete_objects(
                Delete={"Objects": [{"Key": f"{prefix}/test_file.txt"}], "Quiet": True}
            )
            assert response["ResponseMetadata"]["HTTPStatusCode"] == 200, (
                f"Could not delete objects at S3 output prefix: {self.output_prefix} "
                "Check permissions for execution role."
            )

        graph_conf = os.path.join(self.local_config_path, self.config_filename)
        with open(graph_conf, "r", encoding="utf-8") as f:
            dataset_config_dict: Dict[str, Any] = json.load(f)

        # Load the pre-computed transformations if the file exists
        if os.path.exists(os.path.join(self.local_config_path, TRANSFORMATIONS_FILENAME)):
            with open(
                os.path.join(self.local_config_path, TRANSFORMATIONS_FILENAME),
                "r",
                encoding="utf-8",
            ) as f:
                self.precomputed_transformations = json.load(f)
        else:
            self.precomputed_transformations = {}

        if "version" in dataset_config_dict:
            config_version: str = dataset_config_dict["version"]
            if config_version.startswith("gsprocessing"):
                logging.info("Parsing config file as GSProcessing config")
                self.gsp_config_dict = dataset_config_dict["graph"]
            elif config_version.startswith("gconstruct"):
                logging.info("Parsing config file as GConstruct config")
                converter = GConstructConfigConverter()
                self.gsp_config_dict = converter.convert_to_gsprocessing(dataset_config_dict)[
                    "graph"
                ]
            else:
                logging.warning("Unrecognized configuration file version name: %s", config_version)
                try:
                    converter = GConstructConfigConverter()
                    self.gsp_config_dict = converter.convert_to_gsprocessing(dataset_config_dict)[
                        "graph"
                    ]
                except Exception:  # pylint: disable=broad-exception-caught
                    logging.warning("Could not parse config as GConstruct, trying GSProcessing")
                    assert (
                        "graph" in dataset_config_dict
                    ), "Top-level element 'graph' needs to exist in a GSProcessing config"
                    self.gsp_config_dict = dataset_config_dict["graph"]
                    logging.info("Parsed config file as GSProcessing config")
        else:
            # Older versions of GConstruct configs might be missing a version entry
            logging.warning("No configuration file version name, trying to parse as GConstruct...")
            converter = GConstructConfigConverter()
            self.gsp_config_dict = converter.convert_to_gsprocessing(dataset_config_dict)["graph"]

        # Create the Spark session for execution
        self.spark = spark_utils.create_spark_session(self.execution_env, self.filesystem_type)

        # Initialize the graph loader
        data_configs = create_config_objects(self.gsp_config_dict)
        loader_config = HeterogeneousLoaderConfig(
            add_reverse_edges=self.add_reverse_edges,
            data_configs=data_configs,
            enable_assertions=False,
            graph_name=self.graph_name,
            input_prefix=self.input_prefix,
            local_input_path=self.local_config_path,
            local_metadata_output_path=self.local_metadata_output_path,
            num_output_files=self.num_output_files,
            output_prefix=self.output_prefix,
            precomputed_transformations=self.precomputed_transformations,
        )
        self.loader = DistHeterogeneousGraphLoader(
            self.spark,
            loader_config,
        )

    def _upload_output_files(self, loader: DistHeterogeneousGraphLoader, force=False):
        """Upload output files to S3

        Parameters
        ----------
        loader : DistHeterogeneousGraphLoader
            The graph loader instance that performed the processing
        force : bool, optional
            Enforce upload even in SageMaker, by default False
        """
        if (
            not self.execution_env == ExecutionEnv.SAGEMAKER
            and self.filesystem_type == FilesystemType.S3
        ) or force:
            # Output files need to be manually uploaded to S3 when not on SM
            bucket, s3_prefix = s3_utils.extract_bucket_and_key(self.output_prefix)
            s3 = boto3.resource("s3")

            output_files = os.listdir(loader.output_path)
            for output_file in output_files:
                s3.meta.client.upload_file(
                    f"{os.path.join(loader.output_path, output_file)}",
                    bucket,
                    f"{s3_prefix}/{output_file}",
                )

    def run(self) -> None:
        """
        Executes the Spark processing job.
        """
        logging.info("Performing data processing with PySpark...")

        t0 = time.time()

        processed_representations: ProcessedGraphRepresentation = self.loader.load()
        graph_meta_dict = processed_representations.processed_graph_metadata_dict

        t1 = time.time()
        logging.info("Time to transform data for distributed partitioning: %s sec", t1 - t0)
        # Stop the Spark context
        self.spark.stop()

        all_match = verify_metadata_match(graph_meta_dict)
        repartitioner = ParquetRepartitioner(
            self.output_prefix,
            self.filesystem_type,
            region=None,
            verify_outputs=True,
            streaming_repartitioning=False,
        )

        repartition_start = time.perf_counter()
        updated_metadata = {}
        if all_match:
            logging.info(
                "All file row counts match, applying Parquet metadata modification on Spark leader."
            )
            modify_flat_array_metadata(graph_meta_dict, repartitioner)
            # modify_flat_array_metadata modifies file metadata in-place,
            # so the original meta dict still applies
            updated_metadata = graph_meta_dict
        else:
            if self.repartition_on_leader:
                logging.info("Attempting to repartition graph data on Spark leader...")
                try:
                    # Upload existing output files before trying to re-partition
                    if self.filesystem_type == FilesystemType.S3:
                        self._upload_output_files(self.loader, force=True)
                    updated_metadata = repartition_files(graph_meta_dict, repartitioner)
                except Exception as e:  # pylint: disable=broad-exception-caught
                    # If an error happens during re-partition, we don't want to fail the entire
                    # gs-processing job, so we just post an error and continue
                    logging.error(
                        "Failed to repartition data on Spark leader, "
                        "will need to run follow-up re-partition job. "
                        "Original error: %s",
                        str(e),
                    )
            else:
                logging.warning("gs-repartition will need to run as a follow-up job on the data!")

        processed_representations.timers["repartition"] = time.perf_counter() - repartition_start

        # If any of the metadata modification took place, write an updated metadata file
        if updated_metadata:
            updated_meta_path = os.path.join(
                self.loader.output_path, "updated_row_counts_metadata.json"
            )
            with open(
                updated_meta_path,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(updated_metadata, f, indent=4)
                f.flush()
                logging.info("Updated metadata written to %s", updated_meta_path)
            logging.info(
                "Data are now prepared for processing by the Distributed Partition pipeline."
            )

        with open(
            os.path.join(self.local_metadata_output_path, "perf_counters.json"),
            "w",
            encoding="utf-8",
        ) as f:
            sorted_timers = dict(
                sorted(processed_representations.timers.items(), key=lambda x: x[1], reverse=True)
            )
            json.dump(sorted_timers, f, indent=4)

        # If pre-computed representations exist, merge them with the input dict and save to disk
        with open(
            os.path.join(
                self.local_metadata_output_path,
                f"{os.path.splitext(self.config_filename)[0]}_with_transformations.json",
            ),
            "w",
            encoding="utf-8",
        ) as f:
            input_dict_with_transforms = self._merge_config_with_transformations(
                self.gsp_config_dict, processed_representations.transformation_representations
            )
            json.dump(input_dict_with_transforms, f, indent=4)

        # This is used to upload the output output JSON files to S3 on non-SageMaker runs,
        # since we can't rely on SageMaker to do it
        if self.filesystem_type == FilesystemType.S3:
            self._upload_output_files(
                self.loader, force=(not self.execution_env == ExecutionEnv.SAGEMAKER)
            )

    def _merge_config_with_transformations(
        self,
        gsp_config_dict: dict,
        transformations: Mapping[str, Mapping[str, Mapping]],
    ) -> dict:
        """Merge the config dict with the transformations dict and return a copy.

        Parameters
        ----------
        gsp_config_dict : dict
            The input configuration dictionary, using GSProcessing schema
        transformations : Mapping[str, Mapping[str, Mapping]]
            The processed graph representations containing the transformations.
            Expected dict schema:

            {
                "node_features": {
                    "node_type1": {
                        "feature_name1": {
                            "transformation_name": # transformation name, e.g. "numerical"
                            # feature1 representation goes here
                        },
                        "feature_name2": {}, ...
                    },
                    "node_type2": {...}
                },
                "edges_features": {...}
            }

        Returns
        -------
        dict
            A copy of the ``gsp_config_dict`` modified to be merged with
            the transformation representations.
        """
        gsp_config_dict_copy = copy.deepcopy(gsp_config_dict)

        edge_transformations: Mapping[str, Mapping[str, Mapping]] = transformations["edge_features"]
        node_transformations: Mapping[str, Mapping[str, Mapping]] = transformations["node_features"]
        edge_input_dicts: list[dict] = gsp_config_dict_copy["edges"]
        node_input_dicts: list[dict] = gsp_config_dict_copy["nodes"]

        def get_structure_type(single_input_dict: dict, structure_type: str) -> str:
            """Gets the node or edge type name from input dict."""
            if structure_type == "node":
                type_name = single_input_dict["type"]
            elif structure_type == "edge":
                src_type = single_input_dict["source"]["type"]
                dst_type = single_input_dict["dest"]["type"]
                relation = single_input_dict["relation"]["type"]
                type_name = f"{src_type}:{relation}:{dst_type}"
            else:
                raise ValueError(
                    f"Invalid structure type: {structure_type}. Expected 'node' or 'edge'."
                )

            return type_name

        def append_transformations(
            structure_input_dicts: list[dict],
            structure_transforms: Mapping[str, Mapping[str, Mapping]],
            structure_type: str,
        ):
            """Appends the pre-computed transformations to the input dicts."""
            assert structure_type in ["edge", "node"]
            for input_dict in structure_input_dicts:
                # type_name is the name of either a node type or edge type
                type_name = get_structure_type(input_dict, structure_type)
                # If we have pre-computed transformations for this type
                if type_name in structure_transforms:
                    # type_transforms holds the transformation representations for
                    # every feature that has one for type_name, from feature name to
                    # feature representation dict.
                    type_transforms: Mapping[str, Mapping] = structure_transforms[type_name]
                    assert (
                        "features" in input_dict
                    ), f"Expected type {type_name} to have have features in the input config"

                    # Iterate over every feature for the node/edge type,
                    # and append representation to its input dict, if one exists
                    for type_feat_dict in input_dict["features"]:
                        # We take a feature's name either explicitly if it exists,
                        # or from the column name otherwise.
                        feat_name = (
                            type_feat_dict["name"]
                            if "name" in type_feat_dict
                            else type_feat_dict["column"]
                        )
                        if feat_name in type_transforms:
                            # Feature representation needs to contain all the
                            # necessary information to re-apply the feature transformation
                            feature_representation = type_transforms[feat_name]
                            type_feat_dict["precomputed_transformation"] = feature_representation

        if edge_transformations:
            append_transformations(edge_input_dicts, edge_transformations, "edge")
        if node_transformations:
            append_transformations(node_input_dicts, node_transformations, "node")

        gsp_top_level_dict = {
            "graph": gsp_config_dict_copy,
            "version": "gsprocessing-v1.0",
        }

        return gsp_top_level_dict


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
        help="GConstruct or GSProcessing data configuration filename.",
        required=True,
    )
    parser.add_argument(
        "--num-output-files",
        type=int,
        default=None,
        help=(
            "The desired number of output files to be created per type. "
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
    parser.add_argument(
        "--do-repartition",
        type=lambda x: (str(x).lower() in ["true", "1"]),
        default=False,
        help=(
            "When set to 'True', will try to re-partition the output "
            "data on the Spark leader if necessary."
        ),
    )

    return parser.parse_args()


def main():
    """Main entry point for GSProcessing"""
    # Allows us to get typed arguments from the command line
    gsprocessing_args = GSProcessingArguments(**vars(parse_args()))
    logging.basicConfig(
        level=gsprocessing_args.log_level,
        format="[GSPROCESSING] %(asctime)s %(levelname)-8s %(message)s",
    )

    # Determine execution environment
    if os.path.exists("/opt/ml/config/processingjobconfig.json"):
        execution_env = ExecutionEnv.SAGEMAKER
    elif os.path.exists("/usr/lib/spark/code/EMR_SERVERLESS_EXECUTION"):
        execution_env = ExecutionEnv.EMR_SERVERLESS
    elif os.path.exists("/usr/lib/spark/code/EMR_EXECUTION"):
        execution_env = ExecutionEnv.EMR_ON_EC2
    else:
        execution_env = ExecutionEnv.LOCAL

    if gsprocessing_args.input_prefix.startswith("s3://"):
        assert gsprocessing_args.output_prefix.startswith("s3://"), (
            "When providing S3 input and output prefixes, they must both be S3 URIs, got: "
            f"input: '{gsprocessing_args.input_prefix}' "
            f"and output: '{gsprocessing_args.output_prefix}'."
        )

        filesystem_type = FilesystemType.S3
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
        filesystem_type = FilesystemType.LOCAL

    # local input location for config file and execution script
    if execution_env == ExecutionEnv.SAGEMAKER:
        local_config_path = "/opt/ml/processing/input/data"
    else:
        # When not running on SM, we need to manually pull the config file from S3
        if filesystem_type == FilesystemType.LOCAL:
            local_config_path = gsprocessing_args.input_prefix
        else:
            tempdir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
            input_bucket, input_s3_prefix = s3_utils.extract_bucket_and_key(
                gsprocessing_args.input_prefix
            )
            s3 = boto3.client("s3")
            try:
                s3.download_file(
                    input_bucket,
                    f"{input_s3_prefix}/{gsprocessing_args.config_filename}",
                    os.path.join(tempdir.name, gsprocessing_args.config_filename),
                )
            except botocore.exceptions.ClientError as e:  # type: ignore
                raise RuntimeError(
                    "Unable to download config file at"
                    f"s3://{input_bucket}/{input_s3_prefix}/"
                    f"{gsprocessing_args.config_filename}"
                ) from e
            # Try to download the pre-computed transformations file, if it exists
            try:
                s3.download_file(
                    input_bucket,
                    f"{input_s3_prefix}/{TRANSFORMATIONS_FILENAME}",
                    os.path.join(tempdir.name, TRANSFORMATIONS_FILENAME),
                )
            except botocore.exceptions.ClientError as _:
                pass
            local_config_path = tempdir.name

    # local output location for metadata files
    if execution_env == ExecutionEnv.SAGEMAKER:
        local_metadata_output_path = "/opt/ml/processing/output"
    else:
        if filesystem_type == FilesystemType.LOCAL:
            local_metadata_output_path = gsprocessing_args.output_prefix
        else:
            # Only needed for local execution with S3 data
            local_metadata_output_path = tempdir.name

    if not gsprocessing_args.num_output_files:
        gsprocessing_args.num_output_files = -1

    # Save arguments to file for posterity
    with open(
        os.path.join(local_metadata_output_path, "launch_arguments.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(dataclasses.asdict(gsprocessing_args), f, indent=4)
        f.flush()

    executor_configuration = ExecutorConfig(
        local_config_path=local_config_path,
        local_metadata_output_path=local_metadata_output_path,
        input_prefix=gsprocessing_args.input_prefix,
        output_prefix=gsprocessing_args.output_prefix,
        num_output_files=gsprocessing_args.num_output_files,
        config_filename=gsprocessing_args.config_filename,
        execution_env=execution_env,
        filesystem_type=filesystem_type,
        add_reverse_edges=gsprocessing_args.add_reverse_edges,
        graph_name=gsprocessing_args.graph_name,
        do_repartition=gsprocessing_args.do_repartition,
    )

    dist_executor = DistributedExecutor(executor_configuration)

    dist_executor.run()


if __name__ == "__main__":
    main()
