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

Execute a SageMaker pipeline for GraphStorm.
"""

import argparse
import logging
import os
import subprocess
import sys
import warnings

import boto3
import psutil
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import LocalPipelineSession, PipelineSession

from create_sm_pipeline import GraphStormPipelineGenerator
from pipeline_parameters import load_pipeline_args


def parse_args():
    """Parse pipeline execution arguments"""
    parser = argparse.ArgumentParser(
        description="Execute GraphStorm SageMaker Pipeline"
    )

    parser.add_argument(
        "--pipeline-name",
        type=str,
        required=True,
        help="Name of the pipeline to execute. Required.",
    )
    parser.add_argument(
        "--region",
        type=str,
        required=False,
        help="AWS region. Required for SageMaker execution.",
    )
    parser.add_argument(
        "--async-execution",
        action="store_true",
        help="Run pipeline asynchronously on SageMaker, return after printing execution ARN.",
    )
    parser.add_argument(
        "--local-execution",
        action="store_true",
        help="Use a local pipeline session to execute the pipeline.",
    )
    parser.add_argument(
        "--pipeline-args-json-file",
        type=str,
        help=(
            "When executing locally, optionally provide a JSON representation of the pipeline "
            "arguments. By default we look for '<pipeline-name>-pipeline-args.json' "
            "in the working dir."
        ),
    )

    overrides = parser.add_argument_group(
        "Pipeline overrides", "Override default pipeline parameters at execution time."
    )

    # Optional override parameters
    overrides.add_argument("--instance-count", type=int, help="Override instance count")
    overrides.add_argument(
        "--cpu-instance-type",
        type=str,
        help="Override CPU instance type. "
        "Always used in DistPart step and if '--train-on-cpu' is provided, "
        "in Train and Inference steps.",
    )
    overrides.add_argument(
        "--gpu-instance-type",
        type=str,
        help="Override GPU instance type. "
        "Used by default in in Train and Inference steps, unless '--train-on-cpu' is provided.",
    )
    overrides.add_argument(
        "--graphconstruct-instance-type",
        type=str,
        help="Override graph construction instance type",
    )
    overrides.add_argument(
        "--graphconstruct-config-file",
        type=str,
        help="Override graph construction config file",
    )
    overrides.add_argument(
        "--partition-algorithm",
        type=str,
        choices=["random", "parmetis", "metis"],
        help="Override partition algorithm",
    )
    overrides.add_argument("--graph-name", type=str, help="Override graph name")
    overrides.add_argument(
        "--num-trainers", type=int, help="Override number of trainers"
    )
    overrides.add_argument(
        "--use-graphbolt",
        type=str,
        choices=["true", "false"],
        help="Override whether to use GraphBolt",
    )
    overrides.add_argument("--input-data", type=str, help="Override input data S3 path")
    overrides.add_argument("--output-prefix", type=str, help="Override output prefix")
    overrides.add_argument(
        "--train-yaml-file", type=str, help="Override train yaml file S3 path"
    )
    overrides.add_argument(
        "--inference-yaml-file",
        type=str,
        help="Override inference yaml file S3 path",
    )
    overrides.add_argument(
        "--inference-model-snapshot", type=str, help="Override inference model snapshot"
    )
    overrides.add_argument(
        "--execution-subpath",
        type=str,
        help=(
            "Override execution subpath. "
            "By default it's derived from a hash of the input arguments"
        ),
    )

    return parser.parse_args()


def main():
    """Execute GraphStorm SageMaker pipeline"""
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    pipeline_deploy_args = load_pipeline_args(
        args.pipeline_args_json_file or f"{args.pipeline_name}-pipeline-args.json"
    )
    deploy_time_hash = pipeline_deploy_args.get_hash_hex()

    if args.local_execution:
        # Ensure GPU is available if trying to execute with GPU locally
        if not pipeline_deploy_args.instance_config.train_on_cpu:
            try:
                subprocess.check_output("nvidia-smi")
            except Exception:
                raise RuntimeError(
                    "Need host with NVidia GPU to run training on GPU! "
                    "Try re-deploying the pipeline with --train-on-cpu set."
                )
        # Use local pipeline and session
        local_session = LocalPipelineSession()
        pipeline_generator = GraphStormPipelineGenerator(
            pipeline_deploy_args, input_session=local_session
        )
        # Set shared memory to half of the host's size, as SM does
        instance_mem_mb = int(psutil.virtual_memory().total // (1024 * 1024))
        local_session.config = {
            "local": {"container_config": {"shm_size": f"{instance_mem_mb//2}M"}}
        }
        pipeline = pipeline_generator.create_pipeline()
        pipeline.sagemaker_session = local_session
        pipeline.create(role_arn=pipeline_deploy_args.aws_config.execution_role)
    else:
        assert args.region, "Need to provide --region for remote SageMaker execution"
        boto_session = boto3.Session(region_name=args.region)
        # Use remote pipeline and session
        remote_session = PipelineSession(boto_session)
        pipeline = Pipeline(name=args.pipeline_name, sagemaker_session=remote_session)

    # Prepare parameter overrides
    execution_params = {}
    if args.instance_count:
        execution_params["InstanceCount"] = args.instance_count
        pipeline_deploy_args.instance_config.train_infer_instance_count = (
            args.instance_count
        )
    if args.cpu_instance_type:
        execution_params["CPUInstanceType"] = args.cpu_instance_type
        pipeline_deploy_args.instance_config.cpu_instance_type = args.cpu_instance_type
    if args.gpu_instance_type:
        execution_params["GPUInstanceType"] = args.gpu_instance_type
        pipeline_deploy_args.instance_config.gpu_instance_type = args.gpu_instance_type
    if args.graphconstruct_instance_type:
        execution_params["GraphConstructInstanceType"] = (
            args.graphconstruct_instance_type
        )
        pipeline_deploy_args.instance_config.graph_construction_instance_type = (
            args.graphconstruct_instance_type
        )
    if args.graphconstruct_config_file:
        execution_params["GraphConstructConfigFile"] = args.graphconstruct_config_file
        pipeline_deploy_args.graph_construction_config.config_filename = (
            args.graphconstruct_config_file
        )
    if args.partition_algorithm:
        execution_params["PartitionAlgorithm"] = args.partition_algorithm
        pipeline_deploy_args.partition_config.partition_algorithm = (
            args.partition_algorithm
        )
    if args.graph_name:
        execution_params["GraphName"] = args.graph_name
        pipeline_deploy_args.task_config.graph_name = args.graph_name
    if args.num_trainers is not None:
        execution_params["NumTrainers"] = args.num_trainers
        pipeline_deploy_args.training_config.num_trainers = args.num_trainers
    if args.use_graphbolt:
        execution_params["UseGraphBolt"] = args.use_graphbolt
        pipeline_deploy_args.training_config.use_graphbolt_str = args.use_graphbolt
    if args.input_data:
        execution_params["InputData"] = args.input_data
        pipeline_deploy_args.task_config.input_data_s3 = args.input_data
    if args.output_prefix:
        execution_params["OutputPrefix"] = args.output_prefix
        pipeline_deploy_args.task_config.output_prefix = args.output_prefix
    if args.train_yaml_file:
        execution_params["TrainConfigFile"] = args.train_yaml_file
        pipeline_deploy_args.training_config.train_yaml_file = args.train_yaml_file
    if args.inference_yaml_file:
        execution_params["InferenceConfigFile"] = args.inference_yaml_file
        pipeline_deploy_args.inference_config.inference_yaml_file = (
            args.inference_yaml_file
        )
    if args.inference_model_snapshot:
        execution_params["InferenceModelSnapshot"] = args.inference_model_snapshot
        pipeline_deploy_args.inference_config.inference_model_snapshot = (
            args.inference_model_snapshot
        )
    # If user specified a subpath use that, otherwise let the execution parameters determine it
    if args.execution_subpath:
        execution_params["ExecutionSubpath"] = args.execution_subpath
    else:
        execution_params["ExecutionSubpath"] = pipeline_deploy_args.get_hash_hex()

        if pipeline_deploy_args.get_hash_hex() != deploy_time_hash:
            new_prefix = os.path.join(
                pipeline_deploy_args.task_config.output_prefix,
                args.pipeline_name,
                pipeline_deploy_args.get_hash_hex(),
            )
            warnings.warn(
                "The pipeline execution arguments have been modified "
                "compared to the deployment parameters. "
                f"This execution will use a new unique output prefix, : {new_prefix}."
            )

    # If no parameters are provided, use an empty dict to use all defaults
    execution = pipeline.start(
        parameters=execution_params or {},
    )

    if args.local_execution:
        sys.exit(0)

    logging.info("Pipeline execution started: %s", execution.describe())
    logging.info("Execution ARN: %s", execution.arn)
    logging.info(
        "Output will be created under: %s", execution_params["ExecutionSubpath"]
    )

    if not args.async_execution:
        logging.info("Waiting for pipeline execution to complete...")
        execution.wait()
        logging.info("Pipeline execution completed.")
        logging.info(
            "Final status: %s", execution.describe()["PipelineExecutionStatus"]
        )


if __name__ == "__main__":
    main()
