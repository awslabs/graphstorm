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

import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline


def parse_args():
    """Parse pipeline execution arguments"""
    parser = argparse.ArgumentParser(
        description="Execute GraphStorm SageMaker Pipeline"
    )

    parser.add_argument(
        "--pipeline-name",
        type=str,
        required=True,
        help="Name of the pipeline to execute",
    )
    parser.add_argument("--region", type=str, required=True, help="AWS region")

    # Optional override parameters
    parser.add_argument("--instance-count", type=int, help="Override instance count")
    parser.add_argument(
        "--cpu-instance-type", type=str, help="Override CPU instance type"
    )
    parser.add_argument(
        "--gpu-instance-type", type=str, help="Override GPU instance type"
    )
    parser.add_argument(
        "--graphconstruct-instance-type",
        type=str,
        help="Override graph construction instance type",
    )
    parser.add_argument(
        "--graphconstruct-config-file",
        type=str,
        help="Override graph construction config file",
    )
    parser.add_argument(
        "--partition-algorithm",
        type=str,
        choices=["random", "parmetis"],
        help="Override partition algorithm",
    )
    parser.add_argument("--graph-name", type=str, help="Override graph name")
    parser.add_argument("--num-trainers", type=int, help="Override number of trainers")
    parser.add_argument(
        "--use-graphbolt",
        type=str,
        choices=["true", "false"],
        help="Override whether to use GraphBolt",
    )
    parser.add_argument("--input-data", type=str, help="Override input data S3 path")
    parser.add_argument("--output-prefix", type=str, help="Override output prefix")
    parser.add_argument(
        "--train-config-file", type=str, help="Override train config file S3 path"
    )
    parser.add_argument(
        "--inference-config-file",
        type=str,
        help="Override inference config file S3 path",
    )
    parser.add_argument(
        "--inference-model-snapshot", type=str, help="Override inference model snapshot"
    )

    parser.add_argument(
        "--async-execution", action="store_true", help="Run pipeline asynchronously"
    )

    return parser.parse_args()


def main():
    """Execute GraphStorm SageMaker pipeline"""
    args = parse_args()

    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session)

    pipeline = Pipeline(name=args.pipeline_name, sagemaker_session=sagemaker_session)

    # Prepare parameter overrides
    execution_params = {}
    if args.instance_count is not None:
        execution_params["InstanceCount"] = args.instance_count
    if args.cpu_instance_type:
        execution_params["CPUInstanceType"] = args.cpu_instance_type
    if args.gpu_instance_type:
        execution_params["GPUInstanceType"] = args.gpu_instance_type
    if args.graphconstruct_instance_type:
        execution_params["GraphConstructInstanceType"] = (
            args.graphconstruct_instance_type
        )
    if args.graphconstruct_config_file:
        execution_params["GraphConstructConfigFile"] = args.graphconstruct_config_file
    if args.partition_algorithm:
        execution_params["PartitionAlgorithm"] = args.partition_algorithm
    if args.graph_name:
        execution_params["GraphName"] = args.graph_name
    if args.num_trainers is not None:
        execution_params["NumTrainers"] = args.num_trainers
    if args.use_graphbolt:
        execution_params["UseGraphBolt"] = args.use_graphbolt
    if args.input_data:
        execution_params["InputData"] = args.input_data
    if args.output_prefix:
        execution_params["OutputPrefix"] = args.output_prefix
    if args.train_config_file:
        execution_params["TrainConfigFile"] = args.train_config_file
    if args.inference_config_file:
        execution_params["InferenceConfigFile"] = args.inference_config_file
    if args.inference_model_snapshot:
        execution_params["InferenceModelSnapshot"] = args.inference_model_snapshot

    # If no parameters are provided, use an empty dict to use all defaults
    execution = pipeline.start(
        parameters=execution_params or {},
    )

    print(f"Pipeline execution started: {execution.describe()}")
    print(f"Execution ARN: {execution.arn}")

    if not args.async_execution:
        print("Waiting for pipeline execution to complete...")
        execution.wait()
        print("Pipeline execution completed.")
        print(f"Final status: {execution.describe()['PipelineExecutionStatus']}")


if __name__ == "__main__":
    main()
