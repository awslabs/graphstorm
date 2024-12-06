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
    parser.add_argument("--graph-name", type=str, help="Override graph name")
    parser.add_argument("--input-data", type=str, help="Override input data S3 path")
    parser.add_argument(
        "--model-output-path", type=str, help="Override model output S3 path"
    )
    parser.add_argument("--num-epochs", type=int, help="Override number of epochs")
    parser.add_argument("--num-trainers", type=int, help="Override number of trainers")
    parser.add_argument("--instance-count", type=int, help="Override instance count")
    parser.add_argument(
        "--partition-algorithm",
        type=str,
        choices=["random", "parmetis"],
        help="Override partition algorithm",
    )
    parser.add_argument("--subpath", type=str, help="Override subpath")
    parser.add_argument(
        "--train-config-file", type=str, help="Override train config file S3 path"
    )
    parser.add_argument(
        "--use-graphbolt",
        type=str,
        choices=["true", "false"],
        help="Override whether to use GraphBolt",
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
    # pipeline_session = PipelineSession(
    #     boto_session=boto_session, sagemaker_client=sagemaker_client
    # )

    pipeline = Pipeline(name=args.pipeline_name, sagemaker_session=sagemaker_session)

    # Prepare parameter overrides
    execution_params = {}
    if args.graph_name:
        execution_params["GraphName"] = args.graph_name
    if args.input_data:
        execution_params["InputData"] = args.input_data
    if args.model_output_path:
        execution_params["ModelOutputPath"] = args.model_output_path
    if args.num_epochs is not None:
        execution_params["NumEpochs"] = args.num_epochs
    if args.num_trainers is not None:
        execution_params["NumTrainers"] = args.num_trainers
    if args.instance_count is not None:
        execution_params["InstanceCount"] = args.instance_count
    if args.partition_algorithm:
        execution_params["PartitionAlgorithm"] = args.partition_algorithm
    if args.subpath:
        execution_params["Subpath"] = args.subpath
    if args.train_config_file:
        execution_params["TrainConfigFile"] = args.train_config_file
    if args.use_graphbolt:
        execution_params["UseGraphBolt"] = args.use_graphbolt

    # If no parameters are provided, use an empty dict to use all defaults
    execution = pipeline.start(
        parameters=execution_params or None,
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
