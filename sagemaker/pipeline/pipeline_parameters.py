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

Parameter parsing for GraphStorm SageMaker Pipeline
"""

import argparse
import os
from dataclasses import dataclass
from typing import List

# TODO: Add support for gsprocessing
SUPPORTED_JOB_TYPES = {"gconstruct", "dist_part", "train", "inference"}
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


@dataclass
class AWSConfig:
    """AWS-related configuration"""

    role: str
    region: str
    image_url: str


@dataclass
class InstanceConfig:
    """Configuration for SageMaker instances"""

    instance_count: int
    cpu_instance_type: str
    gpu_instance_type: str
    graph_construction_instance_type: str
    train_on_cpu: bool = False

    def __post_init__(self):
        assert (
            self.cpu_instance_type or self.gpu_instance_type
        ), "At least one instance type (CPU or GPU) should be specified."
        assert (
            self.train_on_cpu and self.cpu_instance_type
        ), "Need to provide a CPU instance type when training on CPU"


@dataclass
class TaskConfig:
    """Pipeline/task-level configuration"""

    base_job_name: str
    graph_name: str
    input_data_s3: str
    jobs_to_run: List[str]
    log_level: str
    output_prefix: str
    pipeline_name: str

    def __post_init__(self):
        for job_type in self.jobs_to_run:
            assert (
                job_type in SUPPORTED_JOB_TYPES
            ), f"Unsupported job type: {job_type}, expected one of {SUPPORTED_JOB_TYPES}"


@dataclass
class GraphConstructionConfig:
    """Configuration for the graph construction step"""

    graph_construction_config_file: str
    graph_construction_args: str


@dataclass
class PartitionConfig:
    """Configuration for the partition step"""

    partition_algorithm: str
    input_json_filename: str
    output_json_filename: str


@dataclass
class TrainingConfig:
    """Configuration for the training step"""

    model_output_path: str
    train_inference_task: str
    train_yaml_file: str
    num_trainers: int
    use_graphbolt: str


@dataclass
class InferenceConfig:
    """Configuration for the inference step"""

    save_embeddings: bool
    save_predictions: bool
    inference_model_snapshot: str
    inference_yaml_file: str


@dataclass
class ScriptPaths:
    """Entry point script locations"""

    dist_part_script: str
    gb_part_script: str
    train_script: str
    inference_script: str
    gconstruct_script: str


@dataclass()
class PipelineArgs:
    """Wrapper class for all pipeline configurations"""

    aws_config: AWSConfig
    graph_construction_config: GraphConstructionConfig
    instance_config: InstanceConfig
    task_config: TaskConfig
    partition_config: PartitionConfig
    training_config: TrainingConfig
    inference_config: InferenceConfig
    script_paths: ScriptPaths
    step_cache_expiration: str
    update: bool

    def __post_init__(self):
        if not self.instance_config.train_on_cpu:
            assert self.instance_config.gpu_instance_type, (
                "GPU instance type must be specified if not training on CPU, "
                f"got {self.instance_config.train_on_cpu=} "
                f"{self.instance_config.gpu_instance_type=}"
            )


def parse_pipeline_args() -> PipelineArgs:
    """Parses all the arguments for the pipeline definition.

    Returns
    -------
    PipelineArgs
        Pipeline configuration object.
    """
    parser = argparse.ArgumentParser(description="Create GraphStorm SageMaker Pipeline")

    required_args = parser.add_argument_group("Required arguments")
    optional_args = parser.add_argument_group("Optional arguments")

    # AWS Configuration
    required_args.add_argument(
        "--role", type=str, required=True, help="SageMaker IAM role ARN. Required"
    )
    required_args.add_argument(
        "--region", type=str, required=True, help="AWS region. Required"
    )
    required_args.add_argument(
        "--image-url", type=str, required=True, help="ECR image URL. Required"
    )

    # Instance Configuration
    required_args.add_argument(
        "--instance-count",
        type=int,
        help="Number of worker instances.",
        required=True,
    )
    optional_args.add_argument(
        "--cpu-instance-type",
        type=str,
        help="CPU instance type.",
        default="ml.m5.4xlarge",
    )
    optional_args.add_argument(
        "--gpu-instance-type",
        type=str,
        help="GPU instance type.",
        default="ml.g5.4xlarge",
    )
    optional_args.add_argument(
        "--train-on-cpu", action="store_true", help="Train on CPU instead of GPU"
    )

    # Pipeline/Task Configuration
    required_args.add_argument(
        "--graph-name", type=str, required=True, help="Name of the graph. Required."
    )
    required_args.add_argument(
        "--input-data-s3",
        type=str,
        required=True,
        help="S3 path to the input graph data. Required.",
    )
    required_args.add_argument(
        "--output-prefix-s3",
        type=str,
        required=True,
        help="S3 prefix for the output data. Required.",
    )
    optional_args.add_argument(
        "--pipeline-name",
        type=str,
        help="Name for the pipeline. Needs to be unique per account and region.",
    )
    optional_args.add_argument(
        "--base-job-name",
        type=str,
        default="gs",
        help="Base job name for SageMaker jobs. Default: 'sm'",
    )
    optional_args.add_argument(
        "--jobs-to-run",
        nargs="+",
        default=["gconstruct", "train", "inference"],
        help="Jobs to run in the pipeline. "
        f"Should be one or more of: {list(SUPPORTED_JOB_TYPES)} "
        "Default ['gconstruct', 'train', 'inference']",
    )
    optional_args.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "CRITICAL", "FATAL"],
        help="Logging level for the jobs. Default: INFO",
    )
    optional_args.add_argument(
        "--step-cache-expiration",
        type=str,
        default="30d",
        help="Expiration time for the step cache, as a string in ISO 8601 duration. Default: 30d",
    )
    optional_args.add_argument(
        "--update-pipeline",
        action="store_true",
        help="Update an existing pipeline instead of creating a new one.",
    )

    # Graph construction configuration
    optional_args.add_argument(
        "--graph-construction-config-filename",
        type=str,
        default="",
        help="Filename for the graph construction config. "
        "Needs to exist at the top level of the S3 input data.",
    )
    optional_args.add_argument(
        "--graph-construction-instance-type",
        type=str,
        default="ml.m5.4xlarge",
        help="Instance type for graph construction. Default: ml.m5.4xlarge",
    )
    optional_args.add_argument(
        "--graph-construction-args",
        type=str,
        default="",
        help="Parameters to be passed directly to the GConstruct job, "
        "wrap these in double quotes to avoid splitting, e.g."
        '--graph-construction-args "--num-processes 8 "',
    )

    # Partition configuration
    optional_args.add_argument(
        "--partition-algorithm",
        type=str,
        default="random",
        choices=["random", "parmetis"],
        help="Partitioning algorithm. Default: 'random'",
    )
    optional_args.add_argument(
        "--partition-output-json",
        type=str,
        default="metadata.json",
        help="Name for the output JSON file that describes the partitioned data. "
        "Will be metadata.json if you use GSPartition, <graph-name>.json if you use GConstruct.",
    )
    optional_args.add_argument(
        "--partition-input-json",
        type=str,
        default="updated_row_counts_metadata.json",
        help="Name for the JSON file that describes the input data for partitioning. "
        "Will be updated_row_counts_metadata.json if you use GSProcessing",
    )

    # Training Configuration
    optional_args.add_argument(
        "--model-output-path",
        type=str,
        default="",
        help="S3 path for model output. Default is determined by graph name and subpath.",
    )
    # TODO: Make this a pipeline param or derive from instance type during launch?
    optional_args.add_argument(
        "--num-trainers",
        type=int,
        default=4,
        help="Number of trainers to use during training/inference. Set this to the number of GPUs",
    )
    required_args.add_argument(
        "--train-inference-task",
        type=str,
        required=True,
        help="Task type for training and inference, e.g. 'node_classification'. Required",
    )
    required_args.add_argument(
        "--train-yaml-s3",
        type=str,
        help="S3 path to train YAML configuration file",
    )
    optional_args.add_argument(
        "--use-graphbolt",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Whether to use GraphBolt. Default: 'false'",
    )

    # Inference Configuration
    optional_args.add_argument(
        "--inference-yaml-s3",
        type=str,
        default=None,
        help="S3 path to inference YAML configuration file",
    )
    optional_args.add_argument(
        "--inference-model-snapshot",
        type=str,
        default="best_model",
        help="Which model snapshot to choose to run inference with, e.g. 'epoch-0'.",
    )
    optional_args.add_argument(
        "--save-predictions",
        action="store_true",
        help="Whether to save predictions to S3 during inference",
    )
    optional_args.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Whether to save predictions to S3 during inference.",
    )

    # Script Paths
    optional_args.add_argument(
        "--dist-part-script",
        type=str,
        default=f"{SCRIPT_PATH}/../run/partition_entry.py",
        help="Path to DistPartition SageMaker entry point script.",
    )
    optional_args.add_argument(
        "--gb-part-script",
        type=str,
        default=f"{SCRIPT_PATH}/../run/gb_convert_entry.py",
        help="Path to GraphBolt partition script.",
    )
    optional_args.add_argument(
        "--train-script",
        type=str,
        default=f"{SCRIPT_PATH}/../run/train_entry.py",
        help="Path to training SageMaker entry point script.",
    )
    optional_args.add_argument(
        "--inference-script",
        type=str,
        default=f"{SCRIPT_PATH}/../run/infer_entry.py",
        help="Path to inference SageMaker entry point script.",
    )
    optional_args.add_argument(
        "--gconstruct-script",
        type=str,
        default=f"{SCRIPT_PATH}/../run/gconstruct_entry.py",
        help="Path to GConstruct SageMaker entry point script.",
    )

    # TODO: Support arbitrary SM estimator args

    args = parser.parse_args()

    return PipelineArgs(
        aws_config=AWSConfig(
            role=args.role, region=args.region, image_url=args.image_url
        ),
        instance_config=InstanceConfig(
            instance_count=args.instance_count,
            graph_construction_instance_type=args.graph_construction_instance_type,
            cpu_instance_type=args.cpu_instance_type,
            gpu_instance_type=args.gpu_instance_type,
            train_on_cpu=args.train_on_cpu,
        ),
        task_config=TaskConfig(
            graph_name=args.graph_name,
            input_data_s3=args.input_data_s3,
            output_prefix=args.output_prefix_s3,
            pipeline_name=args.pipeline_name,
            base_job_name=args.base_job_name,
            jobs_to_run=args.jobs_to_run,
            log_level=args.log_level,
        ),
        graph_construction_config=GraphConstructionConfig(
            graph_construction_config_file=args.graph_construction_config_filename,
            graph_construction_args=args.graph_construction_args,
        ),
        partition_config=PartitionConfig(
            input_json_filename=args.partition_input_json,
            output_json_filename=args.partition_output_json,
            partition_algorithm=args.partition_algorithm,
        ),
        training_config=TrainingConfig(
            model_output_path=args.model_output_path,
            num_trainers=args.num_trainers,
            train_inference_task=args.train_inference_task,
            train_yaml_file=args.train_yaml_s3,
            use_graphbolt=args.use_graphbolt,
        ),
        inference_config=InferenceConfig(
            save_predictions=args.save_predictions,
            save_embeddings=args.save_embeddings,
            inference_model_snapshot=args.inference_model_snapshot,
            inference_yaml_file=args.inference_yaml_s3,
        ),
        script_paths=ScriptPaths(
            dist_part_script=args.dist_part_script,
            gconstruct_script=args.gconstruct_script,
            gb_part_script=args.gb_part_script,
            train_script=args.train_script,
            inference_script=args.inference_script,
        ),
        step_cache_expiration=args.step_cache_expiration,
        update=args.update_pipeline,
    )
