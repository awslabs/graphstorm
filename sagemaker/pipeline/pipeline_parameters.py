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

Parameter parsing and validation for GraphStorm SageMaker Pipeline
"""

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import List


JOB_ORDER = {
    "gconstruct": 0,
    "gsprocessing": 1,
    "dist_part": 2,
    "gb_convert": 3,
    "train": 4,
    "inference": 5,
}

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


@dataclass
class AWSConfig:
    """AWS-related configuration.

    Parameters
    ----------
    execution_role : str
        SageMaker execution IAM role ARN.
    region : str
        AWS region.
    graphstorm_pytorch_cpu_image_uri : str
        GraphStorm GConstruct/dist_part/train/inference CPU ECR image URI.
    graphstorm_pytorch_gpu_image_uri : str
        GraphStorm GConstruct/dist_part/train/inference GPU ECR image URI.
    gsprocessing_pyspark_image_uri : str
        GSProcessing SageMaker PySpark ECR image URI.
    """

    execution_role: str
    region: str
    graphstorm_pytorch_cpu_image_uri: str
    graphstorm_pytorch_gpu_image_uri: str
    gsprocessing_pyspark_image_uri: str


@dataclass
class InstanceConfig:
    """Configuration for SageMaker instances.

    Parameters
    ----------
    train_infer_instance_count : int
        Number of worker instances/partitions for partition, training, inference.
    cpu_instance_type : str
        CPU instance type.
    gpu_instance_type : str
        GPU instance type.
    graph_construction_instance_type : str
        Instance type for graph construction.
    gsprocessing_instance_count : int
        Number of GSProcessing instances.
    train_on_cpu : bool
        Whether to run training and inference on CPU instances.
    volume_size_gb : int
        Additional volume size for SageMaker instances in GB.
    """

    train_infer_instance_count: int
    cpu_instance_type: str
    gpu_instance_type: str
    graph_construction_instance_type: str
    gsprocessing_instance_count: int
    train_on_cpu: bool
    volume_size_gb: int

    def __post_init__(self):
        assert (
            self.cpu_instance_type or self.gpu_instance_type
        ), "At least one instance type (CPU or GPU) should be specified."

        if self.train_on_cpu:
            assert (
                self.cpu_instance_type
            ), "Need to provide a CPU instance type when training on CPU"
        else:
            assert (
                self.gpu_instance_type
            ), "Need to provide a GPU instance type when training on GPU"

        if not self.graph_construction_instance_type:
            self.graph_construction_instance_type = self.cpu_instance_type
            assert (
                self.cpu_instance_type
            ), "Need to provide a CPU instance for graph construction."
            logging.warning(
                "No graph processing instance type specified, using the CPU instance type: %s",
                self.cpu_instance_type,
            )


@dataclass
class TaskConfig:
    """Pipeline/task-level configuration.

    Parameters
    ----------
    base_job_name : str
        Base job name for SageMaker jobs.
    graph_name : str
        Name of the graph.
    input_data_s3 : str
        S3 path to the input graph data.
    jobs_to_run : List[str]
        List of jobs to run in the pipeline.
    log_level : str
        Logging level for the jobs.
    output_prefix : str
        S3 prefix for the output data.
    pipeline_name : str
        Name for the pipeline.
    """

    base_job_name: str
    graph_name: str
    input_data_s3: str
    jobs_to_run: List[str]
    log_level: str
    output_prefix: str
    pipeline_name: str

    def __post_init__(self):
        # Ensure job names are valid
        for job_type in self.jobs_to_run:
            assert (
                job_type in JOB_ORDER
            ), f"Unsupported job type: '{job_type}', expected one of {list(JOB_ORDER.keys())}"

        # Ensure jobs are in the right order
        self.jobs_to_run.sort(key=lambda x: JOB_ORDER[x])

        # We should only run either gconstruct or gsprocessing
        if "gconstruct" in self.jobs_to_run and "gsprocessing" in self.jobs_to_run:
            raise ValueError(
                "Should not try to run both GConstruct and GSProcessing steps, "
                f"got job sequence: {self.jobs_to_run}"
            )

        # When running gsprocessing ensure we run dist_part as well
        if "gsprocessing" in self.jobs_to_run and "dist_part" not in self.jobs_to_run:
            raise ValueError(
                "When running GSProcessing need to run 'dist_part' as the following job, "
                f"got job sequence: {self.jobs_to_run}"
            )

        # Fix trailing slash for S3 input paths, otherwise can get
        # ClientError: Failed to invoke sagemaker:CreateProcessingJob. Error Details: null.
        if self.input_data_s3.endswith("/"):
            self.input_data_s3 = self.input_data_s3[:-1]


@dataclass
class GraphConstructionConfig:
    """Configuration for the graph construction step.

    Parameters
    ----------
    config_filename : str
        Filename for the graph construction config.
    graph_construction_args : str
        Parameters to be passed directly to the GConstruct job.
    """

    config_filename: str
    graph_construction_args: str


@dataclass
class PartitionConfig:
    """Configuration for the partition step.

    Parameters
    ----------
    partition_algorithm : str
        Partitioning algorithm.
    input_json_filename : str
        Name for the JSON file that describes the input data for partitioning.
    output_json_filename : str
        Name for the output JSON file that describes the partitioned data.
    """

    partition_algorithm: str
    input_json_filename: str
    output_json_filename: str


@dataclass
class TrainingConfig:
    """Configuration for the training step.

    Parameters
    ----------
    model_output_path : str
        S3 path for model output.
    train_inference_task : str
        Task type for training and inference.
    train_yaml_file : str
        S3 path to train YAML configuration file.
    num_trainers : int
        Number of trainers to use during training/inference.
    use_graphbolt_str : str
        Whether to use GraphBolt ('true' or 'false').
    """

    model_output_path: str
    train_inference_task: str
    train_yaml_file: str
    num_trainers: int
    use_graphbolt_str: str

    def __post_init__(self):
        self.use_graphbolt_str = self.use_graphbolt_str.lower()


@dataclass
class InferenceConfig:
    """Configuration for the inference step.

    Parameters
    ----------
    save_embeddings : bool
        Whether to save embeddings to S3 during inference.
    save_predictions : bool
        Whether to save predictions to S3 during inference.
    inference_model_snapshot : str
        Which model snapshot to choose to run inference with.
    inference_yaml_file : str
        S3 path to inference YAML configuration file.
    """

    save_embeddings: bool
    save_predictions: bool
    inference_model_snapshot: str
    inference_yaml_file: str


@dataclass
class ScriptPaths:
    """Entry point script locations.

    Parameters
    ----------
    dist_part_script : str
        Path to DistPartition SageMaker entry point script.
    gb_convert_script : str
        Path to GraphBolt partition script.
    train_script : str
        Path to training SageMaker entry point script.
    inference_script : str
        Path to inference SageMaker entry point script.
    gconstruct_script : str
        Path to GConstruct SageMaker entry point script.
    gsprocessing_script : str
        Path to GSProcessing SageMaker entry point script.
    """

    dist_part_script: str
    gb_convert_script: str
    train_script: str
    inference_script: str
    gconstruct_script: str
    gsprocessing_script: str


@dataclass()
class PipelineArgs:
    """Wrapper class for all pipeline configurations.

    Parameters
    ----------
    aws_config : AWSConfig
        AWS configuration settings.
    graph_construction_config : GraphConstructionConfig
        Graph construction configuration.
    instance_config : InstanceConfig
        Instance configuration settings.
    task_config : TaskConfig
        Task-level configuration settings.
    partition_config : PartitionConfig
        Partition configuration settings.
    training_config : TrainingConfig
        Training configuration settings.
    inference_config : InferenceConfig
        Inference configuration settings.
    script_paths : ScriptPaths
        Paths to SageMaker entry point scripts.
    step_cache_expiration : str
        Cache expiration for pipeline steps.
    update : bool
        Whether to update existing pipeline or create a new one.
    """

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

    def __eq__(self, other):
        if not isinstance(other, PipelineArgs):
            return False
        return asdict(self) == asdict(other)

    def __hash__(self):
        # Convert the object to a dictionary
        obj_dict = asdict(self)

        # Convert the dictionary to a JSON string
        json_str = json.dumps(obj_dict, sort_keys=True)

        # Create a hash of the JSON string
        return hash(json_str)

    def get_hash_hex(self):
        """Get unique string hash to differentiate between pipeline settings.

        We use this hash as a unique identifier of execution parameters
        in the intermediate output paths of the pipeline. This enables
        caching steps when the input parameters are the same for the same
        pipeline.

        Returns
        -------
        str
            A hexadecimal string representation of the object's hash.
        """
        # Convert the object to a dictionary
        obj_dict = asdict(self)
        # Remove non-functional keys from the dictionary,
        # these do not affect the outcome of the execution
        obj_dict.pop("step_cache_expiration")
        obj_dict.pop("update")

        # Convert the dictionary to a JSON string and create an md5 hash
        json_str = json.dumps(obj_dict, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()

    def __post_init__(self):
        if not self.instance_config.train_on_cpu:
            assert self.aws_config.graphstorm_pytorch_gpu_image_uri, (
                "Must use provide GPU image when training on GPU. "
                "use --graphstorm-pytorch-gpu-image-uri"
            )
        else:
            assert self.aws_config.graphstorm_pytorch_cpu_image_uri, (
                "Must use provide CPU image when training on CPU. "
                "use --graphstorm-pytorch-cpu-image-uri"
            )

        # Ensure we provide a GConstruct/GSProcessing config file when running construction
        if (
            "gconstruct" in self.task_config.jobs_to_run
            or "gsprocessing" in self.task_config.jobs_to_run
        ):
            assert self.graph_construction_config.config_filename, (
                "Need to provide a GConstruct/GSProcessing config file "
                "when running graph construction."
            )

        # When using DistPart and GBConvert ensure the metadata.json filename is used
        if (
            "dist_part" in self.task_config.jobs_to_run
            and "gb_convert" in self.task_config.jobs_to_run
        ):
            if self.partition_config.output_json_filename != "metadata.json":
                logging.warning(
                    "When running DistPart or GBConvert, the partition output JSON "
                    "filename should be 'metadata.json'. "
                    "Got %s, setting to 'metadata.json' instead",
                    self.partition_config.output_json_filename,
                )
                self.partition_config.output_json_filename = "metadata.json"

        # Ensure we have a GSProcessing image to run GSProcessing
        if "gsprocessing" in self.task_config.jobs_to_run:
            assert (
                self.aws_config.gsprocessing_pyspark_image_uri
            ), "Need to provide a GSProcessing PySpark image URI when running GSProcessing"

        # Ensure we run gb_convert after dist_part when training with graphbolt
        if (
            self.training_config.use_graphbolt_str == "true"
            and "dist_part" in self.task_config.jobs_to_run
        ):
            assert "gb_convert" in self.task_config.jobs_to_run, (
                "When using Graphbolt and running distributed partitioning, "
                "need to run 'gb_convert' as a follow-up to 'dist_part', "
                f"got job sequence: {self.task_config.jobs_to_run}"
            )

        # When running gconstruct and train with graphbolt enabled, add '--use-graphbolt true'
        # to gconstruct args
        if (
            "gconstruct" in self.task_config.jobs_to_run
            and self.training_config.use_graphbolt_str == "true"
        ):
            self.graph_construction_config.graph_construction_args += (
                " --use-graphbolt true"
            )

        # If running gsprocessing but do not set instance count for it, use train instance count
        if (
            "gsprocessing" in self.task_config.jobs_to_run
            and not self.instance_config.gsprocessing_instance_count
        ):
            self.instance_config.gsprocessing_instance_count = (
                self.instance_config.train_infer_instance_count
            )
            logging.warning(
                "No GSProcessing instance count specified, using the training instance count: %s",
                self.instance_config.gsprocessing_instance_count,
            )

        # GConstruct uses 'metis', so just translate that if needed
        if (
            "gconstruct" in self.task_config.jobs_to_run
            and self.partition_config.partition_algorithm.lower() == "parmetis"
        ):
            self.partition_config.partition_algorithm = "metis"


def save_pipeline_args(pipeline_args: PipelineArgs, filepath: str) -> None:
    """Save PipelineArgs configuration to a JSON file.

    Parameters
    ----------
    pipeline_args : PipelineArgs
        The PipelineArgs instance to save
    filepath : str
        Path the JSON file to save
    """
    # Convert dataclass to dictionary
    pipeline_args_dict = asdict(pipeline_args)

    # Save to JSON file with nice formatting
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(pipeline_args_dict, f, indent=2)


def dict_to_dataclass(cls: type, data: dict) -> object:
    """Convert a dictionary to a dataclass instance.

    Parameters
    ----------
    cls : type
        The dataclass type to instantiate
    data : dict
        The dictionary used to initialize the dataclass object

    Returns
    -------
    object
        An instance of the dataclass
    """
    if not is_dataclass(cls):
        return data
    field_types = {f.name: f.type for f in fields(cls)}
    return cls(
        **{
            k: dict_to_dataclass(field_types[k], v)
            for k, v in data.items()
            if k in field_types
        }
    )


def load_pipeline_args(filepath: str) -> PipelineArgs:
    """Create PipelineArgs object from a JSON file.

    Parameters
    ----------
    filepath : str
        Path to the JSON representation of a PipelineArgs
        instance.

    Returns
    -------
    PipelineArgs
        PipelineArgs instance with the loaded configuration
    """
    with open(filepath, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    args_instance = dict_to_dataclass(PipelineArgs, config_dict)

    # Assertion to make the type checker happy
    assert isinstance(args_instance, PipelineArgs)

    return args_instance


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
        "--execution-role",
        type=str,
        required=True,
        help="SageMaker execution IAM role ARN. Required.",
    )
    required_args.add_argument(
        "--region", type=str, required=True, help="AWS region. Required"
    )
    required_args.add_argument(
        "--graphstorm-pytorch-cpu-image-uri",
        type=str,
        required=True,
        help="GraphStorm GConstruct/dist_part/train/inference CPU ECR image URI. Required",
    )
    optional_args.add_argument(
        "--graphstorm-pytorch-gpu-image-uri",
        type=str,
        help="GraphStorm GConstruct/dist_part/train/inference GPU ECR image URI.",
    )
    optional_args.add_argument(
        "--gsprocessing-pyspark-image-uri",
        type=str,
        help="GSProcessing SageMaker PySpark ECR image URI. Required if running GSProcessing",
    )

    # Instance Configuration
    required_args.add_argument(
        "--instance-count",
        "--num-parts",
        type=int,
        help="Number of worker instances/partitions for partition, training, inference.",
        required=True,
    )
    optional_args.add_argument(
        "--cpu-instance-type",
        type=str,
        help=(
            "CPU instance type. "
            "Always used in DistPart step and if '--train-on-cpu' is provided, "
            "in Train and Inference steps."
        ),
        default="ml.m5.4xlarge",
    )
    optional_args.add_argument(
        "--gpu-instance-type",
        type=str,
        help=(
            "GPU instance type. Used by default in in Train and Inference steps, "
            "unless '--train-on-cpu' is set."
        ),
        default="ml.g5.4xlarge",
    )
    optional_args.add_argument(
        "--train-on-cpu",
        action="store_true",
        help="Run training and inference on CPU instances instead of GPU.",
    )
    optional_args.add_argument(
        "--graph-construction-instance-type",
        type=str,
        default=None,
        help=(
            "Instance type for graph construction. "
            "Used in GSProcessing, GConstruct, GraphBoltConversion steps. "
            "Default: same as CPU instance type"
        ),
    )
    optional_args.add_argument(
        "--gsprocessing-instance-count",
        type=int,
        default=None,
        help="Number of GSProcessing instances. Default is equal to number of training instances.",
    )
    optional_args.add_argument(
        "--volume-size-gb",
        type=int,
        help="Additional volume size for SageMaker instances in GB.",
        default=100,
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
        help="Base job name for SageMaker jobs. Default: 'gs'",
    )
    required_args.add_argument(
        "--jobs-to-run",
        nargs="+",
        required=True,
        help=(
            "Space-separated string of jobs to run in the pipeline, "
            "e.g. '--jobs-to-run gconstruct train inference'. "
            f"Should be one or more of: {list(JOB_ORDER.keys())}. Required. "
        ),
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
        choices=["random", "parmetis", "metis"],
        help="Partitioning algorithm. Default: 'random'",
    )
    optional_args.add_argument(
        "--partition-output-json",
        type=str,
        default="metadata.json",
        help="Name for the output JSON file that describes the partitioned data. "
        "Will be metadata.json if you use GSPartition to partition the data, "
        "<graph-name>.json if you use GConstruct.",
    )
    optional_args.add_argument(
        "--partition-input-json",
        type=str,
        default="updated_row_counts_metadata.json",
        help="Name for the JSON file that describes the input data for partitioning. "
        "Will be 'updated_row_counts_metadata.json' if you used GSProcessing to process the data, "
        "or '<graph-name>.json' if you used GConstruct.",
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
        help="Number of trainers to use during training/inference. Set this to the number of GPUs."
        "Default: 4",
    )
    required_args.add_argument(
        "--train-inference-task",
        type=str,
        required=True,
        help="Task type for training and inference, e.g. 'node_classification'. Required",
    )
    optional_args.add_argument(
        "--train-yaml-s3",
        type=str,
        help="S3 path to train YAML configuration file. Required if you include train step.",
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
        help="S3 path to inference YAML configuration file. Default: same as --train-yaml-config",
    )
    # TODO: Need a good way to initialize this. Ideally "best-model" should pick up best epoch
    # from training
    optional_args.add_argument(
        "--inference-model-snapshot",
        type=str,
        help="Which model snapshot to choose to run inference with, e.g. 'epoch-0'.",
    )
    optional_args.add_argument(
        "--save-predictions",
        action="store_true",
        help="Whether to save predictions to S3 during inference.",
    )
    optional_args.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Whether to save embeddings to S3 during inference.",
    )

    # Script Paths
    optional_args.add_argument(
        "--dist-part-script",
        type=str,
        default=f"{SCRIPT_PATH}/../run/partition_entry.py",
        help="Path to DistPartition SageMaker entry point script.",
    )
    optional_args.add_argument(
        "--gb-convert-script",
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
    optional_args.add_argument(
        "--gsprocessing-script",
        type=str,
        default=(
            f"{SCRIPT_PATH}/../../graphstorm-processing/"
            "graphstorm_processing/distributed_executor.py"
        ),
        help="Path to GSProcessing SageMaker entry point script.",
    )

    # TODO: Support arbitrary SM estimator args

    args = parser.parse_args()

    return PipelineArgs(
        aws_config=AWSConfig(
            execution_role=args.execution_role,
            region=args.region,
            graphstorm_pytorch_cpu_image_uri=args.graphstorm_pytorch_cpu_image_uri,
            graphstorm_pytorch_gpu_image_uri=args.graphstorm_pytorch_gpu_image_uri,
            gsprocessing_pyspark_image_uri=args.gsprocessing_pyspark_image_uri,
        ),
        instance_config=InstanceConfig(
            train_infer_instance_count=args.instance_count,
            graph_construction_instance_type=args.graph_construction_instance_type,
            gsprocessing_instance_count=args.gsprocessing_instance_count,
            cpu_instance_type=args.cpu_instance_type,
            gpu_instance_type=args.gpu_instance_type,
            train_on_cpu=args.train_on_cpu,
            volume_size_gb=args.volume_size_gb,
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
            config_filename=args.graph_construction_config_filename,
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
            use_graphbolt_str=args.use_graphbolt,
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
            gb_convert_script=args.gb_convert_script,
            train_script=args.train_script,
            inference_script=args.inference_script,
            gsprocessing_script=args.gsprocessing_script,
        ),
        step_cache_expiration=args.step_cache_expiration,
        update=args.update_pipeline,
    )
