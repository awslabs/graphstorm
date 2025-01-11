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

Create a SageMaker pipeline for GraphStorm.
"""

import logging
import os
import re
from typing import List, Optional, Sequence, Union

import boto3
from sagemaker.processing import ScriptProcessor
from sagemaker.spark.processing import PySparkProcessor
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import (
    CacheConfig,
    ProcessingStep,
    TrainingStep,
    ProcessingInput,
    ProcessingOutput,
)

from pipeline_parameters import (
    PipelineArgs,
    parse_pipeline_args,
    save_pipeline_args,
)

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class GraphStormPipelineGenerator:
    """Creates a SageMaker pipeline for a sequence of GraphStorm jobs.

    Parameters
    ----------
    args : PipelineArgs
        Complete set of arguments for the pipeline this generator will create.
    input_session: Optional[PipelineSession]
        Existing PipelineSession to use for the pipeline. Default: None.
    """

    def __init__(
        self, args: PipelineArgs, input_session: Optional[PipelineSession] = None
    ):
        self.args = args
        self.pipeline_session = self._get_or_create_pipeline_session(input_session)
        self.pipeline_params: List[Union[ParameterString, ParameterInteger]] = []
        self._create_pipeline_parameters(args)

        self.next_step_data_input = self.args.task_config.input_data_s3
        self.cache_config = CacheConfig(
            enable_caching=True, expire_after=self.args.step_cache_expiration
        )

        # Build up the output prefix
        # We use a hash of the execution parameters dict and
        # add that to the prefix to have consistent intermediate paths between executions
        # that share all the same parameters.
        self.output_subpath = Join(
            on="/",
            values=[
                self.output_prefix_param,
                self._get_pipeline_name(args),
                self.execution_subpath_param,
            ],
        )
        self.train_infer_instance = (
            self.cpu_instance_type_param
            if self.args.instance_config.train_on_cpu
            else self.gpu_instance_type_param
        )
        self.train_infer_image = (
            args.aws_config.graphstorm_pytorch_cpu_image_uri
            if self.args.instance_config.train_on_cpu
            else args.aws_config.graphstorm_pytorch_gpu_image_uri
        )

    def _get_or_create_pipeline_session(
        self, input_session: Optional[PipelineSession] = None
    ):
        """Will return the input session or create a new one if ``None`` is passed."""
        if input_session is not None:
            return input_session
        else:
            boto_session = boto3.Session(region_name=self.args.aws_config.region)
            sagemaker_client = boto_session.client(service_name="sagemaker")
            return PipelineSession(
                boto_session=boto_session, sagemaker_client=sagemaker_client
            )

    def create_pipeline(self) -> Pipeline:
        """Create a SageMaker pipeline for GraphStorm.

        The pipeline can consist of the following steps:
        1. Pre-process graph data using GConstruct or GSProcessing
        2. Partition graph
        4. Train model
        5. Inference

        Returns
        -------
        Pipeline
            _description_
        """
        pipeline_steps = self._create_pipeline_steps(self.args)
        return self._assemble_pipeline(pipeline_steps)

    def _assemble_pipeline(
        self, pipeline_steps: Sequence[Union[ProcessingStep, TrainingStep]]
    ) -> Pipeline:
        """Assemble the pipeline from the given steps, adding sequential dependencies.

        Parameters
        ----------
        pipeline_steps : Sequence[Union[ProcessingStep, TrainingStep]]
            A list of pipeline steps to be executed in sequence

        Returns
        -------
        Pipeline
            The sequence of GraphStorm SageMaker jobs as a SageMaker Pipeline
        """
        for i in range(1, len(pipeline_steps)):
            pipeline_steps[i].add_depends_on([pipeline_steps[i - 1]])

        return Pipeline(
            name=self._get_pipeline_name(self.args),
            parameters=self.pipeline_params,
            steps=pipeline_steps,
            sagemaker_session=self.pipeline_session,
        )

    def _create_string_parameter(self, name: str, default_val: str) -> ParameterString:
        """Create a string pipeline parameter and add it to the list of parameters"""
        param = ParameterString(
            name=name,
            default_value=default_val,
        )
        self.pipeline_params.append(param)
        return param

    def _create_int_parameter(self, name: str, default_val: int) -> ParameterInteger:
        """Create a string pipeline parameter and add it to the list of parameters"""
        param = ParameterInteger(
            name=name,
            default_value=default_val,
        )
        self.pipeline_params.append(param)
        return param

    def _create_pipeline_parameters(self, args: PipelineArgs):
        """Create all pipeline parameters and update self.pipeline_params

        These parameters can be modified at pipeline execution.

        """

        self.instance_count_param = self._create_int_parameter(
            "InstanceCount", args.instance_config.train_infer_instance_count
        )
        self.cpu_instance_type_param = self._create_string_parameter(
            "CPUInstanceType", args.instance_config.cpu_instance_type
        )
        self.gpu_instance_type_param = self._create_string_parameter(
            "GPUInstanceType", args.instance_config.gpu_instance_type
        )
        self.graphconstruct_instance_type_param = self._create_string_parameter(
            "GraphConstructInstanceType",
            args.instance_config.graph_construction_instance_type,
        )
        self.volume_size_gb_param = self._create_int_parameter(
            "InstanceVolumeSizeGB",
            args.instance_config.volume_size_gb,
        )
        self.execution_subpath_param = self._create_string_parameter(
            "ExecutionSubpath", args.get_hash_hex()
        )
        self.graphconstruct_config_param = self._create_string_parameter(
            "GraphConstructConfigFile", args.graph_construction_config.config_filename
        )
        self.partition_algorithm_param = self._create_string_parameter(
            "PartitionAlgorithm", args.partition_config.partition_algorithm
        )
        # TODO: Probably should not be a parameter
        self.graph_name_param = self._create_string_parameter(
            "GraphName", args.task_config.graph_name
        )
        self.num_trainers_param = self._create_int_parameter(
            "NumTrainers", args.training_config.num_trainers
        )
        self.input_data_param = self._create_string_parameter(
            "InputData", args.task_config.input_data_s3
        )
        self.output_prefix_param = self._create_string_parameter(
            "OutputPrefix", args.task_config.output_prefix
        )
        self.train_config_file_param = self._create_string_parameter(
            "TrainConfigFile", args.training_config.train_yaml_file
        )

        # If inference yaml is not provided, re-use the training one
        inference_yaml_default = (
            args.inference_config.inference_yaml_file
            or args.training_config.train_yaml_file
        )
        self.inference_config_file_param = self._create_string_parameter(
            "InferenceConfigFile", inference_yaml_default
        )
        # User-defined model snapshot to use, e.g. epoch-5
        self.inference_model_snapshot_param = self._create_string_parameter(
            "InferenceModelSnapshot", args.inference_config.inference_model_snapshot
        )

    def _create_pipeline_steps(
        self, args: PipelineArgs
    ) -> List[Union[ProcessingStep, TrainingStep]]:
        """Creates the pipeline steps and returns them as a list

        Parameters
        ----------
        args : PipelineArgs
            Pipeline arguments

        Returns
        -------
        List[Union[ProcessingStep, TrainingStep]]
            A list of pipeline steps to be executed in sequence
        """
        # NOTE: GConstruct, GSProcessing, DistPart, GBConvert run as Processing jobs.
        # Training, Inference run as Training jobs.
        pipeline_steps: List[Union[ProcessingStep, TrainingStep]] = []
        for job in self.args.task_config.jobs_to_run:
            step = self._create_step(job, args)
            if step:
                pipeline_steps.append(step)

        return pipeline_steps

    def _create_step(
        self, job_type: str, args: PipelineArgs
    ) -> Union[ProcessingStep, TrainingStep, None]:
        """Create a pipeline step for the provided job type

        Parameters
        ----------
        job_type : str
            A job type from the task_config.jobs_to_run list

        Returns
        -------
        Union[ProcessingStep, TrainingStep, None]
            A pipeline step for the provided job type, or None if the job type is not supported
        """
        step_generators = {
            "gconstruct": self._create_gconstruct_step,
            "gsprocessing": self._create_gsprocessing_step,
            "dist_part": self._create_dist_part_step,
            "gb_convert": self._create_gb_convert_step,
            "train": self._create_train_step,
            "inference": self._create_inference_step,
        }
        step_generator = step_generators.get(job_type)
        return step_generator(args) if step_generator else None

    def _create_gconstruct_step(self, args: PipelineArgs) -> ProcessingStep:
        # Implementation for GConstruct step
        assert args.graph_construction_config.config_filename, (
            "Graph construction config file must be specified for GConstruct step. "
            "Use --graph-construction-config-filename"
        )

        gconstruct_processor = ScriptProcessor(
            image_uri=args.aws_config.graphstorm_pytorch_cpu_image_uri,
            role=args.aws_config.execution_role,
            instance_count=1,
            instance_type=self.graphconstruct_instance_type_param,
            command=["python3"],
            sagemaker_session=self.pipeline_session,
            volume_size_in_gb=self.volume_size_gb_param,
        )

        gconstruct_s3_output = Join(
            on="/",
            values=[
                self.output_subpath,
                "gconstruct",
            ],
        )

        gc_local_input_path = "/opt/ml/processing/input"
        # GConstruct should always be the first step and start with the source data
        gc_proc_input = ProcessingInput(
            source=self.input_data_param, destination=gc_local_input_path
        )
        gc_local_output_path = "/opt/ml/processing/output"
        gc_proc_output = ProcessingOutput(
            source=gc_local_output_path,
            destination=gconstruct_s3_output,
            output_name=f"{self.args.task_config.graph_name}-gconstruct",
        )

        gconstruct_arguments = [
            "--graph-config-path",
            self.graphconstruct_config_param,
            "--input-path",
            gc_local_input_path,
            "--output-path",
            gc_local_output_path,
            "--graph-name",
            self.graph_name_param,
            "--num-parts",
            self.instance_count_param.to_string(),
            "--part-method",
            self.partition_algorithm_param,
        ]

        # TODO: Make this a pipeline parameter?
        if args.graph_construction_config.graph_construction_args:
            # TODO: Support nargs='+' arguments, .split(" ") won't work with those.
            # Only --output-format seems to be an nargs='+' argument in GConstruct
            gconstruct_arguments.extend(
                args.graph_construction_config.graph_construction_args.split(" ")
            )

        gconstruct_step = ProcessingStep(
            name="GConstruct",
            processor=gconstruct_processor,
            inputs=[
                gc_proc_input,
            ],
            outputs=[
                gc_proc_output,
            ],
            job_arguments=gconstruct_arguments,
            code=args.script_paths.gconstruct_script,
            cache_config=self.cache_config,
        )

        self.next_step_data_input = gconstruct_s3_output

        return gconstruct_step

    def _create_gsprocessing_step(self, args: PipelineArgs) -> ProcessingStep:
        # Implementation for GSProcessing step
        pyspark_processor = PySparkProcessor(
            role=args.aws_config.execution_role,
            instance_type=args.instance_config.graph_construction_instance_type,
            instance_count=args.instance_config.gsprocessing_instance_count,
            image_uri=args.aws_config.gsprocessing_pyspark_image_uri,
            sagemaker_session=self.pipeline_session,
            volume_size_in_gb=self.volume_size_gb_param,
        )

        gsprocessing_output = Join(
            on="/",
            values=[
                self.output_subpath,
                "gsprocessing",
            ],
        )

        # GSProcessing should always be the first step and start with the source data
        gsprocessing_config = Join(
            on="/",
            values=[
                self.input_data_param,
                self.graphconstruct_config_param,
            ],
        )

        gsprocessing_arguments = [
            "--config-filename",
            self.graphconstruct_config_param,
            "--graph-name",
            self.graph_name_param,
            "--input-prefix",
            self.input_data_param,
            "--output-prefix",
            gsprocessing_output,
            "--do-repartition",
            "True",
            "--log-level",
            args.task_config.log_level,
        ]

        if args.graph_construction_config.graph_construction_args:
            gsprocessing_arguments.extend(
                args.graph_construction_config.graph_construction_args.split(" ")
            )

        gsprocessing_config_input = ProcessingInput(
            input_name="gsprocessing_config",
            source=gsprocessing_config,
            destination="/opt/ml/processing/input/data",
        )
        gsprocessing_meta_output = ProcessingOutput(
            output_name="partition-input-metadata",
            destination=gsprocessing_output,
            source="/opt/ml/processing/output",
        )

        pyspark_step = ProcessingStep(
            name="GSProcessing",
            code=args.script_paths.gsprocessing_script,
            processor=pyspark_processor,
            inputs=[
                gsprocessing_config_input,
            ],
            outputs=[gsprocessing_meta_output],
            job_arguments=gsprocessing_arguments,
            cache_config=self.cache_config,
        )

        self.next_step_data_input = gsprocessing_output

        return pyspark_step

    def _create_dist_part_step(self, args: PipelineArgs) -> ProcessingStep:
        # Implementation for DistPartition step
        dist_part_processor = ScriptProcessor(
            image_uri=args.aws_config.graphstorm_pytorch_cpu_image_uri,
            role=args.aws_config.execution_role,
            instance_count=self.instance_count_param,
            instance_type=self.graphconstruct_instance_type_param,
            command=["python3"],
            sagemaker_session=self.pipeline_session,
            volume_size_in_gb=self.volume_size_gb_param,
        )

        partition_output = Join(
            on="/",
            values=[
                self.output_subpath,
                "gspartition",
            ],
        )

        # Expected to be a list of strings
        dist_part_arguments = [
            "--graph-data-s3",
            self.next_step_data_input,
            "--metadata-filename",
            args.partition_config.input_json_filename,
            "--num-parts",
            self.instance_count_param.to_string(),
            "--output-data-s3",
            partition_output,
            "--partition-algorithm",
            self.partition_algorithm_param,
        ]

        distpart_processor_args = dist_part_processor.run(
            inputs=[],
            outputs=[],
            code=args.script_paths.dist_part_script,
            arguments=dist_part_arguments,
        )

        dist_part_step = ProcessingStep(
            name="DistPartition",
            step_args=distpart_processor_args,  # type: ignore
            cache_config=self.cache_config,
        )

        # Append dist_graph to input path, because it's added by GSPartition
        self.next_step_data_input = Join(
            on="/",
            values=[
                partition_output,
                "dist_graph",
            ],
        )

        return dist_part_step

    def _create_gb_convert_step(self, args: PipelineArgs) -> ProcessingStep:
        # Implementation for GraphBolt partition step
        gb_part_processor = ScriptProcessor(
            image_uri=args.aws_config.graphstorm_pytorch_cpu_image_uri,
            role=args.aws_config.execution_role,
            instance_count=1,
            instance_type=self.graphconstruct_instance_type_param,
            command=["python3"],
            sagemaker_session=self.pipeline_session,
            volume_size_in_gb=self.volume_size_gb_param,
        )

        gb_convert_arguments = [
            "--metadata-filename",
            args.partition_config.output_json_filename,
        ]

        gb_convert_step = ProcessingStep(
            name="GraphBoltConversion",
            code=args.script_paths.gb_convert_script,
            processor=gb_part_processor,
            inputs=[
                ProcessingInput(
                    input_name="dist_graph_s3_input",
                    destination="/opt/ml/processing/dist_graph/",
                    source=self.next_step_data_input,
                    s3_input_mode="File",
                )
            ],
            outputs=[],
            job_arguments=gb_convert_arguments,
            cache_config=self.cache_config,
        )
        # No need to update next step input for GB convert
        return gb_convert_step

    def _create_train_step(self, args: PipelineArgs) -> TrainingStep:
        # Implementation for Training step
        model_output_path = Join(
            on="/",
            values=[
                self.output_subpath,
                "model",
            ],
        )

        train_params = {
            "graph-data-s3": self.next_step_data_input,
            "graph-name": self.graph_name_param,
            "log-level": args.task_config.log_level,
            "model-artifact-s3": model_output_path,
            "task-type": args.training_config.train_inference_task,
            "train-yaml-s3": self.train_config_file_param,
            "num-trainers": self.num_trainers_param,
            "use-graphbolt": args.training_config.use_graphbolt_str,
        }

        # Training step
        train_estimator = PyTorch(
            entry_point=os.path.basename(args.script_paths.train_script),
            source_dir=os.path.dirname(args.script_paths.train_script),
            image_uri=self.train_infer_image,
            role=args.aws_config.execution_role,
            instance_count=self.instance_count_param,
            instance_type=self.train_infer_instance,
            py_version="py3",
            hyperparameters=train_params,
            sagemaker_session=self.pipeline_session,
            disable_profiler=True,
            debugger_hook_config=False,
            volume_size=self.volume_size_gb_param,
        )

        train_step = TrainingStep(
            name="Training",
            estimator=train_estimator,
            inputs={"train": self.train_config_file_param},
            cache_config=self.cache_config,
        )

        self.model_input_path = model_output_path

        return train_step

    def _create_inference_step(self, args: PipelineArgs) -> TrainingStep:
        # Implementation for Inference step
        # TODO: During training we should save the best model under '/best_model`
        # to make getting the best model for inference easier
        inference_model_path = Join(
            on="/",
            values=[
                self.model_input_path,
                self.inference_model_snapshot_param,
            ],
        )

        inference_output_path = Join(
            on="/",
            values=[
                self.output_subpath,
                "inference",
            ],
        )

        inference_params = {
            "graph-data-s3": self.next_step_data_input,
            "graph-name": self.graph_name_param,
            "log-level": args.task_config.log_level,
            "model-artifact-s3": inference_model_path,
            "task-type": args.training_config.train_inference_task,
            "infer-yaml-s3": self.inference_config_file_param,
            "num-trainers": self.num_trainers_param,
            "use-graphbolt": args.training_config.use_graphbolt_str,
        }

        if args.inference_config.save_embeddings:
            inference_params["output-emb-s3"] = Join(
                on="/",
                values=[
                    inference_output_path,
                    "embeddings",
                ],
            )
        if args.inference_config.save_predictions:
            inference_params["output-prediction-s3"] = Join(
                on="/",
                values=[
                    inference_output_path,
                    "predictions",
                ],
            )

        # Inference step
        inference_estimator = PyTorch(
            entry_point=os.path.basename(args.script_paths.inference_script),
            source_dir=os.path.dirname(args.script_paths.inference_script),
            image_uri=self.train_infer_image,
            role=args.aws_config.execution_role,
            instance_count=self.instance_count_param,
            instance_type=self.train_infer_instance,
            py_version="py3",
            hyperparameters=inference_params,
            sagemaker_session=self.pipeline_session,
            disable_profiler=True,
            debugger_hook_config=False,
            volume_size=self.volume_size_gb_param,
        )

        inference_step = TrainingStep(
            name="Inference",
            estimator=inference_estimator,
            inputs={"train": self.inference_config_file_param},
            cache_config=self.cache_config,
        )

        return inference_step

    @staticmethod
    def _get_pipeline_name(args: PipelineArgs) -> str:
        # Logic for determining pipeline name
        if not args.task_config.pipeline_name:
            # Assert graph name is valid for ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}
            assert re.match(
                r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}$", args.task_config.graph_name
            ), (
                "Graph name must be valid for a pipeline name, "
                "matching ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,255}"
            )
            pipeline_name = (
                f"GS-{args.task_config.graph_name}-"
                f"{args.training_config.train_inference_task.replace('_', '-')}"
            )
        else:
            pipeline_name = args.task_config.pipeline_name

        return pipeline_name


def main():
    """Create or update a GraphStorm SageMaker Pipeline."""
    pipeline_args = parse_pipeline_args()
    logging.basicConfig(level=logging.INFO)

    save_pipeline_args(
        pipeline_args, f"{pipeline_args.task_config.pipeline_name}-pipeline-args.json"
    )

    pipeline_generator = GraphStormPipelineGenerator(pipeline_args)

    pipeline = pipeline_generator.create_pipeline()

    if pipeline_args.update:
        # TODO: If updating, ensure pipeline exists first to get more informative error
        pipeline.update(role_arn=pipeline_args.aws_config.execution_role)
        logging.info("Pipeline '%s' updated successfully.", pipeline.name)
    else:
        pipeline.create(role_arn=pipeline_args.aws_config.execution_role)
        logging.info("Pipeline '%s' created successfully.", pipeline.name)


if __name__ == "__main__":
    main()
