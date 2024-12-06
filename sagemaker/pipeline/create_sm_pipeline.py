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

import os
import re
from pprint import pp
from typing import List, Union

import boto3
from sagemaker.processing import ScriptProcessor
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.execution_variables import ExecutionVariables
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

from pipeline_parameters import PipelineArgs, parse_pipeline_args

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


def create_pipeline(args: PipelineArgs) -> Pipeline:
    """Creates a GraphStorm SageMaker pipeline object.

    Parameters
    ----------
    args : PipelineArgs
        Pipeline configuration object, contains all the necessary
        settings to set up the pipeline.

    Returns
    -------
    Pipeline
        A SageMaker pipeline object.
    """
    boto_session = boto3.Session(region_name=args.aws_config.region)
    sagemaker_client = boto_session.client(service_name="sagemaker")
    # TODO: Add option for local pipeline session
    pipeline_session = PipelineSession(
        boto_session=boto_session, sagemaker_client=sagemaker_client
    )

    # Define pipeline parameters
    instance_count_param = ParameterInteger(
        name="InstanceCount", default_value=args.instance_config.instance_count
    )
    cpu_instance_type_param = ParameterString(
        name="CPUInstanceType", default_value=args.instance_config.cpu_instance_type
    )
    gpu_instance_type_param = ParameterString(
        name="GPUInstanceType", default_value=args.instance_config.gpu_instance_type
    )
    graphconstruct_instance_type_param = ParameterString(
        name="GraphConstructInstanceType",
        default_value=args.instance_config.graph_construction_instance_type,
    )
    graphconstruct_config_param = ParameterString(
        name="GraphConstructConfigFile",
        default_value=args.graph_construction_config.graph_construction_config_file,
    )
    partition_algorithm_param = ParameterString(
        name="PartitionAlgorithm",
        default_value=args.partition_config.partition_algorithm,
    )
    graph_name_param = ParameterString(
        name="GraphName", default_value=args.task_config.graph_name
    )
    num_trainers_param = ParameterInteger(
        name="NumTrainers", default_value=args.training_config.num_trainers
    )
    use_graphbolt_param = ParameterString(
        name="UseGraphBolt", default_value=args.training_config.use_graphbolt
    )
    input_data_param = ParameterString(
        name="InputData", default_value=args.task_config.input_data_s3
    )
    output_prefix_param = ParameterString(
        name="OutputPrefix", default_value=args.task_config.output_prefix
    )
    train_config_file_param = ParameterString(
        name="TrainConfigFile", default_value=args.training_config.train_yaml_file
    )
    if args.inference_config.inference_yaml_file:
        inference_yaml_default = args.inference_config.inference_yaml_file
    else:
        inference_yaml_default = args.training_config.train_yaml_file
    inference_config_file_param = ParameterString(
        name="InferenceConfigFile",
        default_value=inference_yaml_default,
    )
    inference_model_snapshot_param = ParameterString(
        "InferenceModelSnapshot",
        default_value=args.inference_config.inference_model_snapshot,
    )
    cache_config = CacheConfig(
        enable_caching=True, expire_after=args.step_cache_expiration
    )

    pipeline_steps: List[Union[ProcessingStep, TrainingStep]] = []

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

    # Build up the output prefix
    # TODO: Using PIPELINE_EXECUTION_ID in the output path invalidates cached results,
    # maybe have the output path be static between executions (but unique per pipeline)?
    # One option might be to use a hash of the execution parameters dict and add that to the prefix?
    # Could be passed as another parameter to the pipeline
    output_subpath = Join(
        on="/",
        values=[
            output_prefix_param,
            pipeline_name,
            ExecutionVariables.PIPELINE_EXECUTION_ID,
        ],
    )

    # We use this to update the input to each step in the pipeline
    next_step_input = input_data_param.to_string()

    # GConstruct step
    if "gconstruct" in args.task_config.jobs_to_run:
        assert args.graph_construction_config.graph_construction_config_file, (
            "Graph construction config file must be specified for GConstruct step. "
            "Use --graph-construction-config-filename"
        )

        gconstruct_processor = ScriptProcessor(
            image_uri=args.aws_config.image_url,
            role=args.aws_config.role,
            instance_count=1,
            instance_type=graphconstruct_instance_type_param,
            command=["python3"],
            sagemaker_session=pipeline_session,
        )

        gc_local_input_path = "/opt/ml/processing/input"
        gc_local_output_path = "/opt/ml/processing/output"

        gconstruct_arguments = [
            "--graph-config-path",
            graphconstruct_config_param,
            "--input-path",
            gc_local_input_path,
            "--output-path",
            gc_local_output_path,
            "--graph-name",
            graph_name_param,
            "--num-parts",
            instance_count_param.to_string(),
            "--add-reverse-edges",
        ]

        if args.graph_construction_config.graph_construction_args:
            gconstruct_arguments.extend(
                args.graph_construction_config.graph_construction_args
            )

        gconstruct_output = Join(
            on="/",
            values=[
                output_subpath,
                "gconstruct",
            ],
        )

        gconstruct_step = ProcessingStep(
            name="GConstruct",
            processor=gconstruct_processor,
            inputs=[
                ProcessingInput(
                    source=next_step_input, destination=gc_local_input_path
                ),
            ],
            outputs=[
                ProcessingOutput(
                    source=gc_local_output_path,
                    destination=gconstruct_output,
                    output_name=graph_name_param,
                ),
            ],
            job_arguments=gconstruct_arguments,
            code=args.script_paths.gconstruct_script,
        )
        pipeline_steps.append(gconstruct_step)
        next_step_input = gconstruct_output

    # DistPartition step
    if "dist_part" in args.task_config.jobs_to_run:
        dist_part_processor = ScriptProcessor(
            image_uri=args.aws_config.image_url,
            role=args.aws_config.role,
            instance_count=instance_count_param,
            instance_type=cpu_instance_type_param,
            command=["python3"],
            sagemaker_session=pipeline_session,
        )

        partition_output = Join(
            on="/",
            values=[
                output_subpath,
                "gspartition",
            ],
        )

        # Expected to be a list of strings
        dist_part_arguments = [
            "--graph-data-s3",
            next_step_input,
            "--metadata-filename",
            args.partition_config.input_json_filename,
            "--num-parts",
            instance_count_param.to_string(),
            "--output-data-s3",
            partition_output,
            "--partition-algorithm",
            partition_algorithm_param,
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
            cache_config=cache_config,
        )
        pipeline_steps.append(dist_part_step)

        # Append dist_graph to input path, because it's added by GSPartition
        next_step_input = Join(
            on="/",
            values=[
                partition_output,
                "dist_graph",
            ],
        )

    # GraphBolt partition conversion step
    if "gb_part" in args.task_config.jobs_to_run:
        gb_part_processor = ScriptProcessor(
            image_uri=args.aws_config.image_url,
            role=args.aws_config.role,
            instance_count=1,
            instance_type=cpu_instance_type_param,
            command=["python3"],
            sagemaker_session=pipeline_session,
        )

        gb_part_arguments = [
            "--entry-point",
            args.script_paths.gb_part_script,
            "--graph-data-s3",
            next_step_input,
            "--metadata-filename",
            args.partition_config.output_json_filename,
            "--log-level",
            args.task_config.log_level,
        ]

        gb_part_step = ProcessingStep(
            name="GraphBoltConversion",
            processor=gb_part_processor,
            inputs=[
                ProcessingInput(
                    input_name="dist_graph_s3_input",
                    destination="/opt/ml/processing/dist_graph/",
                    source=next_step_input,
                    s3_input_mode="File",
                )
            ],
            outputs=[],
            job_arguments=gb_part_arguments,
            cache_config=cache_config,
        )
        pipeline_steps.append(gb_part_step)
        # No need to update next step input for GB convert

    model_output_path = Join(
        on="/",
        values=[
            output_subpath,
            "model",
        ],
    )

    # TODO: Should train_on_cpu be a pipeline param?
    train_infer_instance = (
        cpu_instance_type_param
        if args.instance_config.train_on_cpu
        else gpu_instance_type_param
    )

    # Training step
    if "train" in args.task_config.jobs_to_run:
        train_estimator = PyTorch(
            entry_point=os.path.basename(args.script_paths.train_script),
            source_dir=os.path.dirname(args.script_paths.train_script),
            image_uri=args.aws_config.image_url,
            role=args.aws_config.role,
            instance_count=instance_count_param,
            instance_type=train_infer_instance,
            py_version="py3",
            hyperparameters={
                "graph-data-s3": next_step_input,
                "graph-name": graph_name_param,
                "log-level": args.task_config.log_level,
                "model-artifact-s3": model_output_path,
                "task-type": args.training_config.train_inference_task,
                "train-yaml-s3": train_config_file_param,
                "num-trainers": num_trainers_param,
                "use-graphbolt": use_graphbolt_param,
            },
            sagemaker_session=pipeline_session,
            disable_profiler=True,
            debugger_hook_config=False,
        )

        train_step = TrainingStep(
            name="Training",
            estimator=train_estimator,
            inputs={"train": train_config_file_param},
            cache_config=cache_config,
        )
        pipeline_steps.append(train_step)

    # TODO: During training we should save the best model under '/best_model`
    # to make getting the best model for inference easier
    inference_model_path = Join(
        on="/",
        values=[
            model_output_path,
            inference_model_snapshot_param,
        ],
    )

    inference_output_path = Join(
        on="/",
        values=[
            output_subpath,
            "inference",
        ],
    )

    inference_params = {
        "graph-data-s3": next_step_input,
        "graph-name": graph_name_param,
        "log-level": args.task_config.log_level,
        "model-artifact-s3": inference_model_path,
        "task-type": args.training_config.train_inference_task,
        "infer-yaml-s3": inference_config_file_param,
        "num-trainers": num_trainers_param,
        "use-graphbolt": use_graphbolt_param,
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
    if "inference" in args.task_config.jobs_to_run:
        inference_estimator = PyTorch(
            entry_point=os.path.basename(args.script_paths.inference_script),
            source_dir=os.path.dirname(args.script_paths.inference_script),
            image_uri=args.aws_config.image_url,
            role=args.aws_config.role,
            instance_count=instance_count_param,
            instance_type=train_infer_instance,
            py_version="py3",
            hyperparameters=inference_params,
            sagemaker_session=pipeline_session,
            disable_profiler=True,
            debugger_hook_config=False,
        )

        inference_step = TrainingStep(
            name="Inference",
            estimator=inference_estimator,
            inputs={"train": inference_config_file_param},
            cache_config=cache_config,
        )
        pipeline_steps.append(inference_step)

    # Introduce sequential dependencies
    last_step = pipeline_steps[0]
    for step in pipeline_steps[1:]:
        step.add_depends_on([last_step])
        last_step = step

    # Create the pipeline

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            cpu_instance_type_param,
            gpu_instance_type_param,
            graph_name_param,
            graphconstruct_instance_type_param,
            graphconstruct_config_param,
            inference_config_file_param,
            inference_model_snapshot_param,
            input_data_param,
            instance_count_param,
            num_trainers_param,
            output_prefix_param,
            partition_algorithm_param,
            train_config_file_param,
            use_graphbolt_param,
        ],
        steps=pipeline_steps,
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main():
    """Create or update a GraphStorm SageMaker Pipeline."""
    pipeline_args = parse_pipeline_args()

    pp(pipeline_args)

    pipeline = create_pipeline(pipeline_args)

    if pipeline_args.update:
        pipeline.update(role_arn=pipeline_args.aws_config.role)
        print(f"Pipeline '{pipeline.name}' updated successfully.")
    else:
        pipeline.create(role_arn=pipeline_args.aws_config.role)
        print(f"Pipeline '{pipeline.name}' created successfully.")


if __name__ == "__main__":
    main()
