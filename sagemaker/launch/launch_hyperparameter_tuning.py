r"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Launch SageMaker HPO jobs.

    Example:

        python launch_hyperparameter_tuning.py \
            --task-name my-gnn-hpo-job \
            --role arn:aws:iam::123456789012:role/SageMakerRole \
            --region us-west-2 \
            --image-url 123456789012.dkr.ecr.us-west-2.amazonaws.com/my-graphstorm-image:latest \
            --graph-name my-graph \
            --task-type node_classification \
            --graph-data-s3 s3://my-bucket/graph-data/ \
            --yaml-s3 s3://my-bucket/train.yaml \
            --model-artifact-s3 s3://my-bucket/model-artifacts/ \
            --max-jobs 20 \
            --max-parallel-jobs 4 \
            --hyperparameter-ranges ./my-parameter-ranges.json
            --metric-name "accuracy" \
            --metric-dataset "val" \
            --objective-type Maximize
"""
import os
import json
from typing import Mapping

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tuner import (
    ParameterRange,
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
)

from common_parser import (
    parse_estimator_kwargs,
    parse_unknown_gs_args,
)
from launch_train import get_train_parser

INSTANCE_TYPE = "ml.g4dn.12xlarge"


def parse_hyperparameter_ranges(hyperparameter_ranges_json: str) -> dict[str, ParameterRange]:
    """Parse the hyperparameter ranges from JSON string and determine if autotune is needed.

    Expected JSON structure is at
    https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html

    Parameters
    ----------
    hyperparameter_ranges_json : str
        Path to a JSON file or JSON as a string.

    Returns
    -------
    dict
        Hyperparameter dict that can be used to create a HyperparameterTuner
        object.

    Raises
    ------
    ValueError
        If the JSON dict contains an invalid parameter type.
    """
    if os.path.exists(hyperparameter_ranges_json):
        with open(hyperparameter_ranges_json, "r", encoding="utf-8") as f:
            str_ranges: dict[str, list[dict]] = json.load(f)["ParameterRanges"]
    else:
        str_ranges = json.loads(hyperparameter_ranges_json)["ParameterRanges"]

    hyperparameter_ranges: Mapping[str, ParameterRange] = {}
    for params_type, config_list in str_ranges.items():
        if params_type == "ContinuousParameterRanges":
            for config in config_list:
                param_name = config["Name"]
                hyperparameter_ranges[param_name] = ContinuousParameter(
                    config["MinValue"],
                    config["MaxValue"],
                    config.get("ScalingType", "Auto"),
                )
        elif params_type == "IntegerParameterRanges":
            for config in config_list:
                param_name = config["Name"]
                hyperparameter_ranges[param_name] = IntegerParameter(
                    config["MinValue"],
                    config["MaxValue"],
                    config.get("ScalingType", "Auto"),
                )
        elif params_type == "CategoricalParameterRanges":
            for config in config_list:
                param_name = config["Name"]
                hyperparameter_ranges[param_name] = CategoricalParameter(
                    config["Values"]
                )
        else:
            raise ValueError(
                f"Unknown parameter type {params_type}. "
                "Expect one of 'CategoricalParameterRanges', 'ContinuousParameterRanges', "
                "'IntegerParameterRanges'"
            )
    return hyperparameter_ranges


def run_hyperparameter_tuning_job(args, image, unknownargs):
    """Run hyperparameter tuning job using SageMaker HyperparameterTuner"""

    container_image_uri = image

    prefix = f"gs-hpo-{args.graph_name}"

    params = {
        "eval-metric": args.metric_name,
        "graph-data-s3": args.graph_data_s3,
        "graph-name": args.graph_name,
        "log-level": args.log_level,
        "task-type": args.task_type,
        "train-yaml-s3": args.yaml_s3,
    }
    if args.custom_script is not None:
        params["custom-script"] = args.custom_script
    if args.model_checkpoint_to_load is not None:
        params["model-checkpoint-to-load"] = args.model_checkpoint_to_load

    unknown_args_dict = parse_unknown_gs_args(unknownargs)
    params.update(unknown_args_dict)

    print(f"SageMaker launch parameters {params}")
    print(f"GraphStorm forwarded parameters {unknown_args_dict}")

    estimator_kwargs = parse_estimator_kwargs(args.sm_estimator_parameters)

    est = PyTorch(
        entry_point=os.path.basename(args.entry_point),
        source_dir=os.path.dirname(args.entry_point),
        image_uri=container_image_uri,
        role=args.role,
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        output_path=args.model_artifact_s3,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        tags=[
            {"Key": "GraphStorm", "Value": "oss"},
            {"Key": "GraphStorm_Task", "Value": "HPO"},
        ],
        **estimator_kwargs,
    )

    hyperparameter_ranges = parse_hyperparameter_ranges(args.hyperparameter_ranges)

    # Construct the full metric name based on user input
    full_metric_name = f"best_{args.metric_dataset}_score:{args.metric_name}"

    tuner = HyperparameterTuner(
        estimator=est,
        objective_metric_name=full_metric_name,
        hyperparameter_ranges=hyperparameter_ranges,
        objective_type=args.objective_type,
        max_jobs=args.max_jobs,
        max_parallel_jobs=args.max_parallel_jobs,
        metric_definitions=[
            {
                "Name": full_metric_name,
                "Regex": (
                    f"INFO:root:best_{args.metric_dataset}_score: "
                    f"{{'{args.metric_name}': ([0-9\\.]+)}}"
                ),
            }
        ],
        strategy=args.strategy,
    )

    tuner.fit({"train": args.yaml_s3}, wait=not args.async_execution)


def get_hpo_parser():
    """Return a parser for GraphStorm hyperparameter tuning task."""
    parser = get_train_parser()

    hpo_group = parser.add_argument_group("Hyperparameter tuning arguments")

    hpo_group.add_argument(
        "--max-jobs",
        type=int,
        default=10,
        help="Maximum number of training jobs to run",
    )
    hpo_group.add_argument(
        "--max-parallel-jobs",
        type=int,
        default=2,
        help="Maximum number of parallel training jobs",
    )
    hpo_group.add_argument(
        "--hyperparameter-ranges",
        type=str,
        required=True,
        help="Path to a JSON file, or a JSON string defining hyperparameter ranges. "
        "For syntax see 'Dynamic hyperparameters' in "
        "https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html"
        ,
    )
    hpo_group.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Name of the metric to optimize for (e.g., 'accuracy', 'amri')",
    )
    hpo_group.add_argument(
        "--metric-dataset",
        type=str,
        required=True,
        choices=["test", "val"],
        help="Whether to use test or validation metrics for HPO.",
    )
    hpo_group.add_argument(
        "--objective-type",
        type=str,
        default="Maximize",
        choices=["Maximize", "Minimize"],
        help="Type of objective, can be 'Maximize' or 'Minimize'",
    )
    hpo_group.add_argument(
        "--strategy",
        type=str,
        default="Bayesian",
        choices=["Bayesian", "Random", "Hyperband", "Grid"],
        help="Optimization strategy. Default: 'Bayesian'.",
    )

    return parser


if __name__ == "__main__":
    arg_parser = get_hpo_parser()
    args, unknownargs = arg_parser.parse_known_args()
    print(f"HPO launch known args: '{args}'")
    print(f"HPO launch unknown args:{type(unknownargs)=} '{unknownargs=}'")

    run_hyperparameter_tuning_job(args, args.image_url, unknownargs)
