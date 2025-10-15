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
import logging
from typing import Literal

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.tuner import (
    ParameterRange,
    HyperparameterTuner,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
    HyperbandStrategyConfig,
    StrategyConfig
)

from common_parser import (
    parse_estimator_kwargs,
    parse_unknown_gs_args,
)
from launch_train import get_train_parser

INSTANCE_TYPE = "ml.g4dn.12xlarge"


def get_metric_definitions(
    metric_name: str,
    eval_mask: Literal["val", "test"],
    strategy: Literal["Grid", "Random", "Bayesian", "Hyperband"],
) -> list[dict]:
    """Generate metric definitions based on the HPO strategy.

    Parameters
    ----------
    metric_name : str
        Name of the metric to optimize, as it appears in GraphStorm logs (e.g., 'roc_auc')
    eval_mask : str
        Dataset mask to use when collecting metric values ('test' or 'val')
    strategy : str
        HPO strategy ('Bayesian', 'Random', 'Hyperband', or 'Grid')

    Returns
    -------
    list[dict]
        List of metric definitions for SageMaker HPO
    """
    base_metric_name = f"{eval_mask.lower()}_{metric_name.lower()}"

    if strategy == "Hyperband":
        # For Hyperband, capture intermediate metrics during training
        test_set_string = "Validation" if eval_mask.lower() == "val" else "Test"
        return [
            {
                "Name": f"{base_metric_name}",
                "Regex": f"INFO:root:Step [0-9]+ \\| {test_set_string} {metric_name}: ([0-9\\.]+)",
            }
        ]
    else:
        # For other strategies, only capture the final best metric
        return [
            {
                "Name": f"best_{base_metric_name}",
                "Regex": f"INFO:root:best_{eval_mask.lower()}_score: {{'{metric_name}': ([0-9\\.]+)}}",
            }
        ]


def parse_hyperparameter_ranges(
    hyperparameter_ranges_json: str,
) -> dict[str, ParameterRange]:
    """Parse the hyperparameter ranges from JSON file or string.

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

    .. versionadded:: 0.4.1
    """
    if os.path.exists(hyperparameter_ranges_json):
        with open(hyperparameter_ranges_json, "r", encoding="utf-8") as f:
            str_ranges: dict[str, list[dict]] = json.load(f)["ParameterRanges"]
    else:
        try:
            str_ranges = json.loads(hyperparameter_ranges_json)["ParameterRanges"]
        except json.decoder.JSONDecodeError as e:
            logging.fatal("Invalid JSON string: %s", e)
            raise e

    hyperparameter_ranges: dict[str, ParameterRange] = {}
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
        "topk-model-to-save": "1",
    }

    # Add strategy-specific parameters
    if args.strategy == "Hyperband":
        # Hyperband must control max epochs and early stopping
        # Set num-epochs of GraphStorm to hyperband max epoch
        # Disable early-stop of GraphStorm
        params.update(
            {
                "num-epochs": args.hb_max_epochs,
                "use_early_stop": "false",
            }
        )

    if args.custom_script is not None:
        params["custom-script"] = args.custom_script
    if args.model_checkpoint_to_load is not None:
        params["model-checkpoint-to-load"] = args.model_checkpoint_to_load

    unknown_args_dict = parse_unknown_gs_args(unknownargs)
    params.update(unknown_args_dict)

    logging.info("SageMaker launch parameters %s", params)
    logging.info("Parameters forwarded to GraphStorm  %s", unknown_args_dict)

    estimator_kwargs = parse_estimator_kwargs(args.sm_estimator_parameters)

    # Get estimator kwargs and configure early stopping
    estimator_kwargs = parse_estimator_kwargs(args.sm_estimator_parameters)
    if args.strategy == "Hyperband":
        # Let Hyperband handle early stopping
        estimator_kwargs["early_stopping_type"] = "OFF"

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

    # Get metric definitions based on strategy
    metric_definitions = get_metric_definitions(
        args.metric_name, args.eval_mask, args.strategy
    )
    # Configure the tuner
    tuner_config = {
        "estimator": est,
        # Use first metric as objective
        "objective_metric_name": metric_definitions[0][
            "Name"
        ],
        "hyperparameter_ranges": hyperparameter_ranges,
        "objective_type": args.objective_type,
        "max_jobs": args.max_jobs,
        "max_parallel_jobs": args.max_parallel_jobs,
        "metric_definitions": metric_definitions,
        "strategy": args.strategy,
    }

    # Add Hyperband-specific configuration if needed
    if args.strategy == "Hyperband":
        tuner_config["strategy_config"] = StrategyConfig(
            HyperbandStrategyConfig(
                max_resource=args.hb_max_epochs,
                min_resource=args.hb_min_epochs,
            )
        )

    tuner = HyperparameterTuner(**tuner_config)

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
        "https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-define-ranges.html",
    )
    hpo_group.add_argument(
        "--metric-name",
        type=str,
        required=True,
        help="Name of the metric to optimize for (e.g., 'accuracy', 'amri')",
    )
    hpo_group.add_argument(
        "--eval-mask",
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
    # Hyperband-specific arguments
    hpo_group.add_argument(
        "--hb-min-epochs",
        type=int,
        default=1,
        help="Minimum number of epochs for Hyperband strategy. Default: 1",
    )
    hpo_group.add_argument(
        "--hb-max-epochs",
        type=int,
        default=20,
        help="Maximum number of epochs for Hyperband strategy. Default: 20",
    )

    return parser


if __name__ == "__main__":
    arg_parser = get_hpo_parser()
    args, unknownargs = arg_parser.parse_known_args()
    print(f"HPO launch known args: '{args}'")
    print(f"HPO launch unknown args: '{unknownargs}'")

    run_hyperparameter_tuning_job(args, args.image_url, unknownargs)
