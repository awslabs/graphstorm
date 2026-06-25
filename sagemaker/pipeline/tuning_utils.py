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

Duplicates the HPO utilities from launch/launch_hyperparameter_tuning.py
"""

import json
import logging
import os
from typing import Literal

from sagemaker.parameter import (
    ParameterRange,
    ContinuousParameter,
    IntegerParameter,
    CategoricalParameter,
)

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