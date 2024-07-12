"""
Common parsers for all launcher scripts.
"""
from typing import Any, Dict
from ast import literal_eval
import argparse

# Note: Keep SUPPORTED_TASKS and SUPPORTED_INFER_TASKS
# synced with graphstorm.config SUPPORTED_TASKS.
# We do not import graphstorm.config.SUPPORTED_TASKS here
# as we do not want the launch script to rely on graphstorm
# package.
SUPPORTED_TASKS = {
    "node_classification",
    "node_regression",
    "edge_classification",
    "edge_regression",
    "link_prediction",
    "multi_task"
}

SUPPORTED_INFER_TASKS = {
    "node_classification",
    "node_regression",
    "edge_classification",
    "edge_regression",
    "link_prediction",
    "comput_emb",
    "multi_task"
}

def get_common_parser() -> argparse.ArgumentParser:
    """
    Returns an argument parser that can be used by all
    GraphStorm SageMaker launcher scripts.
    """
    parser = argparse.ArgumentParser()

    common_args = parser.add_argument_group("Common arguments")

    common_args.add_argument("--graph-data-s3", type=str,
        help="S3 location of input graph data", required=True)
    common_args.add_argument("--image-url", type=str,
        help="GraphStorm SageMaker docker image URI",
        required=True)
    common_args.add_argument("--role", type=str,
        help="SageMaker execution role",
        required=True)
    common_args.add_argument("--instance-type", type=str,
        help="instance type for the SageMaker job")
    common_args.add_argument("--instance-count", type=int,
        default=2,
        help="number of instances")
    common_args.add_argument("--region", type=str,
        default="us-west-2",
        help="Region")
    common_args.add_argument("--task-name", type=str,
        default=None, help="User defined SageMaker task name")
    common_args.add_argument("--sm-estimator-parameters", type=str,
        default=None, help='Parameters to be passed to the SageMaker Estimator as kwargs, '
            'in <key>=<val> format, separated by spaces. Do not include spaces in the values themselves, e.g.: '
            '--sm-estimator-parameters "volume_size=100 subnets=[\'subnet-123\',\'subnet-345\'] '
            'security_group_ids=[\'sg-1234\',\'sg-3456\']"')
    common_args.add_argument("--async-execution", action='store_true',
        help="When set will launch the job in async mode, returning immediately. "
            "When not set, will block until the job completes")

    return parser

def parse_estimator_kwargs(arg_string: str) -> Dict[str, Any]:
    """
    Parses Estimator/Processor arguments for SageMaker tasks. See
    https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html
    for the full list of arguments for train/infer/partition tasks.
    For GConstruct see
    https://sagemaker.readthedocs.io/en/stable/api/training/processing.html
    Argument values are evaluated as Python
    literals using ast.literal_eval.

    :param arg_string: String of arguments in the form of
        <key>=<val> separated by spaces.
    :return: Dictionary of parsed arguments.
    """
    if arg_string is None:
        return {}
    typed_args_dict = {}
    for param in arg_string.split(" "):
        k, v =param.split("=")
        typed_args_dict[k] = literal_eval(v)

    return typed_args_dict
