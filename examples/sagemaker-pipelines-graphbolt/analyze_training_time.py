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

Analyzes the epoch and evaluation time for GraphStorm training jobs.
"""

import argparse
import re
import time
from datetime import datetime, timedelta
from typing import Iterator, Dict, List, Union

import boto3

LOG_GROUP = "/aws/sagemaker/TrainingJobs"


def parse_args():
    """Parse log analysis args."""
    parser = argparse.ArgumentParser(
        description="Analyze training epoch and eval time."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    # Add pipeline name as arg
    source_group.add_argument(
        "--pipeline-name",
        type=str,
        help="The name of the pipeline.",
    )
    # Add execution id as arg
    parser.add_argument(
        "--execution-name",
        type=str,
        help="The display name of the execution.",
    )
    source_group.add_argument(
        "--log-file",
        type=str,
        help="The name of a file containing logs from a local pipeline execution.",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="The region of the log stream.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Whether to print verbose output.",
    )
    # Add days past as arg
    parser.add_argument(
        "--logs-days-before",
        type=int,
        default=2,
        help="Limit log analysis to logs created this many days before today.",
    )
    return parser.parse_args()


def read_local_logs(file_path: str) -> Iterator[Dict]:
    """Read logs from a local file and yield them in a format similar to CloudWatch events."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield {
                "message": line.strip(),
                "timestamp": int(time.time() * 1000),  # Current time in milliseconds
            }


def get_pipeline_execution_arn(pipeline_name: str, execution_name: str) -> str:
    """Get the execution ARN from a pipeline name and display name for the execution."""
    sm_client = boto3.client("sagemaker")

    try:
        # List pipeline executions and find the matching one
        paginator = sm_client.get_paginator("list_pipeline_executions")
        for page in paginator.paginate(PipelineName=pipeline_name):
            for execution in page["PipelineExecutionSummaries"]:
                if execution_name in execution["PipelineExecutionDisplayName"]:
                    return execution["PipelineExecutionArn"]

        raise ValueError(
            f"No execution found with display name {execution_name} in pipeline {pipeline_name}"
        )

    except Exception as e:
        print(f"Error getting pipeline execution ARN: {e}")
        raise e


def get_cloudwatch_logs(
    logs_client, log_group: str, log_stream: str, start_time: int, end_time: int
) -> Iterator[Dict]:
    """Get logs containing 'INFO' and either 'Epoch' or 'eval' from CloudWatch as a generator."""
    paginator = logs_client.get_paginator("filter_log_events")

    for page in paginator.paginate(
        logGroupName=log_group,
        logStreamNames=[log_stream],
        startTime=start_time,
        endTime=end_time,
        filterPattern="INFO ?Epoch ?eval",
    ):
        events_generator: Iterator = page.get("events", [])
        yield from events_generator


def analyze_logs(
    log_source: Union[str, tuple[str, str, str]],
    days_before: int = 2,
):
    """
    Analyze logs from either CloudWatch or a local file.

    Args:
        log_source: Either a path to a local file (str) or a tuple of
        (pipeline_name, execution_id, step_name)
        days_before: Number of days in the past to start analyzing logs
    """

    # Gather events, either from file or from CloudWatch
    if isinstance(log_source, str):
        print(f"Reading logs from file: {log_source}")
        log_events = read_local_logs(log_source)
    else:
        try:
            start_time = int(
                (datetime.now() - timedelta(days=days_before)).timestamp() * 1000
            )
            end_time = int(datetime.now().timestamp() * 1000)

            # Unpack the logs source
            pipeline_name, execution_name, step_name = log_source

            # Get the training job name which is the prefix of the log stream
            train_job_id = get_training_job_name(
                pipeline_name, execution_name, step_name
            )

            # Get the log stream
            logs_client = boto3.client("logs")
            # Get log streams that match the prefix
            log_streams_response = logs_client.describe_log_streams(
                logGroupName=LOG_GROUP,
                logStreamNamePrefix=train_job_id,
            )

            for log_stream in log_streams_response["logStreams"]:
                if "algo-1" in log_stream["logStreamName"]:
                    log_stream_name = log_stream["logStreamName"]
                    break
            else:
                raise RuntimeError(
                    f"No log stream found with prefix {train_job_id}/algo-1"
                )

            print(f"Analyzing log stream: {log_stream_name}")
            print(f"Time range: {datetime.fromtimestamp(start_time/1000)}")
            print(f"         to: {datetime.fromtimestamp(end_time/1000)}")

            log_events = get_cloudwatch_logs(
                logs_client, LOG_GROUP, log_stream, start_time, end_time
            )
        except Exception as e:
            print(f"Error while retrieving logs from CloudWatch: {e}")
            raise e

    # Patterns for both types of logs
    epoch_pattern = re.compile(r"INFO:root:Epoch (\d+) take (\d+\.\d+) seconds")
    eval_pattern = re.compile(
        r"INFO:root: Eval time: (\d+\.\d+), Evaluation step: (\d+)"
    )
    epochs_data = []
    eval_data = []

    for event in log_events:
        epoch_match = epoch_pattern.search(event["message"])
        eval_match = eval_pattern.search(event["message"])

        if epoch_match:
            epochs_data.append(
                {
                    "epoch": int(epoch_match.group(1)),
                    "time": float(epoch_match.group(2)),
                    "timestamp": datetime.fromtimestamp(event["timestamp"] / 1000),
                }
            )
        elif eval_match:
            eval_data.append(
                {
                    "time": float(eval_match.group(1)),
                    "step": int(eval_match.group(2)),
                    "timestamp": datetime.fromtimestamp(event["timestamp"] / 1000),
                }
            )

    # We have gathered the relevant events, return for processing
    return epochs_data, eval_data


def get_training_job_name(pipeline_name: str, execution_id: str, step_name: str) -> str:
    """Get training job name for a step in a specific pipeline execution"""
    sm_client = boto3.client("sagemaker")

    try:
        # Get the full execution ARN first
        execution_arn = get_pipeline_execution_arn(pipeline_name, execution_id)
        print(f"Found execution ARN: {execution_arn}")

        # Get the pipeline execution details
        execution_steps = sm_client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )

        # Find the desired step
        target_step = None
        for step in execution_steps["PipelineExecutionSteps"]:
            if step["StepName"] == step_name:
                target_step = step
                break
        else:
            raise ValueError(f"Step '{step_name}' not found in pipeline execution")

        # Get the training job name from metadata
        metadata = target_step["Metadata"]
        if "TrainingJob" not in metadata:
            raise ValueError(f"No training job found in step '{step_name}'")

        training_job_name = metadata["TrainingJob"]["Arn"].split("/")[-1]

        return training_job_name

    except Exception as e:
        print(f"Error while getting training job name: {e}")
        raise e


def print_training_summary(
    epochs_data: List[Dict], eval_data: List[Dict], verbose: bool
):
    """Prints a summary of the epoch time and eval time for a GraphStorm training job"""

    print("\n=== Training Epochs Summary ===")
    if epochs_data:
        total_epochs = len(epochs_data)
        avg_time = sum(e["time"] for e in epochs_data) / total_epochs

        print(f"Total epochs completed: {total_epochs}")
        print(f"Average epoch time: {avg_time:.2f} seconds")

        if verbose:
            print("\nEpoch Details:")
            for data in epochs_data:
                print(
                    f"Epoch {data['epoch']:3d}: {data['time']:6.2f} seconds  "
                    f"[{data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}]"
                )

    print("\n=== Evaluation Summary ===")
    if eval_data:
        total_evals = len(eval_data)
        avg_eval_time = sum(e["time"] for e in eval_data) / total_evals

        print(f"Total evaluations: {total_evals}")
        print(f"Average evaluation time: {avg_eval_time:.2f} seconds")

        if verbose:
            print("\nEvaluation Details:")
            for data in eval_data:
                print(
                    f"Step {data['step']:4d}: {data['time']:6.2f} seconds  "
                    f"[{data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}]"
                )


if __name__ == "__main__":
    args = parse_args()
    client = boto3.client("logs", region_name=args.region)
    if args.log_file:
        log_representation = args.log_file
    else:
        log_stream_prefix = get_training_job_name(
            args.pipeline_name, args.execution_name, "Training"
        )
        log_representation = (args.pipeline_name, args.execution_name, "Training")
        # Get the training job name which is the prefix of the log stream
        print(f"Found training job: {log_stream_prefix}")

    retrieved_epochs_data, retrieved_eval_data = analyze_logs(
        log_representation, args.logs_days_before
    )

    print_training_summary(retrieved_epochs_data, retrieved_eval_data, args.verbose)
