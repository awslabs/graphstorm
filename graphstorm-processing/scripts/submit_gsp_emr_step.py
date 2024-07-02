"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Launches an EMR on EC2 cluster a submits a GSProcessing job as an EMR Step.

Usage:
    python submit_gsp_emr_step.py \
            --entry-point-s3 ${S3_ENTRY_POINT} \
            --worker-count ${NUM_WORKERS} \
            --instance-type ${INSTANCE_TYPE} \
            --gsp-arguments "--config-filename ${CONFIG_FILE} \
                --input-prefix ${INPUT_PREFIX} \
                --output-prefix ${OUTPUT_PREFIX} \
                --add-reverse-edges ${GENERATE_REVERSE} \
                --do-repartition ${REPARTITION_ON_LEADER}"
"""

import argparse
import os
from pprint import pprint

import boto3


def parse_args():
    """Parses script arguments."""
    parser = argparse.ArgumentParser(description="Launch a GSP EMR task")

    # Cluster setup params
    parser.add_argument(
        "--gsp-ecr-repository-name",
        default="graphstorm-processing-emr",
        help=("GSProcessing EMR on EC2 ECR repository name, default graphstorm-processing-emr"),
    )
    parser.add_argument(
        "--gsp-ecr-registry",
        required=False,
        default=None,
        help=("GSProcessing ECR registry, default <ACCOUNT>.dkr.ecr.<REGION>.amazonaws.com"),
    )
    parser.add_argument(
        "--gsp-image-tag",
        default="latest-x86_64",
        help=("GSProcessing EMR image tag, default is latest-x86_64"),
    )
    parser.add_argument("--timeout-hours", type=int, default=1)
    parser.add_argument(
        "--instance-count",
        type=int,
        required=True,
        help="Number of worker instances. Required",
    )
    parser.add_argument(
        "--instance-type",
        required=True,
        help="EC2 instance type to use for the EMR cluster. Required.",
    )
    parser.add_argument(
        "--log-uri",
        default=None,
        help="S3 URI to use for log storage. If not provided, no logs are stored.",
    )
    parser.add_argument(
        "--service-role",
        default="EMR_DefaultRole_V2",
        help="EMR service role to use for the EMR cluster. Defaults to EMR_DefaultRole.",
    )
    parser.add_argument(
        "--jobflow-role",
        default="EMR_EC2_DefaultRole",
        help="EC2 role to use for the EMR cluster instances. Defaults to EMR_EC2_DefaultRole.",
    )
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument(
        "--emr-version",
        default="emr-7.0.0",
        help="EMR version to use for the EMR cluster. Defaults to emr-7.0.0.",
    )

    # GSP params
    parser.add_argument(
        "--entry-point-s3",
        type=str,
        required=True,
        help="S3 URI for the GSProcessing entry point script, distributed_executor.py. Required",
    )
    parser.add_argument(
        "--gsp-arguments",
        type=str,
        required=True,
        help=(
            "GSProcessing arguments, in a space-separated string, e.g. "
            "'--input-prefix s3://my-bucket/raw_data --config-file gsp_input.json'. Required."
        ),
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    if not args.region:
        region = (
            os.environ["AWS_REGION"]
            if "AWS_REGION" in os.environ
            else boto3.client("s3").meta.region_name
        )
    else:
        region = args.region
    if not args.gsp_ecr_registry:
        account_id = boto3.client("sts").get_caller_identity().get("Account")
        gsp_ecr_registry = f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    ecr_repo_name = args.gsp_ecr_repository_name
    image_tag = args.gsp_image_tag
    timeout_hours = args.timeout_hours
    instance_type = args.instance_type
    core_instance_count = args.instance_count

    s3_entry_point = args.entry_point_s3

    bucket = s3_entry_point.split("/")[2]
    key = s3_entry_point.split("/", maxsplit=3)[3]
    # Use boto to ensure the entrypoint file exists
    s3 = boto3.resource("s3")
    try:
        s3.Object(bucket, key).load()
    except boto3.client.ClientError as e:  # pylint: disable=no-member
        raise ValueError(f"Invalid entry point S3 URI: {s3_entry_point}") from e

    # Set up EMR client
    emr = boto3.client("emr", region_name=region)

    assert isinstance(args.gsp_arguments, str)
    task_arguments = [
        "spark-submit",
        "--master",
        "yarn",
        "--deploy-mode",
        "cluster",
        "--conf",
        "spark.executorEnv.YARN_CONTAINER_RUNTIME_TYPE=docker",
        "--conf",
        (
            "spark.executorEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE="
            f"{gsp_ecr_registry}/{ecr_repo_name}:{image_tag}"
        ),
        "--conf",
        "spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_TYPE=docker",
        "--conf",
        (
            "spark.yarn.appMasterEnv.YARN_CONTAINER_RUNTIME_DOCKER_IMAGE="
            f"{gsp_ecr_registry}/{ecr_repo_name}:{image_tag}"
        ),
        "--conf",
        "spark.executorEnv.PYSPARK_PYTHON=/.pyenv/shims/python",
        "--conf",
        "spark.yarn.appMasterEnv.PYSPARK_PYTHON=/.pyenv/shims/python",
        s3_entry_point,
    ] + args.gsp_arguments.split()

    # Define cluster configuration
    cluster_config = {
        "Name": f"gsp-emr-{core_instance_count}x-{instance_type}",
        "ReleaseLabel": args.emr_version,
        "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}],
        "Configurations": [
            {
                "Classification": "container-executor",
                "Configurations": [
                    {
                        "Classification": "docker",
                        "Properties": {
                            "docker.privileged-containers.registries": f"local,centos,{gsp_ecr_registry}",  # pylint: disable=line-too-long
                            "docker.trusted.registries": f"local,centos,{gsp_ecr_registry}",
                        },
                    }
                ],
                "Properties": {},
            },
            {
                "Classification": "spark",
                "Properties": {"maximizeResourceAllocation": "true"},
            },
        ],
        "ServiceRole": args.service_role,
        "JobFlowRole": args.jobflow_role,
        "VisibleToAllUsers": True,
        "AutoTerminationPolicy": {"IdleTimeout": timeout_hours * 60 * 60},
        "Instances": {
            "InstanceGroups": [
                {
                    "Name": "Master node",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "MASTER",
                    "InstanceType": instance_type,
                    "InstanceCount": 1,
                },
                {
                    "Name": "Core nodes",
                    "Market": "ON_DEMAND",
                    "InstanceRole": "CORE",
                    "InstanceType": instance_type,
                    "InstanceCount": core_instance_count,
                },
            ],
            "KeepJobFlowAliveWhenNoSteps": False,
        },
        "ScaleDownBehavior": "TERMINATE_AT_TASK_COMPLETION",
        "Steps": [
            {
                "Name": "GSProcessing",
                "ActionOnFailure": "TERMINATE_CLUSTER",
                "HadoopJarStep": {"Jar": "command-runner.jar", "Args": task_arguments},
            }
        ],
        "Tags": [{"Key": "Launching user", "Value": f"{os.environ.get('USER', 'None')}"}],
    }

    print("Submitted task:")
    pprint(task_arguments)

    if args.log_uri:
        cluster_config["LogUri"] = args.log_uri

    # Create the cluster with GSP EMR step attached
    response = emr.run_job_flow(**cluster_config)
    cluster_id = response["JobFlowId"]
    print(f"Cluster being created with ID: {cluster_id}")


if __name__ == "__main__":
    main()
