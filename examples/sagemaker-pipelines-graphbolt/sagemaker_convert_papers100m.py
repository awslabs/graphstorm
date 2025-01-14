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

Launch SageMaker job to convert papers100M data and prepare for input to GConstruct
"""
import argparse
import os

from sagemaker.processing import ScriptProcessor
from sagemaker.network import NetworkConfig
from sagemaker import get_execution_role

_ROOT = os.path.abspath(os.path.dirname(__file__))


def parse_args() -> argparse.Namespace:
    """Parse job launch arguments"""
    parser = argparse.ArgumentParser(
        description="Convert Papers100M dataset to GConstruct format using SageMaker Processing."
    )

    parser.add_argument(
        "--execution-role-arn",
        type=str,
        default=None,
        help="SageMaker Execution Role ARN",
    )
    parser.add_argument(
        "--region", type=str, required=True, help="SageMaker Processing region."
    )
    parser.add_argument("--image-uri", type=str, required=True,
        help="URI for the 'papers100m-processor' image.")
    parser.add_argument(
        "--output-bucket",
        type=str,
        required=True,
        help="S3 output bucket for processed papers100M data. "
        "Data will be saved under ``<output-bucket>/ogb-papers100M-input/``",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.m5.4xlarge",
        help="SageMaker Processing Instance type.",
    )

    return parser.parse_args()


def main():
    """Launch the papers100M conversion job on SageMaker"""
    args = parse_args()

    # Create a ScriptProcessor to run the processing bash script
    script_processor = ScriptProcessor(
        command=["bash"],
        image_uri=args.image_uri,
        role=args.execution_role_arn or get_execution_role(),
        instance_count=1,
        instance_type=args.instance_type,
        volume_size_in_gb=400,
        max_runtime_in_seconds=8 * 60 * 60,  # Adjust as needed
        base_job_name="papers100m-processing",
        network_config=NetworkConfig(
            enable_network_isolation=False
        ),  # Enable internet access to be able to download the data
    )

    # Submit the processing job
    script_processor.run(
        code="process_papers100M.sh",
        inputs=[],
        outputs=[],
        arguments=[
            "convert_ogb_papers100m_to_gconstruct.py",
            f"s3://{args.output_bucket}/papers-100M-input",
        ],
        wait=False,
    )


if __name__ == "__main__":
    main()
