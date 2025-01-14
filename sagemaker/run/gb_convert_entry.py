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

    SageMaker GraphBolt partition conversion entry point.
"""
import argparse
import json
import logging
import os

from graphstorm.sagemaker.sagemaker_gb_convert import run_gb_convert


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="Number of parallel jobs to run for partition conversion.",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata.json",
        help="Partition metadata file name. Default: 'metadata.json'. If GConstruct was "
        "used should be set to <graph-name>.json",
    )
    args = parser.parse_args()

    # NOTE: Ensure no logging has been done before setting logging configuration
    logging.basicConfig(
        level=getattr(logging, "INFO", None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Read processing config file
    with open("/opt/ml/config/processingjobconfig.json", "r", encoding="utf-8") as f:
        processing_config = json.load(f)

    processing_inputs: list[dict] = processing_config["ProcessingInputs"]

    # Find the S3 input path from the job's configuration
    for input_entry in processing_inputs:
        if input_entry["InputName"] == "dist_graph_s3_input":
            local_partition_path = input_entry["S3Input"]["LocalPath"]
            s3_partition_path = input_entry["S3Input"]["S3Uri"]
            break
    else:
        raise ValueError(
            "Could not find a Processing input named 'dist_graph_s3_input'"
        )

    # Run the conversion locally
    logging.info("Running graph partition conversion...")
    print(f"{local_partition_path=}, {s3_partition_path=}")
    run_gb_convert(
        s3_partition_path,
        local_dist_part_config=os.path.join(
            local_partition_path, args.metadata_filename
        ),
        njobs=args.njobs,
    )
