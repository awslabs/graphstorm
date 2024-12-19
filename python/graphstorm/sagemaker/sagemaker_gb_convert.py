""" Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    SageMaker implementation of DistDGL to GraphBolt conversion.
"""

import importlib.metadata
import logging
import os
import time
from collections import defaultdict
from packaging import version

import boto3
import sagemaker
from sagemaker.s3 import S3Uploader

from graphstorm.gpartition.convert_to_graphbolt import run_gb_conversion


def run_gb_convert(s3_output_path: str, local_dist_part_config: str, njobs: int):
    """Convert DistDGL partitions to GraphBolt format on SageMaker Processing.


    Parameters
    ----------
    s3_output_path : str
        The S3 prefix under which we will upload the GB converted graph files.
    local_dist_part_config : str
        The local path to the downloaded partition data metadata.json file.
    njobs: int
        Number of parallel processes to use during GraphBolt conversion.
        One process per partition is used, so ``njobs`` partitions needs
        to be able to fit into memory.

    Raises
    ------
    ValueError
        If the version of DGL used is under 2.1.0
    """
    assert os.path.isfile(
        local_dist_part_config
    ), f"{local_dist_part_config=} does not exist."

    dgl_version = importlib.metadata.version("dgl")
    if version.parse(dgl_version) < version.parse("2.1.0"):
        raise ValueError(
            "GraphBolt conversion requires DGL version >= 2.1.0, "
            f"but DGL version was {dgl_version}. "
        )

    boto_session = boto3.Session(region_name=os.environ["AWS_REGION"])
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # Run the actual conversion, this will create the fused_csc_sampling_graph.pt
    # under each partition in local_dist_part_path and a new metadata.json
    gb_start = time.time()
    run_gb_conversion(local_dist_part_config, njobs)
    logging.info("GraphBolt conversion took %f sec.", time.time() - gb_start)

    # Iterate through the partition data and upload only the modified/new
    # files to the corresponding path on S3
    upload_start = time.time()
    fused_files_exist = defaultdict(lambda: False)
    for root, _, files in os.walk(os.path.dirname(local_dist_part_config)):
        for file in files:
            if file.endswith("fused_csc_sampling_graph.pt"):
                partition_id = root.split("/")[-1]
                # Set fused file existence to true for this partition
                fused_files_exist[partition_id] = True
                # Partition data need to be uploaded to partition-id dirs
                s3_path = os.path.join(s3_output_path, f"{partition_id}")
            elif file.endswith(".json"):
                # Partition output metadata file needs to be uploaded to root dir
                s3_path = s3_output_path
            else:
                # We skip other files
                partition_id = root.split("/")[-1]
                if "part" in partition_id:
                    # Set file existence to False only if
                    # we haven't encountered a fused file already
                    fused_files_exist[partition_id] = (
                        False or fused_files_exist[partition_id]
                    )
                continue

            logging.info("Uploading local %s to %s", os.path.join(root, file), s3_path)
            S3Uploader.upload(
                local_path=os.path.join(root, file),
                desired_s3_uri=s3_path,
                sagemaker_session=sagemaker_session,
            )

    for partition_id, fused_file_exists in fused_files_exist.items():
        if not fused_file_exists:
            raise RuntimeError(
                f"Partition {partition_id} did not have "
                "a fused_csc_sampling_graph.pt file."
            )

    logging.info("Uploading took %f sec.", time.time() - upload_start)
