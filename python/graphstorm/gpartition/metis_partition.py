"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Single-instance metis partition assignment
"""
import os
import logging
import json

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from .partition_algo_base import LocalPartitionAlgorithm


class MetisPartitionAlgorithm(LocalPartitionAlgorithm):
    """
    Multiple-instances metis partitioning algorithm.

    The partition algorithm accepts the intermediate output from GraphStorm
    gs-processing which matches the requirements of the DGL distributed
    partitioning pipeline.


    Parameters
    ----------
    metadata: dict
        DGL "Chunked graph data" JSON, as defined in
        https://docs.dgl.ai/guide/distributed-preprocessing.html#specification
    """

    def _launch_preprocess(self, num_parts, input_data_path, metadata_filename, output_path):
        return "Not Implmentation error"

    def _launch_parmetis(
            self,
            num_parts,
            net_ifname,
            input_data_path,
            metis_input_path):
        """ Launch parmetis script

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        net_ifname: str
            Network interface used by MPI
        input_data_path: str
            Path to the input graph data
        metis_input_path: str
            Path to metis input
        """
        return "Not Implmentation error"

    def _launch_postprocess(self, meta_data_config, parmetis_output_file, partitions_dir):
        """ Launch postprocess which translates nid-partid mapping into
            Per-node-type partid mappings.

        Parameters
        ----------
        meta_data_config: str
            Path to the meta data configuration.
        parmetis_output_file: str
            Path to ParMetis output.
        partitions_dir: str
            Output path
        """
        return "Not Implmentation error"

    def run_command(command):
        """Function to execute a command and check for its success."""
        print(f"Executing command: {command}")
        try:
            # Execute the command and check if it completes successfully
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            print(f"Command output: {result.stdout}")
            return True  # Return True if the command was successful
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e.stderr}")
            return False  # Return False if the command failed

    def _assign_partitions(self, num_partitions: int, partition_dir: str):
        # Execute each command function in sequence and stop if any fails
        if not self._launch_preprocess():
            raise RuntimeError("Stopping execution due to failure in preprocess")
        elif not self._launch_parmetis():
            raise RuntimeError("Stopping execution due to failure in parmetis partition process")
        elif not self._launch_postprocess():
            raise RuntimeError("Stopping execution due to failure in postprocess process")

        logging.info("Finish all parmetis steps.")

    def _create_metadata(self, num_partitions: int, partition_dir: str):
        partition_meta = {
            "algo_name": "parmetis",
            "num_parts": num_partitions,
            "version": "1.0.0"
        }
        partition_meta_filepath = os.path.join(partition_dir, "partition_meta.json")
        with open(partition_meta_filepath, "w", encoding='utf-8') as metafile:
            json.dump(partition_meta, metafile)