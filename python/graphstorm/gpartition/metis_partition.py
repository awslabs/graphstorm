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
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from .partition_algo_base import LocalPartitionAlgorithm


class ParMetisPartitionAlgorithm(LocalPartitionAlgorithm):
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

    def __init__(self, metadata_dict, metis_config):
        super().__init__(metadata_dict)
        self.metis_config = metis_config

    def _launch_preprocess(self, num_parts, input_path, ip_list, dgl_tool_path, metadata_filename):
        command = f"mpirun -np 1 --allow-run-as-root --hostfile {ip_list} \
                    -wdir {input_path} \
                  python3 {dgl_tool_path}/distpartitioning/parmetis_preprocess.py \
                    --input_dir {input_path} \
                    --schema_file {metadata_filename} \
                    --output_dir {input_path} --num_parts {num_parts}"

        if self.run_command(command):
            logging.info("Successfully execute parmetis preprocess.")
            return True
        else:
            logging.info("Failed to execute parmetis preprocess.")
            return False


    def _launch_parmetis(self, num_parts, input_path, ip_list, graph_name):
        """ Launch parmetis script

        Parameters
        ----------
        """
        command = f"mpirun -np 1 --allow-run-as-root \
                    --hostfile {ip_list} \
                    --mca orte_base_help_aggregate 0 -mca btl_tcp_if_include eth0 \
                    -wdir {input_path} \
                ~/local/bin/pm_dglpart {graph_name} {num_parts} {input_path}/parmetis_nfiles.txt \
                  {input_path}/parmetis_efiles.txt"

        if self.run_command(command):
            logging.info("Successfully execute parmetis preprocess.")
            return True
        else:
            logging.info("Failed to execute parmetis preprocess.")
            return False

    def _launch_postprocess(self, num_parts, input_data_path, dgl_tool_path, metadata_filename, graph_name, partition_dir):
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
        command = f"python3 {dgl_tool_path}/distpartitioning/parmetis_postprocess.py \
                        --postproc_input_dir {input_data_path} \
                        --schema_file {metadata_filename} \
                        --parmetis_output_file {input_data_path}/{graph_name}_part.{num_parts} \
                        --partitions_dir {partition_dir}"

        if self.run_command(command):
            logging.info("Successfully execute post parmetis preprocess.")
            return True
        else:
            logging.info("Failed to execute post parmetis preprocess.")
            return False

    def run_command(self, command):
        """Function to execute a command and check for its success."""
        logging.info(f"Executing command: {command}")
        try:
            # Execute the command and check if it completes successfully
            result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            logging.info(f"Command output: {result.stdout}\n")
            return True  # Return True if the command was successful
        except subprocess.CalledProcessError as e:
            logging.info(f"Error executing command: {e.stderr}\n")
            return False  # Return False if the command failed

    def assigned_port(self, ip_file, port="2222"):
        # MPI run will need to explicitly assign port=2222 in the ip list file
        # when running in the docker environment
        with open(ip_file, 'r') as file:
            # Read all lines from the input file
            ip_addresses = file.readlines()

        parts = ip_file.rsplit('.', 1)
        if len(parts) == 2 and parts[1] == 'txt':
            output_file = f"{parts[0]}_parmetis}.{parts[1]}"
        else:
            raise ValueError("Input file should be a txt file.")
        with open(output_file, 'w') as file:
            # Write each IP address with the appended port information
            for ip in ip_addresses:
                ip = ip.strip()  # Remove any leading/trailing whitespace
                file.write(f"{ip} port={port}\n")
        return output_file

    def _assign_partitions(self, num_partitions: int, partition_dir: str):
        # TODO: adjust ip_list file input format inside
        ip_file = self.assigned_port(self.metis_config.ip_list)
        # Execute each command function in sequence and stop if any fails
        if not self._launch_preprocess(num_partitions, self.metis_config.input_path,
                                       ip_file, self.metis_config.dgl_tool_path,
                                       self.metis_config.metadata_filename):
            raise RuntimeError("Stopping execution due to failure in preprocess")
        if not self._launch_parmetis(num_partitions, self.metis_config.input_path,
                                       ip_file, self.metadata_dict["graph_name"]):
            raise RuntimeError("Stopping execution due to failure in parmetis partition process")
        if not self._launch_postprocess(num_partitions, self.metis_config.input_path, self.metis_config.dgl_tool_path,
                                          self.metis_config.metadata_filename, self.metadata_dict["graph_name"],
                                          partition_dir):
            raise RuntimeError("Stopping execution due to failure in postprocess process")

        logging.info("Finish all parmetis steps.")

    def _create_metadata(self, num_partitions: int, partition_dir: str):
        partition_meta = {
            "algo_name": "metis",
            "num_parts": num_partitions,
            "version": "1.0.0"
        }
        partition_meta_filepath = os.path.join(partition_dir, "partition_meta.json")
        with open(partition_meta_filepath, "w", encoding='utf-8') as metafile:
            json.dump(partition_meta, metafile)
