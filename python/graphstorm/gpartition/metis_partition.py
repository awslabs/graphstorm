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

    Parmetis partition assignment
"""
import os
import logging
import json
import subprocess
import sys
import shutil

from .partition_algo_base import LocalPartitionAlgorithm
from .partition_config import ParMETISConfig


class ParMetisPartitionAlgorithm(LocalPartitionAlgorithm):
    """
    Multiple-instances metis partitioning algorithm.

    The partition algorithm accepts the intermediate output from GraphStorm
    gs-processing which matches the requirements of the DGL distributed
    partitioning pipeline.


    Parameters
    ----------
    metadata_dict: dict
        DGL "Chunked graph data" JSON, as defined in
        https://docs.dgl.ai/guide/distributed-preprocessing.html#specification
    metis_config: ParMETISConfig
        Configuration object for ParMETIS.
    """

    def __init__(self, metadata_dict: dict, metis_config: ParMETISConfig):
        super().__init__(metadata_dict)
        self.metis_config = metis_config

    def _launch_preprocess(self, num_parts, input_path, ip_list, dgl_tool_path, metadata_filename):
        """ Launch preprocessing script

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        input_path: str
            Path to the input graph data
        ip_list: str
            ip list file after port assigned
        dgl_tool_path: str
            Path to the dgl tool added in the PYTHONPATH
        metadata_filename: str
            Meta data configuration name
        """
        command = f"mpirun -np {num_parts} --allow-run-as-root --hostfile {ip_list} \
                    -wdir {input_path} \
                  {sys.executable} {dgl_tool_path}/distpartitioning/parmetis_preprocess.py \
                    --input_dir {input_path} \
                    --schema_file {metadata_filename} \
                    --output_dir {input_path} --num_parts {num_parts}"

        if self.run_command(command, "preprocess"):
            # parmetis_preprocess.py creates this file, but doesn't put it in the cwd,
            # where the parmetis program (pm_dglpart) expects it to be.
            # So we copy it from the location parmetis_preprocess saves it to the cwd.
            # https://github.com/dmlc/dgl/blob/cbad2f0af317dce2af1771c131b7eea92ae7c8a7/tools/distpartitioning/parmetis_preprocess.py#L318
            with open(os.path.join(input_path, metadata_filename), encoding="utf-8") as f:
                graph_meta = json.load(f)
            graph_name = graph_meta["graph_name"]
            shutil.copy(
                os.path.join(input_path, f"{graph_name}_stats.txt"),
                f"{graph_name}_stats.txt",
            )

            logging.info("Successfully executed parmetis preprocess.")
            return True
        else:
            logging.info("Failed to execute parmetis preprocess.")
            return False


    def _launch_parmetis(self, num_parts, input_path, ip_list, graph_name):
        """ Launch parmetis script

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        input_path: str
            Path to the input graph data
        ip_list: str
            ip list
        graph_name: str
            Graph name
        """
        assert os.path.exists(os.path.expanduser("~/local/bin/pm_dglpart")), \
            "pm_dglpart not found in ~/local/bin/"
        # TODO: ParMETIS also claims to support num_workers != num_parts, we can test
        # if it's possible to speed the process up by using more workers than partitions
        command = f"mpirun -np {num_parts} --allow-run-as-root \
                    --hostfile {ip_list} \
                    --mca orte_base_help_aggregate 0 -mca btl_tcp_if_include eth0 \
                    -wdir {input_path} \
                ~/local/bin/pm_dglpart {graph_name} {num_parts} {input_path}/parmetis_nfiles.txt \
                  {input_path}/parmetis_efiles.txt"

        if self.run_command(command, "parmetis"):
            logging.info("Successfully execute parmetis process.")
            return True
        else:
            logging.info("Failed to execute parmetis process.")
            return False

    def _launch_postprocess(self, num_parts, input_data_path, dgl_tool_path,
                            metadata_filename, graph_name, partition_dir):
        """ Launch postprocess which translates nid-partid mapping into
            Per-node-type partid mappings.

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        input_data_path: str
            Path to the input graph data
        dgl_tool_path: str
            Path to the dgl tool added in the PYTHONPATH
        metadata_filename: str
            Meta data configuration name
        graph_name: str
            name of the graph in the parmetis step
        partition_dir: str
            output path
        """
        command = f"{sys.executable} {dgl_tool_path}/distpartitioning/parmetis_postprocess.py \
                        --postproc_input_dir {input_data_path} \
                        --schema_file {metadata_filename} \
                        --parmetis_output_file {input_data_path}/{graph_name}_part.{num_parts} \
                        --partitions_dir {partition_dir}"

        if self.run_command(command, "postprocess"):
            logging.info("Successfully execute post parmetis process.")
            return True
        else:
            logging.info("Failed to execute post parmetis process.")
            return False

    def run_command(self, command, stream):
        """Function to execute a command and check for its success."""
        logging.info("Executing command: %s", command)
        try:
            # Execute the command and check if it completes successfully
            result = subprocess.run(command, shell=True, check=True, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if stream in ("preprocess", "postprocess"):
                logging.info("Command output: %s", result.stderr)
            else:
                logging.info("Command output: %s", result.stdout)
            return True  # Return True if the command was successful
        except subprocess.CalledProcessError as e:
            logging.info("Error executing command: %s", e.stderr)
            return False  # Return False if the command failed

    def assigned_port(self, ip_file, port="2222"):
        """Function to assigned port to each ip prepared for mpi."""
        if not os.path.isfile(ip_file):
            raise ValueError("ip file does not exist")
        # MPI run will need to explicitly assign port=2222 in the ip list file
        # when running in the docker environment
        with open(ip_file, 'r', encoding='utf-8') as file:
            # Read all lines from the input file
            ip_addresses = file.readlines()

        base, ext = os.path.splitext(ip_file)
        output_file = f"{base}_parmetis{ext if ext else ''}"
        with open(output_file, 'w', encoding='utf-8') as file:
            # Write each IP address with the appended port information
            for ip in ip_addresses:
                ip = ip.strip()  # Remove any leading/trailing whitespace
                file.write(f"{ip} port={port}\n")
        return output_file

    def _assign_partitions(self, num_partitions: int, partition_dir: str):
        ip_file = self.assigned_port(self.metis_config.ip_list)
        # Execute each command function in sequence and stop if any fails
        if not self._launch_preprocess(num_partitions, self.metis_config.input_path,
                                       ip_file, self.metis_config.dgl_tool_path,
                                       self.metis_config.metadata_filename):
            raise RuntimeError("Stopping execution due to failure in preprocess")
        if not self._launch_parmetis(num_partitions, self.metis_config.input_path,
                                       ip_file, self.metadata_dict["graph_name"]):
            raise RuntimeError("Stopping execution due to failure in parmetis partition process")
        if not self._launch_postprocess(num_partitions, self.metis_config.input_path,
                                        self.metis_config.dgl_tool_path,
                                        self.metis_config.metadata_filename,
                                        self.metadata_dict["graph_name"], partition_dir):
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
