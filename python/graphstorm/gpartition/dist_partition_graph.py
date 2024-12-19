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

    Run local partition for distributed data processing.
    This script only works with data in DGL's chunked format,
    for example, the output of graphstorm-processing.
    See https://docs.dgl.ai/guide/distributed-preprocessing.html#specification
"""
import argparse
import json
import logging
import os
import queue
import time
import shutil
import subprocess
import sys
import importlib.metadata
from typing import Dict
from threading import Thread

from dgl import distributed as dgl_distributed
from packaging import version

from graphstorm.gpartition import (
    ParMetisPartitionAlgorithm,
    ParMETISConfig,
    RandomPartitionAlgorithm,
    shuffle_hard_negative_nids,
)
from graphstorm.utils import get_log_level


def run_build_dglgraph(
        input_data_path,
        partitions_dir,
        ip_list,
        output_path,
        metadata_filename,
        dgl_tool_path,
        ssh_port,
        process_group_timeout):
    """ Build DistDGL Graph

    Parameters
    ----------
    input_data_path: str
        Path to the input graph data
    partitions_dir: str
        Path to Per-node-type partid mappings.
    ip_list: str
        Path to a file storing ips of instances
    output_path: str
        Output Path
    metadata_filename: str
        The filename for the graph partitioning metadata file we'll use to determine data sources.
    ssh_port: int
        SSH port
    process_group_timeout: int
        Timeout[seconds] for operations executed against the process group.
    """
    # Get the python interpreter used right now.
    # If we can not get it we go with the default `python3`
    python_bin = sys.executable \
        if sys.executable is not None and sys.executable != "" \
        else "python3 "
    state_q = queue.Queue()

    launch_cmd = ["python3", f"{dgl_tool_path}/dispatch_data.py",
        "--in-dir", input_data_path,
        "--metadata-filename", metadata_filename,
        "--partitions-dir", partitions_dir,
        "--ip-config", ip_list,
        "--out-dir", output_path,
        "--ssh-port", f"{ssh_port}",
        "--python-path", f"{python_bin}",
        "--log-level", logging.getLevelName(logging.root.getEffectiveLevel()),
        "--process-group-timeout", str(process_group_timeout),
        "--save-orig-nids",
        "--save-orig-eids"]

    # thread func to run the job
    def run(cmd, state_q):
        try:
            subprocess.check_call(cmd, shell=False)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            logging.error("Called process error %s", err)
            state_q.put(err.returncode)
        except Exception: # pylint: disable=broad-exception-caught
            state_q.put(-1)

    # launch postprocess task
    thread = Thread(target=run, args=(launch_cmd, state_q,), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)

    thread.join()
    err_code = state_q.get()
    if err_code != 0:
        raise RuntimeError("Building DistDGL graph failed")


def main():
    """Main entry point"""
    args = parse_args()
    # Configure logging
    logging.basicConfig(level=get_log_level(args.logging_level))

    output_path: str = args.output_path
    metadata_file: str = args.metadata_filename

    with open(os.path.join(args.input_path, metadata_file), "r", encoding="utf-8") as f:
        metadata_dict: Dict = json.load(f)

    part_start = time.time()
    if args.partition_algorithm == "random":
        partitioner = RandomPartitionAlgorithm(metadata_dict)
    elif args.partition_algorithm == "parmetis":
        partition_config = ParMETISConfig(args.ip_config, args.input_path,
                                          args.dgl_tool_path, args.metadata_filename)
        partitioner = ParMetisPartitionAlgorithm(metadata_dict, partition_config)
    else:
        raise RuntimeError(f"Unknown partition algorithm {args.part_algorithm}")

    part_assignment_dir = os.path.join(output_path, "partition_assignment")
    os.makedirs(part_assignment_dir, exist_ok=True)

    partitioner.create_partitions(
        args.num_parts,
        part_assignment_dir)

    logging.info("Partition assignment with algorithm '%s' took %f sec",
                 args.partition_algorithm,
                 time.time() - part_start,
    )

    if not args.partition_assignment_only:
        dgl_graph_start = time.time()
        run_build_dglgraph(
            args.input_path,
            part_assignment_dir,
            args.ip_config,
            os.path.join(output_path, "dist_graph"),
            args.metadata_filename,
            args.dgl_tool_path,
            args.ssh_port,
            args.process_group_timeout)

        logging.info("DGL graph building took %f sec", dgl_graph_start - time.time())

        if args.use_graphbolt:
            gb_start = time.time()
            dgl_version = importlib.metadata.version('dgl')
            if version.parse(dgl_version) >= version.parse("2.1.0"):
                # TODO: Implement distributed conversion using
                # dgl.distributed.partition.gb_convert_single_dgl_partition()
                logging.info("Converting partitions to GraphBolt format")
                # NOTE: The partition conversion happens on just the leader
                # and a shared filesystem is assumed to hold the graph data.
                dgl_distributed.dgl_partition_to_graphbolt(
                    os.path.join(output_path, "dist_graph", "metadata.json"),
                    store_eids=True,
                    graph_formats="coo",
                )
                logging.info("GraphBolt conversion took %f sec.",
                             time.time() - gb_start)
            else:
                raise ValueError(
                    f"use_graphbolt was 'true' but but DGL version was {dgl_version}. "
                    "GraphBolt graph construction requires DGL version >= 2.1.0"
                )

    # Copy raw_id_mappings to dist_graph if they exist in the input
    raw_id_mappings_path = os.path.join(args.input_path, "raw_id_mappings")

    if os.path.exists(raw_id_mappings_path):
        logging.info("Copying raw_id_mappings to dist_graph")
        shutil.copytree(
            raw_id_mappings_path,
            os.path.join(output_path, 'dist_graph/raw_id_mappings'),
            dirs_exist_ok=True,
        )

    # Hard Negative Mapping
    # Load GSProcessing config from launch_arguments generated by GSProcessing
    # Generated GSProcessing config will have _with_transformation suffix.
    launch_arguments_path = os.path.join(args.input_path, "launch_arguments.json")
    if os.path.exists(launch_arguments_path):
        with open(launch_arguments_path, "r", encoding="utf-8") as f:
            gsprocessing_launch_arguments: Dict = json.load(f)
        gsprocessing_config = gsprocessing_launch_arguments["config_filename"]
        gsprocessing_config = gsprocessing_config.replace(".json", "_with_transformations.json")
        shuffle_hard_negative_nids(f"{args.input_path}/{gsprocessing_config}",
                                   args.num_parts, args.output_path)
    else:
        logging.info("Skip the hard negative node ID mapping, "
                     "the processed data is not generated by GSProcessing.")


def parse_args() -> argparse.Namespace:
    """Parses arguments for the script"""
    argparser = argparse.ArgumentParser("Partition DGL graphs for node and edge classification "
                                        + "or regression tasks")
    argparser.add_argument("--input-path", type=str, required=True,
                           help="Path to input DGL chunked data.")
    argparser.add_argument("--metadata-filename", type=str, default="metadata.json",
                           help="Name for the chunked DGL data metadata file.")
    argparser.add_argument("--output-path", type=str, required=True,
                           help="Path to store the partitioned data")
    argparser.add_argument("--num-parts", type=int, required=True,
                           help="Number of partitions to generate")
    argparser.add_argument("--ssh-port", type=int, default=22, help="SSH Port")
    argparser.add_argument("--dgl-tool-path", type=str, default="/root/dgl/tools",
                           help="The path to dgl/tools")
    argparser.add_argument("--partition-algorithm", type=str, default="random",
                           choices=["random", "parmetis"], help="Partition algorithm to use.")
    argparser.add_argument("--ip-config", type=str,
                           help=("A file storing a list of IPs, one line for "
                                "each instance of the partition cluster."))
    argparser.add_argument("--partition-assignment-only", action='store_true',
                           help="Only generate partition assignments for nodes, \
                                 the process will not build the partitioned DGL graph")
    argparser.add_argument("--logging-level", type=str, default="info",
                           help="The logging level. The possible values: debug, info, warning, \
                                   error. The default value is info.")
    argparser.add_argument("--use-graphbolt", type=lambda x: (str(x).lower() in ['true', '1']),
                           default="false",
                           help=("Whether to convert the partitioned data to the GraphBolt format "
                               "after creating the DistDGL graph."))
    argparser.add_argument("--process-group-timeout", type=int, default=1800,
                           help="Timeout[seconds] for operations executed "
                                "against the process group.")

    return argparser.parse_args()


if __name__ == '__main__':
    main()
