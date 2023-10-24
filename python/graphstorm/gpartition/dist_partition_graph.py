"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Do random partition for distributed data processing.
    This script only works with graphstorm-processing (Distirbuted graph processing).
"""
import os
import argparse
import time
import logging
import sys
import queue
import subprocess
from shutil import rmtree

from threading import Thread

from .random_partition import LocalRandomPartitioner
from ..utils import get_log_level

def run_build_dglgraph(
        input_data_path,
        partitions_dir,
        ip_list,
        output_path,
        metadata_filename,
        dgl_tool_path):
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
            "--ssh-port", "22",
            "--python-path", f"{python_bin}",
            "--log-level", logging.getLevelName(logging.root.getEffectiveLevel()),
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
            raise RuntimeError("build dglgrah failed")


def main(args):
    logging.basicConfig(level=get_log_level(args.logging_level))
    output_path = args.output_path
    metadata_file=args.meta_info_path
    local_output_path = os.path.join(output_path, "tmp")

    part_start = time.time()
    if args.part_algorithm == "random":
        partitioner = LocalRandomPartitioner(metadata_file, local_output_path)
        local_partition_path = partitioner.create_partitions(args.num_parts)
    else:
        raise RuntimeError(f"Unknow partition algorithm {args.part_algorighm}")
    part_end = time.time()
    logging.info("Partition takes %f sec", part_end - part_start)

    if args.do_dispatch:
        meta_info = args.meta_info_path.rsplit("/", 1)
        input_data_path = meta_info[0]
        metadata_filename = meta_info[1]
        partitions_dir = local_partition_path
        ip_list = args.ip_list
        dgl_tool_path = args.dgl_tool_path
        run_build_dglgraph(
            input_data_path,
            partitions_dir,
            ip_list,
            output_path,
            metadata_filename,
            dgl_tool_path)

        rmtree(local_partition_path)
        logging.info("Dispatch takes %f sec", part_end - time.time())

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition DGL graphs for node and edge classification "
                                        + "or regression tasks")
    argparser.add_argument("--meta-info-path", type=str, required=True,
                           help="Path to meta-info")
    argparser.add_argument("--output-path", type=str, required=True,
                           help="Path to store the partitioned data")
    argparser.add_argument("--num-parts", type=int, required=True,
                           help="Number of partitions to generate")
    argparser.add_argument("--dgl-tool-path", type=str,
                           help="The path to dgl/tools")
    argparser.add_argument("--part-algorithm", type=str, default="random",
                           choices=["random"], help="Partition algorithm to use.")
    argparser.add_argument("--ip-list", type=str,
                           help="A file storing the ip list of instances of the partition cluster.")
    argparser.add_argument("--do-dispatch", action='store_true')
    argparser.add_argument("--logging-level", type=str, default="info",
                           help="The logging level. The possible values: debug, info, warning, \
                                   error. The default value is info.")

    args = argparser.parse_args()
    start = time.time()

    main(args)
    print(f'Partition takes {time.time() - start:.3f} seconds')
