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

    SageMaker partitioning entry point.
"""
from dataclasses import dataclass
import os
import json
import logging
import queue
import socket
import time
import subprocess
from threading import Thread, Event

import numpy as np
import boto3
import sagemaker

from graphstorm.sagemaker import utils
from .s3_utils import download_data_from_s3, upload_file_to_s3
from .sm_partition_algorithm import (SageMakerRandomPartitioner,
                                  SageMakerPartitionerConfig)

DGL_TOOL_PATH = "/root/dgl/tools"

@dataclass
class PartitionJobConfig():
    """
    Configuration object for a SageMaker partitioning job.

    Parameters
    ----------
    graph_data_s3: str
        S3 path to the graph data in chunked format.
    output_data_s3: str
        S3 path to store the partitioned graph data.
    num_parts: int
        Number of partitions to create.
    metadata_filename: str
        The filename for the graph partitioning metadata file we'll use to determine data sources.
    partition_algorithm: str
        The name of the partition algorithm to use.
    skip_partitioning: bool
        When true we skip partitioning and skip to the DistDGL file creation step.
    log_level: str
        The log level to use. Choose from [DEBUG, INFO, WARNING, ERROR, CRITICAL].
    """
    graph_data_s3: str
    output_data_s3: str
    num_parts: int
    metadata_filename: str
    partition_algorithm: str
    skip_partitioning: bool
    log_level: str

def launch_build_dglgraph(
        input_data_path,
        partitions_dir,
        ip_list,
        output_path,
        metadata_filename,
        state_q):
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
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    """
    launch_cmd = ["python3", f"{DGL_TOOL_PATH}/dispatch_data.py",
        "--in-dir", input_data_path,
        "--metadata-filename", metadata_filename,
        "--partitions-dir", partitions_dir,
        "--ip-config", ip_list,
        "--out-dir", output_path,
        "--ssh-port", "22",
        "--python-path", "/opt/conda/bin/python3",
        "--log-level", logging.getLevelName(logging.root.getEffectiveLevel()),
        "--save-orig-nids",
        "--save-orig-eids"]

    # launch postprocess task
    thread = Thread(target=utils.run, args=(launch_cmd, state_q,), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread


def download_graph(graph_data_s3, graph_config, world_size,
        local_rank, local_path, sagemaker_session):
    """ download graph structure data

    Parameters
    ----------
    graph_data_s3: str
        S3 uri storing the partitioned graph data
    graph_config: dict
        metadata config
    world_size: int
        Size of the cluster
    local_rank: str
        Path to store graph data
    local_path: str
        directory path under which the data will be downloaded
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download

    Return
    ------
    local_path: str
        Local path to downloaded graph data
    """

    # download edge info
    edges = graph_config["edges"]
    for etype, edge_data in edges.items():
        if local_rank == 0:
            logging.info("Downloading edge structure for edge type '%s'", etype)
        edge_file_list = edge_data["data"]
        read_list = np.array_split(np.arange(len(edge_file_list)), world_size)
        for i, efile in enumerate(edge_file_list):
            # TODO: Only download round-robin if ParMETIS will run, skip otherwise
            # Download files both in round robin and sequential assignment
            if i % world_size == local_rank or i in read_list[local_rank]:
                efile = edge_file_list[i]
                file_s3_path = os.path.join(graph_data_s3, efile.strip('./'))
                logging.debug("Download %s from %s",
                    efile, file_s3_path)
                local_dir = local_path \
                    if len(efile.rpartition('/')) <= 1 else \
                    os.path.join(local_path, efile.rpartition('/')[0])
                download_data_from_s3(file_s3_path, local_dir,
                    sagemaker_session=sagemaker_session)

        # download node feature
        node_data = graph_config["node_data"]
        for ntype, ndata in node_data.items():
            for feat_name, feat_data in ndata.items():
                if local_rank == 0:
                    logging.info("Downloading node feature '%s' of node type '%s'",
                    feat_name, ntype)
                num_files = len(feat_data["data"])
                # TODO: Use dgl.tools.distpartitioning.utils.generate_read_list
                # once we move the code over from DGL
                read_list = np.array_split(np.arange(num_files), world_size)
                for i in read_list[local_rank].tolist():
                    nf_file = feat_data["data"][i]
                    file_s3_path = os.path.join(graph_data_s3, nf_file.strip('./'))
                    logging.debug("Download %s from %s",
                        nf_file, file_s3_path)
                    local_dir = local_path \
                        if len(nf_file.rpartition('/')) <= 1 else \
                        os.path.join(local_path, nf_file.rpartition('/')[0])
                    download_data_from_s3(file_s3_path, local_dir,
                        sagemaker_session=sagemaker_session)

        # download edge feature
        edge_data = graph_config["edge_data"]
        for e_feat_type, edata in edge_data.items():
            for feat_name, feat_data in edata.items():
                if local_rank == 0:
                    logging.info("Downloading edge feature '%s' of '%s'",
                        feat_name, e_feat_type)
                num_files = len(feat_data["data"])
                read_list = np.array_split(np.arange(num_files), world_size)
                for i in read_list[local_rank].tolist():
                    ef_file = feat_data["data"][i]
                    file_s3_path = os.path.join(graph_data_s3, ef_file.strip('./'))
                    logging.debug("Download %s from %s",
                        ef_file, file_s3_path)
                    local_dir = local_path \
                        if len(ef_file.rpartition('/')) <= 1 else \
                        os.path.join(local_path, ef_file.rpartition('/')[0])
                    download_data_from_s3(file_s3_path, local_dir,
                        sagemaker_session=sagemaker_session)

    return local_path

def run_partition(job_config: PartitionJobConfig):
    """
    Performs the partitioning on SageMaker.
    """
    # start the ssh server
    subprocess.run(["service", "ssh", "start"], check=True)

    tmp_data_path = "/tmp/"

    graph_data_s3 = job_config.graph_data_s3
    num_parts = job_config.num_parts
    output_s3 = job_config.output_data_s3
    metadata_filename = job_config.metadata_filename
    skip_partitioning = job_config.skip_partitioning == 'true'

    with open("/opt/ml/config/resourceconfig.json", "r", encoding="utf-8") as f:
        sm_env = json.load(f)
    hosts = sm_env['hosts']
    current_host = sm_env['current_host']
    world_size = len(hosts)
    os.environ['WORLD_SIZE'] = str(world_size)
    host_rank = hosts.index(current_host)

    # NOTE: Ensure no logging has been done before setting logging configuration
    logging.basicConfig(
        level=getattr(logging, job_config.log_level.upper(), None),
        format=f'{current_host}: %(asctime)s - %(levelname)s - %(message)s'
        )

    boto_session = boto3.session.Session(region_name=os.environ['AWS_REGION'])
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session)

    if host_rank == 0:
        for host in hosts:
            logging.info("The %s IP is %s",
                host, socket.gethostbyname(host))


    for key, val in os.environ.items():
        logging.debug("%s: %s", key, val)

    leader_addr = socket.gethostbyname('algo-1')
    # sync with all instances in the cluster
    if host_rank == 0:
        # sync with workers
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((leader_addr, 12000))
        sock.listen(world_size)

        client_list = [None] * world_size
        for i in range(1, world_size):
            client, _ = sock.accept()
            client_list[i] = client
    else:
        # sync with master
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for _ in range(30):
            try:
                sock.connect((leader_addr, 12000))
                break
            except: # pylint: disable=bare-except
                logging.debug("Rank %s trying to connect to leader %s",
                    host_rank, leader_addr)
                time.sleep(10)
        logging.debug("Connected")
        client_list = None

    # write ip list info to disk
    ip_list = []
    ip_list_path = os.path.join(tmp_data_path, 'ip_list.txt')
    with open(ip_list_path, 'w', encoding='utf-8') as f:
        for host in hosts:
            f.write(f"{socket.gethostbyname(host)}\n")
            ip_list.append(socket.gethostbyname(host))

    graph_data_s3_no_trailing = graph_data_s3[:-1] if graph_data_s3.endswith('/') else graph_data_s3
    graph_config_s3_path = f"{graph_data_s3_no_trailing}/{metadata_filename}"
    meta_info_file = os.path.join(tmp_data_path, metadata_filename)

    if not os.path.exists(meta_info_file):
        if host_rank == 0:
            logging.debug("Downloading metadata file from %s into %s",
                graph_config_s3_path, meta_info_file)
        download_data_from_s3(graph_config_s3_path, tmp_data_path,
            sagemaker_session=sagemaker_session)

    with open(meta_info_file, 'r') as f: # pylint: disable=unspecified-encoding
        graph_config = json.load(f)

    logging.info("Downloading graph data from %s into %s",
        graph_data_s3, tmp_data_path)
    graph_data_path = download_graph(
        graph_data_s3,
        graph_config,
        world_size,
        host_rank,
        tmp_data_path,
        sagemaker_session)

    partition_config = SageMakerPartitionerConfig(
        metadata_file=meta_info_file,
        local_output_path=tmp_data_path,
        rank=host_rank,
        sagemaker_session=sagemaker_session)

    if job_config.partition_algorithm == 'random':
        sm_partitioner = SageMakerRandomPartitioner(partition_config)
    else:
        raise RuntimeError(f"Unknown partition algorithm: '{job_config.partition_algorithm}'", )

    # Conditionally skip partitioning
    if skip_partitioning:
        # All workers + leader need to download partition data from S3
        s3_partition_path = os.path.join(output_s3, "partition")
        local_partition_path = os.path.join(graph_data_path, "partition")
        logging.warning("Skipped partitioning step, trying to download partition data from %s",
            s3_partition_path)

        download_data_from_s3(s3_partition_path, local_partition_path, sagemaker_session)
    else:
        # Perform the partitioning. Leader uploads partition assignments to S3.
        local_partition_path, s3_partition_path = sm_partitioner.create_partitions(
            output_s3, num_parts)
        if host_rank == 0:
            sm_partitioner.broadcast_partition_done(client_list, world_size, success=True)
            # Wait for signal from all workers that they have downloaded partition assignments
            # before moving on
            utils.barrier_master(client_list, world_size)
        else:
            # Workers need to download partition assignments from S3
            logging.debug("Worker %s is waiting for partition done...", host_rank)
            # wait for signal from leader that partition is finised and files are ready to download
            sm_partitioner.wait_for_partition_done(sock)

            logging.debug("Worker %s is downloading partition data from %s into %s",
                    host_rank,  s3_partition_path, local_partition_path)
            download_data_from_s3(s3_partition_path, local_partition_path, sagemaker_session)

            logging.debug("Partition data: %s: %s",
                local_partition_path, os.listdir(local_partition_path))
            # Signal to leader that we're done downloading partition assignments
            utils.barrier(sock)

    s3_dglgraph_output = os.path.join(output_s3, "dist_graph")
    dglgraph_output = os.path.join(tmp_data_path, "dist_graph")
    logging.debug("Worker %s s3_dglgraph_output: %s",
            host_rank, s3_dglgraph_output)
    if host_rank == 0:
        state_q = queue.Queue()
        def data_dispatch_step(partition_dir):
            # Build DistDGL graph

            build_dglgraph_task = launch_build_dglgraph(graph_data_path,
                partition_dir,
                ip_list_path,
                dglgraph_output,
                metadata_filename,
                state_q)

            build_dglgraph_task.join()
            err_code = state_q.get()
            if err_code != 0:
                raise RuntimeError("build dglgrah failed")

        task_end = Event()
        thread = Thread(target=utils.keep_alive,
            args=(client_list, world_size, task_end),
            daemon=True)
        thread.start()

        data_dispatch_step(local_partition_path)

        # Indicate we can stop sending keepalive messages
        task_end.set()
        # Ensure the keepalive thread has finished before closing sockets
        thread.join()
        # Close connections with workers
        utils.terminate_workers(client_list, world_size)
    else:
        # Block until dispatch_data finished
        # Listen to end command
        utils.wait_for_exit(sock)

    # All instances (leader+workers) upload local DGL objects to S3
    upload_file_to_s3(s3_dglgraph_output, dglgraph_output, sagemaker_session)
    logging.info("Rank %s completed all tasks, exiting...", host_rank)

    sock.close()
