""" Distributed metis partion entry point
"""

import os
import json
import sys
import queue
import socket
import time
import subprocess
import argparse
from threading import Thread, Event

import psutil
import numpy as np

import boto3
import utils
import sagemaker

from sagemaker.s3 import S3Downloader
from sagemaker.s3 import S3Uploader

DGL_TOOL_PATH = "/root/dgl/tools"

def run(launch_cmd, state_q, env=None):
    """ Running cmd using shell

    Parameters
    ----------
    launch_cmd: str
        cmd to launch
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    env: dict
        System environment. If None, subprocess will use the inherited one.
    """
    try:
        subprocess.check_call(launch_cmd, shell=False, env=env)
        state_q.put(0)
    except subprocess.CalledProcessError as err:
        print(f"Called process error {err}")
        state_q.put(err.returncode)
    except Exception as err: # pylint: disable=broad-except
        print(f"Called process error {err}")
        state_q.put(-1)

########### Sychronized operations ############
def broadcast_preprocess_done(client_list, world_size, success=True):
    """ Notify each worker process the preprocess process is done

    Parameters
    ----------
    client_list: list
        List of socket clients
    world_size: int
        Size of the distributed training/inference cluster
    success: bool
        True if preprocess success
    """
    msg = b"PreprocessDone" if success else b"PreprocessFail"
    for rank in range(1, world_size):
        client_list[rank].sendall(msg)

def wait_for_preprocess_done(master_sock):
    """ Waiting for preprocessing done

    Parameters
    ----------
    master_sock: socket
        Socket connecting master
    """
    msg = master_sock.recv(20)
    msg = msg.decode()
    if msg != "PreprocessDone":
        raise RuntimeError("wait for Preprocess Error detected")

def broadcast_parmetis_done(client_list, world_size, success=True):
    """ Notify each worker process the parmetis process is done

    Parameters
    ----------
    client_list: list
        List of socket clients
    world_size: int
        Size of the distributed training/inference cluster
    success: bool
        True if preprocess success
    """
    msg = b"ParmetisDone" if success else b"ParmetisFail"
    for rank in range(1, world_size):
        client_list[rank].sendall(msg)

def wait_for_parmetis_done(master_sock):
    """ Waiting for parmetis done

    Parameters
    ----------
    master_sock: socket
        Socket connecting master
    """
    msg = master_sock.recv(20)
    msg = msg.decode()
    if msg != "ParmetisDone":
        raise RuntimeError("Wait for Parmetis Done Error detected")

def launch_preprocess(num_parts, ip_list, input_data_path,
    meta_data_config, output_path, state_q):
    """ Launch preprocessing script

    Parameters
    ----------
    num_parts: int
        Number of graph partitions
    ip_list: str
        ip list
    input_data_path: str
        Path to the input graph data
    meta_data_config: str
        Path to the meta data configuration
    output_path: str
        Path to store preprocess output
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    """

    launch_cmd = ["mpirun", "-np", f"{num_parts}",
        "--host", f"{ip_list}",
        "-wdir", f"{input_data_path}",
        "--rank-by", "node",
        "--mca", "orte_base_help_aggregate", "0",
        "/opt/conda/bin/python3", f"{DGL_TOOL_PATH}/distpartitioning/parmetis_preprocess.py",
        "--schema_file", f"{meta_data_config}",
        "--output_dir", f"{output_path}"]

    print(f"RUN {launch_cmd}")
    # launch preprocess task
    thread = Thread(target=run, args=(launch_cmd, state_q,), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread

def launch_parmetis(num_parts, net_ifname, ip_list, input_data_path,
    graph_name, metis_input_path, state_q):
    """ Launch parmetis script

    Parameters
    ----------
    num_parts: int
        Number of graph partitions
    net_ifname: str
        Network interface used by MPI
    ip_list: str
        ip list
    input_data_path: str
        Path to the input graph data
    graph_name: str
        Graph name
    metis_input_path: str
        Path to metis input
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    """
    parmetis_nfiles = os.path.join(metis_input_path, "parmetis_nfiles.txt")
    parmetis_efiles = os.path.join(metis_input_path, "parmetis_efiles.txt")

    launch_cmd = ["mpirun", "-np", f"{num_parts}",
        "--host", f"{ip_list}",
        "--rank-by", "node",
        "--mca", "orte_base_help_aggregate", "0",
        "--mca", "opal_warn_on_missing_libcuda", "0",
        "-mca", "btl_tcp_if_include", f"{net_ifname}",
        "-wdir", f"{input_data_path}",
        "-v", "/root/ParMETIS/bin/pm_dglpart3",
        f"{graph_name}", f"{num_parts}", f"{parmetis_nfiles}", f"{parmetis_efiles}"]

    print(f"RUN {launch_cmd}")
    # launch ParMetis task
    thread = Thread(target=run, args=(launch_cmd, state_q,), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread

def launch_postprocess(meta_data_config, parmetis_output_file, partitions_dir, state_q):
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
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    """
    launch_cmd = ["python3",
        f"{DGL_TOOL_PATH}/distpartitioning/parmetis_postprocess.py",
        "--schema_file", f"{meta_data_config}",
        "--parmetis_output_file", f"{parmetis_output_file}",
        "--partitions_dir", f"{partitions_dir}"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{DGL_TOOL_PATH}:{env['PYTHONPATH']}"
    print(launch_cmd)
    print(env)

    # launch postprocess task
    thread = Thread(target=run, args=(launch_cmd, state_q, env), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread

def launch_build_dglgraph(input_data_path, partitions_dir, ip_list, output_path, state_q):
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
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    """
    launch_cmd = ["python3", f"{DGL_TOOL_PATH}/dispatch_data.py",
        "--in-dir", f"{input_data_path}",
        "--partitions-dir", f"{partitions_dir}",
        "--ip-config", f"{ip_list}",
        "--out-dir", f"{output_path}",
        "--ssh-port", "22",
        "--python-path", "/opt/conda/bin/python3"]

    # launch postprocess task
    thread = Thread(target=run, args=(launch_cmd, state_q,), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread

def download_graph(graph_data_s3, metadata_filename, world_size,
    local_rank, local_path, sagemaker_session):
    """ download graph structure data

    Parameters
    ----------
    graph_data_s3: str
        S3 uri storing the partitioned graph data
    metadata_filename: str
        File name of metadata config
    world_size: int
        Size of the cluster
    local_rank: str
        Path to store graph data
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download

    Return
    ------
    local_path: str
        Local path to downloaded graph data
    local_config: str
        Local path to graph metadata config
    """
    local_path = os.path.join(local_path, "input_graph")
    graph_config = os.path.join(graph_data_s3, metadata_filename)
    local_config = os.path.join(local_path, metadata_filename)
    print(f"download {graph_config} into {local_config}")
    download_data_from_s3(graph_config, local_path,
        sagemaker_session=sagemaker_session)

    with open(local_config, encoding='utf-8') as f:
        graph_config = json.load(f)

        # download edge info
        edges = graph_config["edges"]
        for etype, edge_data in edges.items():
            print(f"Downloading edges of {etype}")
            for i, efile in enumerate(edge_data["data"]):
                if i % world_size == local_rank:
                    file_s3_path = os.path.join(graph_data_s3, efile.strip('./'))
                    print(f"Download {efile} from {file_s3_path}")
                    local_dir = local_path \
                        if len(efile.rpartition('/')) <= 1 else \
                        os.path.join(local_path, efile.rpartition('/')[0])
                    download_data_from_s3(file_s3_path, local_dir,
                        sagemaker_session=sagemaker_session)

        # download node feature
        node_data = graph_config["node_data"]
        for ntype, ndata in node_data.items():
            for feat_name, feat_data in ndata.items():
                print(f"Downloading node feature {feat_name} of {ntype}")
                # we follow the logic in DGL:
                # tools/distpartitioning/dataset_utils.py: L268
                num_files = len(feat_data["data"])
                read_list = np.array_split(np.arange(num_files), world_size)
                print(read_list)
                print(read_list[local_rank].tolist())
                for i in read_list[local_rank].tolist():
                    nf_file = feat_data["data"][i]
                    file_s3_path = os.path.join(graph_data_s3, nf_file.strip('./'))
                    print(f"Download {nf_file} from {file_s3_path}")
                    local_dir = local_path \
                        if len(nf_file.rpartition('/')) <= 1 else \
                        os.path.join(local_path, nf_file.rpartition('/')[0])
                    download_data_from_s3(file_s3_path, local_dir,
                        sagemaker_session=sagemaker_session)

        # download edge feature
        edge_data = graph_config["edge_data"]
        for etype, edata in edge_data.items():
            for feat_name, feat_data in edata.items():
                print(f"Downloading node feature {feat_name} of {etype}")
                # we follow the logic in DGL:
                # tools/distpartitioning/dataset_utils.py: L268
                num_files = len(feat_data["data"])
                read_list = np.array_split(np.arange(num_files), world_size)
                print(read_list)
                print(read_list[local_rank].tolist())

                for i in read_list[local_rank].tolist():
                    ef_file = feat_data["data"][i]
                    file_s3_path = os.path.join(graph_data_s3, ef_file.strip('./'))
                    print(f"Download {ef_file} from {file_s3_path}")
                    local_dir = local_path \
                        if len(ef_file.rpartition('/')) <= 1 else \
                        os.path.join(local_path, ef_file.rpartition('/')[0])
                    download_data_from_s3(file_s3_path, local_dir,
                        sagemaker_session=sagemaker_session)

    return local_path, local_config

def download_data_from_s3(input_s3, local_data_path, sagemaker_session):
    """ Download intermediate data info into S3

    Parameters
    ----------
    input_s3: str
        S3 uri of the input file
    local_data_path: str
        Local file path to store the downloaded file.
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    print(f"download {input_s3} into {local_data_path}")
    try:
        S3Downloader.download(input_s3,
            local_data_path, sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print(f"Can not download {input_s3}.")
        raise RuntimeError(f"Can not download {input_s3}.")

def upload_file_to_s3(data_path_s3_path, local_data_path, sagemaker_session):
    """ Upload intermediate data info into S3

    Parameters
    ----------
    data_path_s3_path: str
        S3 uri to upload the data
    local_data_path: str
        Path to local data
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run upload
    """
    print(f"upload {local_data_path} into {data_path_s3_path}")
    try:
        ret = S3Uploader.upload(local_data_path, data_path_s3_path,
            sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print(f"Can not upload data into {data_path_s3_path}")
        raise RuntimeError(f"Can not upload data into {data_path_s3_path}")
    return ret

def parse_partition_args():
    """ Add arguments for graph partition
    """
    parser = argparse.ArgumentParser(description='gs sagemaker train pipeline')

    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input graph")
    parser.add_argument("--num-parts", type=int, help="Number of partitions")
    parser.add_argument("--output-data-s3", type=str,
        help="S3 location to store the partitioned graph")
    parser.add_argument("--metadata-filename", type=str,
        default="metadata.json", help="file name of metadata config file")

    return parser

def main():
    """ main logic
    """
    for key, val in os.environ.items():
        print(f"{key}: {val}")

    # start the ssh server
    subprocess.run(["service", "ssh", "start"], check=True)

    tmp_data_path = "/opt/ml/"
    parser = parse_partition_args()
    args = parser.parse_args()
    sm_env = json.loads(os.environ['SM_TRAINING_ENV'])
    hosts = sm_env['hosts']
    current_host = sm_env['current_host']
    world_size = len(hosts)
    os.environ['WORLD_SIZE'] = str(world_size)
    host_rank = hosts.index(current_host)
    assert args.graph_name is not None, "Graph name must be provided"

    try:
        for host in hosts:
            print(f"The {host} IP is {socket.gethostbyname(host)}")
    except:
        raise RuntimeError(f"Can not get host name of {hosts}")

    master_addr = os.environ['MASTER_ADDR']
    # sync with all instances in the cluster
    if host_rank == 0:
        # sync with workers
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((master_addr, 12000))
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
                sock.connect((master_addr, 12000))
                break
            except: # pylint: disable=bare-except
                print(f"Try to connect {master_addr}")
                time.sleep(10)
        print("Connected")

    # write ip list info into disk
    ip_list = []
    ip_list_path = os.path.join(tmp_data_path, 'ip_list.txt')
    with open(ip_list_path, 'w', encoding='utf-8') as f:
        for host in hosts:
            f.write(f"{socket.gethostbyname(host)}\n")
            ip_list.append(socket.gethostbyname(host))

    graph_name = args.graph_name
    graph_data_s3 = args.graph_data_s3
    num_parts = args.num_parts
    output_s3 = args.output_data_s3
    metadata_filename = args.metadata_filename

    boto_session = boto3.session.Session(region_name=os.environ['AWS_REGION'])
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session)
    graph_data_path, meta_info_file = download_graph(graph_data_s3,
        metadata_filename,
        world_size,
        host_rank,
        tmp_data_path,
        sagemaker_session)

    err_code = 0
    if host_rank == 0:
        # To get which net device to use for MPI communication
        def get_ifname():
            nics = psutil.net_if_addrs()
            for ifname, if_info in nics.items():
                for info in if_info:
                    if info.address == ip_list[0] and info.family==socket.AF_INET:
                        return ifname
            raise RuntimeError("Can not find network interface")
        net_ifname = get_ifname()
        print(net_ifname)

        # get ip list
        ip_list = ",".join(ip_list)
        utils.barrier_master(client_list, world_size)

        state_q = queue.Queue()
        def leader_prepare_partition_graph():
            metis_input_path = os.path.join(tmp_data_path, "metis_input")
            # launch pre-processing
            preproc_task = launch_preprocess(num_parts,
                                            ip_list,
                                            graph_data_path,
                                            meta_info_file,
                                            metis_input_path,
                                            state_q)
            preproc_task.join()
            err_code = state_q.get()
            if err_code != 0:
                # Notify failure to workers
                broadcast_preprocess_done(client_list, world_size, success=False)
                raise RuntimeError("Partition preprocessing failed")

            # Upload processed node/edge data into S3.
            # It will also upload parmetis_nfiles.txt and parmetis_efiles.txt.
            # We upload the intermediate metis_input to S3 for data backup.
            #
            # TODO(xinagsx): Allow skiping the preprocess step and
            # resuming metis_input to do partition.
            metis_input_s3 = os.path.join(graph_data_s3, "metis_input")
            upload_file_to_s3(metis_input_s3, metis_input_path, sagemaker_session)

            # Upload <graph_name>_stats.txt
            # This file is generated by preprocess and consumed by parmetis
            state_file_name = f"{graph_name}_stats.txt"
            state_file_path = os.path.join(graph_data_path, state_file_name)
            upload_file_to_s3(metis_input_s3, state_file_path, sagemaker_session)

            broadcast_preprocess_done(client_list, world_size)

            print(f"{host_rank} {os.listdir(metis_input_path)}")
            print(f"{host_rank} {os.listdir(graph_data_path)}")

            # sync on uploading metis_input data
            utils.barrier_master(client_list, world_size)

            return metis_input_path

        def partition_graph(metis_input_path):
            # launch parmetis
            metis_task = launch_parmetis(num_parts,
                                         net_ifname,
                                         ip_list,
                                         graph_data_path,
                                         graph_name,
                                         metis_input_path,
                                         state_q)
            metis_task.join()
            err_code = state_q.get()
            if err_code != 0:
                raise RuntimeError("Parallel metis partition failed")

            # launch post processing
            parmetis_output_file = os.path.join(graph_data_path,
                f"{graph_name}_part.{num_parts}")
            partition_dir = os.path.join(graph_data_path, "partition")
            postproc_task = launch_postprocess(meta_info_file,
                                               parmetis_output_file,
                                               partition_dir,
                                               state_q)
            postproc_task.join()
            err_code = state_q.get()
            if err_code != 0:
                broadcast_parmetis_done(client_list, world_size, success=False)
                raise RuntimeError("Post processing failed")

            # parmetis done
            # Upload per node type partition-id into S3
            partition_input_s3 = os.path.join(graph_data_s3, "partition")
            upload_file_to_s3(partition_input_s3, partition_dir, sagemaker_session)
            broadcast_parmetis_done(client_list, world_size)

            print(f"{host_rank} {partition_dir} {os.listdir(partition_dir)}")

            # sync on downloading parmetis data
            utils.barrier_master(client_list, world_size)

            # Build DistDGL graph
            dglgraph_output = os.path.join(tmp_data_path, "dist_graph")
            build_dglgraph_task = launch_build_dglgraph(graph_data_path,
                partition_dir,
                ip_list_path,
                dglgraph_output,
                state_q)

            build_dglgraph_task.join()
            err_code = state_q.get()
            if err_code != 0:
                raise RuntimeError("build dglgrah failed")

            return dglgraph_output

        try:
            metis_input_path = leader_prepare_partition_graph()
        except RuntimeError as e:
            print(e)
            err_code = -1

        if err_code != -1:
            task_end = Event()
            thread = Thread(target=utils.keep_alive,
                args=(client_list, world_size, task_end),
                daemon=True)
            thread.start()

            try:
                dglgraph_output = partition_graph(metis_input_path)
            except RuntimeError as e:
                print(e)
                err_code = -1

            utils.terminate_workers(client_list, world_size, task_end)
        print("Master End")
        if err_code != -1:
            upload_file_to_s3(output_s3, dglgraph_output, sagemaker_session)
            # clean up the downloaded graph and the generated graph
            utils.remove_data(graph_data_path)
            utils.remove_data(dglgraph_output)
    else:
        utils.barrier(sock)

        def worker_prepare_partition_graph():
            wait_for_preprocess_done(sock)

            # download parmetis_nfiles.txt and parmetis_efiles.txt
            metis_input_s3 = os.path.join(graph_data_s3, "metis_input")
            metis_input_path = os.path.join(tmp_data_path, "metis_input")
            print(f"{host_rank} {os.listdir(metis_input_path)}")
            download_data_from_s3(os.path.join(metis_input_s3, "parmetis_nfiles.txt"),
                metis_input_path,
                sagemaker_session)
            download_data_from_s3(os.path.join(metis_input_s3, "parmetis_efiles.txt"),
                metis_input_path,
                sagemaker_session)

            # Download <graph_name>_stats.txt
            # This file is generated by preprocess and consumed by parmetis
            state_file_name = f"{graph_name}_stats.txt"
            state_file_s3_path = os.path.join(metis_input_s3, state_file_name)
            download_data_from_s3(state_file_s3_path, graph_data_path, sagemaker_session)

            print(f"{host_rank} {metis_input_path} {os.listdir(metis_input_path)}")
            print(f"{host_rank} {graph_data_path} {os.listdir(graph_data_path)}")

            # we upload local metis_input
            upload_file_to_s3(metis_input_s3, metis_input_path, sagemaker_session)

            # done download parmetis info
            utils.barrier(sock)

            # wait for parmetis finish
            wait_for_parmetis_done(sock)

            partition_input_s3 = os.path.join(graph_data_s3, "partition")
            partition_dir = os.path.join(graph_data_path, "partition")
            download_data_from_s3(partition_input_s3, partition_dir, sagemaker_session)

            print(f"{host_rank} {partition_dir} {os.listdir(partition_dir)}")

            # done download parmetis data
            utils.barrier(sock)

        try:
            worker_prepare_partition_graph()
        except RuntimeError as e:
            print(e)
            err_code = -1

        print(os.listdir(graph_data_path))

        # Block util dispatch_data finished
        # Listen to end command
        utils.wait_for_exit(sock)
        if err_code != -1:
            dglgraph_output = os.path.join(tmp_data_path, "dist_graph")
            upload_file_to_s3(output_s3, dglgraph_output, sagemaker_session)
            # clean up the downloaded graph and the generated graph
            utils.remove_data(dglgraph_output)
            print("Worker End")
        utils.remove_data(graph_data_path)

    sock.close()
    if err_code != 0:
        # Report an error
        print("Task failed")
        sys.exit(-1)

if __name__ == '__main__':
    main()
