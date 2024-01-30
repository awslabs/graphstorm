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

    sagemaker script utilities
"""
import hashlib
import math
import logging
import os
import shutil
import socket
import subprocess
import time
import warnings
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import boto3
import botocore
from botocore.errorfactory import ClientError
from joblib import delayed, Parallel
from sagemaker.s3 import S3Downloader
from sagemaker.s3 import S3Uploader

from graphstorm import get_rank

PORT_MIN = 10000  # Avoid privileged ports
PORT_MAX = 65535  # Maximum TCP port number

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
        logging.error("Called process error %s", err)
        state_q.put(err.returncode)
    except Exception as err: # pylint: disable=broad-except
        logging.error("Called process error %s", err)
        state_q.put(-1)

def barrier_master(client_list, world_size):
    """ Master barrier, called by host_rank == 0

    Parameters
    ----------
    client_list: list
        List of socket clients
    world_size: int
        Size of the distributed training/inference cluster
    """
    for rank in range(1, world_size):
        client_list[rank].send(b"sync")

    for rank in range(1, world_size):
        msg = client_list[rank].recv(8)
        msg = msg.decode()
        assert msg == "sync_ack", f"Rank {rank} did not send 'sync_ack', got msg {msg}"

    for rank in range(1, world_size):
        client_list[rank].send(b"synced")

    for rank in range(1, world_size):
        msg = client_list[rank].recv(12)
        msg = msg.decode()
        assert msg == "synced_ack"

def barrier(master_sock):
    """ Worker node barrier, called by host_rank > 0

    Parameters
    ----------
    master_sock: socket
        Socket connecting master
    """
    msg = master_sock.recv(8)
    msg = msg.decode()
    assert msg == "sync", f"Incorrect message received from master: {msg}"

    master_sock.send(b"sync_ack")

    msg = master_sock.recv(8)
    msg = msg.decode()
    assert msg == "synced", f"Incorrect message received from master: {msg}"

    master_sock.send(b"synced_ack")

def keep_alive(client_list, world_size, task_end):
    """ Keep the communication between master and workers alive

    Parameters
    ----------
    client_list: list
        List of socket clients
    world_size: int
        Size of the distributed training/inference cluster
    task_end: threading.Event
        Indicate whether the task has finished.
    """
    while task_end.is_set() is False:
        time.sleep(60)
        for rank in range(1, world_size):
            client_list[rank].send(b"Dummy")

    logging.info("keepalive thread exiting...")

def terminate_workers(client_list, world_size):
    """ terminate all worker daemons.

    Parameters
    ----------
    client_list: list
        List of socket clients
    world_size: int
        Size of the distributed training/inference cluster
    """
    for rank in range(1, world_size):
        client_list[rank].send(b"Done")
        msg = client_list[rank].recv(8)
        logging.info("Client %d exit %s",
            rank, msg.decode())

    # close connections with clients
    for rank in range(1, world_size):
        client_list[rank].close()

def wait_for_exit(master_sock):
    """ Worker processes wait for exit

    Parameters
    ----------
    master_sock: socket
        Socket connecting master
    """
    msg = master_sock.recv(8)
    while msg.decode() != "Done":
        msg = master_sock.recv(8)
        logging.debug(msg.decode())
    master_sock.send(b"Exit")

def download_yaml_config(yaml_s3, local_path, sagemaker_session):
    """ download yaml config file

    Parameters
    ----------
    yaml_s3: str
        S3 uri storing the yaml config file
    yaml_file_name: str
        Yaml config file name
    local_path: str
        Path to store the yaml file
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download

    Return
    ------
    yaml_path: str
        Path to downloaded file
    """
    # Download training yaml file
    s3_url = urlparse(yaml_s3)
    yaml_file_name = s3_url.path.split('/')[-1]
    assert yaml_file_name.endswith('yaml') or yaml_file_name.endswith('yml'), \
        f"{yaml_s3} must be a yaml file."
    yaml_path = os.path.join(local_path, yaml_file_name)
    ### Download Partitioned graph data

    os.makedirs(local_path, exist_ok=True)
    try:
        S3Downloader.download(yaml_s3, local_path,
            sagemaker_session=sagemaker_session)
    except Exception as err: # pylint: disable=broad-except
        raise RuntimeError(f"Fail to download yaml file {yaml_s3}: {err}")

    return yaml_path

def download_model(model_artifact_s3, model_path, sagemaker_session):
    """ Download graph model

    Parameters
    ----------
    model_artifact_s3: str
        S3 uri storing the model artifacts
    model_path: str
        Path to store the  graph model locally
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    # download model
    # TODO: Each instance will download the full set of learnable
    # sparse embedding. We need to find a more space efficient method.
    try:
        S3Downloader.download(model_artifact_s3,
            model_path, sagemaker_session=sagemaker_session)
    except Exception as err: # pylint: disable=broad-except
        raise RuntimeError("Can not download saved model artifact" \
                           f"model.bin from {model_artifact_s3}." \
                           f"{err}")

def download_graph(graph_data_s3, graph_name, part_id, world_size,
                   local_path, sagemaker_session,
                   raw_node_mapping_prefix_s3=None,
                   s3_client=None):
    """ download graph data

    Parameters
    ----------
    graph_data_s3: str
        S3 uri storing the partitioned graph data
    graph_name: str
        Graph name
    part_id: int
        Graph partition id
    world_size: int
        Number of instances in the cluster.
    local_path: str
        Path to store graph data
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    raw_node_mapping_prefix_s3: str, optional
        S3 prefix to where the node_id_mapping data are stored

    Return
    ------
    graph_config_path: str
        Path to downloaded graph config file
    """
    # Download partitioned graph data.
    # Each training instance only download 1 partition.
    DOWNLOAD_THREADS = 64
    rank = get_rank()
    graph_part = f"part{part_id}"

    graph_path = os.path.join(local_path, graph_name)
    graph_part_path = os.path.join(graph_path, graph_part)
    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(graph_part_path, exist_ok=True)

    graph_data_s3 = graph_data_s3[:-1] if graph_data_s3.endswith('/') else graph_data_s3

    # We split on '/' to get the bucket, as it's always the third split element in an S3 URI
    s3_input_bucket = graph_data_s3.split("/")[2]
    # Similarly, by having maxsplit=3 we get the S3 key value as the fourth element
    s3_input_key = graph_data_s3.split("/", maxsplit=3)[3]

    s3_client = boto3.client('s3') if s3_client is None else s3_client
    graph_config = None
    for config_name  in [f"{graph_name}.json", "metadata.json"]:
        try:
            s3_client.head_object(Bucket=s3_input_bucket, Key=f"{s3_input_key}/{config_name}")
            # This will only be accessed if the above doesn't trigger an exception
            graph_config = config_name
        except ClientError as err:
            if err.response['ResponseMetadata']['HTTPStatusCode'] == 404:
                # The key does not exist.
                logging.debug("Metadata key %s does not exist",
                            f"{s3_input_key}/{graph_config}")
            elif err.response['Error']['Code'] == 403:
                # Unauthorized, including invalid bucket
                logging.error("Authorization error, check the path again: %s",
                            f"{s3_input_key}/{graph_config}")
            else:
                # Something else has gone wrong.
                raise err

    assert graph_config, \
        (f"Could not find a graph config file named {graph_name}.json or metadata.json "
         f"under {graph_data_s3}")
    graph_part_start = time.perf_counter()
    # Download partition metadata file
    S3Downloader.download(os.path.join(graph_data_s3, graph_config),
            graph_path, sagemaker_session=sagemaker_session)

    def s3_get_meta_data(client, bucket, key):
        meta_data = client.head_object(
            Bucket=bucket,
            Key=key
        )
        return meta_data

    def convert_size(size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])


    def get_cunks(size_bytes, desired_sections):
        return size_bytes / desired_sections

    def download_large_file(client, bucket, key, local_filepath, parallel_threads):
        start = time.time()
        md = s3_get_meta_data(client, bucket, key)
        chunk = get_cunks(md["ContentLength"], parallel_threads)
        logging.debug("Making %s parallel s3 calls with a chunk size of %s each..." % (
            parallel_threads, convert_size(chunk))
        )
        client.download_file(
            Bucket=bucket,
            Filename=local_filepath,
            Key=key,
            Config=boto3.s3.transfer.TransferConfig(
                max_concurrency=parallel_threads
            )
        )
        end = time.time() - start
        logging.debug("Finished downloading %s in %s seconds" % (key, end))


    graph_part_s3_prefix = os.path.join(os.path.join(graph_data_s3, graph_part), "")
    s3_graph_part_files = S3Downloader.list(
        graph_part_s3_prefix,
        sagemaker_session=sagemaker_session)

    # Download graph structure, features and DGL mapping files
    for s3_graph_part_file in s3_graph_part_files:
        graph_part_key = s3_graph_part_file.split("/", maxsplit=3)[3]
        local_part_path = os.path.join(graph_part_path, os.path.basename(graph_part_key))
        download_large_file(
            s3_client,
            s3_input_bucket,
            graph_part_key,
            local_part_path,
            min(DOWNLOAD_THREADS, os.cpu_count())
        )

    logging.info("Rank %d: Time to download graph part %s: %.2f seconds",
                 rank, graph_part, time.perf_counter() - graph_part_start)

    node_id_mapping = "node_mapping.pt"
    # Try to download node id mapping file if any
    try:
        logging.info("Download graph id mapping from %s to %s",
                     os.path.join(graph_data_s3, node_id_mapping),
                     graph_path)
        S3Downloader.download(os.path.join(graph_data_s3, node_id_mapping),
            graph_path, sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        logging.warning("Node id mapping file does not exist."
                        "If you are running GraphStorm on a graph with "
                        "more than 1 partition, it is recommended to provide "
                        "the node id mapping file created by gconstruct or gsprocessing.")

    if part_id == 0:
        # The leader needs to download the DGL intermediate mapping files
        lead_mapping_start = time.perf_counter()
        # It is possible that id mappings are generated by
        # dgl tools/distpartitioning/convert_partition.py
        for i in range(1, world_size):
            local_graph_part = f"part{i}"
            local_graph_part_path = os.path.join(graph_path, local_graph_part)
            os.makedirs(local_graph_part_path, exist_ok=True)

            # Try to download node id mapping file if any
            filename = "orig_nids.dgl"
            s3_path = os.path.join(graph_data_s3, local_graph_part, filename)
            try:
                logging.debug("Try to download %s to %s", s3_path, local_graph_part_path)
                dgl_mapping_key = s3_path.split("/", maxsplit=3)[3]
                download_large_file(
                    s3_client,
                    s3_input_bucket,
                    dgl_mapping_key,
                    os.path.join(local_graph_part_path, filename),
                    min(DOWNLOAD_THREADS, os.cpu_count())
                )
            except Exception: # pylint: disable=broad-except
                logging.info("Could not download DGL node id mapping file %s", s3_path)
        logging.info("Time to download DGL node ID mappings on leader: %f seconds",
                     time.perf_counter() - lead_mapping_start)

    # Try to get GraphStorm ID to Original ID remapping files if any
    # The S3 path can be empty, which means no Raw ID mapping is needed.
    # For exampling during SageMaker training.
    raw_id_mappings_start = time.perf_counter()

    # By default we assume the node mappings exist
    # under the same path as the rest of the graph data
    if not raw_node_mapping_prefix_s3:
        raw_node_mapping_prefix_s3 = f"{graph_data_s3}/raw_id_mappings"
    else:
        raw_node_mapping_prefix_s3 = (
            raw_node_mapping_prefix_s3[:-1] if raw_node_mapping_prefix_s3.endswith('/')
            else raw_node_mapping_prefix_s3)

    # If no mappings exist this list will be empty
    s3_id_map_files = S3Downloader.list(
        raw_node_mapping_prefix_s3, sagemaker_session=sagemaker_session)
    for mapping_file in s3_id_map_files:
        # The expected layout for GConstruct mapping files on S3 is:
        # raw_id_mappings/node_type/part-xxxxx.parquet
        ntype = mapping_file.split("/")[-2]
        # This is the case where the output was generated by GSProcessing
        if ntype == "parquet":
            # Then we have raw_id_mappings/node_type/parquet/part-xxxxx.parquet
            ntype = mapping_file.split("/")[-3]
        os.makedirs(os.path.join(graph_path, "raw_id_mappings", ntype), exist_ok=True)

    def download_raw_mapping_file(s3_mapping_file):
        ntype = s3_mapping_file.split("/")[-2]
        # This is the case where the output was generated by GSProcessing
        if ntype == "parquet":
            # Then we have raw_id_mappings/node_type/parquet/part-xxxxx.parquet
            ntype = s3_mapping_file.split("/")[-3]
        mapping_key = s3_mapping_file.split("/", maxsplit=3)[3]
        filename = os.path.basename(mapping_key)
        local_dl_path = os.path.join(graph_path, "raw_id_mappings", ntype, filename)
        s3_client.download_file(
            s3_input_bucket, mapping_key, local_dl_path)

    # We expect the raw id mapping files to be many small files, so we download in parallel
    Parallel(n_jobs=min(DOWNLOAD_THREADS, os.cpu_count()), prefer="threads")(
        delayed(download_raw_mapping_file)(mapping_file) for mapping_file in s3_id_map_files)
    logging.info("Rank %d: Time to download %d raw id mapping files: %f seconds",
                 rank, len(s3_id_map_files), time.perf_counter() - raw_id_mappings_start)

    logging.info("Finished downloading graph data from %s", graph_data_s3)
    return os.path.join(graph_path, graph_config)


def upload_data_to_s3(s3_path, data_path, sagemaker_session):
    """ Upload data into S3

    Parameters
    ----------
    s3_path: str
        S3 uri to upload the data
    data_path: str
        Local data path
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    try:
        ret = S3Uploader.upload(data_path, s3_path,
            sagemaker_session=sagemaker_session)
    except Exception as err: # pylint: disable=broad-except
        logging.error("Can not upload data into %s", s3_path)
        raise RuntimeError(f"Can not upload data into {s3_path}. {err}")
    return ret

def upload_model_artifacts(model_s3_path, model_path, sagemaker_session):
    """ Upload trained model into S3

    The trained model includes all dense layers of input encoders, GNNs and
    output decoders and learnable sparse embeddings which are stored in
    a distributed manner.

    Parameters
    ----------
    model_s3_path: str
        S3 uri to upload the model artifacts.
    model_path: str
        Local path of the model artifacts.
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    logging.info("Uploading model artifacts to %s", model_s3_path)
    # Rank0 will upload both dense models and learnable embeddings owned by Rank0.
    # Other ranks will only upload learnable embeddings owned by themselves.
    return upload_data_to_s3(model_s3_path, model_path, sagemaker_session)

def upload_directory_parallel(local_prefix: str, s3_prefix: str, s3_client=None):
    """Upload all files under a local prefix to an S3 prefix

    Parameters
    ----------
    local_prefix : str
        Local directory prefix
    s3_prefix : str
        S3 prefix under which files will be uploaded
    s3_client : boto3.client, optional
        S3 boto client, by default None
    """
    if not s3_client:
        s3_client = boto3.client(
            "s3",
            config=botocore.config.Config(max_pool_connections=150),
            region_name=os.environ.get("AWS_REGION", None)
        )
    rank = get_rank()
    UPLOAD_THREADS=min(64, os.cpu_count()*2)

    local_src_s3_dst_tuples = get_upload_tuples(local_prefix, s3_prefix, include_filename=True)

    logging.info("Rank %d: Uploading %d embeddings files to %s",
                rank, len(local_src_s3_dst_tuples), s3_prefix)

    def upload_file(local_path: str, s3_uri: str):
        bucket = s3_uri.split("/")[2]
        key = s3_uri.split("/", maxsplit=3)[3]
        s3_client.upload_file(local_path, bucket, key)

    verbosity = 10 if rank == 0 else 0
    # We know we'll get many 'WARNING - Connection pool is full' warnings here so we suppress them
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Parallel(n_jobs=min(UPLOAD_THREADS, os.cpu_count()), prefer="threads", verbose=verbosity)(
                delayed(upload_file)(local_path, s3_path)
                    for (local_path, s3_path) in local_src_s3_dst_tuples
            )

def update_gs_params(gs_params, param_name, param_value):
    """ Update the graphstorm parameter `param_name` with a new
        value `param_value`. If the `param_name` does not exist,
        add it into gs_params

        Parameters
        ----------
        gs_params: list
            List of input parameters
        param_name: str
            The parameter to update
        param_value: str
            The new value
    """
    for i, pname in enumerate(gs_params):
        if pname == param_name:
            gs_params[i+1] = param_value
            return
    gs_params.append(param_name)
    gs_params.append(param_value)

def remove_data(path):
    """ Clean up local data under path

    Parameters
    ----------
    path: str
        Path to local data
    """
    shutil.rmtree(path)

def remove_embs(emb_path):
    """ Clean up saved embeddings locally,
    so SageMaker does not need to upload embs again

    Parameters
    ----------
    emb_path: str
        Local embedding path
    """
    remove_data(emb_path)

# From https://github.com/aws/sagemaker-python-sdk/blob/fb16a269daf4db6a717ef26c1a6bf7631c0c8d2d/src/sagemaker/session.py#L390-L406
def get_upload_tuples(local_path: str, key_prefix: str, include_filename: bool = False) -> List[Tuple[str]]:
    """Walks a directory to create a list of (local_src, s3_dst) paths for upload.

    Parameters
    ----------
    local_path : str
        A local path, can be a directory or single file.
    key_prefix : str
        An S3 key prefix under we want all local files uploaded.
    include_filename: bool (default: False)
        When True, will include the filename in the returned S3 URIs, otherwise
        will just return the prefix
    Returns
    -------
    List[Tuple[str]]
        A list of (local_src_path, s3_dist_path) tuples, one for each file
        under the input local_path.
    """
    # Generate a tuple for each file that we want to upload of the form (local_path, s3_key).
    files = []
    if os.path.isdir(local_path):
        for dirpath, _, filenames in os.walk(local_path):
            for name in filenames:
                file_path = os.path.join(dirpath, name)
                s3_relative_prefix = (
                    "" if local_path == dirpath else os.path.relpath(dirpath, start=local_path) + "/"
                )
                if include_filename:
                    s3_key = "{}/{}{}".format(key_prefix, s3_relative_prefix, name)
                else:
                    s3_key = "{}/{}".format(key_prefix, s3_relative_prefix)
                files.append((file_path, s3_key))
    else:
        _, name = os.path.split(local_path)
        s3_key = "{}/{}".format(key_prefix, name)
        files.append((local_path, s3_key))

    return files

def is_port_available(port):
    """Check if a port is available."""
    try:
        # Try to bind to all interfaces with a timeout
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Add one second timeout
            s.settimeout(1)
            s.bind(('', port))
            # Also try listening to ensure port is fully available
            s.listen(1)
            return True
    except (OSError, socket.timeout):
        return False

def find_free_port(start_port: Optional[int]=None):
    """Find next available port, starting from start_port."""
    if start_port is None:
        # Let OS choose
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]
        except OSError:
            # Fall back to manual search if OS assignment fails
            start_port = PORT_MIN

    # Try ports sequentially starting from start_port
    port = start_port
    while port <= PORT_MAX:
        if is_port_available(port):
            return port
        port += 1

    raise RuntimeError("No available ports found")

def get_job_port(job_str_identifier: Optional[str] = None):
    """Get port number based on per-job unique ID

    Parameters
    ----------
    unique_identifier : str, optional
        An identifier that should be unique to each SM job, by default None

    Returns
    -------
    int
        A common port number that master and workers will use
    """
    if not job_str_identifier:
        job_str_identifier = os.getenv('SM_USER_ARGS', '')

    # Create a hash of the unique identifier
    hash_object = hashlib.md5(job_str_identifier.encode())
    hash_hex = hash_object.hexdigest()

    # Convert first 4 chars of hash to int and scale to valid port range
    # Using 10000-65000 to avoid privileged ports and common ports
    base_port = PORT_MIN + (int(hash_hex[:4], 16) % (PORT_MAX - PORT_MIN))

    # Ensure we return an open port, starting at base_port
    port = find_free_port(base_port)
    return port
