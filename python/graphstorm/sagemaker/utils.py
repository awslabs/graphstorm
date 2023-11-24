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
import subprocess
import logging
import os
import time
import shutil
from urllib.parse import urlparse

import boto3
from botocore.errorfactory import ClientError
from sagemaker.s3 import S3Downloader
from sagemaker.s3 import S3Uploader


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

def terminate_workers(client_list, world_size, task_end):
    """ termiate all worker deamons.

    Parameters
    ----------
    client_list: list
        List of socket clients
    world_size: int
        Size of the distributed training/inference cluster
    task_end: threading.Event
        Indicate whether the task has finished.
    """
    task_end.set()
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
    except Exception: # pylint: disable=broad-except
        raise RuntimeError(f"Fail to download yaml file {yaml_s3}")

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
    except Exception: # pylint: disable=broad-except
        raise RuntimeError("Can not download saved model artifact" \
                           f"model.bin from {model_artifact_s3}.")

def download_graph(graph_data_s3, graph_name, part_id, world_size,
                   local_path, sagemaker_session):
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

    Return
    ------
    graph_config_path: str
        Path to downloaded graph config file
    """
    # Download partitioned graph data.
    # Each training instance only download 1 partition.
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

    s3_client = boto3.client('s3')
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

    S3Downloader.download(os.path.join(graph_data_s3, graph_config),
            graph_path, sagemaker_session=sagemaker_session)
    try:
        S3Downloader.download(os.path.join(graph_data_s3, graph_part),
            graph_part_path, sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print(f"Can not download graph_data from {graph_data_s3}.")
        raise RuntimeError(f"Can not download graph_data from {graph_data_s3}.")

    node_id_mapping = "node_mapping.pt"
    # Try to download node id mapping file if any
    try:
        S3Downloader.download(os.path.join(graph_data_s3, node_id_mapping),
            graph_path, sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print("node id mapping file does not exist")

    if part_id == 0:
        # It is possible that id mappings are generated by
        # dgl tools/distpartitioning/convert_partition.py
        for i in range(1, world_size):
            local_graph_part = f"part{i}"
            graph_part_path = os.path.join(graph_path, local_graph_part)
            os.makedirs(graph_part_path, exist_ok=True)

            # Try to download node id mapping file if any
            s3_path = os.path.join(graph_data_s3, local_graph_part, "orig_nids.dgl")
            try:
                logging.info("Try to download %s to %s", s3_path, graph_part_path)
                S3Downloader.download(s3_path,
                    graph_part_path, sagemaker_session=sagemaker_session)
            except Exception: # pylint: disable=broad-except
                logging.info("node id mapping file %s does not exist", s3_path)

    # Try to get GraphStorm ID to Original ID remaping files if any
    files = S3Downloader.list(graph_data_s3, sagemaker_session=sagemaker_session)
    id_map_files = [file for file in files if file.endswith("id_remap.parquet")]
    for file in id_map_files:
        try:
            S3Downloader.download(file, graph_path,
                                  sagemaker_session=sagemaker_session)
        except Exception: # pylint: disable=broad-except
            print(f"node id remap file {file} does not exist")

    print(f"Finish download graph data from {graph_data_s3}")
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
    except Exception: # pylint: disable=broad-except
        print(f"Can not upload data into {s3_path}")
        raise RuntimeError(f"Can not upload data into {s3_path}")
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
    print(f"Upload model artifacts to {model_s3_path}")
    # Rank0 will upload both dense models and learnable embeddings owned by Rank0.
    # Other ranks will only upload learnable embeddings owned by themselves.
    return upload_data_to_s3(model_s3_path, model_path, sagemaker_session)

def upload_embs(emb_s3_path, emb_path, sagemaker_session):
    """ Upload generated node embeddings into S3

    As embeddding table is huge and each trainer/inferrer only
    stores part of the embedding, we need to upload them
    into S3.

    Parameters
    ----------
    emb_s3_path: str
        S3 uri to upload node embeddings
    emb_path: str
        Local embedding path
    sagemaker_session: sagemaker.session.Session
        sagemaker_session to run download
    """
    return upload_data_to_s3(emb_s3_path, emb_path, sagemaker_session)

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
