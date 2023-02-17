""" sagemaker script utilities
"""

import os
import time
import shutil

from sagemaker.s3 import S3Downloader
from sagemaker.s3 import S3Uploader

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
        assert msg == "sync_ack"

    for rank in range(1, world_size):
        client_list[rank].send(b"synced")

def barrier(master_sock):
    """ Worker node barrier, called by host_rank > 0

    Parameters
    ----------
    master_sock: socket
        Socket connecting master
    """
    msg = master_sock.recv(8)
    msg = msg.decode()
    assert msg == "sync"

    master_sock.send(b"sync_ack")

    msg = master_sock.recv(8)
    msg = msg.decode()
    assert msg == "synced"

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

    print("Exit")

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
        print(f"Client {rank} exit {msg.decode()}")

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
        print(msg.decode())
    master_sock.send(b"Exit")

def download_yaml(yaml_s3, yaml_file_name, local_path, sagemaker_session):
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
    yaml_file_s3 = yaml_s3 + yaml_file_name
    yaml_path = os.path.join(local_path, yaml_file_name)
    ### Download Partitioned graph data

    os.makedirs(local_path, exist_ok=True)
    try:
        S3Downloader.download(yaml_file_s3, local_path,
            sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print(f"Can not download yaml config from {yaml_file_s3}.")
        print(f"Please check S3 folder {yaml_s3} and yaml file {yaml_file_name}")
        raise RuntimeError(f"Fail to download yaml file {yaml_file_s3}")

    return yaml_path

def download_graph(graph_data_s3, graph_name, part_id, local_path, sagemaker_session):
    """ download graph data

    Parameters
    ----------
    graph_data_s3: str
        S3 uri storing the partitioned graph data
    graph_name: str
        Graph name
    part_id: int
        Graph partition id
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
    graph_config = f"{graph_name}.json"
    graph_part = f"part{part_id}"

    graph_path = os.path.join(local_path, graph_name)
    graph_part_path = os.path.join(graph_path, graph_part)
    os.makedirs(graph_path, exist_ok=True)
    os.makedirs(graph_part_path, exist_ok=True)
    try:
        S3Downloader.download(os.path.join(graph_data_s3, graph_config),
            graph_path, sagemaker_session=sagemaker_session)
        S3Downloader.download(os.path.join(graph_data_s3, graph_part),
            graph_part_path, sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print(f"Can not download graph_data from {graph_data_s3}.")
        raise RuntimeError(f"Can not download graph_data from {graph_data_s3}.")

    return os.path.join(graph_path, graph_config)

def upload_embs(emb_s3_path, emb_path, sagemaker_session):
    """ Upload generated node embeddings into S3

    As embeddding table is huge and each trainer/inferer only
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
    try:
        ret = S3Uploader.upload(emb_path, emb_s3_path,
            sagemaker_session=sagemaker_session)
    except Exception: # pylint: disable=broad-except
        print(f"Can not upload data into {emb_s3_path}")
        raise RuntimeError(f"Can not upload data into {emb_s3_path}")
    return ret

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
