""" training entry point

    As SageMaker only accept a single script as the entry point.
    We have to put all code in one file.
"""
# Install additional requirements
import os
import socket
import time
import json
import subprocess
import argparse
from threading import Thread, Event
import sys
import queue

import boto3
from graphstorm.config.config import SUPPORTED_TASKS
from graphstorm.config.config import BUILTIN_TASK_NODE_CLASSIFICATION
from graphstorm.config.config import BUILTIN_TASK_NODE_REGRESSION
from graphstorm.config.config import BUILTIN_TASK_EDGE_CLASSIFICATOIN
from graphstorm.config.config import BUILTIN_TASK_EDGE_REGRESSION
from graphstorm.config.config import BUILTIN_TASK_LINK_PREDICTION

import sagemaker
from sagemaker.s3 import S3Downloader

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
    """ terminate all worker deamons.

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
    yaml_file_s3 = os.path.join(yaml_s3, yaml_file_name)
    yaml_path = os.path.join(local_path, yaml_file_name)
    ### Download Partitioned graph data

    os.makedirs(local_path, exist_ok=True)
    S3Downloader.download(yaml_file_s3, local_path, sagemaker_session=sagemaker_session)

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
    S3Downloader.download(os.path.join(graph_data_s3, graph_config),
        graph_path, sagemaker_session=sagemaker_session)
    S3Downloader.download(os.path.join(graph_data_s3, graph_part),
        graph_part_path, sagemaker_session=sagemaker_session)

    return os.path.join(graph_path, graph_config)

def launch_train_task(task_type, num_gpus, graph_config,
    save_model_path, ip_list, enable_bert,
    yaml_path, extra_args, state_q):
    """ Launch SageMaker training task

    Parameters
    ----------
    task_type: str
        Task type. It can be node classification/regression,
        edge classification/regression, link prediction, etc.
        Refer to graphstorm.config.config.SUPPORTED_TASKS for more details.
    num_gpus: int
        Number of gpus per instance
    graph_config: str
        Where does the graph partition config reside.
    save_model_path: str
        Output path to save models
    ip_list: str
        Where does the ip list reside.
    enable_bert: bool
        Whether BERT model is used during training.
    yaml_path: str
        Where does the yaml config file reside.
    extra_args: list
        Training args
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)

    Return
    ------
    Thread: training task thread
    """
    if task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
        workspace = "/graph-storm/training_scripts/gsgnn_nc"
        cmd = "gsgnn_nc_huggingface.py" if enable_bert else "gsgnn_pure_gnn_nc.py"
    elif task_type == BUILTIN_TASK_NODE_REGRESSION:
        workspace = "/graph-storm/training_scripts/gsgnn_nr"
        assert enable_bert is True, "gsgnn_pure_gnn_nr.py needs to be supported"
        cmd = "gsgnn_nr_huggingface.py" if enable_bert else "gsgnn_pure_gnn_nr.py"
    elif task_type == BUILTIN_TASK_EDGE_CLASSIFICATOIN:
        workspace = "/graph-storm/training_scripts/gsgnn_ec"
        cmd = "gsgnn_ec_huggingface.py" if enable_bert else "gsgnn_pure_gnn_ec.py"
    elif task_type == BUILTIN_TASK_EDGE_REGRESSION:
        workspace = "/graph-storm/training_scripts/gsgnn_er"
        assert enable_bert is True, "gsgnn_pure_gnn_er.py needs to be supported"
        cmd = "gsgnn_er_huggingface.py" if enable_bert else "gsgnn_pure_gnn_er.py"
    elif task_type == BUILTIN_TASK_LINK_PREDICTION:
        workspace = "/graph-storm/training_scripts/gsgnn_lp"
        cmd = "gsgnn_lp_huggingface.py" if enable_bert else "gsgnn_pure_gnn_lp.py"
    else:
        raise RuntimeError(f"Unsupported task type {task_type}")

    extra_args = " ".join(extra_args)

    launch_cmd = "python3 ~/dgl/tools/launch.py " \
        f"--workspace {workspace} " \
        f"--num_trainers {num_gpus} " \
        "--num_servers 1 " \
        "--num_samplers 0 " \
        f"--part_config {graph_config} " \
        f"--ip_config {ip_list} " \
        "--ssh_port 22 " \
        f"'python3 {cmd} --cf {yaml_path} --ip-config {ip_list} " \
        f"--part-config {graph_config} --save-model-path {save_model_path} {extra_args}'"

    def run(launch_cmd, state_q):
        try:
            subprocess.check_call(launch_cmd, shell=True)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception: # pylint: disable=broad-except
            state_q.put(-1)

    thread = Thread(target=run, args=(launch_cmd, state_q,), daemon=True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread

def parse_train_args():
    """ Add arguments for model training
    """
    parser = argparse.ArgumentParser(description='gs sagemaker train pipeline')

    parser.add_argument("--task_type", type=str,
        help=f"task type, builtin task type includes: {SUPPORTED_TASKS}")

    # distributed training
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}")
    parser.add_argument("--train-yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--train-yaml-name", type=str,
        help="Training yaml config file name")
    parser.add_argument("--enable-bert", type=bool, default=False,
        help="Whether enable cotraining Bert with GNN")

    return parser

def main():
    """ main logic
    """
    if 'SM_NUM_GPUS' in os.environ:
        num_gpus = int(os.environ['SM_NUM_GPUS'])

    for key, val in os.environ.items():
        print(f"{key}: {val}")

    assert 'SM_CHANNEL_TRAIN' in os.environ, \
        "SageMaker trainer should have the data path in os.environ."
    data_path = str(os.environ['SM_CHANNEL_TRAIN'])

    output_path = "/opt/ml/model/"
    save_model_path = os.path.join(output_path, "model_checkpoint")

    # start the ssh server
    subprocess.run(["service", "ssh", "start"], check=True)

    parser = parse_train_args()
    args, unknownargs = parser.parse_known_args()
    print(args)
    print(unknownargs)

    train_env = json.loads(os.environ['SM_TRAINING_ENV'])
    hosts = train_env['hosts']
    current_host = train_env['current_host']
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
        sock.bind((master_addr, 12345))
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
                sock.connect((master_addr, 12345))
                break
            except: # pylint: disable=bare-except
                print(f"Try to connect {master_addr}")
                time.sleep(10)
        print("Connected")

    # write ip list info into disk
    ip_list_path = os.path.join(data_path, 'ip_list.txt')
    with open(ip_list_path, 'w', encoding='utf-8') as f:
        for host in hosts:
            f.write(f"{socket.gethostbyname(host)}\n")

    gs_params = unknownargs
    graph_name = args.graph_name
    graph_data_s3 = args.graph_data_s3
    task_type = args.task_type
    train_yaml_s3 = args.train_yaml_s3
    train_yaml_name = args.train_yaml_name
    enable_bert = args.enable_bert

    boto_session = boto3.session.Session(region_name=os.environ['AWS_REGION'])
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session)
    yaml_path = download_yaml(train_yaml_s3, train_yaml_name,
        data_path, sagemaker_session)
    graph_config_path = download_graph(graph_data_s3, graph_name,
        host_rank, data_path, sagemaker_session)

    err_code = 0
    if host_rank == 0:
        # launch a thread to send keep alive message to all workers
        task_end = Event()
        thread = Thread(target=keep_alive, args=(client_list, world_size, task_end), daemon=True)
        thread.start()

        try:
            # launch distributed training here
            state_q = queue.Queue()
            train_task = launch_train_task(task_type,
                                        num_gpus,
                                        graph_config_path,
                                        save_model_path,
                                        ip_list_path,
                                        enable_bert,
                                        yaml_path,
                                        gs_params,
                                        state_q)
            train_task.join()
            err_code = state_q.get()
        except RuntimeError as e:
            print(e)
            err_code = -1
        terminate_workers(client_list, world_size, task_end)
        print("Master End")
    else:
        # Block util training finished
        # Listen to end command
        wait_for_exit(sock)
        print("Worker End")

    print(os.listdir(output_path))

    sock.close()
    if err_code != 0:
        # Report an error
        print("Task failed")
        sys.exit(-1)

if __name__ == '__main__':
    main()
