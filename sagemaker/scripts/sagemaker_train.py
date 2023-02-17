""" training entry point
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
from graphstorm.config.config import BUILTIN_TASK_EDGE_CLASSIFICATION
from graphstorm.config.config import BUILTIN_TASK_EDGE_REGRESSION
from graphstorm.config.config import BUILTIN_TASK_LINK_PREDICTION

import utils
import sagemaker

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
    assert not enable_bert, 'We do not support BERT+GNN right now.'
    if task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        workspace = "/opt/ml/code/graphstorm/training_scripts/gsgnn_np"
        cmd = "gsgnn_np.py"
    elif task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        workspace = "/opt/ml/code/graphstorm/training_scripts/gsgnn_ep"
        cmd = "gsgnn_ep.py"
    elif task_type == BUILTIN_TASK_LINK_PREDICTION:
        workspace = "/opt/ml/code/graphstorm/training_scripts/gsgnn_lp"
        cmd = "gsgnn_lp.py"
    else:
        raise RuntimeError(f"Unsupported task type {task_type}")

    extra_args = " ".join(extra_args)

    launch_cmd = ["python3", "/root/dgl/tools/launch.py",
        "--workspace", f"{workspace}",
        "--num_trainers", f"{num_gpus}",
        "--num_servers", "1",
        "--num_samplers", "0",
        "--part_config", f"{graph_config}",
        "--ip_config", f"{ip_list}",
        "--extra_envs", f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']} ",
        "--ssh_port", "22",
        f"python3 {cmd} --cf {yaml_path} --ip-config {ip_list} " \
        f"--part-config {graph_config} --save-model-path {save_model_path} {extra_args}"]

    def run(launch_cmd, state_q):
        try:
            subprocess.check_call(launch_cmd, shell=False)
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
    parser.add_argument("--enable-bert",
        type=lambda x: (str(x).lower() in ['true', '1']), default=False,
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
    print(f"Know args {args}")
    print(f"Unknow args {unknownargs}")

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
    yaml_path = utils.download_yaml(train_yaml_s3, train_yaml_name,
        data_path, sagemaker_session)
    graph_config_path = utils.download_graph(graph_data_s3, graph_name,
        host_rank, data_path, sagemaker_session)

    err_code = 0
    if host_rank == 0:
        utils.barrier_master(client_list, world_size)

        # launch a thread to send keep alive message to all workers
        task_end = Event()
        thread = Thread(target=utils.keep_alive,
            args=(client_list, world_size, task_end),
            daemon=True)
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
        utils.terminate_workers(client_list, world_size, task_end)
        print("Master End")
    else:
        utils.barrier(sock)
        # Block util training finished
        # Listen to end command
        utils.wait_for_exit(sock)
        print("Worker End")

    sock.close()
    if err_code != 0:
        # Report an error
        print("Task failed")
        sys.exit(-1)

if __name__ == '__main__':
    main()
