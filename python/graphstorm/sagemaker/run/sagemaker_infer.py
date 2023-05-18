""" Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Inference entry point.

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
from graphstorm.config.config import BUILTIN_TASK_EDGE_CLASSIFICATION
from graphstorm.config.config import BUILTIN_TASK_EDGE_REGRESSION
from graphstorm.config.config import BUILTIN_TASK_LINK_PREDICTION

import sagemaker
from graphstorm.sagemaker.run.utils import download_yaml_config
from graphstorm.sagemaker.run.utils import download_graph
from graphstorm.sagemaker.run.utils import keep_alive
from graphstorm.sagemaker.run.utils import barrier_master
from graphstorm.sagemaker.run.utils import barrier
from graphstorm.sagemaker.run.utils import terminate_workers
from graphstorm.sagemaker.run.utils import wait_for_exit
from graphstorm.sagemaker.run.utils import upload_data_to_s3
from graphstorm.sagemaker.run.utils import update_gs_params
from graphstorm.sagemaker.run.utils import download_model
from graphstorm.sagemaker.run.utils import upload_embs
from graphstorm.sagemaker.run.utils import remove_embs

def launch_infer_task(task_type, num_gpus, graph_config,
    load_model_path, save_emb_path, ip_list,
    yaml_path, extra_args, state_q, custom_script):
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
    load_model_path: str
        Where to load graph model.
    save_emb_path: str
        Output path to save inference result and node embeddings.
    ip_list: str
        Where does the ip list reside.
    yaml_path: str
        Where does the yaml config file reside.
    extra_args: list
        Training args
    state_q: queue.Queue()
        A queue used to return execution result (success or failure)
    custom_script: str
        Custom training script provided by a customer to run customer training logic.
    """
    if custom_script is not None:
        cmd = "graphstorm.run.launch"
    elif task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
        cmd = "graphstorm.run.gs_node_classification"
    elif task_type == BUILTIN_TASK_NODE_REGRESSION:
        cmd = "graphstorm.run.gs_node_classification"
    elif task_type == BUILTIN_TASK_EDGE_CLASSIFICATION:
        cmd = "graphstorm.run.gs_edge_classification"
    elif task_type == BUILTIN_TASK_EDGE_REGRESSION:
        cmd = "graphstorm.run.gs_edge_regression"
    elif task_type == BUILTIN_TASK_LINK_PREDICTION:
        cmd = "graphstorm.run.gs_link_prediction"
    else:
        raise RuntimeError(f"Unsupported task type {task_type}")

    launch_cmd = ["python3", "-m", cmd,
        "--num-trainers", f"{num_gpus}",
        "--num-servers", "1",
        "--num-samplers", "0",
        "--part-config", f"{graph_config}",
        "--ip-config", f"{ip_list}",
        "--extra-envs", f"LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']} ",
        "--ssh-port", "22", "--inference"]
    launch_cmd += [custom_script] if custom_script is not None else []
    launch_cmd += ["--cf", f"{yaml_path}",
         "--restore-model-path", f"{load_model_path}",
         "--save-embed-path", f"{save_emb_path}"] + extra_args

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
    """ Add arguments for model offline inference
    """
    parser = argparse.ArgumentParser(description='gs sagemaker train pipeline')

    parser.add_argument("--task_type", type=str,
        help=f"task type, builtin task type includes: {SUPPORTED_TASKS}")

    # disrributed training
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}")
    parser.add_argument("--infer-yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--output-emb-s3", type=str,
        help="S3 location to store GraphStorm generated node embeddings.",
        default=None)
    parser.add_argument("--output-prediction-s3", type=str,
        help="S3 location to store prediction results. " \
             "(Only works with node classification/regression " \
             "and edge classification/regression tasks)",
        default=None)
    parser.add_argument("--model-artifact-s3", type=str,
        help="S3 bucket to load the saved model artifacts")
    parser.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by a customer to run customer training logic. \
            Please provide the path of the script within the docker image")

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
    model_path = '/opt/ml/model'
    output_path = '/opt/ml/checkpoints'

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
    model_artifact_s3 = args.model_artifact_s3
    task_type = args.task_type
    infer_yaml_s3 = args.infer_yaml_s3
    # remove tailing /
    output_emb_s3 = args.output_emb_s3.rstrip('/')
    custom_script = args.custom_script
    emb_path = os.path.join(output_path, "embs")

    if args.output_emb_s3 is not None:
        update_gs_params(gs_params, "--save-embed-path", emb_path)
    if args.output_prediction_s3 is not None:
        update_gs_params(gs_params, "--save-prediction-path", os.path.join(output_path, "predict"))

    ### Download Partitioned graph data
    boto_session = boto3.session.Session(region_name=os.environ['AWS_REGION'])
    sagemaker_session = sagemaker.session.Session(boto_session=boto_session)

    yaml_path = download_yaml_config(infer_yaml_s3,
        data_path, sagemaker_session)

    graph_config_path = download_graph(graph_data_s3, graph_name,
        host_rank, data_path, sagemaker_session)

    # Download Saved model
    download_model(model_artifact_s3, model_path,
        host_rank, world_size, graph_config_path, sagemaker_session)
    print(f"{model_path} {os.listdir(model_path)}")

    err_code = 0
    if host_rank == 0:
        barrier_master(client_list, world_size)

        # launch a thread to send keep alive message to all workers
        task_end = Event()
        thread = Thread(target=keep_alive,
            args=(client_list, world_size, task_end),
            daemon=True)
        thread.start()

        try:
            # launch distributed training here
            state_q = queue.Queue()
            train_task = launch_infer_task(task_type,
                                           num_gpus,
                                           graph_config_path,
                                           model_path,
                                           emb_path,
                                           ip_list_path,
                                           yaml_path,
                                           gs_params,
                                           state_q,
                                           custom_script)
            train_task.join()
            err_code = state_q.get()
        except RuntimeError as e:
            print(e)
            err_code = -1

        terminate_workers(client_list, world_size, task_end)
        print("Master End")
        if err_code != -1:
            upload_embs(output_emb_s3, emb_path, sagemaker_session)
            # clean embs, so SageMaker does not need to upload embs again
            remove_embs(emb_path)
    else:
        barrier(sock)

        # Block util training finished
        # Listen to end command
        wait_for_exit(sock)
        upload_embs(output_emb_s3, emb_path, sagemaker_session)
        # clean embs, so SageMaker does not need to upload embs again
        remove_embs(emb_path)
        print("Worker End")

    sock.close()
    if err_code != 0:
        # Report an error
        print("Task failed")
        sys.exit(-1)

    if args.output_prediction_s3 is not None:
        # remove tailing /
        output_prediction_s3 = args.output_prediction_s3.rstrip('/')
        upload_data_to_s3(output_prediction_s3,
                          os.path.join(output_path, "predict"),
                          sagemaker_session)

if __name__ == '__main__':
    main()
