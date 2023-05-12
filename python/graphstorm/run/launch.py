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

    Launching tool for launching GraphStorm distributed training and inference
    based on DGL distributed training.

    Run as:
    python3 -m graphstorm.run.launch <Launch args> YOUR_SCRIPT.py <Train/Infer args>
"""
import argparse
import json
import logging
import multiprocessing
import os
import queue
import re
import signal
import subprocess
import sys
import time
from functools import partial
from threading import Thread
from typing import Optional
from argparse import REMAINDER


def cleanup_proc(get_all_remote_pids_func, conn):
    """This process tries to clean up the remote training tasks.

        Parameters
        ----------
        get_all_remote_pids: func
            Function to get all remote pids
        conn:
            connection
    """
    print("cleanupu process runs")
    # This process should not handle SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    data = conn.recv()
    # If the launch process exits normally, this process doesn't need to do anything.
    if data == "exit":
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids_func()
        # Otherwise, we need to ssh to each machine and kill the training jobs.
        for (ip, port), pids in remote_pids.items():
            kill_process(ip, port, pids)
    print("cleanup process exits")


def kill_process(ip, port, pids):
    """ssh to a remote machine and kill the specified processes.

        Parameters
        ----------
        ip: str
            IP adrress of the remote machine
        port: str
            SSH port
        pids: list
            Pid list
    """
    curr_pid = os.getpid()
    killed_pids = []
    # If we kill child processes first, the parent process may create more again. This happens
    # to Python's process pool. After sorting, we always kill parent processes first.
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print("kill process {} on {}:{}".format(pid, ip, port), flush=True)
        kill_cmd = (
            "ssh -o StrictHostKeyChecking=no -p "
            + str(port)
            + " "
            + ip
            + " 'kill {}'".format(pid)
        )
        subprocess.run(kill_cmd, shell=True, check=False)
        killed_pids.append(pid)
    # It's possible that some of the processes are not killed. Let's try again.
    for _ in range(3):
        killed_pids = get_killed_pids(ip, port, killed_pids)
        if len(killed_pids) == 0:
            break

        killed_pids.sort()
        for pid in killed_pids:
            print(
                "kill process {} on {}:{}".format(pid, ip, port), flush=True
            )
            kill_cmd = (
                "ssh -o StrictHostKeyChecking=no -p "
                + str(port)
                + " "
                + ip
                + " 'kill -9 {}'".format(pid)
            )
            subprocess.run(kill_cmd, shell=True, check=False)


def get_killed_pids(ip, port, killed_pids):
    """Get the process IDs that we want to kill but are still alive.

        Parameters
        ----------
        ip: str
            IP adrress of the remote machine
        port: str
            SSH port
        killed_pids: list
            Pid list
    """
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ",".join(killed_pids)
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'ps -p {} -h'".format(killed_pids)
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE, check=False)
    pids = []
    for process in res.stdout.decode("utf-8").split("\n"):
        process_list = process.split()
        if len(process_list) > 0:
            pids.append(int(process_list[0]))
    return pids


def execute_remote(
    cmd: str,
    state_q: queue.Queue,
    ip: str,
    port: int,
    username: Optional[str] = "",
) -> Thread:
    """Execute command line on remote machine via ssh.

        Parameters
        ----------
        cmd:
            User-defined command (udf) to execute on the remote host.
        state_q:
            A queue collecting Thread exit states.
        ip:
            The ip-address of the host to run the command on.
        port:
            Port number that the host is listening on.
        username: Optional. If given, this will specify a username to use
            when issuing commands over SSH.
            Useful when your infra requires you to explicitly specify a
            username to avoid permission issues.

    Returns:
        thread: The Thread whose run() is to run the `cmd` on the remote host.
        Returns when the cmd completes on the remote host.
    """
    ip_prefix = ""
    if username:
        ip_prefix += "{username}@".format(username=username)

    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} '{cmd}'".format(
        port=str(port),
        ip_prefix=ip_prefix,
        ip=ip,
        cmd=cmd,
    )

    # thread func to run the job
    def run(ssh_cmd, state_q):
        try:
            subprocess.check_call(ssh_cmd, shell=True)
            state_q.put(0)
        except subprocess.CalledProcessError as err:
            print(f"Called process error {err}")
            state_q.put(err.returncode)
        except Exception: # pylint: disable=broad-exception-caught
            state_q.put(-1)

    thread = Thread(
        target=run,
        args=(
            ssh_cmd,
            state_q,
        ),
    )
    thread.setDaemon(True)
    thread.start()
    # sleep for a while in case of ssh is rejected by peer due to busy connection
    time.sleep(0.2)
    return thread


def get_remote_pids(ip, port, cmd_regex):
    """Get the process IDs that run the command in the remote machine.

        Parameters
        ----------
        ip: str
            IP adrress of the remote machine
        port: str
            SSH port
        cmd_regex:
            command pattern
    """
    pids = []
    curr_pid = os.getpid()
    # Here we want to get the python processes.
    # We may get some ssh processes, so we should filter them out.
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'ps -aux | grep python | grep -v StrictHostKeyChecking'"
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE, check=False)
    for process in res.stdout.decode("utf-8").split("\n"):
        process_list = process.split()
        if len(process_list) < 2:
            continue
        # We only get the processes that run the specified command.
        res = re.search(cmd_regex, process)
        if res is not None and int(process_list[1]) != curr_pid:
            pids.append(process_list[1])

    pid_str = ",".join([str(pid) for pid in pids])
    ps_cmd = (
        "ssh -o StrictHostKeyChecking=no -p "
        + str(port)
        + " "
        + ip
        + " 'pgrep -P {}'".format(pid_str)
    )
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE, check=False)
    pids1 = res.stdout.decode("utf-8").split("\n")
    all_pids = []
    for pid in set(pids + pids1):
        if pid == "" or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids


def get_all_remote_pids(hosts, ssh_port, udf_command):
    """Get all remote processes.

        Parameters
        ----------
        hosts: list
            list of hosts
        ssh_port: str
            SSH port
        udf_command:
            command
    """
    remote_pids = {}
    for _, host in enumerate(hosts):
        ip, _ = host
        # When creating training processes in remote machines, we may insert some arguments
        # in the commands. We need to use regular expressions to match the modified command.
        new_udf_command = " .*".join(udf_command)
        pids = get_remote_pids(ip, ssh_port, new_udf_command)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids


def construct_torch_dist_launcher_cmd(
    num_trainers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> str:
    """Constructs the torch distributed launcher command.
       Helper function.

        Parameters
        ----------
        num_trainers:
            Number of trainers on each machine.
        num_nodes:
            Number of machines
        node_rank:
            Rank of current node
        master_addr:
            Master address
        master_port:
            Master port

        Returns
        -------
            cmd_str.
    """
    torch_cmd_template = (
        "-m torch.distributed.launch "
        "--nproc_per_node={nproc_per_node} "
        "--nnodes={nnodes} "
        "--node_rank={node_rank} "
        "--master_addr={master_addr} "
        "--master_port={master_port}"
    )
    return torch_cmd_template.format(
        nproc_per_node=num_trainers,
        nnodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )


def wrap_udf_in_torch_dist_launcher(
    udf_command: str,
    num_trainers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
    master_port: int,
) -> str:
    """Wraps the user-defined function (udf_command) with the torch.distributed.launch module:

         "python3 -m torch.distributed.launch <TORCH DIST ARGS> run/some/trainer.py arg1 arg2

        Parameters
        ----------
        udf_command:
            Execution command
        num_trainers:
            Number of trainers on each machine.
        num_nodes:
            Number of machines
        node_rank:
            Rank of current node
        master_addr:
            Master address
        master_port:
            Master port

    Returns:
        New command
    """
    torch_dist_cmd = construct_torch_dist_launcher_cmd(
        num_trainers=num_trainers,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr,
        master_port=master_port,
    )

    # Get the python interpreter used right now.
    # If we can not get it we go with the default `python3`
    python_bin = sys.executable \
        if sys.executable is not None and sys.executable != "" \
        else "python3 "

    # transforms the udf_command from:
    #     path/to/dist_trainer.py arg0 arg1
    # to:
    #     python -m torch.distributed.launch [DIST TORCH ARGS] path/to/dist_trainer.py arg0 arg1
    udf_command = " ".join(udf_command)
    new_udf_command = f"{python_bin} {torch_dist_cmd} {udf_command}"

    return new_udf_command


def construct_dgl_server_env_vars(
    num_samplers: int,
    num_server_threads: int,
    tot_num_clients: int,
    part_config: str,
    ip_config: str,
    num_servers: int,
    graph_format: str,
    pythonpath: Optional[str] = "",
) -> str:
    """Constructs the DGL server-specific env vars string that
       are required for DGL code to behave in the correct server role.
    Convenience function.

        Parameters
        ----------
        num_samplers:
            Number of sampler per server
        num_server_threads:
            Number of OMP threads per server
        tot_num_clients:
            Total number of clients
        part_config:
            Partition config.
        ip_config:
            IP config file containing IP addresses of cluster hosts.
        num_servers:
            Number of servers
        graph_format:
            Graph format
        pythonpath:
            Optional. If given, this will pass this as PYTHONPATH.

    Returns:
        server_env_vars: The server-specific env-vars in a string format,
        friendly for CLI execution.

    """
    server_env_vars_template = (
        "DGL_ROLE={DGL_ROLE} "
        "DGL_NUM_SAMPLER={DGL_NUM_SAMPLER} "
        "OMP_NUM_THREADS={OMP_NUM_THREADS} "
        "DGL_NUM_CLIENT={DGL_NUM_CLIENT} "
        "DGL_CONF_PATH={DGL_CONF_PATH} "
        "DGL_IP_CONFIG={DGL_IP_CONFIG} "
        "DGL_NUM_SERVER={DGL_NUM_SERVER} "
        "DGL_GRAPH_FORMAT={DGL_GRAPH_FORMAT} "
        "{suffix_optional_envvars}"
    )
    suffix_optional_envvars = ""
    if pythonpath:
        suffix_optional_envvars += f"PYTHONPATH={pythonpath} "
    return server_env_vars_template.format(
        DGL_ROLE="server",
        DGL_NUM_SAMPLER=num_samplers,
        OMP_NUM_THREADS=num_server_threads,
        DGL_NUM_CLIENT=tot_num_clients,
        DGL_CONF_PATH=part_config,
        DGL_IP_CONFIG=ip_config,
        DGL_NUM_SERVER=num_servers,
        DGL_GRAPH_FORMAT=graph_format,
        suffix_optional_envvars=suffix_optional_envvars,
    )


def construct_dgl_client_env_vars(
    num_samplers: int,
    tot_num_clients: int,
    part_config: str,
    ip_config: str,
    num_servers: int,
    graph_format: str,
    num_omp_threads: int,
    group_id: int,
    pythonpath: Optional[str] = "",
) -> str:
    """Constructs the DGL client-specific env vars string that are
        required for DGL code to behave in the correct client role.
    Convenience function.

    Parameters
    ----------
    num_samplers:
        Number of sampler per server
    tot_num_clients:
        Total number of clients
    part_config:
        Partition config.
    ip_config:
        IP config file containing IP addresses of cluster hosts.
    num_servers:
        Number of servers per machine.
    graph_format:
        Graph format
    num_omp_threads:
        Number of OMP threads per trainer
    group_id:
        Used in client processes to indicate which group it belongs to.
    pythonpath:
        Optional. If given, this will pass this as PYTHONPATH.

    Returns:
        client_env_vars: The client-specific env-vars in a string format,
        friendly for CLI execution.
    """
    client_env_vars_template = (
        "DGL_DIST_MODE={DGL_DIST_MODE} "
        "DGL_ROLE={DGL_ROLE} "
        "DGL_NUM_SAMPLER={DGL_NUM_SAMPLER} "
        "DGL_NUM_CLIENT={DGL_NUM_CLIENT} "
        "DGL_CONF_PATH={DGL_CONF_PATH} "
        "DGL_IP_CONFIG={DGL_IP_CONFIG} "
        "DGL_NUM_SERVER={DGL_NUM_SERVER} "
        "DGL_GRAPH_FORMAT={DGL_GRAPH_FORMAT} "
        "OMP_NUM_THREADS={OMP_NUM_THREADS} "
        "DGL_GROUP_ID={DGL_GROUP_ID} "
        "{suffix_optional_envvars}"
    )
    # append optional additional env-vars
    suffix_optional_envvars = ""
    if pythonpath:
        suffix_optional_envvars += f"PYTHONPATH={pythonpath} "
    return client_env_vars_template.format(
        DGL_DIST_MODE="distributed",
        DGL_ROLE="client",
        DGL_NUM_SAMPLER=num_samplers,
        DGL_NUM_CLIENT=tot_num_clients,
        DGL_CONF_PATH=part_config,
        DGL_IP_CONFIG=ip_config,
        DGL_NUM_SERVER=num_servers,
        DGL_GRAPH_FORMAT=graph_format,
        OMP_NUM_THREADS=num_omp_threads,
        DGL_GROUP_ID=group_id,
        suffix_optional_envvars=suffix_optional_envvars,
    )


def wrap_cmd_with_local_envvars(cmd: str, env_vars: str) -> str:
    """Wraps a CLI command with desired env vars with the following properties:
    (1) env vars persist for the entire `cmd`, even if it consists of multiple
        "chained" commands like:
        cmd = "ls && pwd && python run/something.py"
    (2) env vars don't pollute the environment after `cmd` completes.

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = "VAR1=value1 VAR2=value2"
        >>> wrap_cmd_with_local_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Parameters
    ----------
    cmd:
        cmd
    env_vars:
        A string containing env vars, eg "VAR1=val1 VAR2=val2"

    Returns:
        cmd_with_env_vars:

    """
    # use `export` to persist env vars for entire cmd block. required if
    # udf_command is a chain of commands
    # also: wrap in parens to not pollute env:
    #     https://stackoverflow.com/a/45993803
    return f"(export {env_vars}; {cmd})"


def wrap_cmd_with_extra_envvars(cmd: str, env_vars: list) -> str:
    """Wraps a CLI command with extra env vars

    Example:
        >>> cmd = "ls && pwd"
        >>> env_vars = ["VAR1=value1", "VAR2=value2"]
        >>> wrap_cmd_with_extra_envvars(cmd, env_vars)
        "(export VAR1=value1 VAR2=value2; ls && pwd)"

    Parameters
    ----------
    cmd:
        cmd
    env_vars:
        A list of strings containing env vars, e.g., ["VAR1=value1", "VAR2=value2"]

    Returns:
        cmd_with_env_vars:
    """
    env_vars = " ".join(env_vars)
    return wrap_cmd_with_local_envvars(cmd, env_vars)

GLOBAL_GROUP_ID = 0

def update_udf_command(udf_command, args):
    """ Update udf_command with arguments from args.

        The arguments to update includes:
        1. ip-config
        2. part-config
        3. verbose

        Parameters
        ----------
        udf_command: list
            Execution arguments to update
        args:
            Launch arguments
    """
    udf_command.append("--ip-config")
    udf_command.append(args.ip_config)

    udf_command.append("--part-config")
    udf_command.append(args.part_config)

    udf_command.append("--verbose")
    udf_command.append(str(args.verbose))

    return udf_command

def get_available_port(ip):
    """Get available port with specified ip.

        Parameters
        ----------
        ip: str
            Current ip
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(1234, 65535):
        try:
            sock.connect((ip, port))
        except Exception: # pylint: disable=broad-exception-caught
            return port
    raise RuntimeError("Failed to get available port for ip~{}".format(ip))


def submit_jobs(args, udf_command):
    """Submit distributed jobs (server and client processes) via ssh

        Parameters
        ----------
        args:
            Launch arguments
        udf_command: list
            Execution arguments to update
    """
    servers_cmd = []
    clients_cmd = []
    hosts = []
    thread_list = []
    server_count_per_machine = 0

    # Get the IP addresses of the cluster.
    # The path to the ip config file can be a absolute path or
    # a relative path to the workspace
    ip_config = args.ip_config if os.path.isabs(args.ip_config) else \
        os.path.join(args.workspace, args.ip_config)
    args.ip_config = ip_config
    assert os.path.isfile(ip_config), \
        f"IP config file must be provided but got {ip_config}"

    with open(ip_config, encoding='utf-8') as f:
        for line in f:
            result = line.strip().split()
            if len(result) == 2:
                ip = result[0]
                port = int(result[1])
                hosts.append((ip, port))
            elif len(result) == 1:
                ip = result[0]
                port = get_available_port(ip)
                hosts.append((ip, port))
            else:
                raise RuntimeError("Format error of ip_config.")
            server_count_per_machine = args.num_servers

    # Get partition info of the graph data
    # The path to the partition config file can be a absolute path or
    # a relative path to the workspace
    part_config = args.part_config if os.path.isabs(args.part_config) else \
        os.path.join(args.workspace, args.part_config)
    args.part_config = part_config

    with open(part_config, encoding='utf-8') as conf_f:
        part_metadata = json.load(conf_f)
    assert "num_parts" in part_metadata, "num_parts does not exist."
    # The number of partitions must match the number of machines in the cluster.
    assert part_metadata["num_parts"] == len(
        hosts
    ), "The number of graph partitions has to match the number of machines in the cluster."

    state_q = queue.Queue()
    tot_num_clients = args.num_trainers * (1 + args.num_samplers) * len(hosts)

    udf_command = update_udf_command(udf_command, args)
    # launch server tasks
    server_cmd = f"{sys.executable} {' '.join(udf_command)}" \
        if sys.executable is not None and sys.executable != "" \
        else f"python3 {' '.join(udf_command)}"

    server_env_vars = construct_dgl_server_env_vars(
        num_samplers=args.num_samplers,
        num_server_threads=args.num_server_threads,
        tot_num_clients=tot_num_clients,
        part_config=args.part_config,
        ip_config=args.ip_config,
        num_servers=args.num_servers,
        graph_format=args.graph_format,
        pythonpath=os.environ.get("PYTHONPATH", ""),
    )
    for i in range(len(hosts) * server_count_per_machine):
        ip, _ = hosts[int(i / server_count_per_machine)]
        server_env_vars_cur = f"{server_env_vars} DGL_SERVER_ID={i}"
        cmd = wrap_cmd_with_local_envvars(server_cmd, server_env_vars_cur)
        cmd = (
            wrap_cmd_with_extra_envvars(cmd, args.extra_envs)
            if len(args.extra_envs) > 0
            else cmd
        )
        cmd = "cd " + str(args.workspace) + "; " + cmd
        servers_cmd.append(cmd)

        thread_list.append(
            execute_remote(
                cmd,
                state_q,
                ip,
                args.ssh_port,
                username=args.ssh_username,
            )
        )

    # launch client tasks
    client_env_vars = construct_dgl_client_env_vars(
        num_samplers=args.num_samplers,
        tot_num_clients=tot_num_clients,
        part_config=args.part_config,
        ip_config=args.ip_config,
        num_servers=args.num_servers,
        graph_format=args.graph_format,
        num_omp_threads=os.environ.get(
            "OMP_NUM_THREADS", str(args.num_omp_threads)
        ),
        group_id=GLOBAL_GROUP_ID,
        pythonpath=os.environ.get("PYTHONPATH", ""),
    )

    master_addr = hosts[0][0]
    master_port = get_available_port(master_addr)
    for node_id, host in enumerate(hosts):
        ip, _ = host
        # Transform udf_command to follow torch's dist launcher format:
        # `PYTHON_BIN -m torch.distributed.launch ... UDF`
        torch_dist_udf_command = wrap_udf_in_torch_dist_launcher(
            udf_command=udf_command,
            num_trainers=args.num_trainers,
            num_nodes=len(hosts),
            node_rank=node_id,
            master_addr=master_addr,
            master_port=master_port,
        )
        cmd = wrap_cmd_with_local_envvars(
            torch_dist_udf_command, client_env_vars
        )
        cmd = (
            wrap_cmd_with_extra_envvars(cmd, args.extra_envs)
            if len(args.extra_envs) > 0
            else cmd
        )
        cmd = "cd " + str(args.workspace) + "; " + cmd
        clients_cmd.append(cmd)
        thread_list.append(
            execute_remote(
                cmd, state_q, ip, args.ssh_port, username=args.ssh_username
            )
        )

        if args.verbose:
            print(torch_dist_udf_command)

    # Start a cleanup process dedicated for cleaning up remote training jobs.
    conn1, conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, udf_command)
    process = multiprocessing.Process(target=cleanup_proc, args=(func, conn1))
    process.start()

    def signal_handler(sig, frame): # pylint: disable=unused-argument
        logging.info("Stop launcher")
        # We need to tell the cleanup process to kill remote training jobs.
        conn2.send("cleanup")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    err = 0
    for thread in thread_list:
        thread.join()
        err_code = state_q.get()
        if err_code != 0:
            # Record err_code
            # We record one of the error if there are multiple
            err = err_code

    # The training processes complete. We should tell the cleanup process to exit.
    conn2.send("exit")
    process.join()
    if err != 0:
        print("Task failed")
        sys.exit(-1)

def get_argument_parser():
    """ Arguments listed here are those used by the launch script to launch
        a distribute task.
    """
    parser = argparse.ArgumentParser(description="Launch a distributed job")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH Port.")
    parser.add_argument(
        "--ssh-username",
        default="",
        help="Optional. When issuing commands (via ssh) to cluster, \
              use the provided username in the ssh cmd. "
             "Example: If you provide --ssh_username=bob, \
              then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
             "instead of 'ssh 1.2.3.4 CMD'",
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: (str(x).lower() in ['true', '1']),
        default=False,
        help="Print more information.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        type=str,
        help="Path of user directory of distributed tasks. \
              This is used to specify a destination location treated \
              as PWD",
    )
    parser.add_argument(
        "--num-trainers",
        type=int,
        help="The number of trainer processes per machine",
    )
    parser.add_argument(
        "--num-omp-threads",
        type=int,
        help="The number of OMP threads per trainer",
    )
    parser.add_argument(
        "--num-samplers",
        type=int,
        default=0,
        help="The number of sampler processes per trainer process",
    )
    parser.add_argument(
        "--num-servers",
        type=int,
        default=1,
        help="The number of server processes per machine",
    )
    parser.add_argument(
        "--part-config",
        type=str,
        help="The file of the partition config. Absolute path is preferred. \
              Otherwise, the file should be in workspace.",
    )
    parser.add_argument(
        "--ip-config",
        type=str,
        help="The file of IP configuration for server processes. \
              Absolute path is preferred. \
              Otherwise, the file should be in workspace.",
    )
    parser.add_argument(
        "--num-server-threads",
        type=int,
        default=1,
        help="The number of OMP threads in the server process. \
                        It should be small if server processes and trainer processes run on \
                        the same machine. By default, it is 1.",
    )
    parser.add_argument(
        "--graph-format",
        type=str,
        default="csc",
        help='The format of the graph structure of each partition. \
              The allowed formats are csr, csc and coo. A user can specify multiple \
              formats, separated by ",". For example, the graph format is "csr,csc".',
    )
    parser.add_argument(
        "--extra-envs",
        nargs="+",
        type=str,
        default=[],
        help="Extra environment parameters need to be set. For example, \
              you can set the LD_LIBRARY_PATH and NCCL_DEBUG by adding: \
              --extra-envs LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
              NCCL_DEBUG=INFO ",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Inidcate that it is a inference task. \
              Used with built-in training/inference scripts"
    )
    parser.add_argument(
        "--lm-encoder-only",
        action="store_true",
        help="Inidcate that the model is using language model + decoder only. \
            No GNN is involved, only graph structure. \
            Used with built-in training/inference scripts"
    )
    return parser

def check_input_arguments(args):
    """ Check the correctness of input arguments

        Parameters
        ----------
        args:
            Input argument.
    """
    assert (
        args.num_trainers is not None and args.num_trainers > 0
    ), "--num-trainers must be a positive number."
    assert (
        args.num_samplers is not None and args.num_samplers >= 0
    ), "--num-samplers must be a non-negative number."
    assert (
        args.num_servers is not None and args.num_servers > 0
    ), "--num-servers must be a positive number."
    assert (
        args.num_server_threads > 0
    ), "--num-server-threads must be a positive number."
    assert (
        args.part_config is not None
    ), "A user has to specify a partition configuration file with --part-onfig."
    assert (
        args.ip_config is not None
    ), "A user has to specify an IP configuration file with --ip-config."

    if args.workspace is None:
        # Get PWD
        args.workspace = os.getcwd()
    else:
        args.workspace = os.path.abspath(args.workspace)

    if args.num_omp_threads is None:
        # Here we assume all machines have the same number of CPU cores as the machine
        # where the launch script runs.
        args.num_omp_threads = max(
            multiprocessing.cpu_count() // 2 // args.num_trainers, 1
        )
        if args.verbose:
            print(f"The number of OMP threads per trainer is set to {args.num_omp_threads}")
    else:
        assert args.num_omp_threads > 0, \
            "The number of OMP threads per trainer should be larger than 0"

def main():
    """Main func"""
    parser = get_argument_parser()
    # Positional arguments.
    parser.add_argument(
        "exec_script",
        type=str,
        help="Full path to the (single GPU) training program/script to be launched in parallel, "
        "followed by all the arguments for the training script.",
    )

    # Rest from the training program.
    parser.add_argument("exec_script_args", nargs=REMAINDER)
    args = parser.parse_args()
    check_input_arguments(args)

    exec_script_args = [args.exec_script] + args.exec_script_args
    submit_jobs(args, exec_script_args)

if __name__ == "__main__":
    FMT = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=FMT, level=logging.INFO)
    main()
