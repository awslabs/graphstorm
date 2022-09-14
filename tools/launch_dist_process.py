"""Launching tool for distributed data processing"""
import os
import stat
import sys
import subprocess
import argparse
import signal
import logging
import time
import json
import multiprocessing
import re
from functools import partial
from threading import Thread
from typing import Optional

def cleanup_proc(get_all_remote_pids, conn):
    '''This process tries to clean up the remote training tasks.
    '''
    print('cleanupu process runs')
    # This process should not handle SIGINT.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    data = conn.recv()
    # If the launch process exits normally, this process doesn't need to do anything.
    if data == 'exit':
        sys.exit(0)
    else:
        remote_pids = get_all_remote_pids()
        # Otherwise, we need to ssh to each machine and kill the training jobs.
        for (ip, port), pids in remote_pids.items():
            kill_process(ip, port, pids)
    print('cleanup process exits')

def kill_process(ip, port, pids):
    '''ssh to a remote machine and kill the specified processes.
    '''
    curr_pid = os.getpid()
    killed_pids = []
    # If we kill child processes first, the parent process may create more again. This happens
    # to Python's process pool. After sorting, we always kill parent processes first.
    pids.sort()
    for pid in pids:
        assert curr_pid != pid
        print('kill process {} on {}:{}'.format(pid, ip, port), flush=True)
        kill_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'kill {}\''.format(pid)
        subprocess.run(kill_cmd, shell=True)
        killed_pids.append(pid)
    # It's possible that some of the processes are not killed. Let's try again.
    for i in range(3):
        killed_pids = get_killed_pids(ip, port, killed_pids)
        if len(killed_pids) == 0:
            break
        else:
            killed_pids.sort()
            for pid in killed_pids:
                print('kill process {} on {}:{}'.format(pid, ip, port), flush=True)
                kill_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'kill -9 {}\''.format(pid)
                subprocess.run(kill_cmd, shell=True)

def get_killed_pids(ip, port, killed_pids):
    '''Get the process IDs that we want to kill but are still alive.
    '''
    killed_pids = [str(pid) for pid in killed_pids]
    killed_pids = ','.join(killed_pids)
    ps_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'ps -p {} -h\''.format(killed_pids)
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids = []
    for p in res.stdout.decode('utf-8').split('\n'):
        l = p.split()
        if len(l) > 0:
            pids.append(int(l[0]))
    return pids

def execute_remote(
    cmd: str,
    ip: str,
    port: int,
    username: Optional[str] = ""
) -> Thread:
    """Execute command line on remote machine via ssh.
    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        ip: The ip-address of the host to run the command on.
        port: Port number that the host is listening on.
        thread_list:
        username: Optional. If given, this will specify a username to use when issuing commands over SSH.
            Useful when your infra requires you to explicitly specify a username to avoid permission issues.
    Returns:
        thread: The Thread whose run() is to run the `cmd` on the remote host. Returns when the cmd completes
            on the remote host.
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

    print(ssh_cmd)

    # thread func to run the job
    def run(ssh_cmd):
        subprocess.check_call(ssh_cmd, shell=True)

    thread = Thread(target=run, args=(ssh_cmd,))
    thread.setDaemon(True)
    thread.start()
    return thread

def get_remote_pids(ip, port, cmd_regex):
    """Get the process IDs that run the command in the remote machine.
    """
    pids = []
    curr_pid = os.getpid()
    # Here we want to get the python processes. We may get some ssh processes, so we should filter them out.
    ps_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'ps -aux | grep python | grep -v StrictHostKeyChecking\''
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    for p in res.stdout.decode('utf-8').split('\n'):
        l = p.split()
        if len(l) < 2:
            continue
        # We only get the processes that run the specified command.
        res = re.search(cmd_regex, p)
        if res is not None and int(l[1]) != curr_pid:
            pids.append(l[1])

    pid_str = ','.join([str(pid) for pid in pids])
    ps_cmd = 'ssh -o StrictHostKeyChecking=no -p ' + str(port) + ' ' + ip + ' \'pgrep -P {}\''.format(pid_str)
    res = subprocess.run(ps_cmd, shell=True, stdout=subprocess.PIPE)
    pids1 = res.stdout.decode('utf-8').split('\n')
    all_pids = []
    for pid in set(pids + pids1):
        if pid == '' or int(pid) == curr_pid:
            continue
        all_pids.append(int(pid))
    all_pids.sort()
    return all_pids

def get_all_remote_pids(hosts, ssh_port, udf_command):
    '''Get all remote processes.
    '''
    remote_pids = {}
    for node_id, ip in enumerate(hosts):
        # When creating training processes in remote machines, we may insert some arguments
        # in the commands. We need to use regular expressions to match the modified command.
        cmds = udf_command.split()
        new_udf_command = ' .*'.join(cmds)
        pids = get_remote_pids(ip, ssh_port, new_udf_command)
        remote_pids[(ip, ssh_port)] = pids
    return remote_pids

def construct_torch_dist_launcher_cmd(
    num_workers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
) -> str:
    """Constructs the torch distributed launcher command.
    Helper function.
    Args:
        num_workers:
        num_nodes:
        node_rank:
        master_addr:
    Returns:
        cmd_str.
    """
    torch_cmd_template = "-m torch.distributed.launch " \
                         "--nproc_per_node={nproc_per_node} " \
                         "--nnodes={nnodes} " \
                         "--node_rank={node_rank} " \
                         "--master_addr={master_addr}"
    return torch_cmd_template.format(
        nproc_per_node=num_workers,
        nnodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr
    )


def wrap_udf_torch_dist_launcher(
    udf_command: str,
    num_workers: int,
    num_nodes: int,
    node_rank: int,
    master_addr: str,
) -> str:
    """Wraps the user-defined function (udf_command) with the torch.distributed.launch module.
     Example: if udf_command is "python3 preprocess_dist_graph.py arg1 arg2", then new_df_command becomes:
         "python3 -m torch.distributed.launch <TORCH DIST ARGS> preprocess_dist_graph.py arg1 arg2

    Args:
        udf_command:
        num_workers:
        num_nodes:
        node_rank:
        master_addr:
    Returns:
    """
    torch_dist_cmd = construct_torch_dist_launcher_cmd(
        num_workers=num_workers,
        num_nodes=num_nodes,
        node_rank=node_rank,
        master_addr=master_addr
    )

    # Auto-detect the python binary that kicks off the distributed trainer code.
    # Note: This allowlist order matters, this will match with the FIRST matching entry. Thus, please add names to this
    #       from most-specific to least-specific order eg:
    #           (python3.7, python3.8) -> (python3)
    # The allowed python versions are from this: https://www.dgl.ai/pages/start.html
    python_bin_allowlist = (
        "python3.6", "python3.7", "python3.8", "python3.9", "python3")

    # If none of the candidate python bins match, then we go with the default `python`
    python_bin = "python"
    for candidate_python_bin in python_bin_allowlist:
        if candidate_python_bin in udf_command:
            python_bin = candidate_python_bin
            break

    # transforms the udf_command from:
    #     python path/to/dist_trainer.py arg0 arg1
    # to:
    #     python -m torch.distributed.launch [DIST TORCH ARGS] path/to/dist_trainer.py arg0 arg1
    # Note: if there are multiple python commands in `udf_command`, this may do the Wrong Thing, eg launch each
    #       python command within the torch distributed launcher.
    new_udf_command = udf_command.replace(python_bin, f"{python_bin} {torch_dist_cmd}")

    return new_udf_command

def submit_jobs(args, udf_command):
    """Submit distributed jobs via ssh"""
    hosts = []
    thread_list = []
    # Get the IP addresses of the cluster.
    ip_config = os.path.join(args.workspace, args.ip_config)

    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            assert len(result) == 1, "Format error of ip_config."
            hosts.append(result[0])

    print(hosts)
    for node_id, host in enumerate(hosts):
        cmd = wrap_udf_torch_dist_launcher(
            udf_command=udf_command,
            num_workers=args.num_workers,
            num_nodes=len(hosts),
            node_rank=node_id,
            master_addr=hosts[0],
        )

        pythonpath=os.environ.get("PYTHONPATH", "")
        cmd = f"cd {str(args.workspace)}; (export PYTHONPATH={pythonpath}; {cmd})"
        thread_list.append(execute_remote(cmd, host, args.ssh_port, username=args.ssh_username))

    # Start a cleanup process dedicated for cleaning up remote training jobs.
    conn1,conn2 = multiprocessing.Pipe()
    func = partial(get_all_remote_pids, hosts, args.ssh_port, udf_command)
    process = multiprocessing.Process(target=cleanup_proc, args=(func, conn1))
    process.start()

    def signal_handler(signal, frame):
        logging.info('Stop launcher')
        # We need to tell the cleanup process to kill remote training jobs.
        conn2.send('cleanup')
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    for thread in thread_list:
        thread.join()
    # The training processes complete. We should tell the cleanup process to exit.
    conn2.send('exit')
    process.join()

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')
    parser.add_argument('--ssh_port', type=int, default=22, help='SSH Port.')
    parser.add_argument(
        "--ssh_username", default="",
        help="Optional. When issuing commands (via ssh) to cluster, use the provided username in the ssh cmd. "
             "Example: If you provide --ssh_username=bob, then the ssh command will be like: 'ssh bob@1.2.3.4 CMD' "
             "instead of 'ssh 1.2.3.4 CMD'"
    )
    parser.add_argument('--workspace', type=str, default="./",
                        help='Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        the contents of current directory will be rsyncd')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='The number of worker processes per machine')
    parser.add_argument('--ip_config', type=str,
                        help='The file (in workspace) of IP configuration for instances')
    args, udf_command = parser.parse_known_args()

    assert len(udf_command) == 1, 'Please provide user command line.'
    assert args.num_workers is not None and args.num_workers > 0, \
            '--num_workers must be a positive number.'
    assert args.ip_config is not None, \
            'A user has to specify an IP configuration file with --ip_config.'
    udf_command = str(udf_command[0])
    if 'python' not in udf_command:
        raise RuntimeError("Data processing launching script can only support Python executable file.")

    submit_jobs(args, udf_command)

if __name__ == '__main__':
    main()