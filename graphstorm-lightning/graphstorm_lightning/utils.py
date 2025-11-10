import atexit
import os
import signal
import socket
import tempfile
from multiprocessing import Process
from pathlib import Path
from typing import IO, Any, Dict, Optional, List

import dgl
import graphstorm as gs
import pyarrow as pa
import pytorch_lightning as pl
import torch
import yaml


def get_config(trainer: pl.Trainer, cfg: Dict[str, Any]) -> gs.config.GSConfig:
    with tempfile.NamedTemporaryFile(prefix="gs_", mode="w", suffix=".yaml") as yaml_file:
        yaml.safe_dump(cfg, yaml_file)
        gs_config_args = [
            "--cf",
            yaml_file.name,
            "--local-rank",
            str(trainer.local_rank),
        ]
        gs_parser = gs.config.get_argument_parser()
        gs_args = gs_parser.parse_args(gs_config_args)
        config = gs.config.GSConfig(gs_args)
    return config


def load_data(trainer: pl.Trainer, config: Dict[str, Any], from_: str):
    to = get_part_config(config)
    fs, path = pa.fs.FileSystem.from_uri(from_)  # type: ignore[call-arg,arg-type]
    num_devices = trainer.num_devices
    rank = trainer.global_rank
    local_ranks = set(range(rank, rank + num_devices))
    dir = fs.get_file_info(path)
    if dir.type != pa.fs.FileType.Directory:
        raise ValueError("Directory path must be provided")
    local_root = Path(to)
    local_root = local_root.parent if local_root.name else local_root
    local_root.mkdir(exist_ok=True)
    selector = pa.fs.FileSelector(path, recursive=False)
    for file in fs.get_file_info(selector):
        file_path = Path(file.path).name
        if file_path.startswith("part"):
            rank = int(file_path.replace("part", ""))
            if rank not in local_ranks:
                continue
        local_file = local_root / file_path
        if file.type == pa.fs.FileType.Directory:
            local_file.mkdir(exist_ok=True)
        pa.fs.copy_files(source=file.path, destination=local_file.as_uri(), source_filesystem=fs)


def get_ip_tensor() -> torch.ByteTensor:
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    return torch.ByteTensor(list(ip.encode("ascii")))


def get_part_config(config: Dict[str, Any]) -> str:
    return config.get("gsf", {}).get("basic", {}).get("part_config")


def start_server(role: str, server_id: int, num_server: int, num_client: int, conf_path: str, ip_config: str) -> None:
    os.environ["DGL_ROLE"] = role
    os.environ["DGL_DIST_MODE"] = "distributed"
    os.environ["DGL_SERVER_ID"] = str(server_id)
    os.environ["DGL_NUM_SERVER"] = str(num_server)
    os.environ["DGL_NUM_CLIENT"] = str(num_client)
    os.environ["DGL_CONF_PATH"] = conf_path
    os.environ["DGL_IP_CONFIG"] = ip_config
    dgl.distributed.initialize(ip_config, net_type="socket")


def ip_addresses(trainer: pl.Trainer) -> Optional[List[str]]:
    if not torch.distributed.is_initialized():
        return None
    gloo = torch.distributed.new_group(backend="gloo")
    ip_tensor = get_ip_tensor()
    max_length = max(15, len(ip_tensor))  # assuming IPv4 and fixed xxx.xxx.xxx.xxx format
    ip_tensor_padded = torch.cat((ip_tensor, torch.zeros(max_length - len(ip_tensor), dtype=torch.uint8)))
    gathered_ip_tensors = [torch.zeros(max_length, dtype=torch.uint8) for _ in range(trainer.world_size)]
    torch.distributed.all_gather(gathered_ip_tensors, ip_tensor_padded, group=gloo)
    gathered_ips = sorted(
        set(tensor.numpy().tobytes().decode("ascii").rstrip("\x00") for tensor in gathered_ip_tensors)
    )
    torch.distributed.destroy_process_group(gloo)
    return gathered_ips


def prepare_data(trainer: pl.Trainer, config: Dict[str, Any], graph_data_uri: Optional[str]) -> None:
    part_config = get_part_config(config)
    if part_config and graph_data_uri:
        load_data(from_=graph_data_uri, to=part_config, num_devices=trainer.num_devices, rank=trainer.global_rank)


def initialize_dgl(trainer: pl.Trainer, config: Dict[str, Any]) -> Optional[IO[Any]]:
    gathered_ips = ip_addresses(trainer)
    if not gathered_ips:
        dgl.distributed.initialize(None, net_type="socket")
        return None
    ip_config = tempfile.NamedTemporaryFile(prefix="ip_config", mode="w", suffix=".txt")
    ip_config.write("\n".join(gathered_ips))
    ip_config.flush()

    id = trainer.global_rank
    num_servers = trainer.num_nodes
    num_clients = trainer.world_size
    part_config = get_part_config(config)
    if trainer.local_rank == 0:
        dgl_process = Process(
            target=start_server, args=("server", id, num_servers, num_clients, part_config, ip_config.name)
        )
        dgl_process.start()
    start_server("client", id, num_servers, num_clients, part_config, ip_config.name)

    def teardown(*args: Any) -> None:
        dgl_process.terminate()

    signal.signal(signal.SIGTERM, teardown)  # terminate signal
    signal.signal(signal.SIGINT, teardown)  # keyboard interrupt
    atexit.register(teardown)
    return ip_config
