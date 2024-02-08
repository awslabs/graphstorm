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

    Unitest code for model save and restore
"""
import os
import dgl
import pytest
import tempfile

import numpy as np
import torch.multiprocessing as mp
import time

import torch as th
import torch.nn.functional as F
from torch import nn
from numpy.testing import assert_equal, assert_almost_equal

from unittest.mock import patch

from dgl.distributed import (
    DistGraph,
    DistGraphServer,
    partition_graph,
)

from graphstorm.wholegraph import init_wholegraph, is_wholegraph_init, create_wholememory_optimizer

from graphstorm.model import GSNodeEncoderInputLayer
from graphstorm import get_node_feat_size

from data_utils import generate_dummy_hetero_graph


def generate_ip_config(file_name, num_machines, num_servers):
    import socket
    import random
    """Get local IP and available ports, writes to file."""
    # get available IP in localhost
    ip = "127.0.0.1"
    # scan available PORT
    ports = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start = random.randint(10000, 30000)
    for port in range(start, 65535):
        try:
            sock.connect((ip, port))
            ports = []
        except:
            ports.append(port)
            if len(ports) == num_machines * num_servers:
                break
    sock.close()
    if len(ports) < num_machines * num_servers:
        raise RuntimeError(
            "Failed to get available IP/PORT with required numbers."
        )
    with open(file_name, "w") as f:
        for i in range(num_machines):
            f.write("{} {}\n".format(ip, ports[i * num_servers]))

def reset_envs():
    """Reset common environment variable which are set in tests."""
    for key in [
        "DGL_ROLE",
        "DGL_NUM_SAMPLER",
        "DGL_NUM_SERVER",
        "DGL_DIST_MODE",
        "DGL_NUM_CLIENT",
        "DGL_DIST_MAX_TRY_TIMES",
        "DGL_DIST_DEBUG",
    ]:
        if key in os.environ:
            os.environ.pop(key)

def _initialize(proc_id, nprocs, use_wholegraph=True):
    backend = "nccl"
    assert th.cuda.is_available(), "NCCL backend requires CUDA device(s) to be available."

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = str(proc_id)
    os.environ["WORLD_SIZE"] = str(nprocs)
    os.environ["LOCAL_RANK"] = str(proc_id)
    os.environ["LOCAL_WORLD_SIZE"] = str(nprocs)

    th.cuda.set_device(proc_id) # necessary for this example
    th.distributed.init_process_group(backend=backend, world_size=nprocs, rank=proc_id)
    init_wholegraph()

def _finalize():
    if is_wholegraph_init():
        import pylibwholegraph.torch as wgth
        wgth.finalize()
        # below patch fix (manually reset wg comm) will not be needed
        # once PR: https://github.com/rapidsai/wholegraph/pull/111 is merged.
        import pylibwholegraph.torch.comm as wgth_comm
        wgth_comm.global_communicators = {}
        wgth_comm.local_node_communicator = None
        wgth_comm.local_device_communicator = None
    th.distributed.destroy_process_group() if th.distributed.is_initialized() else None

def _start_server(
    rank,
    ip_config,
    part_config,
    disable_shared_mem,
    num_clients,
):
    g = DistGraphServer(
        rank,
        ip_config,
        1,
        num_clients,
        part_config,
        disable_shared_mem=disable_shared_mem,
        graph_format=["csc", "coo"],
    )
    g.start()

def _start_trainer(
    rank,
    world_size,
    ip_config,
    part_config,
    embed_dim,
    num_indices,
):
    os.environ["DGL_GROUP_ID"] = str(0)
    dgl.distributed.initialize(ip_config)
    dist_graph = DistGraph("test_wholegraph_sparseemb", part_config=part_config)

    _initialize(rank, world_size, use_wholegraph=True)
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False
    lr = 0.01

    dev = th.device('cuda:{}'.format(rank))
    feat_size = get_node_feat_size(dist_graph, {'n0': ['feat']})
    layer_wg = GSNodeEncoderInputLayer(
        dist_graph, feat_size, embed_dim, use_wholegraph_sparse_emb=True
    ).to(dev)

    layer = GSNodeEncoderInputLayer(
        dist_graph, feat_size, embed_dim, use_wholegraph_sparse_emb=False
    ).to(dev)
    # create the optimizer for wholegraph sparse embedding
    emb_optimizer_wg = create_wholememory_optimizer("adam", {})
    emb_optimizer_wg.lr = lr
    layer_wg.sparse_embeds['n1'].attach_wg_optimizer(emb_optimizer_wg)
    # create the optimizer for DistDGL sparse embedding
    emb_optimizer = dgl.distributed.optim.SparseAdam([layer.sparse_embeds['n1']], lr=lr)

    # create the initialized params
    init_sparse_emb = th.zeros((dist_graph.num_nodes('n1'), embed_dim), dtype=th.float32).cuda()
    th.manual_seed(0)
    nn.init.xavier_uniform_(init_sparse_emb)
    # inital the sparse embedding for DistDGL (DistEmbedding)
    layer.sparse_embeds['n1'].weight[0: dist_graph.num_nodes("n1")] = init_sparse_emb.cpu()
    # inital the sparse embedding for WholeGraph (only the local part)
    subpart_size = -(dist_graph.num_nodes('n1')// -world_size)
    local_tensor, _ = layer_wg.sparse_embeds['n1'].get_local_tensor()
    for part in range(world_size):
        l = part*subpart_size
        u = (part+1)*subpart_size if part != (world_size - 1) else dist_graph.num_nodes('n1')
        if part == rank:
            local_tensor.copy_(init_sparse_emb[l:u])
    th.distributed.barrier()

    # create the input nodes and labels for training
    node_feat = {}
    input_nodes = {}
    labels = {}
    for ntype in dist_graph.ntypes:
        num_nodes = dist_graph.num_nodes(ntype)
        idx_g = th.randperm(num_nodes, dtype=th.int64)[:num_indices]
        input_nodes[ntype] = th.tensor_split(idx_g, world_size)[rank].cuda()
        labels[ntype] = th.ones((len(input_nodes[ntype]),)).long()
    nn.init.eye_(layer_wg.input_projs['n0'])
    nn.init.eye_(layer_wg.proj_matrix['n1'])
    node_feat['n0'] = dist_graph.nodes['n0'].data['feat'][input_nodes['n0']].cuda()
    nn.init.eye_(layer.input_projs['n0'])
    nn.init.eye_(layer.proj_matrix['n1'])

    # 5 steps of wholegraph sparse embedding
    for i in range(5):
        embed_wg = layer_wg(node_feat, input_nodes)
        wg_loss = nn.functional.cross_entropy(embed_wg['n1'].cpu(), labels['n1'])
        wg_loss.backward()
        emb_optimizer_wg.step(lr)

    # 5 steps of DistDGL sparse embedding
    for i in range(5):
        emb_optimizer.zero_grad()
        embed = layer(node_feat, input_nodes)
        loss = nn.functional.cross_entropy(embed['n1'].cpu(), labels['n1'])
        loss.backward()
        emb_optimizer.step()

    assert_almost_equal(embed_wg['n1'].detach().cpu().numpy(),
                        embed['n1'].detach().cpu().numpy(), decimal=4)
    dgl.distributed.exit_client()
    _finalize()

@pytest.mark.parametrize("world_size", [1, 4])
@pytest.mark.parametrize("embed_dim", [4, 15, 32])
@pytest.mark.parametrize("num_indices", [4, 32])
def test_wg_sparse_opt(embed_dim, num_indices, world_size):
    """ Test sparse embedding saving logic using wholegraph. (graphstorm.model.utils.save_sparse_embeds)

        It will mimic the logic when multiple trainers are saving the embedding.
        And then check the value of the saved embedding.
    """
    # initialize the torch and wholegraph distributed environment
    pytest.importorskip("pylibwholegraph.torch")
    if world_size > th.cuda.device_count():
        pytest.skip("Skip test_wg_sparse_embed_save_load due to insufficient GPU devices.")
    import pylibwholegraph
    if pylibwholegraph.__version__ < "24.02":
        pytest.skip("Skipping this accuracy test due to a dependency on a fix from rapidsai/wholegraph:PR#108, which is targeted for release in version 24.02.")
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)
    num_groups = 1
    num_server = 1
    with tempfile.TemporaryDirectory() as test_dir:
        ip_config = "ip_config.txt"
        model_path = os.path.join(test_dir, "model")
        os.mkdir(model_path)

        generate_ip_config(ip_config, num_server, num_server)
        g = generate_dummy_hetero_graph()
        num_parts = num_server
        num_trainers = world_size

        partition_graph(
            g,
            "test_wholegraph_sparseemb",
            num_parts,
            test_dir,
            part_method="metis",
        )

        part_config = os.path.join(test_dir, "test_wholegraph_sparseemb.json")
        pserver_list = []
        ctx = mp.get_context("spawn")
        for i in range(num_server):
            p = ctx.Process(
                target=_start_server,
                args=(
                    i,
                    ip_config,
                    part_config,
                    num_server > 1,
                    num_trainers,
                ),
            )
            p.start()
            time.sleep(1)
            pserver_list.append(p)

        os.environ["DGL_DIST_MODE"] = "distributed"
        os.environ["DGL_NUM_SAMPLER"] = "0"
        ptrainer_list = []

        for trainer_id in range(num_trainers):
            for group_id in range(num_groups):
                p = ctx.Process(
                    target=_start_trainer,
                    args=(
                        trainer_id,
                        num_trainers,
                        ip_config,
                        part_config,
                        embed_dim,
                        num_indices,
                    ),
                )
                p.start()
                # avoid race condition when instantiating DistGraph
                time.sleep(1)
                ptrainer_list.append(p)

        for p in ptrainer_list:
            p.join()
            assert p.exitcode == 0
        for p in pserver_list:
            p.join()
            assert p.exitcode == 0


if __name__ == '__main__':
    test_wg_sparse_opt(4, 4, 1)
    test_wg_sparse_opt(15, 32, 4)
