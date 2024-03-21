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
from graphstorm.model.utils import save_sparse_embeds
from graphstorm.model.utils import load_sparse_embeds
from graphstorm import get_node_feat_size

from data_utils import generate_dummy_dist_graph, generate_dummy_hetero_graph


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
    num_server,
    model_path,
):
    os.environ["DGL_GROUP_ID"] = str(0)
    dgl.distributed.initialize(ip_config)
    dist_graph = DistGraph("test_wholegraph_sparseemb", part_config=part_config)

    _initialize(rank, world_size, use_wholegraph=True)
    feat_size = {"n0":0, "n1":0}
    embed_layer = GSNodeEncoderInputLayer(
        dist_graph, feat_size, 32, use_wholegraph_sparse_emb=True
    )
    for ntype in embed_layer.sparse_embeds.keys():
        embed_layer.sparse_embeds[ntype].attach_wg_optimizer(create_wholememory_optimizer("adam", {}))


    def get_wholegraph_sparse_emb(sparse_emb):
        (local_tensor, _) = sparse_emb.get_local_tensor()
        return local_tensor

    saved_embs = \
        {ntype: get_wholegraph_sparse_emb(sparse_emb) \
            for ntype, sparse_emb in embed_layer.sparse_embeds.items()}
    save_sparse_embeds(model_path, embed_layer)
    load_sparse_embeds(model_path, embed_layer)
    load_sparse_embs = \
        {ntype: get_wholegraph_sparse_emb(sparse_emb) \
            for ntype, sparse_emb in embed_layer.sparse_embeds.items()}

    for ntype in embed_layer.sparse_embeds.keys():
        assert_equal(saved_embs[ntype].numpy(), load_sparse_embs[ntype].numpy())
    dgl.distributed.exit_client()
    _finalize()

@pytest.mark.parametrize("world_size", [1, 3, 4])
def test_wg_sparse_embed_save_load(world_size):
    """ Test sparse embedding saving logic using wholegraph. (graphstorm.model.utils.save_sparse_embeds)

        It will mimic the logic when multiple trainers are saving the embedding.
        And then check the value of the saved embedding.
    """
    # initialize the torch and wholegraph distributed environment
    pytest.importorskip("pylibwholegraph.torch")
    if world_size > th.cuda.device_count():
        pytest.skip("Skip test_wg_sparse_embed_save_load due to insufficient GPU devices.")
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
                        num_server,
                        model_path,
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

def _standalone_initialize(use_wholegraph=True):
    from dgl.distributed import role
    reset_envs()
    role.init_role("default")
    os.environ["DGL_DIST_MODE"] = "standalone"

    backend = "nccl"
    assert th.cuda.is_available(), "NCCL backend requires CUDA device(s) to be available."
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    th.cuda.set_device(int(os.environ['LOCAL_RANK']))
    th.distributed.init_process_group(backend=backend, rank=0, world_size=1)
    if use_wholegraph:
        init_wholegraph()

# In this case, we use node feature on one node type and
# use sparse embedding on the other node type.
# Refer to: unit-tests/test_embed.py:test_input_layer3
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_wg_input_layer3(dev):
    # initialize the torch and wholegraph distributed environment
    pytest.importorskip("pylibwholegraph.torch")
    _standalone_initialize(use_wholegraph=True)
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, {'n0' : ['feat']})
    layer = GSNodeEncoderInputLayer(g, feat_size, 2, use_wholegraph_sparse_emb=True)
    assert len(layer.input_projs) == 1
    assert list(layer.input_projs.keys())[0] == 'n0'
    assert len(layer.sparse_embeds) == 1
    layer = layer.to(dev)

    node_feat = {}
    node_embs = {}
    input_nodes = {}
    for ntype in g.ntypes:
        input_nodes[ntype] = np.arange(10)
    nn.init.eye_(layer.input_projs['n0'])
    nn.init.eye_(layer.proj_matrix['n1'])
    node_feat['n0'] = g.nodes['n0'].data['feat'][input_nodes['n0']].to(dev)

    layer.sparse_embeds['n1'].attach_wg_optimizer(None)
    node_embs['n1'] = layer.sparse_embeds['n1'].module(th.from_numpy(input_nodes['n1']).cuda())
    node_embs['n1'] = node_embs['n1'].to(dev)

    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    # check emb device
    for _, emb in embed.items():
        assert emb.get_device() == (-1 if dev == 'cpu' else 0)
    assert_almost_equal(embed['n0'].detach().cpu().numpy(),
                        node_feat['n0'].detach().cpu().numpy())
    assert_almost_equal(embed['n1'].detach().cpu().numpy(),
                        node_embs['n1'].detach().cpu().numpy())

    # test the case that one node type has no input nodes.
    input_nodes['n0'] = np.arange(10)

    # TODO(chang-l): Somehow, WholeGraph does not support empty indices created from numpy then converted to torch, i.e.,
    # empty_nodes = th.from_numpy(np.zeros((0,), dtype=int)) does not work (segfault in wholegraph.gather).
    # Need to submit an issue to WholeGraph team
    input_nodes['n1'] = th.tensor([],dtype=th.int64) #np.zeros((0,), dtype=int) should work but not!!

    nn.init.eye_(layer.input_projs['n0'])
    node_feat['n0'] = g.nodes['n0'].data['feat'][input_nodes['n0']].to(dev)
    node_embs['n1'] = layer.sparse_embeds['n1'].module(input_nodes['n1'].cuda())

    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    # check emb device
    for _, emb in embed.items():
        assert emb.get_device() == (-1 if dev == 'cpu' else 0)
    assert_almost_equal(embed['n0'].detach().cpu().numpy(),
                        node_feat['n0'].detach().cpu().numpy())
    assert_almost_equal(embed['n1'].detach().cpu().numpy(),
                        node_embs['n1'].detach().cpu().numpy())

    _finalize()

# In this case, we use both node features and sparse embeddings.
# Refer to: unit-tests/test_embed.py:test_input_layer2
def test_wg_input_layer2():
    # initialize the torch and wholegraph distributed environment
    pytest.importorskip("pylibwholegraph.torch")
    _standalone_initialize(use_wholegraph=True)
    th.backends.cuda.matmul.allow_tf32 = False
    th.backends.cudnn.allow_tf32 = False
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, 'feat')
    layer = GSNodeEncoderInputLayer(
        g, feat_size, 2, use_node_embeddings=True, use_wholegraph_sparse_emb=True
    )
    assert set(layer.input_projs.keys()) == set(g.ntypes)
    assert set(layer.sparse_embeds.keys()) == set(g.ntypes)
    assert set(layer.proj_matrix.keys()) == set(g.ntypes)
    node_feat = {}
    node_embs = {}
    input_nodes = {}
    for ntype in g.ntypes:
        # We make the projection matrix a diagonal matrix so that
        # the input and output matrices are identical.
        nn.init.eye_(layer.input_projs[ntype])
        assert layer.proj_matrix[ntype].shape == (4, 2)
        layer.sparse_embeds[ntype].attach_wg_optimizer(None)

        # We make the projection matrix that can simply add the node features
        # and the node sparse embeddings after projection.
        with th.no_grad():
            layer.proj_matrix[ntype][:2,:] = layer.input_projs[ntype]
            layer.proj_matrix[ntype][2:,:] = layer.input_projs[ntype]
        input_nodes[ntype] = np.arange(10)
        node_feat[ntype] = g.nodes[ntype].data['feat'][input_nodes[ntype]]
        node_embs[ntype] = layer.sparse_embeds[ntype].module(th.from_numpy(input_nodes[ntype]).cuda())
    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    assert len(embed) == len(node_feat)
    for ntype in embed:
        true_val = node_feat[ntype].detach().numpy() + node_embs[ntype].detach().cpu().numpy()
        assert_almost_equal(embed[ntype].detach().cpu().numpy(), true_val)
    _finalize()


if __name__ == '__main__':
    test_wg_sparse_embed_save_load(3)
    test_wg_sparse_embed_save_load(4)

    test_wg_input_layer2()
    test_wg_input_layer3('cpu')
    test_wg_input_layer3('cuda:0')