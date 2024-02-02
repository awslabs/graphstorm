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
"""
import os
import json
import tempfile
import pytest
import multiprocessing as mp
import h5py

import torch as th
import numpy as np
import dgl
from numpy.testing import assert_equal, assert_almost_equal
from dgl.distributed import DistTensor
from graphstorm.model.utils import save_embeddings, LazyDistTensor, remove_saved_models, TopKList
from graphstorm.model.utils import get_data_range
from graphstorm.model.utils import _exchange_node_id_mapping, distribute_nid_map
from graphstorm.model.utils import shuffle_predict, NodeIDShuffler
from graphstorm.model.utils import pad_file_index
from graphstorm.model.utils import (save_node_prediction_results,
                                    save_edge_prediction_results,
                                    save_shuffled_node_embeddings,
                                    save_full_node_embeddings)
from graphstorm.model.utils import normalize_node_embs
from graphstorm.gconstruct.utils import save_maps
from graphstorm import get_node_feat_size
from graphstorm.model.gnn_encoder_base import prepare_for_wholegraph

from data_utils import generate_dummy_dist_graph
from graphstorm.eval.utils import gen_mrr_score
from graphstorm.utils import setup_device, get_graph_name

from graphstorm.gconstruct.file_io import stream_dist_tensors_to_hdf5

def gen_embedding_with_nid_mapping(num_embs):
    emb = th.rand((num_embs, 12))
    ori_nid_mapping = th.randperm(num_embs)
    _, nid_mapping = th.sort(ori_nid_mapping)
    emb = LazyDistTensor(emb, th.arange(num_embs))
    return emb, ori_nid_mapping, nid_mapping

def gen_predict_with_nid_mapping(num_embs):
    pred = th.rand((num_embs, 12)) * 10
    pred = pred.long()
    ori_nid_mapping = th.randperm(num_embs)
    _, nid_mapping = th.sort(ori_nid_mapping)
    return pred, ori_nid_mapping, nid_mapping

def helper_save_embedding(tmpdirname):
    random_emb = th.rand((103, 12))
    type0_random_emb = LazyDistTensor(random_emb, th.arange(103))
    random_emb = th.rand((205, 12))
    type1_random_emb = LazyDistTensor(random_emb, th.arange(205))

    emb = {
        "type0": type0_random_emb,
        "type1": type1_random_emb
    }
    save_embeddings(tmpdirname, emb, 0, 4)
    emb = {
        "type0": type0_random_emb,
        "type1": type1_random_emb
    }
    save_embeddings(tmpdirname, emb, 1, 4)
    emb = {
        "type0": type0_random_emb,
        "type1": type1_random_emb
    }
    save_embeddings(tmpdirname, emb, 2, 4)
    emb = {
        "type0": type0_random_emb,
        "type1": type1_random_emb
    }
    save_embeddings(tmpdirname, emb, 3, 4)

    return type0_random_emb, type1_random_emb

def test_get_data_range():
    # test get_data_range
    # num_embs < world_size, only latest rank will do the work
    start, end = get_data_range(0, 3, 2)
    assert start == 0
    assert end == 0

    start, end = get_data_range(1, 3, 2)
    assert start == 0
    assert end == 0

    start, end = get_data_range(2, 3, 2)
    assert start == 0
    assert end == 2

    # num_embs > world_size
    start, end = get_data_range(0, 2, 5)
    assert start == 0
    assert end == 2

    start, end = get_data_range(1, 2, 5)
    assert start == 2
    assert end == 5

    start, end = get_data_range(0, 2, 4)
    assert start == 0
    assert end == 2

    start, end = get_data_range(1, 2, 4)
    assert start == 2
    assert end == 4

def run_dist_exchange_node_id_mapping(worker_rank, world_size, backend,
    node_id_mapping, num_embs, target_nid_mapping):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    th.cuda.set_device(worker_rank)
    device = setup_device(worker_rank)

    nid_mapping = _exchange_node_id_mapping(worker_rank, world_size, device, node_id_mapping, num_embs)

    assert_equal(target_nid_mapping.numpy(), nid_mapping.cpu().numpy())

@pytest.mark.parametrize("num_embs", [100, 101])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_exchange_node_id_mapping(num_embs, backend):
    node_id_mapping = th.randperm(num_embs)
    start, end = get_data_range(0, 4, num_embs)
    target_nid_mapping_0 = node_id_mapping[start:end]
    start, end = get_data_range(1, 4, num_embs)
    target_nid_mapping_1 = node_id_mapping[start:end]
    start, end = get_data_range(2, 4, num_embs)
    target_nid_mapping_2 = node_id_mapping[start:end]
    start, end = get_data_range(3, 4, num_embs)
    target_nid_mapping_3 = node_id_mapping[start:end]
    ctx = mp.get_context('spawn')
    p0 = ctx.Process(target=run_dist_exchange_node_id_mapping,
                    args=(0, 4, backend, node_id_mapping.clone(), num_embs, target_nid_mapping_0))
    p1 = ctx.Process(target=run_dist_exchange_node_id_mapping,
                    args=(1, 4, backend, node_id_mapping.clone(), num_embs, target_nid_mapping_1))
    p2 = ctx.Process(target=run_dist_exchange_node_id_mapping,
                    args=(2, 4, backend, node_id_mapping.clone(), num_embs, target_nid_mapping_2))
    p3 = ctx.Process(target=run_dist_exchange_node_id_mapping,
                    args=(3, 4, backend, node_id_mapping.clone(), num_embs, target_nid_mapping_3))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p0.join()
    p1.join()
    p2.join()
    p3.join()
    assert p0.exitcode == 0
    assert p1.exitcode == 0
    assert p2.exitcode == 0
    assert p3.exitcode == 0

def run_distribute_nid_map(embeddings, worker_rank, world_size,
    node_id_mapping_file, backend, target_nid_mapping):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    device = setup_device(worker_rank)
    nid_mapping = distribute_nid_map(embeddings, worker_rank, world_size,
        node_id_mapping_file, device)

    if isinstance(embeddings, (dgl.distributed.DistTensor, LazyDistTensor)):
        assert_equal(target_nid_mapping[worker_rank].numpy(), nid_mapping.cpu().numpy())
    elif isinstance(embeddings, dict):
        for name in embeddings.keys():
            assert_equal(target_nid_mapping[name][worker_rank].numpy(), \
                nid_mapping[name].cpu().numpy())
    if worker_rank == 0:
        th.distributed.destroy_process_group()

def run_distributed_shuffle_nids(part_config, ntype0, ntype1, nids0, nids1, node_id_mapping_file,
                                 worker_rank, world_size, original_nids0, original_nids1):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    dgl.distributed.initialize('')
    g = dgl.distributed.DistGraph(graph_name='dummy', part_config=part_config)
    shuffler = NodeIDShuffler(g,
    node_id_mapping_file, [ntype0, ntype1])
    assert len(shuffler._id_mapping_info) == 2
    shuffled_nids = shuffler.shuffle_nids(ntype0, nids0)
    assert_equal(shuffled_nids.numpy(), original_nids0.numpy())
    shuffled_nids = shuffler.shuffle_nids(ntype1, nids1)
    assert_equal(shuffled_nids.numpy(), original_nids1.numpy())
    del shuffler

    shuffler = NodeIDShuffler(g,
    node_id_mapping_file)
    assert len(shuffler._id_mapping_info) == 2
    shuffled_nids = shuffler.shuffle_nids(ntype0, nids0)
    assert_equal(shuffled_nids.numpy(), original_nids0.numpy())
    shuffled_nids = shuffler.shuffle_nids(ntype1, nids1)
    assert_equal(shuffled_nids.numpy(), original_nids1.numpy())

    if worker_rank == 0:
        th.distributed.destroy_process_group()

def test_shuffle_nids():
    with tempfile.TemporaryDirectory() as tmpdirname:
        g, part_config = generate_dummy_dist_graph(tmpdirname, size="tiny")
        nid_map_dict_path = os.path.join(tmpdirname, "nid_map_dict.pt")

        target_ntype0 = g.ntypes[0]
        target_ntype1 = g.ntypes[1]
        ori_nid_maps = {target_ntype0: th.randperm(g.number_of_nodes(target_ntype0)),
                        target_ntype1: th.randperm(g.number_of_nodes(target_ntype1))}
        th.save(ori_nid_maps, nid_map_dict_path)

        test_nids0 = th.randint(g.number_of_nodes(target_ntype0),
                                (g.number_of_nodes(target_ntype0),))
        orig_nids0 = ori_nid_maps[target_ntype0][test_nids0]
        test_nids1 = th.arange(10)
        orig_nids1 = ori_nid_maps[target_ntype1][test_nids1]

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_distributed_shuffle_nids,
                        args=(part_config, target_ntype0, target_ntype1,
                              test_nids0, test_nids1, nid_map_dict_path,
                              0, 1, orig_nids0, orig_nids1))
        p0.start()
        p0.join()
        assert p0.exitcode == 0

def test_shuffle_nids_dist_part():
    with tempfile.TemporaryDirectory() as tmpdirname:
        g, part_config = generate_dummy_dist_graph(tmpdirname, size="tiny")
        nid_map_dict_path0 = os.path.join(tmpdirname, "part0")
        nid_map_dict_path1 = os.path.join(tmpdirname, "part1")
        # part0 should exists
        os.mkdir(nid_map_dict_path1)

        target_ntype0 = g.ntypes[0]
        target_ntype1 = g.ntypes[1]
        ori_nid_maps = {target_ntype0: th.randperm(g.number_of_nodes(target_ntype0)),
                        target_ntype1: th.randperm(g.number_of_nodes(target_ntype1))}
        mapping0 = {
            target_ntype0: ori_nid_maps[target_ntype0][0:g.number_of_nodes(target_ntype0)//2],
            target_ntype1: ori_nid_maps[target_ntype1][0:g.number_of_nodes(target_ntype0)//2],
        }
        mapping1 = {
            target_ntype0: ori_nid_maps[target_ntype0][g.number_of_nodes(target_ntype0)//2:],
            target_ntype1: ori_nid_maps[target_ntype1][g.number_of_nodes(target_ntype0)//2:],
        }
        dgl.data.utils.save_tensors(os.path.join(nid_map_dict_path0,
                                                 "orig_nids.dgl"),
                                    mapping0)
        dgl.data.utils.save_tensors(os.path.join(nid_map_dict_path1,
                                                 "orig_nids.dgl"),
                                    mapping1)

        test_nids0 = th.randint(g.number_of_nodes(target_ntype0),
                                (g.number_of_nodes(target_ntype0),))
        orig_nids0 = ori_nid_maps[target_ntype0][test_nids0]
        test_nids1 = th.arange(10)
        orig_nids1 = ori_nid_maps[target_ntype1][test_nids1]

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_distributed_shuffle_nids,
                        args=(part_config, target_ntype0, target_ntype1,
                              test_nids0, test_nids1, tmpdirname,
                              0, 1, orig_nids0, orig_nids1))
        p0.start()
        p0.join()
        assert p0.exitcode == 0


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
@pytest.mark.parametrize("map_pattern", ["gconstruct", "distdgl"])
def test_distribute_nid_map(backend, map_pattern):
    # need to force to reset the fork context
    # because dist tensor is the input for mulitiple processes
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname, size="tiny")
        dummy_dist_embeds = {}
        ori_nid_maps = {}
        target_nid_maps = {}
        for ntype in g.ntypes:
            dummy_dist_embeds[ntype] = DistTensor((g.number_of_nodes(ntype), 5),
                      dtype=th.float32, name=f'ntype-{ntype}',
                      part_policy=g.get_node_partition_policy(ntype))
            ori_nid_maps[ntype] = th.randperm(g.number_of_nodes(ntype))

            target_nid_maps[ntype] = []
            _, sorted_nid_map = th.sort(ori_nid_maps[ntype])
            for i in range(4):
                start, end = get_data_range(i, 4, g.number_of_nodes(ntype))
                target_nid_maps[ntype].append(sorted_nid_map[start:end].clone())

        if map_pattern == "gconstruct":
            nid_map_dict_path = os.path.join(tmpdirname, "nid_map_dict.pt")
            th.save(ori_nid_maps, nid_map_dict_path)
        else: # distdgl
            nid_map_dict_path = tmpdirname
            for i in range(4):
                local_map_dict_path = os.path.join(nid_map_dict_path, f"part{i}")
                if i > 0:
                    os.mkdir(local_map_dict_path)
                mapping = {
                    nt: nid_map[i:i+1 if i < 3 else g.number_of_nodes(ntype)] \
                        for nt, nid_map in ori_nid_maps.items()
                }
                dgl.data.utils.save_tensors(os.path.join(local_map_dict_path,
                                                 "orig_nids.dgl"),
                                            mapping)

        nid_map_tensor_path = os.path.join(tmpdirname, "nid_map_tensor.pt")
        dummy_ntype = g.ntypes[0]
        th.save(ori_nid_maps[dummy_ntype], nid_map_tensor_path)

        # when dummy_dist_embeds is a dict
        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds, 0, 4, nid_map_dict_path, backend, \
                            target_nid_maps))
        p1 = ctx.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds, 1, 4, nid_map_dict_path, backend, \
                            target_nid_maps))
        p2 = ctx.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds, 2, 4, nid_map_dict_path, backend, \
                            target_nid_maps))
        p3 = ctx.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds, 3, 4, nid_map_dict_path, backend, \
                            target_nid_maps))
        p0.start()
        p1.start()
        p2.start()
        p3.start()
        p0.join()
        p1.join()
        p2.join()
        p3.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0
        assert p2.exitcode == 0
        assert p3.exitcode == 0

        # when dummy_dist_embeds is a dist tensor
        ctx2 = mp.get_context('spawn')
        p4 = ctx2.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds[dummy_ntype], 0, 4, nid_map_tensor_path, \
                            backend, target_nid_maps[dummy_ntype]))
        p5 = ctx2.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds[dummy_ntype], 1, 4, nid_map_tensor_path, \
                            backend, target_nid_maps[dummy_ntype]))
        p6 = ctx2.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds[dummy_ntype], 2, 4, nid_map_tensor_path, \
                            backend, target_nid_maps[dummy_ntype]))
        p7 = ctx2.Process(target=run_distribute_nid_map,
                        args=(dummy_dist_embeds[dummy_ntype], 3, 4, nid_map_tensor_path, \
                            backend, target_nid_maps[dummy_ntype]))

        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p4.join()
        p5.join()
        p6.join()
        p7.join()
        assert p4.exitcode == 0
        assert p5.exitcode == 0
        assert p6.exitcode == 0
        assert p7.exitcode == 0

def run_dist_save_embeddings(model_path, emb, worker_rank,
    world_size, node_id_mapping_file, backend):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    th.cuda.set_device(worker_rank)
    device = setup_device(worker_rank)

    save_embeddings(model_path, emb, worker_rank, world_size, device, node_id_mapping_file)

    if worker_rank == 0:
        th.distributed.destroy_process_group()

def run_dist_shuffle_predict(pred, worker_rank,
    world_size, node_id_mapping_file, type, backend, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend=backend,
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    th.cuda.set_device(worker_rank)
    device = setup_device(worker_rank)

    pred = shuffle_predict(pred, node_id_mapping_file, type, worker_rank, world_size, device)
    conn.send(pred.detach().cpu().numpy())

    if worker_rank == 0:
        th.distributed.destroy_process_group()

# TODO: Only test gloo now
# Will add test for nccl once we enable nccl
@pytest.mark.parametrize("num_embs", [16, 17])
@pytest.mark.parametrize("backend", ["gloo"])
def test_shuffle_predict(num_embs, backend):
    import tempfile

    # node_mapping is tensor
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred, ori_nid_mapping, nid_mapping = gen_predict_with_nid_mapping(num_embs)
        save_maps(tmpdirname, "node_mapping", ori_nid_mapping)
        nid_mapping_file = os.path.join(tmpdirname, "node_mapping.pt")
        ctx = mp.get_context('spawn')
        conn1, conn2 = mp.Pipe()
        p0 = ctx.Process(target=run_dist_shuffle_predict,
                        args=(pred, 0, 2, nid_mapping_file, None, backend, conn2))
        conn3, conn4 = mp.Pipe()
        p1 = ctx.Process(target=run_dist_shuffle_predict,
                        args=(pred, 1, 2, nid_mapping_file, None, backend, conn4))

        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        shuffled_pred_1 = conn1.recv()
        shuffled_pred_2 = conn3.recv()
        conn1.close()
        conn2.close()
        conn3.close()
        conn4.close()

        shuffled_pred = np.concatenate([shuffled_pred_1, shuffled_pred_2])

        # Load saved embeddings
        assert_equal(pred[nid_mapping].numpy(), shuffled_pred)

     # node mapping is a dict
    with tempfile.TemporaryDirectory() as tmpdirname:
        pred, ori_nid_mapping, nid_mapping = gen_predict_with_nid_mapping(num_embs)
        nid_mapping = {"node": nid_mapping}
        ori_nid_mapping = {"node": ori_nid_mapping}
        save_maps(tmpdirname, "node_mapping", ori_nid_mapping)
        nid_mapping_file = os.path.join(tmpdirname, "node_mapping.pt")
        ctx = mp.get_context('spawn')
        conn1, conn2 = mp.Pipe()
        p0 = ctx.Process(target=run_dist_shuffle_predict,
                        args=(pred, 0, 2, nid_mapping_file, "node", backend, conn2))
        conn3, conn4 = mp.Pipe()
        p1 = ctx.Process(target=run_dist_shuffle_predict,
                        args=(pred, 1, 2, nid_mapping_file, "node", backend, conn4))

        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        shuffled_pred_1 = conn1.recv()
        shuffled_pred_2 = conn3.recv()
        conn1.close()
        conn2.close()
        conn3.close()
        conn4.close()

        shuffled_pred = np.concatenate([shuffled_pred_1, shuffled_pred_2])

        # Load saved embeddings
        assert_equal(pred[nid_mapping["node"]].numpy(), shuffled_pred)

def do_dist_shuffle_nids(part_config, node_id_mapping_file, ntypes, nids,
                         worker_rank, world_size, conn):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    dgl.distributed.initialize('')
    g = dgl.distributed.DistGraph(graph_name='dummy', part_config=part_config)
    shuffler = NodeIDShuffler(g, node_id_mapping_file, ntypes)
    shuffled_nids = []
    for ntype, nid in zip(ntypes, nids):
        shuffled_nid = shuffler.shuffle_nids(ntype, nid)
        shuffled_nids.append(shuffled_nid.detach().cpu().numpy())

    conn.send(shuffled_nids)

    if worker_rank == 0:
        th.distributed.destroy_process_group()

@pytest.mark.parametrize("map_pattern", ["gconstruct", "distdgl"])
def test_shuffle_emb_with_shuffle_nids(map_pattern):
    # multiple embedding
    with tempfile.TemporaryDirectory() as tmpdirname:
        g, part_config = generate_dummy_dist_graph(tmpdirname, size="tiny")
        embs = {}
        ori_nid_mappings = {}
        nid_mappings = {}
        num_n0 = g.num_nodes('n0')
        emb, ori_nid_mapping, nid_mapping = gen_embedding_with_nid_mapping(num_n0)
        embs['n0'] = emb
        ori_nid_mappings['n0'] = ori_nid_mapping
        nid_mappings['n0'] = nid_mapping
        num_n1 = g.num_nodes('n1')
        emb, ori_nid_mapping, nid_mapping = gen_embedding_with_nid_mapping(num_n1)
        embs['n1'] = emb
        ori_nid_mappings['n1'] = ori_nid_mapping
        nid_mappings['n1'] = nid_mapping
        ntypes = ['n0', 'n1']
        nids0 = [th.arange(num_n0), th.arange(num_n1)]

        if map_pattern == "gconstruct":
            save_maps(tmpdirname, "node_mapping", ori_nid_mappings)
            nid_mapping_file = os.path.join(tmpdirname, "node_mapping.pt")
        else:
            dist_nid_mapping_file = os.path.join(os.path.join(tmpdirname, "part0"),
                                                 "orig_nids.dgl")
            dgl.data.utils.save_tensors(dist_nid_mapping_file, ori_nid_mappings)
            nid_mapping_file = tmpdirname
        ctx = mp.get_context('spawn')
        conn1, conn2 = mp.Pipe()
        p0 = ctx.Process(target=do_dist_shuffle_nids,
                        args=(part_config, nid_mapping_file, ntypes, nids0, 0, 1, conn2))
        p0.start()
        p0.join()
        assert p0.exitcode == 0

        shuffled_nids = conn1.recv()
        conn1.close()
        conn2.close()

        shuffled_nids0 = shuffled_nids[0]
        shuffled_nids1 = shuffled_nids[1]
        assert len(shuffled_nids0) == len(embs['n0'])
        assert len(shuffled_nids1) == len(embs['n1'])

        # Load saved embeddings
        ground_truth0 = embs['n0'][nid_mappings['n0']].numpy()
        for i, nid in enumerate(shuffled_nids0):
            assert_equal(ground_truth0[nid], embs['n0'][i].numpy())
        ground_truth1 = embs['n1'][nid_mappings['n1']].numpy()
        for i, nid in enumerate(shuffled_nids1):
            assert_equal(ground_truth1[nid], embs['n1'][i].numpy())

@pytest.mark.parametrize("num_embs", [16, 17])
@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_save_embeddings_with_id_mapping(num_embs, backend):
    # single embedding
    with tempfile.TemporaryDirectory() as tmpdirname:
        emb, ori_nid_mapping, nid_mapping = gen_embedding_with_nid_mapping(num_embs)
        save_maps(tmpdirname, "node_mapping", ori_nid_mapping)
        nid_mapping_file = os.path.join(tmpdirname, "node_mapping.pt")
        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_save_embeddings,
                        args=(tmpdirname, emb, 0, 2, nid_mapping_file, backend))
        p1 = ctx.Process(target=run_dist_save_embeddings,
                        args=(tmpdirname, emb, 1, 2, nid_mapping_file, backend))

        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        # Load saved embeddings
        emb0 = th.load(os.path.join(os.path.join(tmpdirname, dgl.NTYPE),
                                    f'embed-{pad_file_index(0)}.pt'), weights_only=True)
        emb1 = th.load(os.path.join(os.path.join(tmpdirname, dgl.NTYPE),
                                    f'embed-{pad_file_index(1)}.pt'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(emb)
        assert_equal(emb[nid_mapping].numpy(), saved_emb.numpy())

    # multiple embedding
    with tempfile.TemporaryDirectory() as tmpdirname:
        embs = {}
        ori_nid_mappings = {}
        nid_mappings = {}
        emb, ori_nid_mapping, nid_mapping = gen_embedding_with_nid_mapping(num_embs)
        embs['n0'] = emb
        ori_nid_mappings['n0'] = ori_nid_mapping
        nid_mappings['n0'] = nid_mapping
        emb, ori_nid_mapping, nid_mapping = gen_embedding_with_nid_mapping(num_embs*2)
        embs['n1'] = emb
        ori_nid_mappings['n1'] = ori_nid_mapping
        nid_mappings['n1'] = nid_mapping
        emb, ori_nid_mapping, nid_mapping = gen_embedding_with_nid_mapping(num_embs*3)
        embs['n2'] = emb
        ori_nid_mappings['n2'] = ori_nid_mapping
        nid_mappings['n2'] = nid_mapping

        save_maps(tmpdirname, "node_mapping", ori_nid_mappings)
        nid_mapping_file = os.path.join(tmpdirname, "node_mapping.pt")
        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_save_embeddings,
                        args=(tmpdirname, embs, 0, 2, nid_mapping_file, backend))
        p1 = ctx.Process(target=run_dist_save_embeddings,
                        args=(tmpdirname, embs, 1, 2, nid_mapping_file, backend))

        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        # Load saved embeddings
        emb0 = th.load(os.path.join(os.path.join(tmpdirname, 'n0'),
                                    f'embed-{pad_file_index(0)}.pt'), weights_only=True)
        emb1 = th.load(os.path.join(os.path.join(tmpdirname, 'n0'),
                                    f'embed-{pad_file_index(1)}.pt'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(embs['n0'])
        assert_equal(embs['n0'][nid_mappings['n0']].numpy(), saved_emb.numpy())

        emb0 = th.load(os.path.join(os.path.join(tmpdirname, 'n1'),
                                    f'embed-{pad_file_index(0)}.pt'), weights_only=True)
        emb1 = th.load(os.path.join(os.path.join(tmpdirname, 'n1'),
                                    f'embed-{pad_file_index(1)}.pt'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(embs['n1'])
        assert_equal(embs['n1'][nid_mappings['n1']].numpy(), saved_emb.numpy())

        emb0 = th.load(os.path.join(os.path.join(tmpdirname, 'n2'),
                                    f'embed-{pad_file_index(0)}.pt'), weights_only=True)
        emb1 = th.load(os.path.join(os.path.join(tmpdirname, 'n2'),
                                    f'embed-{pad_file_index(1)}.pt'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(embs['n2'])
        assert_equal(embs['n2'][nid_mappings['n2']].numpy(), saved_emb.numpy())

def test_save_embeddings():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        type0_random_emb, type1_random_emb = helper_save_embedding(tmpdirname)

        # Only work with torch 1.13+
        feats_type0 = [th.load(os.path.join(os.path.join(tmpdirname, "type0"),
                                            f"embed-{pad_file_index(i)}.pt"),
                               weights_only=True) for i in range(4)]
        feats_type0 = th.cat(feats_type0, dim=0)
        # Only work with torch 1.13+
        feats_type1 = [th.load(os.path.join(os.path.join(tmpdirname, "type1"),
                                            f"embed-{pad_file_index(i)}.pt"),
                               weights_only=True) for i in range(4)]
        feats_type1 = th.cat(feats_type1, dim=0)

        assert np.all(type0_random_emb.dist_tensor.numpy() == feats_type0.numpy())
        assert np.all(type1_random_emb.dist_tensor.numpy() == feats_type1.numpy())

def test_remove_saved_models():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdirname:
        _, _ = helper_save_embedding(tmpdirname)

        assert os.path.exists(tmpdirname) == True

        remove_saved_models(tmpdirname)

        assert os.path.exists(tmpdirname) == False

def test_topklist():
    # 1. build test data_1
    val_scores = [1.2, 0.4, 5.5, 7.4, -1.7, 3.45]
    perf_list = []

    # 2. define ground truth
    topk = TopKList(3)
    insert_success_list = [True, True, True, True, False, True]
    return_val_list = [0, 1, 2, 1, 4, 0]

    # 3. test for each run and record results
    for epoch, val_score in enumerate(val_scores):

        rank = 1
        for val in perf_list:
            if val >= val_score:
                rank += 1
        perf_list.append(val_score)

        insert_success, return_val = topk.insert(rank, epoch)

        assert insert_success_list[epoch] == insert_success
        assert return_val_list[epoch] == return_val

    # 4. test top 0 case
    topk = TopKList(0)
    insert_success_list = [False, False, False, False, False, False]
    return_val_list = [0, 1, 2, 3, 4, 5]

    for epoch, val_score in enumerate(val_scores):

        rank = 1
        for val in perf_list:
            if val >= val_score:
                rank += 1
        perf_list.append(val_score)

        insert_success, return_val = topk.insert(rank, epoch)

        assert insert_success_list[epoch] == insert_success
        assert return_val_list[epoch] == return_val

def test_get_node_feat_size():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, 'feat')
    assert len(feat_size) == len(g.ntypes)
    for ntype in feat_size:
        assert ntype in g.ntypes
        assert feat_size[ntype] == g.nodes[ntype].data['feat'].shape[1]

    feat_size = get_node_feat_size(g, {'n0': ['feat'], 'n1': ['feat']})
    assert len(feat_size) == len(g.ntypes)
    for ntype in feat_size:
        assert ntype in g.ntypes
        assert feat_size[ntype] == g.nodes[ntype].data['feat'].shape[1]

    feat_size = get_node_feat_size(g, {'n0' : ['feat']})
    assert len(feat_size) == len(g.ntypes)
    assert feat_size['n0'] == g.nodes['n0'].data['feat'].shape[1]
    assert feat_size['n1'] == 0

    try:
        feat_size = get_node_feat_size(g, {'n0': ['feat'], 'n1': ['feat_not_exist']})
    except:
        feat_size = None
    assert feat_size is None

def test_gen_mrr_score():
    ranking = th.rand(500)
    logs = []
    for rank in ranking:
        logs.append(1.0 / rank)
    metrics = {"mrr": th.tensor(sum(log for log in logs) / len(logs))}

    metrics_opti = gen_mrr_score(ranking)

    assert th.isclose(metrics['mrr'], metrics_opti['mrr'])  # Default tolerance: 1e-08

def test_stream_dist_tensors_to_hdf5():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        # medium size has 1,000,000 nodes, which is enough (>chunk_size)
        g, _ = generate_dummy_dist_graph(tmpdirname, size="medium")

        dummy_dist_embeds = {}
        for ntype in g.ntypes:
            dummy_dist_embeds[ntype] = DistTensor((g.number_of_nodes(ntype), 5),
                      dtype=th.float32, name=f'ntype-{ntype}',
                      part_policy=g.get_node_partition_policy(ntype))

        # chunk size needs to be smaller than num of nodes
        chunk_size = g.number_of_nodes(g.ntypes[0]) // 4
        stream_dist_tensors_to_hdf5(dummy_dist_embeds, os.path.join(tmpdirname, "embed_dict.hdf5"), \
            chunk_size=chunk_size)

        read_f = h5py.File(os.path.join(tmpdirname, "embed_dict.hdf5"), "r")
        for ntype in g.ntypes:
            assert g.number_of_nodes(ntype) == len(read_f[ntype])
            assert_equal(dummy_dist_embeds[ntype][0:len(dummy_dist_embeds[ntype])].numpy(), \
                read_f[ntype][0:])

def test_pad_file_index():
    assert pad_file_index(1) == "00001"
    assert pad_file_index(111) == "00111"
    assert pad_file_index(111, 4) == "0111"
    fail = False
    try:
        pad_file_index(111, 0)
    except:
        fail = True
    assert fail

def run_dist_save_predict_results(func, result_path, predictions, worker_rank, world_size):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    func(predictions, result_path)

    th.distributed.barrier()
    if worker_rank == 0:
        th.distributed.destroy_process_group()

def test_save_node_prediction_results():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ntype0 = "ntype0"
        ntype1 = "ntype1"
        predictions0 = {
            ntype0: (th.rand((10, 4)), th.randint(20, (10,))),
            ntype1: (th.rand((10, 4)), th.randint(20, (10,))),
        }
        predictions1 = {
            ntype0: (th.rand((10, 4)), th.randint(20, (10,))),
            ntype1: (th.rand((10, 4)), th.randint(20, (10,))),
        }

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_save_predict_results,
                         args=(save_node_prediction_results,
                               tmpdirname, predictions0, 0, 2))
        p1 = ctx.Process(target=run_dist_save_predict_results,
                         args=(save_node_prediction_results,
                               tmpdirname, predictions1, 1, 2))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        os.path.exists(os.path.join(tmpdirname, "result_info.json"))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype0, "predict-00000.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype0, "predict-00001.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype1, "predict-00000.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype1, "predict-00001.pt")))
        with open(os.path.join(tmpdirname, "result_info.json"), 'r', encoding='utf-8') as f:
            info = json.load(f)
            assert info["format"] == "pytorch"
            assert info["world_size"] == 2
            assert set(info["ntypes"]) == set([ntype0, ntype1])

        n0_feat0 = th.load(os.path.join(tmpdirname, os.path.join(ntype0, "predict-00000.pt")))
        n0_feat1 = th.load(os.path.join(tmpdirname, os.path.join(ntype0, "predict-00001.pt")))
        n0_nid0 = th.load(os.path.join(tmpdirname, os.path.join(ntype0, "predict_nids-00000.pt")))
        n0_nid1 = th.load(os.path.join(tmpdirname, os.path.join(ntype0, "predict_nids-00001.pt")))
        n1_feat0 = th.load(os.path.join(tmpdirname, os.path.join(ntype1, "predict-00000.pt")))
        n1_feat1 = th.load(os.path.join(tmpdirname, os.path.join(ntype1, "predict-00001.pt")))
        n1_nid0 = th.load(os.path.join(tmpdirname, os.path.join(ntype1, "predict_nids-00000.pt")))
        n1_nid1 = th.load(os.path.join(tmpdirname, os.path.join(ntype1, "predict_nids-00001.pt")))

        assert_almost_equal(th.cat([n0_feat0, n0_feat1]).numpy(),
                            th.cat([predictions0[ntype0][0], predictions1[ntype0][0]]).numpy())
        assert_almost_equal(th.cat([n1_feat0, n1_feat1]).numpy(),
                            th.cat([predictions0[ntype1][0], predictions1[ntype1][0]]).numpy())
        assert_almost_equal(th.cat([n0_nid0, n0_nid1]).numpy(),
                            th.cat([predictions0[ntype0][1], predictions1[ntype0][1]]).numpy())
        assert_almost_equal(th.cat([n1_nid0, n1_nid1]).numpy(),
                            th.cat([predictions0[ntype1][1], predictions1[ntype1][1]]).numpy())

def test_save_shuffled_node_embeddings():
    with tempfile.TemporaryDirectory() as tmpdirname:
        ntype0 = "ntype0"
        ntype1 = "ntype1"
        embs0 = {
            ntype0: (th.rand((10, 4)), th.randint(20, (10,))),
            ntype1: (th.rand((10, 4)), th.randint(20, (10,))),
        }
        embs1 = {
            ntype0: (th.rand((10, 4)), th.randint(20, (10,))),
            ntype1: (th.rand((10, 4)), th.randint(20, (10,))),
        }

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_save_predict_results,
                         args=(save_shuffled_node_embeddings,
                               tmpdirname, embs0, 0, 2))
        p1 = ctx.Process(target=run_dist_save_predict_results,
                         args=(save_shuffled_node_embeddings,
                               tmpdirname, embs1, 1, 2))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        os.path.exists(os.path.join(tmpdirname, "emb_info.json"))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype0, "embed-00000.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype0, "embed-00001.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype1, "embed-00000.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join(ntype1, "embed-00001.pt")))
        with open(os.path.join(tmpdirname, "emb_info.json"), 'r', encoding='utf-8') as f:
            info = json.load(f)
            assert info["format"] == "pytorch"
            assert info["world_size"] == 2
            assert set(info["emb_name"]) == set([ntype0, ntype1])

        n0_feat0 = th.load(os.path.join(tmpdirname,
                                        os.path.join(ntype0, "embed-00000.pt")))
        n0_feat1 = th.load(os.path.join(tmpdirname,
                                        os.path.join(ntype0, "embed-00001.pt")))
        n0_nid0 = th.load(os.path.join(tmpdirname,
                                       os.path.join(ntype0, "embed_nids-00000.pt")))
        n0_nid1 = th.load(os.path.join(tmpdirname,
                                       os.path.join(ntype0, "embed_nids-00001.pt")))
        n1_feat0 = th.load(os.path.join(tmpdirname,
                                        os.path.join(ntype1, "embed-00000.pt")))
        n1_feat1 = th.load(os.path.join(tmpdirname,
                                        os.path.join(ntype1, "embed-00001.pt")))
        n1_nid0 = th.load(os.path.join(tmpdirname,
                                       os.path.join(ntype1, "embed_nids-00000.pt")))
        n1_nid1 = th.load(os.path.join(tmpdirname,
                                       os.path.join(ntype1, "embed_nids-00001.pt")))

        assert_almost_equal(th.cat([n0_feat0, n0_feat1]).numpy(),
                            th.cat([embs0[ntype0][0], embs1[ntype0][0]]).numpy())
        assert_almost_equal(th.cat([n1_feat0, n1_feat1]).numpy(),
                            th.cat([embs0[ntype1][0], embs1[ntype1][0]]).numpy())
        assert_almost_equal(th.cat([n0_nid0, n0_nid1]).numpy(),
                            th.cat([embs0[ntype0][1], embs1[ntype0][1]]).numpy())
        assert_almost_equal(th.cat([n1_nid0, n1_nid1]).numpy(),
                            th.cat([embs0[ntype1][1], embs1[ntype1][1]]).numpy())

def test_save_edge_prediction_results():
    with tempfile.TemporaryDirectory() as tmpdirname:
        etype0 = ("ntype0", "rel0", "ntype1")
        etype1 = ("ntype0", "rel1", "ntype2")
        predictions0 = {
            etype0: (th.rand((10, 4)), th.randint(100, (10,)), th.randint(100, (10,))),
            etype1: (th.rand((10, 4)), th.randint(100, (10,)), th.randint(100, (10,)))
        }
        predictions1 = {
            etype0: (th.rand((10, 4)), th.randint(100, (10,)), th.randint(100, (10,))),
            etype1: (th.rand((10, 4)), th.randint(100, (10,)), th.randint(100, (10,)))
        }
        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_save_predict_results,
                         args=(save_edge_prediction_results,
                               tmpdirname, predictions0, 0, 2))
        p1 = ctx.Process(target=run_dist_save_predict_results,
                         args=(save_edge_prediction_results,
                               tmpdirname, predictions1, 1, 2))
        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

        os.path.exists(os.path.join(tmpdirname, "result_info.json"))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join("_".join(etype0), "predict-00000.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join("_".join(etype0), "predict-00001.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join("_".join(etype1), "predict-00000.pt")))
        os.path.exists(os.path.join(tmpdirname,
                                    os.path.join("_".join(etype1), "predict-00001.pt")))
        with open(os.path.join(tmpdirname, "result_info.json"), 'r', encoding='utf-8') as f:
            info = json.load(f)
            assert info["format"] == "pytorch"
            assert info["world_size"] == 2
            assert set([tuple(etype) for etype in info["etypes"]]) == set([etype0, etype1])

        e0_feat0 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype0), "predict-00000.pt")))
        e0_feat1 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype0), "predict-00001.pt")))
        e1_feat0 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype1), "predict-00000.pt")))
        e1_feat1 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype1), "predict-00001.pt")))
        e0_src0 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype0), "src_nids-00000.pt")))
        e0_src1 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype0), "src_nids-00001.pt")))
        e1_src0 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype1), "src_nids-00000.pt")))
        e1_src1 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype1), "src_nids-00001.pt")))
        e0_dst0 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype0), "dst_nids-00000.pt")))
        e0_dst1 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype0), "dst_nids-00001.pt")))
        e1_dst0 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype1), "dst_nids-00000.pt")))
        e1_dst1 = th.load(os.path.join(tmpdirname,
                                        os.path.join("_".join(etype1), "dst_nids-00001.pt")))

        assert_almost_equal(th.cat([e0_feat0, e0_feat1]).numpy(),
                            th.cat([predictions0[etype0][0], predictions1[etype0][0]]).numpy())
        assert_almost_equal(th.cat([e1_feat0, e1_feat1]).numpy(),
                            th.cat([predictions0[etype1][0], predictions1[etype1][0]]).numpy())
        assert_almost_equal(th.cat([e0_src0, e0_src1]).numpy(),
                            th.cat([predictions0[etype0][1], predictions1[etype0][1]]).numpy())
        assert_almost_equal(th.cat([e1_src0, e1_src1]).numpy(),
                            th.cat([predictions0[etype1][1], predictions1[etype1][1]]).numpy())
        assert_almost_equal(th.cat([e0_dst0, e0_dst1]).numpy(),
                            th.cat([predictions0[etype0][2], predictions1[etype0][2]]).numpy())
        assert_almost_equal(th.cat([e1_dst0, e1_dst1]).numpy(),
                            th.cat([predictions0[etype1][2], predictions1[etype1][2]]).numpy())

def run_distributed_full_node_embedding_save(part_config, emb_path, embeddings, node_id_mapping_file,
                                 worker_rank, world_size):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip='127.0.0.1', master_port='12345')
    th.distributed.init_process_group(backend="gloo",
                                      init_method=dist_init_method,
                                      world_size=world_size,
                                      rank=worker_rank)
    dgl.distributed.initialize('')
    g = dgl.distributed.DistGraph(graph_name='dummy', part_config=part_config)
    save_full_node_embeddings(g, emb_path, embeddings, node_id_mapping_file)



def test_full_node_embeddings():
    with tempfile.TemporaryDirectory() as tmpdirname:
        g, part_config = generate_dummy_dist_graph(tmpdirname, size="tiny")
        nid_map_dict_path = os.path.join(tmpdirname, "nid_map_dict.pt")

        target_ntype0 = g.ntypes[0]
        target_ntype1 = g.ntypes[1]
        ori_nid_maps = {target_ntype0: th.randperm(g.number_of_nodes(target_ntype0)),
                        target_ntype1: th.randperm(g.number_of_nodes(target_ntype1))}
        th.save(ori_nid_maps, nid_map_dict_path)

        embeddings = {target_ntype0: th.arange(g.number_of_nodes(target_ntype0)),
                        target_ntype1: th.arange(g.number_of_nodes(target_ntype1))}
        emb_path = os.path.join(tmpdirname, "emb")
        os.mkdir(emb_path)

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_distributed_full_node_embedding_save,
                        args=(part_config, emb_path, embeddings, nid_map_dict_path, 0, 1))
        p0.start()
        p0.join()
        assert p0.exitcode == 0

        ntype0_emb_path = os.path.join(emb_path, target_ntype0)
        ntype0_emb = th.load(os.path.join(ntype0_emb_path, "embed-00000.pt"))
        ntype0_nid = th.load(os.path.join(ntype0_emb_path, "embed_nids-00000.pt"))
        assert_equal(embeddings[target_ntype0].numpy(), ntype0_emb.numpy())
        assert_equal(ori_nid_maps[target_ntype0].numpy(), ntype0_nid.numpy())

        ntype1_emb_path = os.path.join(emb_path, target_ntype1)
        ntype1_emb = th.load(os.path.join(ntype1_emb_path, "embed-00000.pt"))
        ntype1_nid = th.load(os.path.join(ntype1_emb_path, "embed_nids-00000.pt"))
        assert_equal(embeddings[target_ntype1].numpy(), ntype1_emb.numpy())
        assert_equal(ori_nid_maps[target_ntype1].numpy(), ntype1_nid.numpy())

def test_prepare_for_wholegraph():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        input_nodes = {"n0": th.ones((g.num_nodes(),), dtype=g.idtype)}
        prepare_for_wholegraph(g, input_nodes)
        assert list(input_nodes.keys()) == g.ntypes

        input_nodes2 = {}
        prepare_for_wholegraph(g, input_nodes2)
        assert list(input_nodes2.keys()) == g.ntypes

        input_edges = {}
        input_nodes = {}
        prepare_for_wholegraph(g, input_nodes, input_edges)
        assert list(input_nodes.keys()) == g.ntypes
        assert list(input_edges.keys()) == g.canonical_etypes

        input_edges = {}
        input_nodes = None
        prepare_for_wholegraph(g, input_nodes, input_edges)
        assert input_nodes == None
        assert list(input_edges.keys()) == g.canonical_etypes

        input_nodes = {"n0": th.ones((g.num_nodes(),), dtype=g.idtype)}
        input_edges = {("n0", "r0", "n1"): th.ones((g.num_nodes(),), dtype=g.idtype)}
        prepare_for_wholegraph(g, input_nodes, input_edges)
        assert list(input_nodes.keys()) == g.ntypes
        assert list(input_edges.keys()) == g.canonical_etypes

@pytest.mark.parametrize("num_embs", [10, 100])
def test_normalize_node_embs(num_embs):
    embs = {"n1": th.rand((10, num_embs)),
            "n2": th.rand((5, num_embs))}
    new_embs = normalize_node_embs(embs, None)

    assert len(embs) == len(new_embs)
    assert_equal(embs["n1"].numpy(), new_embs["n1"].numpy())
    assert_equal(embs["n2"].numpy(), new_embs["n2"].numpy())

    new_embs = normalize_node_embs(embs, "l2_norm")
    def l2_norm(emb):
        return emb / th.norm(emb, p=2, dim=1).reshape(-1, 1)
    l2norm_embs = {
        "n1": l2_norm(embs["n1"]),
        "n2": l2_norm(embs["n2"])
    }

    assert len(l2norm_embs) == len(new_embs)
    assert_almost_equal(l2norm_embs["n1"].numpy(), new_embs["n1"].numpy())
    assert_almost_equal(l2norm_embs["n2"].numpy(), new_embs["n2"].numpy())

    raise_error = False
    try:
        normalize_node_embs(embs, "unknown")
    except:
        raise_error = True
    assert raise_error

def test_get_graph_name():
    with tempfile.TemporaryDirectory() as tmpdirname:
        _, part_config = generate_dummy_dist_graph(tmpdirname, size="tiny")
        graph_name = get_graph_name(part_config)
        
        assert graph_name == 'dummy'


if __name__ == '__main__':
    test_shuffle_nids_dist_part()
    test_distribute_nid_map(backend='gloo', map_pattern='distdgl')
    test_distribute_nid_map(backend='nccl', map_pattern='gconstruct')
    test_shuffle_emb_with_shuffle_nids("distdgl")
    test_shuffle_emb_with_shuffle_nids("gconstruct")
    test_normalize_node_embs(10000)
    test_full_node_embeddings()

    test_shuffle_nids()
    test_save_node_prediction_results()
    test_save_edge_prediction_results()
    test_save_shuffled_node_embeddings()

    test_shuffle_predict(num_embs=16, backend='gloo')
    test_shuffle_predict(num_embs=17, backend='nccl')

    test_get_data_range()
    test_exchange_node_id_mapping(100, backend='gloo')
    test_exchange_node_id_mapping(101, backend='nccl')
    test_save_embeddings_with_id_mapping(num_embs=16, backend='gloo')
    test_save_embeddings_with_id_mapping(num_embs=17, backend='nccl')

    test_get_node_feat_size()
    test_save_embeddings()
    test_remove_saved_models()
    test_topklist()
    test_gen_mrr_score()

    test_stream_dist_tensors_to_hdf5()
    test_prepare_for_wholegraph()

    test_get_graph_name()
