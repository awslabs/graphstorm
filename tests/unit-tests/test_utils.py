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
import tempfile
import pytest
import multiprocessing as mp

import torch as th
import numpy as np
from numpy.testing import assert_equal
from graphstorm.model.utils import save_embeddings, LazyDistTensor, remove_saved_models, TopKList
from graphstorm.model.utils import _get_data_range
from graphstorm.model.utils import _exchange_node_id_mapping
from graphstorm.model.utils import shuffle_predict
from graphstorm.gconstruct.utils import save_maps
from graphstorm import get_feat_size

from data_utils import generate_dummy_dist_graph
from graphstorm.eval.utils import gen_mrr_score
from graphstorm.utils import setup_device

def gen_embedding_with_nid_mapping(num_embs):
    emb = th.rand((num_embs, 12))
    nid_mapping = th.randperm(num_embs)
    emb = LazyDistTensor(emb, th.arange(num_embs))
    return emb, nid_mapping

def gen_predict_with_nid_mapping(num_embs):
    pred = th.rand((num_embs, 12)) * 10
    pred = pred.long()
    nid_mapping = th.randperm(num_embs)
    return pred, nid_mapping

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
    # test _get_data_range
    # num_embs < world_size, only latest rank will do the work
    start, end = _get_data_range(0, 3, 2)
    assert start == 0
    assert end == 0

    start, end = _get_data_range(1, 3, 2)
    assert start == 0
    assert end == 0

    start, end = _get_data_range(2, 3, 2)
    assert start == 0
    assert end == 2

    # num_embs > world_size
    start, end = _get_data_range(0, 2, 5)
    assert start == 0
    assert end == 2

    start, end = _get_data_range(1, 2, 5)
    assert start == 2
    assert end == 5

    start, end = _get_data_range(0, 2, 4)
    assert start == 0
    assert end == 2

    start, end = _get_data_range(1, 2, 4)
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
@pytest.mark.parametrize("backend", ["gloo"])
def test_exchange_node_id_mapping(num_embs, backend):
    node_id_mapping = th.randperm(num_embs)
    start, end = _get_data_range(0, 4, num_embs)
    target_nid_mapping_0 = node_id_mapping[start:end]
    start, end = _get_data_range(1, 4, num_embs)
    target_nid_mapping_1 = node_id_mapping[start:end]
    start, end = _get_data_range(2, 4, num_embs)
    target_nid_mapping_2 = node_id_mapping[start:end]
    start, end = _get_data_range(3, 4, num_embs)
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
        pred, nid_mapping = gen_predict_with_nid_mapping(num_embs)
        save_maps(tmpdirname, "node_mapping", nid_mapping)
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
        pred, nid_mapping = gen_predict_with_nid_mapping(num_embs)
        nid_mapping = {"node": nid_mapping}
        save_maps(tmpdirname, "node_mapping", nid_mapping)
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

# TODO: Only test gloo now
# Will add test for nccl once we enable nccl
@pytest.mark.parametrize("num_embs", [16, 17])
@pytest.mark.parametrize("backend", ["gloo"])
def test_save_embeddings_with_id_mapping(num_embs, backend):
    import tempfile

    # single embedding
    with tempfile.TemporaryDirectory() as tmpdirname:
        emb, nid_mapping = gen_embedding_with_nid_mapping(num_embs)
        save_maps(tmpdirname, "node_mapping", nid_mapping)
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
        emb0 = th.load(os.path.join(tmpdirname, 'emb.part0.bin'), weights_only=True)
        emb1 = th.load(os.path.join(tmpdirname, 'emb.part1.bin'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(emb)
        assert_equal(emb[nid_mapping].numpy(), saved_emb.numpy())

    # multiple embedding
    with tempfile.TemporaryDirectory() as tmpdirname:
        embs = {}
        nid_mappings = {}
        emb, nid_mapping = gen_embedding_with_nid_mapping(num_embs)
        embs['n0'] = emb
        nid_mappings['n0'] = nid_mapping
        emb, nid_mapping = gen_embedding_with_nid_mapping(num_embs*2)
        embs['n1'] = emb
        nid_mappings['n1'] = nid_mapping
        emb, nid_mapping = gen_embedding_with_nid_mapping(num_embs*3)
        embs['n2'] = emb
        nid_mappings['n2'] = nid_mapping

        save_maps(tmpdirname, "node_mapping", nid_mappings)
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
        emb0 = th.load(os.path.join(tmpdirname, 'n0_emb.part0.bin'), weights_only=True)
        emb1 = th.load(os.path.join(tmpdirname, 'n0_emb.part1.bin'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(embs['n0'])
        assert_equal(embs['n0'][nid_mappings['n0']].numpy(), saved_emb.numpy())

        emb0 = th.load(os.path.join(tmpdirname, 'n1_emb.part0.bin'), weights_only=True)
        emb1 = th.load(os.path.join(tmpdirname, 'n1_emb.part1.bin'), weights_only=True)
        saved_emb = th.cat([emb0, emb1], dim=0)
        assert len(saved_emb) == len(embs['n1'])
        assert_equal(embs['n1'][nid_mappings['n1']].numpy(), saved_emb.numpy())

        emb0 = th.load(os.path.join(tmpdirname, 'n2_emb.part0.bin'), weights_only=True)
        emb1 = th.load(os.path.join(tmpdirname, 'n2_emb.part1.bin'), weights_only=True)
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
        feats_type0 = [th.load(os.path.join(tmpdirname, "type0_emb.part{}.bin".format(i)),
                               weights_only=True) for i in range(4)]
        feats_type0 = th.cat(feats_type0, dim=0)
        # Only work with torch 1.13+
        feats_type1 = [th.load(os.path.join(tmpdirname, "type1_emb.part{}.bin".format(i)),
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

def test_get_feat_size():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_feat_size(g, 'feat')
    assert len(feat_size) == len(g.ntypes)
    for ntype in feat_size:
        assert ntype in g.ntypes
        assert feat_size[ntype] == g.nodes[ntype].data['feat'].shape[1]

    feat_size = get_feat_size(g, {'n0': ['feat'], 'n1': ['feat']})
    assert len(feat_size) == len(g.ntypes)
    for ntype in feat_size:
        assert ntype in g.ntypes
        assert feat_size[ntype] == g.nodes[ntype].data['feat'].shape[1]

    feat_size = get_feat_size(g, {'n0' : ['feat']})
    assert len(feat_size) == len(g.ntypes)
    assert feat_size['n0'] == g.nodes['n0'].data['feat'].shape[1]
    assert feat_size['n1'] == 0

    try:
        feat_size = get_feat_size(g, {'n0': ['feat'], 'n1': ['feat1']})
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

if __name__ == '__main__':
    test_shuffle_predict(num_embs=16, backend='gloo')
    test_shuffle_predict(num_embs=17, backend='nccl')

    test_get_data_range()
    test_exchange_node_id_mapping(100, backend='gloo')
    test_exchange_node_id_mapping(101, backend='nccl')
    test_save_embeddings_with_id_mapping(num_embs=16, backend='gloo')
    test_save_embeddings_with_id_mapping(num_embs=17, backend='nccl')

    test_get_feat_size()
    test_save_embeddings()
    test_remove_saved_models()
    test_topklist()
    test_gen_mrr_score()
