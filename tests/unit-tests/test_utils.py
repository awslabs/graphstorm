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
from pathlib import Path
from argparse import Namespace
import tempfile

import torch as th
import numpy as np
from graphstorm.model.utils import save_embeddings, LazyDistTensor, remove_saved_models, TopKList
from graphstorm import get_feat_size

from data_utils import generate_dummy_dist_graph
from graphstorm.eval.utils import gen_mrr_score

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
    test_get_feat_size()
    test_save_embeddings()
    test_remove_saved_models()
    test_topklist()
    test_gen_mrr_score()
