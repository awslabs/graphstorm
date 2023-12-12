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

import torch as th
import torch.nn.functional as F
from torch import nn
from numpy.testing import assert_equal, assert_almost_equal

from unittest.mock import patch

from graphstorm.gsf import init_wholegraph
from graphstorm.utils import use_wholegraph_sparse_emb, is_wholegraph_sparse_emb
from graphstorm.model import GSNodeEncoderInputLayer
from graphstorm.model.embed import compute_node_input_embeddings
from graphstorm.model.utils import save_sparse_embeds
from graphstorm.model.utils import load_sparse_embeds
from graphstorm.model.utils import _get_sparse_emb_range
from graphstorm.model.utils import pad_file_index
from graphstorm import get_feat_size

from data_utils import generate_dummy_dist_graph

def initialize(use_wholegraph=True):

    from dgl.distributed import role
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

@pytest.mark.parametrize("world_size", [3, 4])
def test_wg_sparse_embed_save(world_size):
    """ Test sparse embedding saving logic using wholegraph. (graphstorm.model.utils.save_sparse_embeds)

        It will mimic the logic when multiple trainers are saving the embedding.
        And then check the value of the saved embedding.
    """
    # initialize the torch and wholegraph distributed environment
    wgth = pytest.importorskip("pylibwholegraph.torch")
    use_wholegraph_sparse_emb()
    initialize(use_wholegraph=is_wholegraph_sparse_emb())

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        feat_size = {"n0":0, "n1":0}
        embed_layer = GSNodeEncoderInputLayer(g, feat_size, 32)
        model_path = os.path.join(tmpdirname, "model")
        os.mkdir(model_path)

        def get_wholegraph_sparse_emb(sparse_emb):
            local_tensor, offset = sparse_emb.wm_embedding.get_embedding_tensor().get_local_tensor(host_view=True)
            return local_tensor

        if is_wholegraph_sparse_emb():
            sparse_embs = \
                {ntype: get_wholegraph_sparse_emb(sparse_emb) \
                    for ntype, sparse_emb in embed_layer.sparse_embeds.items()}
        else:
            sparse_embs = \
                {ntype: sparse_emb._tensor[th.arange(embed_layer.g.number_of_nodes(ntype))] \
                    for ntype, sparse_emb in embed_layer.sparse_embeds.items()}

        @patch("graphstorm.model.utils.get_rank")
        @patch("graphstorm.model.utils.get_world_size")
        def check_saved_sparse_emb(mock_get_world_size, mock_get_rank):
            for i in range(world_size):
                mock_get_rank.side_effect = [i, i]
                mock_get_world_size.side_effect = [world_size] * 2
                save_sparse_embeds(model_path, embed_layer)

            for ntype in embed_layer.sparse_embeds.keys():
                saved_embs = []
                for i in range(world_size):
                    saved_embs.append(th.from_numpy(np.load(
                        os.path.join(os.path.join(model_path, ntype),
                                f'sparse_emb_{pad_file_index(i)}.npy'))))
                saved_embs = th.cat(saved_embs, dim=0)
                assert_equal(saved_embs.numpy(), sparse_embs[ntype].numpy())
        check_saved_sparse_emb()

    if is_wholegraph_sparse_emb():
        wgth.finalize()
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("infer_world_size", [3, 8, 16])
@pytest.mark.parametrize("train_world_size", [8])
def test_wg_sparse_embed_load(infer_world_size, train_world_size):
    """ Test sparse embedding loading logic using wholegraph. (graphstorm.model.utils.load_sparse_embeds)

        It uses save_sparse_embeds to save the modal and loads it using
        load_sparse_embeds with different infer_world_size to mimic
        different number of processes to load the sparse embedding.
        It will compare the embedings stored and loaded.
    """
    # initialize the torch and wholegraph distributed environment
    wgth = pytest.importorskip("pylibwholegraph.torch")
    use_wholegraph_sparse_emb()
    initialize(use_wholegraph=is_wholegraph_sparse_emb())

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        feat_size = {"n0":0, "n1":0}
        embed_layer = GSNodeEncoderInputLayer(g, feat_size, 4)
        model_path = os.path.join(tmpdirname, "model")
        os.mkdir(model_path)
        def get_wholegraph_sparse_emb(sparse_emb):
            local_tensor, offset = sparse_emb.wm_embedding.get_embedding_tensor().get_local_tensor(host_view=True)
            return local_tensor

        if is_wholegraph_sparse_emb():
            saved_embs = \
                {ntype: get_wholegraph_sparse_emb(sparse_emb) \
                    for ntype, sparse_emb in embed_layer.sparse_embeds.items()}
        else:
            saved_embs = \
                {ntype: sparse_emb._tensor[th.arange(embed_layer.g.number_of_nodes(ntype))] \
                    for ntype, sparse_emb in embed_layer.sparse_embeds.items()}

        @patch("graphstorm.model.utils.get_rank")
        @patch("graphstorm.model.utils.get_world_size")
        def check_sparse_emb(mock_get_world_size, mock_get_rank):

            for i in range(train_world_size):
                mock_get_rank.side_effect = [i] * 2
                mock_get_world_size.side_effect = [train_world_size] * 2
                save_sparse_embeds(model_path, embed_layer)

            for i in range(infer_world_size):
                mock_get_rank.side_effect = [i] * 2
                mock_get_world_size.side_effect = [infer_world_size] * 2
                load_sparse_embeds(model_path, embed_layer)
            if is_wholegraph_sparse_emb():
                load_sparse_embs = \
                    {ntype: get_wholegraph_sparse_emb(sparse_emb) \
                        for ntype, sparse_emb in embed_layer.sparse_embeds.items()}
            else:
                load_sparse_embs = \
                    {ntype: sparse_emb._tensor[th.arange(embed_layer.g.number_of_nodes(ntype))] \
                        for ntype, sparse_emb in embed_layer.sparse_embeds.items()}

            for ntype in embed_layer.sparse_embeds.keys():
                assert_equal(saved_embs[ntype].numpy(), load_sparse_embs[ntype].numpy())
        check_sparse_emb()

    if is_wholegraph_sparse_emb():
        wgth.finalize()
    th.distributed.destroy_process_group()


# In this case, we use node feature on one node type and
# use sparse embedding on the other node type.
# Refer to: unit-tests/test_embed.py:test_input_layer3
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_wg_input_layer3(dev):
    # initialize the torch and wholegraph distributed environment
    wgth = pytest.importorskip("pylibwholegraph.torch")
    use_wholegraph_sparse_emb()
    initialize(use_wholegraph=is_wholegraph_sparse_emb())
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_feat_size(g, {'n0' : ['feat']})
    layer = GSNodeEncoderInputLayer(g, feat_size, 2)
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

    node_embs['n1'] = layer.sparse_embeds['n1'](th.from_numpy(input_nodes['n1']).cuda())

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
    node_embs['n1'] = layer.sparse_embeds['n1'](input_nodes['n1'].cuda())

    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    # check emb device
    for _, emb in embed.items():
        assert emb.get_device() == (-1 if dev == 'cpu' else 0)
    assert_almost_equal(embed['n0'].detach().cpu().numpy(),
                        node_feat['n0'].detach().cpu().numpy())
    assert_almost_equal(embed['n1'].detach().cpu().numpy(),
                        node_embs['n1'].detach().cpu().numpy())

    if is_wholegraph_sparse_emb():
        wgth.finalize()
    th.distributed.destroy_process_group()

# In this case, we use both node features and sparse embeddings.
# Refer to: unit-tests/test_embed.py:test_input_layer2
def test_wg_input_layer2():
    # initialize the torch and wholegraph distributed environment
    wgth = pytest.importorskip("pylibwholegraph.torch")
    use_wholegraph_sparse_emb()
    initialize(use_wholegraph=is_wholegraph_sparse_emb())
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_feat_size(g, 'feat')
    layer = GSNodeEncoderInputLayer(g, feat_size, 2, use_node_embeddings=True)
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
        # We make the projection matrix that can simply add the node features
        # and the node sparse embeddings after projection.
        with th.no_grad():
            layer.proj_matrix[ntype][:2,:] = layer.input_projs[ntype]
            layer.proj_matrix[ntype][2:,:] = layer.input_projs[ntype]
        input_nodes[ntype] = np.arange(10)
        node_feat[ntype] = g.nodes[ntype].data['feat'][input_nodes[ntype]]
        node_embs[ntype] = layer.sparse_embeds[ntype](th.from_numpy(input_nodes[ntype]).cuda())
    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    assert len(embed) == len(node_feat)
    for ntype in embed:
        true_val = node_feat[ntype].detach().numpy() + node_embs[ntype].detach().cpu().numpy()
        assert_almost_equal(embed[ntype].detach().cpu().numpy(), true_val)
    if is_wholegraph_sparse_emb():
        wgth.finalize()
    th.distributed.destroy_process_group()

# Refer to: unit-tests/test_embed.py:test_compute_embed
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_wg_compute_embed(dev):
    # initialize the torch and wholegraph distributed environment
    wgth = pytest.importorskip("pylibwholegraph.torch")
    use_wholegraph_sparse_emb()
    initialize(use_wholegraph=is_wholegraph_sparse_emb())
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
    print('g has {} nodes of n0 and {} nodes of n1'.format(
        g.number_of_nodes('n0'), g.number_of_nodes('n1')))

    feat_size = get_feat_size(g, {'n0' : ['feat']})
    layer = GSNodeEncoderInputLayer(g, feat_size, 2)
    nn.init.eye_(layer.input_projs['n0'])
    nn.init.eye_(layer.proj_matrix['n1'])
    layer.to(dev)

    embeds = compute_node_input_embeddings(g, 10, layer,
                                           feat_field={'n0' : ['feat']})
    assert len(embeds) == len(g.ntypes)
    assert_almost_equal(embeds['n0'][0:len(embeds['n1'])].cpu().numpy(),
            g.nodes['n0'].data['feat'][0:g.number_of_nodes('n0')].cpu().numpy())
    indices = th.arange(g.number_of_nodes('n1'))
    assert_almost_equal(embeds['n1'][0:len(embeds['n1'])].cpu().numpy(),
            layer.sparse_embeds['n1'](indices.cuda()).cpu().detach().numpy())
    # Run it again to tigger the branch that access 'input_emb' directly.
    embeds = compute_node_input_embeddings(g, 10, layer,
                                           feat_field={'n0' : ['feat']})
    if is_wholegraph_sparse_emb():
        wgth.finalize()
    th.distributed.destroy_process_group()

if __name__ == '__main__':
    test_wg_sparse_embed_save(4)
    test_wg_sparse_embed_load(3, 8)
    test_wg_sparse_embed_load(8, 8)

    test_wg_input_layer2()
    test_wg_input_layer3('cpu')
    test_wg_input_layer3('cuda:0')
    test_wg_compute_embed('cpu')
    test_wg_compute_embed('cuda:0')