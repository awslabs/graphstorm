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
from typing import Tuple

import torch as th
from numpy.testing import assert_equal
from unittest.mock import patch

from graphstorm.model import GSNodeEncoderInputLayer
from graphstorm.model.utils import save_sparse_embeds
from graphstorm.model.utils import load_sparse_embeds
from graphstorm.model.utils import _get_sparse_emb_range
from graphstorm.model.utils import pad_file_index
from graphstorm import get_node_feat_size

import graphstorm.model.utils as utils

from data_utils import generate_dummy_dist_graph

def test_get_sparse_emb_range():
    # num_embs = 1, local_rank is 0 or 1.
    start, end = _get_sparse_emb_range(1, 0, 2)
    assert start == 0
    assert end == 1
    start, end = _get_sparse_emb_range(1, 1, 2)
    assert start == 1
    assert end == 1

    # num_embs = 16
    for i in range(4):
        start, end = _get_sparse_emb_range(16, i, 4)
        assert start == i * 4
        assert end == i * 4 + 4

    # num_embs = 15
    for i in range(3):
        start, end = _get_sparse_emb_range(15, i, 4)
        assert start == i * 4
        assert end == i * 4 + 4
    start, end = _get_sparse_emb_range(15, 3, 4)
    assert start == 12
    assert end == 15

    # num_embs = 133
    for i in range(66):
        start, end = _get_sparse_emb_range(133, i, 80)
        assert start == i * 2
        assert end == i * 2 + 2
    start, end = _get_sparse_emb_range(133, 66, 80)
    assert start == 132
    assert end == 133
    for i in range(67, 80):
        start, end = _get_sparse_emb_range(133, i, 80)
        assert start == 133
        assert end == 133

    with pytest.raises(AssertionError):
        _get_sparse_emb_range(15, 4, 4)

@pytest.mark.parametrize("world_size", [3, 4])
def test_sparse_embed_save(world_size):
    """ Test sparse embedding saving logic. (graphstorm.model.utils.save_sparse_embeds)

        It will mimic the logic when multiple trainers are saving the embedding.
        And then check the value of the saved embedding.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        feat_size = {"n0":0, "n1":0}
        embed_layer = GSNodeEncoderInputLayer(g, feat_size, 32)

        model_path = os.path.join(tmpdirname, "model")
        os.mkdir(model_path)
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
                    saved_embs.append(th.load(
                        os.path.join(os.path.join(model_path, ntype),
                                    f'sparse_emb_{pad_file_index(i)}.pt')))
                saved_embs = th.cat(saved_embs, dim=0)
                assert_equal(saved_embs.numpy(), sparse_embs[ntype].numpy())
        check_saved_sparse_emb()
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("infer_world_size", [3, 8, 16])
@pytest.mark.parametrize("train_world_size", [8])
def test_sparse_embed_load(infer_world_size, train_world_size):
    """ Test sparse embedding loading logic. (graphstorm.model.utils.load_sparse_embeds)

        It uses save_sparse_embeds to save the modal and loads it using
        load_sparse_embeds with different infer_world_size to mimic
        different number of processes to load the sparse embedding.
        It will compare the embedings stored and loaded.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        feat_size = {"n0":0, "n1":0}
        embed_layer = GSNodeEncoderInputLayer(g, feat_size, 4)
        model_path = os.path.join(tmpdirname, "model")
        os.mkdir(model_path)
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
            load_sparse_embs = \
                {ntype: sparse_emb._tensor[th.arange(embed_layer.g.number_of_nodes(ntype))] \
                    for ntype, sparse_emb in embed_layer.sparse_embeds.items()}

            for ntype in embed_layer.sparse_embeds.keys():
                assert_equal(saved_embs[ntype].numpy(), load_sparse_embs[ntype].numpy())
        check_sparse_emb()

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_sparse_embed_load_corner_cases():
    """ Cover some corner cases load_sparse_embeds handles:
        1) embed_layer is None
        2) embed_layer.sparse_embeds is empty.
        3) invalid value for rank and world size
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_path = os.path.join(tmpdirname, "model")
        # embed_layer is None
        try:
            load_sparse_embeds(model_path, None) # This call should pass
        except:
            raise "load_sparse_embeds call error with embed_layer as None"

        # embed_layer.sparse_embeds is empty.
        g, _ = generate_dummy_dist_graph(tmpdirname)
        feat_size = get_node_feat_size(g, 'feat')
        embed_layer = GSNodeEncoderInputLayer(g, feat_size, 4)
        assert len(embed_layer.sparse_embeds) == 0
        try:
            load_sparse_embeds(model_path, None) # This call should pass
        except:
            raise "load_sparse_embeds call error with embed_layer.sparse_embeds is empty"

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

class DummySparseEmb:
    """ Dummy class for sparse embedding
    """
    def __init__(
        self,
        shape: Tuple,
        dtype: th.dtype = th.float,
    ):
        self._tensor = th.zeros(shape, dtype=dtype)

    @property
    def num_embeddings(self):
        return len(self._tensor)

    def __setitem__(self, idx: th.Tensor, val: th.Tensor):
        self._tensor[idx] = val

    def __getitem__(self, idx: th.Tensor) -> th.Tensor:
        return self._tensor[idx]

@pytest.mark.parametrize("emb_size", [(140, 10), (1, 10), (13, 10)])
@pytest.mark.parametrize("world_size", [80, 1, 12])
def test_save_load_sparse_emb(emb_size, world_size, monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdirname:
        emb_path = os.path.join(tmpdirname, "path")


        sparse_emb = DummySparseEmb(emb_size)
        sparse_emb[th.arange(emb_size[0])] = th.rand(emb_size)
        ntype = "test"
        def mock_get_world_size():
            return world_size
        monkeypatch.setattr(utils , "get_world_size", mock_get_world_size)
        for rank in range(world_size):
            def mock_get_rank():
                return rank

            monkeypatch.setattr(utils , "get_rank", mock_get_rank)
            utils.save_sparse_emb(emb_path, sparse_emb, ntype)

        new_sparse_emb = DummySparseEmb(emb_size)
        load_emb_path = os.path.join(emb_path, ntype)

        for rank in range(world_size):
            def mock_get_rank():
                return rank
            monkeypatch.setattr(utils, "get_rank", mock_get_rank)
            utils.load_sparse_emb(new_sparse_emb, load_emb_path)

        assert_equal(new_sparse_emb._tensor.numpy(), sparse_emb._tensor.numpy())


        new_sparse_emb2 = DummySparseEmb(emb_size)
        load_emb_path = os.path.join(emb_path, ntype)

        # Test the case where the numbers of processes (world_size)
        # are different between emb saving and loading.
        world_size = 2
        for rank in range(world_size):
            def mock_get_rank():
                return rank

            monkeypatch.setattr(utils, "get_rank", mock_get_rank)
            utils.load_sparse_emb(new_sparse_emb2, load_emb_path)

        assert_equal(new_sparse_emb2._tensor.numpy(), sparse_emb._tensor.numpy())

if __name__ == '__main__':
    test_save_load_sparse_emb()
    test_get_sparse_emb_range()
    test_sparse_embed_save(4)
    test_sparse_embed_load(3, 8)
    test_sparse_embed_load(8, 8)

    test_sparse_embed_load_corner_cases()