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

import torch as th
from numpy.testing import assert_equal
from unittest.mock import patch

from graphstorm.gsf import init_wholegraph
from graphstorm.utils import use_wholegraph_sparse_emb, is_wholegraph_sparse_emb
from graphstorm.model import GSNodeEncoderInputLayer
from graphstorm.model.utils import save_sparse_embeds
from graphstorm.model.utils import load_sparse_embeds
from graphstorm.model.utils import _get_sparse_emb_range
from graphstorm.model.utils import pad_file_index
from graphstorm import get_feat_size

from data_utils import generate_dummy_dist_graph
import pylibwholegraph.torch as wgth

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
    # initialize the torch distributed environment
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
                    saved_embs.append(th.load(
                        os.path.join(os.path.join(model_path, ntype),
                                    f'sparse_emb_{pad_file_index(i)}.pt')))
                saved_embs = th.cat(saved_embs, dim=0)
                assert_equal(saved_embs.numpy(), sparse_embs[ntype].numpy())
        check_saved_sparse_emb()

    if is_wholegraph_sparse_emb():
        wgth.finalize()
        th.distributed.destroy_process_group()
    else:
        th.distributed.destroy_process_group()
        dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("infer_world_size", [8])
@pytest.mark.parametrize("train_world_size", [8])
def test_wg_sparse_embed_load(infer_world_size, train_world_size):
    """ Test sparse embedding loading logic using wholegraph. (graphstorm.model.utils.load_sparse_embeds)

        It uses save_sparse_embeds to save the modal and loads it using
        load_sparse_embeds with different infer_world_size to mimic
        different number of processes to load the sparse embedding.
        It will compare the embedings stored and loaded.
    """
    # initialize the torch distributed environment
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
                mock_get_world_size.side_effect = [train_world_size] * 2
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
    else:
        th.distributed.destroy_process_group()
        dgl.distributed.kvstore.close_kvstore()

if __name__ == '__main__':
    test_wg_sparse_embed_save(4)
    test_wg_sparse_embed_load(8, 8)