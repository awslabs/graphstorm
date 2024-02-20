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
import multiprocessing as mp
import pytest
from unittest.mock import patch
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises
import tempfile


import dgl
from transformers import AutoTokenizer
import graphstorm as gs
from graphstorm import get_node_feat_size
from graphstorm.model import GSNodeEncoderInputLayer, GSLMNodeEncoderInputLayer, GSPureLMNodeInputLayer
from graphstorm.model.embed import compute_node_input_embeddings
from graphstorm.dataloading.dataset import prepare_batch_input
from graphstorm.model.lm_model import TOKEN_IDX, ATT_MASK_IDX, VALID_LEN
from graphstorm.model.lm_embed import LMModels, LMCache
from graphstorm.model.utils import (LazyDistTensor,
                                    load_pytorch_embedding,
                                    save_pytorch_embedding)
from graphstorm.wholegraph import init_wholegraph

from data_utils import generate_dummy_dist_graph
from data_utils import create_lm_graph, create_lm_graph2, load_lm_graph
from util import create_tokens

# In this case, we only use the node features to generate node embeddings.
@pytest.mark.parametrize("input_activate", [None, F.relu])
def test_input_layer1(input_activate):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, 'feat')
    layer = GSNodeEncoderInputLayer(g, feat_size, 2, activation=input_activate)
    ntypes = list(layer.input_projs.keys())
    assert set(ntypes) == set(g.ntypes)
    node_feat = {}
    input_nodes = {}
    for ntype in ntypes:
        # We make the projection matrix a diagonal matrix so that
        # the input and output matrices are identical.
        nn.init.eye_(layer.input_projs[ntype])
        input_nodes[ntype] = np.arange(10)
        node_feat[ntype] = g.nodes[ntype].data['feat'][input_nodes[ntype]]
        if input_activate:
            node_feat[ntype] = input_activate(node_feat[ntype])
    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    assert len(embed) == len(node_feat)
    for ntype in embed:
        assert_almost_equal(embed[ntype].detach().numpy(),
                            node_feat[ntype].detach().numpy())
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

# In this case, we use both node features and sparse embeddings.
def test_input_layer2():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, 'feat')
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
        node_embs[ntype] = layer.sparse_embeds[ntype].weight[input_nodes[ntype]]
    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    assert len(embed) == len(node_feat)
    for ntype in embed:
        true_val = node_feat[ntype].detach().numpy() + node_embs[ntype].detach().numpy()
        assert_almost_equal(embed[ntype].detach().numpy(), true_val)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

# In this case, we use node feature on one node type and
# use sparse embedding on the other node type.
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_input_layer3(dev):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, {'n0' : ['feat']})
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
    node_embs['n1'] = layer.sparse_embeds['n1'].weight[input_nodes['n1']]
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
    input_nodes['n1'] = np.zeros((0,))
    nn.init.eye_(layer.input_projs['n0'])
    node_feat['n0'] = g.nodes['n0'].data['feat'][input_nodes['n0']].to(dev)
    node_embs['n1'] = layer.sparse_embeds['n1'].weight[input_nodes['n1']]
    embed = layer(node_feat, input_nodes)
    assert len(embed) == len(input_nodes)
    # check emb device
    for _, emb in embed.items():
        assert emb.get_device() == (-1 if dev == 'cpu' else 0)
    assert_almost_equal(embed['n0'].detach().cpu().numpy(),
                        node_feat['n0'].detach().cpu().numpy())
    assert_almost_equal(embed['n1'].detach().cpu().numpy(),
                        node_embs['n1'].detach().cpu().numpy())
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()


@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_compute_embed(dev):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
    print('g has {} nodes of n0 and {} nodes of n1'.format(
        g.number_of_nodes('n0'), g.number_of_nodes('n1')))

    feat_size = get_node_feat_size(g, {'n0' : ['feat']})
    layer = GSNodeEncoderInputLayer(g, feat_size, 2)
    nn.init.eye_(layer.input_projs['n0'])
    nn.init.eye_(layer.proj_matrix['n1'])
    layer.to(dev)

    embeds = compute_node_input_embeddings(g, 10, layer,
                                           feat_field={'n0' : ['feat']})
    assert len(embeds) == len(g.ntypes)
    assert_almost_equal(embeds['n0'][0:len(embeds['n1'])].cpu().numpy(),
            g.nodes['n0'].data['feat'][0:g.number_of_nodes('n0')].cpu().numpy())
    assert_almost_equal(embeds['n1'][0:len(embeds['n1'])].cpu().numpy(),
            layer.sparse_embeds['n1'].weight[0:g.number_of_nodes('n1')].cpu().numpy())
    # Run it again to tigger the branch that access 'input_emb' directly.
    embeds = compute_node_input_embeddings(g, 10, layer,
                                           feat_field={'n0' : ['feat']})
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_lm_cache():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, g, _ = \
            create_lm_graph(tmpdirname)

        lm_models = LMModels(g, lm_config, 0, 10)
        lm_cache = LMCache(g, lm_models, tmpdirname)
        ret = lm_cache.update_cache(100)
        assert ret == True      # This is the first time we need to compute the BERT embeddings.
        assert len(lm_cache) == 1
        assert len(lm_cache.ntypes) == 1
        assert lm_cache.ntypes[0] == 'n0'

        lm_models._lm_models["n0"].lm_model.init_weights()
        # Create the second cache. It should loads the embeddings from
        # the first cache.
        lm_cache2 = LMCache(g, lm_models, tmpdirname)
        ret = lm_cache2.update_cache(100)
        assert ret == False     # It loads LM embeddings directly.
        assert len(lm_cache2) == 1
        emb1 = lm_cache["n0"]
        emb2 = lm_cache2["n0"]
        assert np.all(emb1[0:len(emb1)].numpy() == emb2[0:len(emb2)].numpy())
        ret = lm_cache2.update_cache(100)
        assert ret == False     # It uses the loaded embeddings.
        lm_cache2.clear_cache()
        ret = lm_cache2.update_cache(100)
        assert ret == False     # Load LM embeddings from the disk.

        # If the model is changed, the model name should also be changed.
        model_name = lm_cache._get_model_name("n0")
        for param in lm_models.get_lm_model("n0").parameters():
            param.data += 1
        model_name1 = lm_cache._get_model_name("n0")
        assert model_name != model_name1

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def run_dist_cache(part_config, tmpdirname):
    gs.initialize(ip_config=None, backend="gloo")
    g, lm_config = load_lm_graph(part_config)
    lm_models = LMModels(g, lm_config, 0, 10)
    lm_cache = LMCache(g, lm_models, tmpdirname)
    lm_cache.update_cache(100)
    assert len(lm_cache) == 1
    assert len(lm_cache.ntypes) == 1
    assert lm_cache.ntypes[0] == 'n0'

    # Create the second cache. It should loads the embeddings from
    # the first cache.
    lm_cache2 = LMCache(g, lm_models, tmpdirname)
    lm_cache2.update_cache(100)
    assert len(lm_cache2) == 1
    emb1 = lm_cache["n0"]
    emb2 = lm_cache2["n0"]
    assert np.all(emb1[0:len(emb1)].numpy() == emb2[0:len(emb2)].numpy())

def test_mp_lm_cache():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, _, part_config = \
            create_lm_graph(tmpdirname)

        ctx = mp.get_context('spawn')
        p0 = ctx.Process(target=run_dist_cache, args=(part_config, tmpdirname))
        p1 = ctx.Process(target=run_dist_cache, args=(part_config, tmpdirname))

        p0.start()
        p1.start()
        p0.join()
        p1.join()
        assert p0.exitcode == 0
        assert p1.exitcode == 0

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_lm_infer():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, g, _ = \
            create_lm_graph(tmpdirname)
    layer = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 2, num_train=0)
    # during infer, no bert emb cache is used.
    assert len(layer.lm_emb_cache) == 0

    # Bert + feat for n0
    nn.init.eye_(layer.input_projs['n0'])
    embeds_with_lm = compute_node_input_embeddings(g, 10, layer,
                                                   feat_field={'n0' : ['feat']})
    ntype = layer._lm_models.ntypes[0]
    lm_model = layer._lm_models.get_lm_model(ntype).lm_model
    lm_model.eval()
    outputs = lm_model(input_ids, attention_mask=attention_mask)
    lm_model.train()
    out_emb = outputs.pooler_output
    feat_size['n0'] += out_emb.shape[1]
    g.nodes['n0'].data['text'] = out_emb
    layer2 = GSNodeEncoderInputLayer(g, feat_size, 2)
    nn.init.eye_(layer2.input_projs['n0'])
    # Treat Bert text as another node feat
    embeds = compute_node_input_embeddings(g, 10, layer2,
                                           feat_field={'n0' : ['feat', 'text']})
    assert_almost_equal(embeds['n0'][th.arange(g.number_of_nodes('n0'))].numpy(),
                        embeds_with_lm['n0'][th.arange(g.number_of_nodes('n0'))].numpy())
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("num_train", [0, 10])
def test_lm_embed(num_train):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, g, _ = \
            create_lm_graph(tmpdirname)

    layer = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 2, num_train=num_train)
    layer.warmup(g)
    if num_train == 0:
        assert len(layer.lm_emb_cache) > 0
    else:
        assert len(layer.lm_emb_cache) == 0

    # Bert + feat for n0
    nn.init.eye_(layer.input_projs['n0'])
    embeds_with_lm = compute_node_input_embeddings(g, 10, layer,
                                                   feat_field={'n0' : ['feat']})
    ntype = layer._lm_models.ntypes[0]
    lm_model = layer._lm_models.get_lm_model(ntype).lm_model
    lm_model.eval()
    outputs = lm_model(input_ids, attention_mask=attention_mask)
    lm_model.train()
    out_emb = outputs.pooler_output

    assert len(embeds_with_lm) == len(g.ntypes)
    assert_almost_equal(embeds_with_lm['n1'][0:len(embeds_with_lm['n1'])].numpy(),
            layer.sparse_embeds['n1'].weight[0:g.number_of_nodes('n1')].numpy())

    feat_size['n0'] += out_emb.shape[1]
    g.nodes['n0'].data['text'] = out_emb
    layer2 = GSNodeEncoderInputLayer(g, feat_size, 2)
    nn.init.eye_(layer2.input_projs['n0'])
    # Treat Bert text as another node feat
    embeds = compute_node_input_embeddings(g, 10, layer2,
                                           feat_field={'n0' : ['feat', 'text']})
    assert_almost_equal(embeds['n0'][th.arange(g.number_of_nodes('n0'))].numpy(),
                        embeds_with_lm['n0'][th.arange(g.number_of_nodes('n0'))].numpy())

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()


@pytest.mark.parametrize("num_train", [0, 10])
def test_lm_embed(num_train):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, g, _ = \
            create_lm_graph(tmpdirname)
    layer = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 128, num_train=num_train)
    layer.prepare(g)
    if num_train == 0:
        assert len(layer.lm_emb_cache) > 0
    else:
        assert len(layer.lm_emb_cache) == 0

    embeds_with_lm = compute_node_input_embeddings(g, 10, layer,
                                                   feat_field={'n0' : ['feat']})

    ntype = layer._lm_models.ntypes[0]
    lm_model = layer._lm_models.get_lm_model(ntype).lm_model
    lm_model.eval()
    outputs = lm_model(input_ids, attention_mask=attention_mask)
    lm_model.train()
    out_emb = outputs.pooler_output

    assert len(embeds_with_lm) == len(g.ntypes)
    # There is a feature projection layer, the output of lm_models does not match
    # the output of GSLMNodeEncoderInputLayer
    assert out_emb.shape[1] != embeds_with_lm['n0'].shape[1]
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("num_train", [0, 10])
def test_pure_lm_embed(num_train):
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids0, attention_mask0, \
            input_ids1, attention_mask1, g, _ = create_lm_graph2(tmpdirname)
    layer = GSPureLMNodeInputLayer(g, lm_config, num_train=num_train)
    layer.prepare(g)
    if num_train == 0:
        assert len(layer.lm_emb_cache) > 0
    else:
        assert len(layer.lm_emb_cache) == 0

    # GSPureLMNodeInputLayer will ignore input feat
    embeds_with_lm = compute_node_input_embeddings(g, 10, layer,
                                                   feat_field={'n0' : ['feat']})

    ntype = layer._lm_models.ntypes[0]
    lm_model = layer._lm_models.get_lm_model(ntype).lm_model.eval()
    outputs0 = lm_model(input_ids0, attention_mask=attention_mask0)
    outputs1 = lm_model(input_ids1, attention_mask=attention_mask1)
    lm_model.train()
    out_emb0 = outputs0.pooler_output
    out_emb1 = outputs1.pooler_output

    assert len(embeds_with_lm) == len(g.ntypes)

    assert_almost_equal(out_emb0.detach().numpy(),
                        embeds_with_lm['n0'][th.arange(g.number_of_nodes('n0'))].numpy(),
                        decimal=2)
    assert_almost_equal(out_emb1.detach().numpy(),
                        embeds_with_lm['n1'][th.arange(g.number_of_nodes('n0'))].numpy(),
                        decimal=2)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_lm_embed_warmup(dev):
    th.manual_seed(10)
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    bert_model_name = "bert-base-uncased"
    max_seq_length = 8
    num_train = 10
    lm_config = [{"lm_type": "bert",
                  "model_name": bert_model_name,
                  "gradient_checkpoint": True,
                  "node_types": ["n0"]}]
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)

    feat_size = get_node_feat_size(g, {'n0' : ['feat']})
    input_text = ["Hello world!"]
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    input_ids, valid_len, attention_mask, _ = \
        create_tokens(tokenizer=tokenizer,
                      input_text=input_text,
                      max_seq_length=max_seq_length,
                      num_node=g.number_of_nodes('n0'))

    g.nodes['n0'].data[TOKEN_IDX] = input_ids
    g.nodes['n0'].data[ATT_MASK_IDX] = valid_len

    layer = GSLMNodeEncoderInputLayer(g, lm_config, feat_size,
                                      2, num_train=num_train)
    layer = layer.to(dev)
    layer.freeze(g)
    assert len(layer.lm_emb_cache) > 0
    feat_field={'n0' : ['feat']}
    input_nodes = {"n0": th.arange(0, 10, dtype=th.int64)}
    layer.eval()
    feat = prepare_batch_input(g, input_nodes, dev=dev, feat_field=feat_field)
    emb_0 = layer(feat, input_nodes)

    # we change the node feature
    def rand_init(m):
        if isinstance(m, th.nn.Embedding):
            th.nn.init.uniform(m.weight.data, b=10.)
    ntype = layer._lm_models.ntypes[0]
    lm_model = layer._lm_models.get_lm_model(ntype).lm_model
    lm_model.apply(rand_init)
    # model has been freezed, still use bert cache.
    feat = prepare_batch_input(g, input_nodes, dev=dev, feat_field=feat_field)
    emb_1 = layer(feat, input_nodes)
    assert_almost_equal(emb_0['n0'].detach().cpu().numpy(), emb_1['n0'].detach().cpu().numpy(), decimal=6)

    # unfreeze the model, compute bert again
    feat = prepare_batch_input(g, input_nodes, dev=dev, feat_field=feat_field)
    layer.unfreeze()
    emb_2 = layer(feat, input_nodes)
    with assert_raises(AssertionError):
         assert_almost_equal(emb_0['n0'].detach().cpu().numpy(), emb_2['n0'].detach().cpu().numpy(), decimal=1)

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("num_embs", [100, 99])
def test_pytroch_emb_load_save(num_embs):
    with tempfile.TemporaryDirectory() as tmpdirname:
        emb = th.rand(num_embs, 10)
        # gen tensor
        lazy_emb = LazyDistTensor(emb, th.arange(num_embs))
        path = os.path.join(tmpdirname, "test")
        os.makedirs(path)
        save_pytorch_embedding(path, lazy_emb, 0, 2)
        save_pytorch_embedding(path, lazy_emb, 1, 2)
        out_emb = th.empty(emb.shape)

        class DummpyPolicy:
            def get_size(self):
                return 0

        @patch("graphstorm.model.utils.create_dist_tensor", side_effect = [out_emb, out_emb])
        @patch("graphstorm.model.utils.get_rank", side_effect = [0, 1])
        @patch("graphstorm.model.utils.get_world_size", side_effect = [2, 2])
        def load_tensor(mode_get_world_size,
                        mock_get_rank,
                        mock_create_dist_tensor):
            load_pytorch_embedding(path, DummpyPolicy(), "dummy")
            load_pytorch_embedding(path, DummpyPolicy(), "dummy")

        load_tensor()
        assert_almost_equal(emb.numpy(), out_emb.numpy())


def _wg_finalize():
    import pylibwholegraph.torch as wgth
    wgth.finalize()
    # below patch fix (manually reset wg comm) will not be needed
    # once PR: https://github.com/rapidsai/wholegraph/pull/111 is merged.
    import pylibwholegraph.torch.comm as wgth_comm
    wgth_comm.global_communicators = {}
    wgth_comm.local_node_communicator = None
    wgth_comm.local_device_communicator = None


def _wg_initialize(proc_id=0, nprocs=1):
    from dgl.distributed import role
    # Need to fix the wholegraph initializer
    # Options.local_rank = get_rank() % role.get_num_trainers() reqiures role to be initialized
    role.init_role("default")
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


def test_wg_lm_cache():
    # initialize the torch distributed environment
    pytest.importorskip("pylibwholegraph.torch")
    if not th.cuda.is_available():
        pytest.skip("Skip test_wholegraph_lm_cache due to no GPU devices.")

    _wg_initialize()
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, g, _ = \
            create_lm_graph(tmpdirname)

        lm_models = LMModels(g, lm_config, 0, 10)
        lm_cache = LMCache(g, lm_models, tmpdirname, use_wg=True)
        ret = lm_cache.update_cache(100)
        assert ret == True      # This is the first time we need to compute the BERT embeddings.
        assert len(lm_cache) == 1
        assert len(lm_cache.ntypes) == 1
        assert lm_cache.ntypes[0] == 'n0'

        lm_models._lm_models["n0"].lm_model.init_weights()
        # Create the second cache. It should loads the embeddings from
        # the first cache.
        lm_cache2 = LMCache(g, lm_models, tmpdirname, use_wg=True)
        ret = lm_cache2.update_cache(100)
        assert ret == False     # It loads LM embeddings directly.
        assert len(lm_cache2) == 1
        emb1 = lm_cache["n0"]
        emb2 = lm_cache2["n0"]
        emb1, _ = emb1.get_local_tensor()
        emb2, _ = emb2.get_local_tensor()
        assert np.all(emb1.numpy() == emb2.numpy())
        ret = lm_cache2.update_cache(100)
        assert ret == False     # It uses the loaded embeddings.
        lm_cache2.clear_cache()
        ret = lm_cache2.update_cache(100)
        assert ret == False     # Load LM embeddings from the disk.

        # If the model is changed, the model name should also be changed.
        model_name = lm_cache._get_model_name("n0")
        for param in lm_models.get_lm_model("n0").parameters():
            param.data += 1
        model_name1 = lm_cache._get_model_name("n0")
        assert model_name != model_name1
    _wg_finalize()
    th.distributed.destroy_process_group() if th.distributed.is_initialized() else None


def run_wg_dist_cache(proc_id, nprocs, part_config, tmpdirname):
    gs.initialize(ip_config=None, backend="nccl")
    _wg_initialize(proc_id, nprocs)
    g, lm_config = load_lm_graph(part_config)
    lm_models = LMModels(g, lm_config, 0, 10)
    lm_cache = LMCache(g, lm_models, tmpdirname, use_wg=True)
    lm_cache.update_cache(100)
    assert len(lm_cache) == 1
    assert len(lm_cache.ntypes) == 1
    assert lm_cache.ntypes[0] == 'n0'

    # Create the second cache. It should loads the embeddings from
    # the first cache.
    lm_cache2 = LMCache(g, lm_models, tmpdirname, use_wg=True)
    lm_cache2.update_cache(100)
    assert len(lm_cache2) == 1
    emb1 = lm_cache["n0"]
    emb2 = lm_cache2["n0"]
    emb1, _ = emb1.get_local_tensor()
    emb2, _ = emb2.get_local_tensor()
    assert np.all(emb1.numpy() == emb2.numpy())
    _wg_finalize()
    th.distributed.destroy_process_group() if th.distributed.is_initialized() else None


@pytest.mark.parametrize("world_size", [1, 3, 4])
def test_mp_wg_lm_cache(world_size):
    pytest.importorskip("pylibwholegraph.torch")
    if world_size > th.cuda.device_count():
        pytest.skip("Skip test_wg_sparse_embed_save_load due to insufficient GPU devices.")
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // world_size)
    ctx = mp.get_context("spawn")
    ptrainer_list = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, feat_size, input_ids, attention_mask, _, part_config = \
            create_lm_graph(tmpdirname)
        for rank in range(world_size):
            p = ctx.Process(target=run_wg_dist_cache, args=(rank, world_size, part_config, tmpdirname))
            p.start()
            ptrainer_list.append(p)

        for p in ptrainer_list:
            p.join()
            assert p.exitcode == 0


if __name__ == '__main__':
    test_pytroch_emb_load_save(11)
    test_lm_cache()
    test_mp_lm_cache()
    test_input_layer1(None)
    test_input_layer1(F.relu)
    test_input_layer2()
    test_input_layer3('cpu')
    test_input_layer3('cuda:0')
    test_compute_embed('cpu')
    test_compute_embed('cuda:0')

    test_pure_lm_embed(0)
    test_pure_lm_embed(10)

    test_lm_embed(0)
    test_lm_embed(10)

    test_lm_embed_warmup('cpu')
    test_lm_embed_warmup('cuda:0')
    test_lm_infer()

    test_wg_lm_cache()
    test_mp_wg_lm_cache(1)
