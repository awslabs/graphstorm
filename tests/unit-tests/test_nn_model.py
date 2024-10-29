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

    Test basic Nueral Network modules
"""
import pytest
import tempfile
import torch as th
import dgl
from numpy.testing import assert_almost_equal, assert_raises

from graphstorm.model.rgat_encoder import RelationalAttLayer
from graphstorm.model.rgcn_encoder import RelGraphConvLayer
from graphstorm.model.hgt_encoder import HGTLayer

from data_utils import (generate_dummy_hetero_graph,
                        generate_dummy_hetero_graph_for_efeat_gnn)

def create_dummy_zero_input_test_graph(dim):
    num_src_nodes = {
        "n0": 1024,
        "n1": 0,
        "n2": 0,
        "n3": 0,
        "n4": 0,
    }
    num_nodes_dict = {
        "n0": 1024,
        "n1": 0,
        "n2": 0,
        "n3": 0,
        "n4": 0,
    }

    edges = {
    ("n1", "r0", "n0"): (th.empty((0,), dtype=th.int64),
                            (th.empty((0,), dtype=th.int64))),
    ("n2", "r1", "n0"): (th.empty((0,), dtype=th.int64),
                            (th.empty((0,), dtype=th.int64))),
    ("n3", "r2", "n1"): (th.empty((0,), dtype=th.int64),
                            (th.empty((0,), dtype=th.int64))),
    ("n4", "r3", "n2"): (th.empty((0,), dtype=th.int64),
                            (th.empty((0,), dtype=th.int64))),
    ("n0", "r4", "n3"): (th.empty((0,), dtype=th.int64),
                            (th.empty((0,), dtype=th.int64))),
    }

    block = dgl.create_block(edges,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_nodes_dict)

    inputs = {"n0": th.zeros((1024,dim)),
              "n1": th.empty((0,dim)),
              "n2": th.empty((0,dim)),
              "n3": th.empty((0,dim)),
              "n4": th.empty((0,dim)),}

    return block, inputs, list(num_src_nodes.keys()), list(edges.keys())


def create_dummy_no_indegree_test_graph(dim=16):
    """ Generate dummy test graph in which some destination nodes have no in-degree.
    This could happen when either these nodes are target nodes, or these nodes have
    outdegree to target nodes but no indegree to themselves.
    """
    num_src_nodes = {
        "n0": 1024,             # The first n0 256 nodes are des nodes w/t indegree
        "n1": 128,              # Half of n1 nodes link to n2 nodes, target nodes
        "n2": 64                # Target nodes are all in src and dst lists.
    }
    num_nodes_dict = {
        "n0": 256,              # The dst nodes w/t indegree
        "n1": 64,               # Half of n1 nodes are dst nodes from n0 nodes
        "n2": 64                # some link from n0, and some from n1
    }

    edges = {
    ("n0", "r0", "n1"): (th.arange(256, 1024), th.concat([th.arange(64), th.randint(0, 64, (704,))])),
    ("n0", "r1", "n2"): (th.arange(256), th.concat([th.arange(64), th.randint(0, 64, (192,))])),
    ("n1", "r2", "n2"): (th.arange(64, 128), th.arange(64))
    }

    block = dgl.create_block(edges,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_nodes_dict)

    inputs = {"n0": th.zeros((1024, dim)),
              "n1": th.empty((128, dim)),
              "n2": th.empty((64, dim))}

    return block, inputs, list(num_src_nodes.keys()), list(edges.keys())

    
@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32])
def test_rgcn_with_zero_input(input_dim, output_dim):
    block, inputs, _, etypes = create_dummy_zero_input_test_graph(input_dim)

    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        2, activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)

    out = layer(block, inputs)
    assert out["n0"].shape[0] == 1024
    assert out["n0"].shape[1] == output_dim
    assert out["n1"].shape[0] == 0
    assert out["n1"].shape[1] == output_dim
    assert out["n2"].shape[0] == 0
    assert out["n2"].shape[1] == output_dim
    assert out["n3"].shape[0] == 0
    assert out["n3"].shape[1] == output_dim
    assert "n4" not in out


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_rgat_with_zero_input(input_dim, output_dim):
    block, inputs, _, etypes = create_dummy_zero_input_test_graph(input_dim)

    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        2, activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)

    out = layer(block, inputs)

    assert out["n0"].shape[0] == 1024
    assert out["n0"].shape[1] == output_dim
    assert out["n1"].shape[0] == 0
    assert out["n1"].shape[1] == output_dim
    assert out["n2"].shape[0] == 0
    assert out["n2"].shape[1] == output_dim
    assert out["n3"].shape[0] == 0
    assert out["n3"].shape[1] == output_dim
    assert "n4" not in out


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_hgt_with_zero_input(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_zero_input_test_graph(input_dim)

    layer = HGTLayer(input_dim,
                     output_dim,
                     ntypes,
                     etypes,
                     num_heads=4)
    out = layer(block, inputs)

    assert out["n0"].shape[0] == 1024
    assert out["n0"].shape[1] == output_dim
    assert out["n1"].shape[0] == 0
    assert out["n1"].shape[1] == output_dim
    assert out["n2"].shape[0] == 0
    assert out["n2"].shape[1] == output_dim
    assert out["n3"].shape[0] == 0
    assert out["n3"].shape[1] == output_dim
    assert "n4" not in out


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32])
def test_rgcn_with_no_indegree_dstnodes(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_no_indegree_test_graph(input_dim)

    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        2, activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)

    outputs = layer(block, inputs)
    
    assert outputs['n0'].shape[0] == 256
    assert outputs['n0'].shape[1] == output_dim
    assert outputs['n1'].shape[0] == 64
    assert outputs['n1'].shape[1] == output_dim
    assert outputs['n2'].shape[0] == 64
    assert outputs['n2'].shape[1] == output_dim


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_rgat_with_no_indegree_dstnodes(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_no_indegree_test_graph(input_dim)

    layer = RelationalAttLayer(input_dim, output_dim, etypes,
                               2, activation=th.nn.ReLU(), self_loop=True,
                               dropout=0.1)
    outputs = layer(block, inputs)
    
    assert outputs['n0'].shape[0] == 256
    assert outputs['n0'].shape[1] == output_dim
    assert outputs['n1'].shape[0] == 64
    assert outputs['n1'].shape[1] == output_dim
    assert outputs['n2'].shape[0] == 64
    assert outputs['n2'].shape[1] == output_dim


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_hgt_with_no_indegree_dstnodes(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_no_indegree_test_graph(input_dim)

    layer = HGTLayer(input_dim,
                     output_dim,
                     ntypes,
                     etypes,
                     num_heads=4)
    outputs = layer(block, inputs)
    
    assert outputs['n0'].shape[0] == 256
    assert outputs['n0'].shape[1] == output_dim
    assert outputs['n1'].shape[0] == 64
    assert outputs['n1'].shape[1] == output_dim
    assert outputs['n2'].shape[0] == 64
    assert outputs['n2'].shape[1] == output_dim

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_rgcn_with_edge_features(input_dim, output_dim, dev):
    """ Test the RelGraphConvLayer that supports edge features
    """
    # construct test block and input features
    heter_graph = generate_dummy_hetero_graph(size='tiny', gen_mask=False, 
                                              add_reverse=False, is_random=False)

    seeds = {'n1': [0]}
    subg = dgl.sampling.sample_neighbors(heter_graph, seeds, 100)
    block = dgl.to_block(subg, seeds).to(dev)

    etypes = [("n0", "r0", "n1"), ("n0", "r1", "n1")]

    src1, dst1, r0_eid = subg.edges(form='all', etype='r0')
    src2, dst2, r1_eid = subg.edges(form='all', etype='r1')

    src_idx = th.unique(th.concat([src1, src2]))
    dst_idx = th.unique(th.concat([dst1, dst2]))

    node_feats = {
        "n0": th.rand(src_idx.shape[0], input_dim).to(dev),
        "n1": th.rand(dst_idx.shape[0], input_dim).to(dev)
    }
    edge_feats = {
        ("n0", "r0", "n1"): th.rand(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.rand(r1_eid.shape[0], input_dim).to(dev)
    }

    # Test case 0: normal case, have both node and edge feature on all node and edge types
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    emb0 = layer(block, node_feats, edge_feats)
    # check output numbers, dimensions and device
    assert 'n0' not in emb0
    assert emb0['n1'].shape[0] == len(seeds['n1'])
    assert emb0['n1'].shape[1] == output_dim
    assert emb0['n1'].get_device() == (-1 if dev == 'cpu' else 0)

    # Test case 1: normal case, one edge type has features
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat']},
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    edge_feats = {
        ("n0", "r0", "n1"): th.rand(r0_eid.shape[0], input_dim).to(dev)
    }

    emb1 = layer(block, node_feats, edge_feats)
    # check output numbers, and dimensions
    assert 'n0' not in emb1
    assert emb1['n1'].shape[0] == len(seeds['n1'])
    assert emb1['n1'].shape[1] == output_dim

    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r1", "n1"): ['feat']},
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    edge_feats = {
        ("n0", "r1", "n1"): th.rand(r1_eid.shape[0], input_dim).to(dev)
    }

    emb1 = layer(block, node_feats, edge_feats)
    # check output numbers, and dimensions
    assert 'n0' not in emb1
    assert emb1['n1'].shape[0] == len(seeds['n1'])
    assert emb1['n1'].shape[1] == output_dim

    # Test case 2: normal case, no model weights nor edge features as inputs
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    emb2 = layer(block, node_feats)
    # check output numbers, and dimensions
    assert 'n0' not in emb2
    assert emb2['n1'].shape[0] == len(seeds['n1'])
    assert emb2['n1'].shape[1] == output_dim

    # Test case 3: normal case, all 5 message passing ops
    # Test 3.1, "add" op
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op="add",
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    edge_feats = {
        ("n0", "r0", "n1"): th.rand(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.rand(r1_eid.shape[0], input_dim).to(dev)
    }

    emb31 = layer(block, node_feats, edge_feats)
    # check output numbers, dimensions and device
    assert 'n0' not in emb31
    assert emb31['n1'].shape[0] == len(seeds['n1'])
    assert emb31['n1'].shape[1] == output_dim

    # Test 3.2, "sub" op
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op="sub",
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    emb32 = layer(block, node_feats, edge_feats)
    # check output numbers, dimensions and device
    assert 'n0' not in emb32
    assert emb32['n1'].shape[0] == len(seeds['n1'])
    assert emb32['n1'].shape[1] == output_dim

    # Test 3.3, "mul" op
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op="mul",
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    emb33 = layer(block, node_feats, edge_feats)
    # check output numbers, dimensions and device
    assert 'n0' not in emb33
    assert emb33['n1'].shape[0] == len(seeds['n1'])
    assert emb33['n1'].shape[1] == output_dim

    # Test 3.4, "div" op
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op="div",
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    emb34 = layer(block, node_feats, edge_feats)
    # check output numbers, dimensions and device
    assert 'n0' not in emb34
    assert emb34['n1'].shape[0] == len(seeds['n1'])
    assert emb34['n1'].shape[1] == output_dim

    # Test case 4: abnormal case, layer has no edge feature weights, but give edge features。
    #              this will trigger an assertion error to let users know that they need to use
    #              GraphConvwithEdgeFeat.
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    with assert_raises(AssertionError):
        layer(block, node_feats, edge_feats)

    # Test case 5: abnormal case, layer has edge feature weights, but not give edge features
    #              this will trigger an assertion error of mismatch of the number of inputs
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    with assert_raises(AssertionError):
        layer(block, node_feats)

    # Test case 6: normal case, checking forward results accuracy.
    #              we set all feature to be 1s and all weights to be 1s.
    #  - 'concat' and 'add' both have the same results for n1:[0], i.e., input_dim*2 + input_dim*4 * 2**-0.5
    #  - 'sub' will have all results to be 0s.
    #  - 'mul' and 'div' both have the same results for n1:[0], i.e., input_dim + input_dim*2 * 2**-0.5
    node_feats = {
        "n0": th.ones(src_idx.shape[0], input_dim).to(dev),
        "n1": th.ones(dst_idx.shape[0], input_dim).to(dev)
    }
    edge_feats = {
        ("n0", "r0", "n1"): th.ones(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.ones(r1_eid.shape[0], input_dim).to(dev)
    }

    # concat
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        weight=False, bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).weights)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).weights)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    actual_1 = (emb6['n1']/input_dim).detach().cpu()
    desired_1 = th.ones(dst_idx.shape[0], output_dim) * (2 + 4*(2**-0.5))
    assert_almost_equal(actual_1.numpy(), desired_1.numpy(), decimal=5)

    # add
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='add',
        weight=False, bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).weights)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).weights)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    actual_1 = (emb6['n1']/input_dim).detach().cpu()
    desired_1 = th.ones(dst_idx.shape[0], output_dim) * (2 + 4*(2**-0.5))
    assert_almost_equal(actual_1.numpy(), desired_1.numpy(), decimal=5)

    # sub
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='sub',
        weight=False, bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).weights)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).weights)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(),
                       (th.zeros(dst_idx.shape[0], output_dim)).numpy(), decimal=5)

    # mul
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='mul',
        weight=False, bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).weights)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).weights)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(),
                       (th.ones(dst_idx.shape[0],
                                output_dim) * (input_dim + input_dim*2*(2**-0.5))).numpy(),
                       decimal=5)

    # div
    layer = RelGraphConvLayer(
        input_dim, output_dim, etypes,
        num_bases=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='div',
        weight=False, bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).weights)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).weights)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    actual_2 = (emb6['n1']/input_dim).detach().cpu()
    desired_2 = th.ones(dst_idx.shape[0], output_dim) * (1 + 2*(2**-0.5))
    assert_almost_equal(actual_2.numpy(), desired_2.numpy(), decimal=5)


if __name__ == '__main__':
    test_rgcn_with_zero_input(32, 64)
    test_rgat_with_zero_input(32, 64)
    test_hgt_with_zero_input(32, 64)

    test_rgcn_with_no_indegree_dstnodes(32, 64)
    test_rgat_with_no_indegree_dstnodes(32, 64)
    test_hgt_with_no_indegree_dstnodes(32, 64)

    test_rgcn_with_edge_features(32, 64, 'cpu')
    test_rgcn_with_edge_features(64, 64, 'cpu')
    test_rgcn_with_edge_features(32, 64, 'cuda:0')
