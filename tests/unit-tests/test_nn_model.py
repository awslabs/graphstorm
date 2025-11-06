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
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from graphstorm.model.rgat_encoder import RelationalAttLayer
from graphstorm.model.rgcn_encoder import RelGraphConvLayer
from graphstorm.model.hgt_encoder import HGTLayer, HGTLayerwithEdgeFeat

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


def create_dummy_zero_dest_test_graph(dim):
    num_src_nodes = {
        "n0": 4,
        "n1": 2,
        "n2": 1,
    }
    num_dst_dict = {
        "n0": 0,
        "n1": 2,
        "n2": 0,
    }

    edges = {
    ("n0", "r0", "n1"): (th.arange(4),th.tensor([0,0,1,1])),
    ("n2", "r1", "n1"): (th.arange(1),th.tensor([0])),
    ("n1", "r2", "n2"): (th.empty((0,), dtype=th.int64),        # In DGL 2.0+, there will be a case
                            (th.empty((0,), dtype=th.int64))),  # where both source and dest nodes
    }                                                           # are 0, but still DGL MFC keep this
                                                                # edge type that might cause errors
    block = dgl.create_block(edges,
        num_src_nodes=num_src_nodes,
        num_dst_nodes=num_dst_dict)

    inputs = {"n0": th.zeros((4,dim)),
              "n1": th.zeros((2,dim)),
              "n2": th.zeros((1,dim)),}

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
@pytest.mark.parametrize("output_dim", [32,64])
def test_gat_with_zero_input(input_dim, output_dim):
    # GAT works with homogeneous graphs, so we need to create a simpler test
    # Create a simple homogeneous block with zero input nodes
    from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
    import torch as th
    import dgl
    
    # Create a block with some nodes having zero features
    num_src_nodes = 1024
    num_dst_nodes = 1024
    
    # Create edges (some nodes will have no incoming edges)
    src_nodes = th.arange(512, 1024)  # Only half the nodes have outgoing edges
    dst_nodes = th.randint(0, 1024, (512,))
    
    block = dgl.create_block((src_nodes, dst_nodes), 
                           num_src_nodes=num_src_nodes, 
                           num_dst_nodes=num_dst_nodes)
    
    # Add required node IDs for GAT encoder
    block.nodes[DEFAULT_NTYPE].data[dgl.NID] = th.arange(num_src_nodes)
    
    # Create input features
    inputs = {DEFAULT_NTYPE: th.zeros((num_src_nodes, input_dim))}
    
    # Create GAT encoder - with 0 hidden layers, it should go directly from input_dim to output_dim
    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=2, num_hidden_layers=0)
    
    # Create blocks for GAT encoder (needs list of blocks)
    blocks = [block]
    
    out = gat_encoder(blocks, inputs)

    assert out[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert out[DEFAULT_NTYPE].shape[1] == output_dim
    # Check output is not NaN or Inf
    assert not th.isnan(out[DEFAULT_NTYPE]).any(), "Output contains NaN values"
    assert not th.isinf(out[DEFAULT_NTYPE]).any(), "Output contains Inf values"

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
def test_gat_with_no_indegree_dstnodes(input_dim, output_dim):
    # Create a homogeneous graph where some destination nodes have no in-degree
    from dgl.distributed.constants import DEFAULT_NTYPE
    import torch as th
    import dgl
    
    num_src_nodes = 1024
    num_dst_nodes = 256
    
    # Create edges where some dst nodes have no incoming edges
    src_nodes = th.randint(0, num_src_nodes, (500,))
    dst_nodes = th.randint(0, 128, (500,))  # Only first 128 dst nodes get edges
    
    block = dgl.create_block((src_nodes, dst_nodes), 
                           num_src_nodes=num_src_nodes, 
                           num_dst_nodes=num_dst_nodes)
    
    # Add required node IDs for GAT encoder
    block.nodes[DEFAULT_NTYPE].data[dgl.NID] = th.arange(num_src_nodes)
    
    inputs = {DEFAULT_NTYPE: th.randn((num_src_nodes, input_dim))}

    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=2, num_hidden_layers=0)
    blocks = [block]
    
    outputs = gat_encoder(blocks, inputs)
    
    assert outputs[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert outputs[DEFAULT_NTYPE].shape[1] == output_dim
    # Check output is not NaN or Inf
    assert not th.isnan(outputs[DEFAULT_NTYPE]).any(), "Output contains NaN values"
    assert not th.isinf(outputs[DEFAULT_NTYPE]).any(), "Output contains Inf values"

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_rgcn_with_zero_dstnodes(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_zero_dest_test_graph(input_dim)

    layer = RelGraphConvLayer(
            input_dim, output_dim, etypes,
            2, activation=th.nn.ReLU(), self_loop=True,
            dropout=0.1)
    outputs = layer(block, inputs)

    assert not 'n0' in outputs                      # No edge type to n0, so not in outputs
    assert outputs['n1'].shape[0] == 2              # 2 n1 destination nodes
    assert outputs['n1'].shape[1] == output_dim     # same as output dim
    assert outputs['n2'].shape[0] == 0              # 0 n2 destinnation node, and the (n2,r2,n1)
                                                    # edge exists in the block, so n2 will be in
                                                    # the outputs, but has 0 in the 1st dim, and
    assert outputs['n2'].shape[1] == output_dim     # output dim on the 2nd dim.

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_rgat_with_zero_dstnodes(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_zero_dest_test_graph(input_dim)

    layer = RelationalAttLayer(input_dim, output_dim, etypes,
                               2, activation=th.nn.ReLU(), self_loop=True,
                               dropout=0.1)
    outputs = layer(block, inputs)

    assert not 'n0' in outputs                      # No edge type to n0, so not in outputs
    assert outputs['n1'].shape[0] == 2              # 2 n1 destination nodes
    assert outputs['n1'].shape[1] == output_dim     # same as output dim
    assert outputs['n2'].shape[0] == 0              # 0 n2 destinnation nodes
    assert outputs['n2'].shape[1] == output_dim     # same as output

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_hgt_with_zero_dstnodes(input_dim, output_dim):
    block, inputs, ntypes, etypes = create_dummy_zero_dest_test_graph(input_dim)

    layer = HGTLayer(input_dim,
                     output_dim,
                     ntypes,
                     etypes,
                     num_heads=4)
    outputs = layer(block, inputs)

    assert not 'n0' in outputs                      # No edge type to n0, so not in outputs
    assert outputs['n1'].shape[0] == 2              # 2 n1 destination nodes
    assert outputs['n1'].shape[1] == output_dim     # same as output dim
    assert outputs['n2'].shape[0] == 0              # 0 n2 destinnation nodes
    assert outputs['n2'].shape[1] == output_dim     # same as output

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_gat_with_zero_dstnodes(input_dim, output_dim):
    # Create a homogeneous graph with zero destination nodes for some cases
    from dgl.distributed.constants import DEFAULT_NTYPE
    import torch as th
    import dgl
    
    num_src_nodes = 4
    num_dst_nodes = 2
    
    # Create edges
    src_nodes = th.tensor([0, 1, 2, 3])
    dst_nodes = th.tensor([0, 0, 1, 1])
    
    block = dgl.create_block((src_nodes, dst_nodes), 
                           num_src_nodes=num_src_nodes, 
                           num_dst_nodes=num_dst_nodes)
    
    # Add required node IDs for GAT encoder
    block.nodes[DEFAULT_NTYPE].data[dgl.NID] = th.arange(num_src_nodes)
    
    inputs = {DEFAULT_NTYPE: th.randn((num_src_nodes, input_dim))}

    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=2, num_hidden_layers=0)
    blocks = [block]
    
    outputs = gat_encoder(blocks, inputs)

    assert outputs[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert outputs[DEFAULT_NTYPE].shape[1] == output_dim
    # Check output is not NaN or Inf
    assert not th.isnan(outputs[DEFAULT_NTYPE]).any(), "Output contains NaN values"
    assert not th.isinf(outputs[DEFAULT_NTYPE]).any(), "Output contains Inf values"

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

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_rgat_with_edge_features(input_dim, output_dim, dev):
    """ Test the RelationalAttLayer that supports edge features """
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
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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

    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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

    # Test case 2: normal case, no edge features as inputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    emb2 = layer(block, node_feats)
    # check output numbers, and dimensions
    assert 'n0' not in emb2
    assert emb2['n1'].shape[0] == len(seeds['n1'])
    assert emb2['n1'].shape[1] == output_dim

    # the value of emb0, emb1 and emb2 should be different,
    # as emb0 integrates edge features from  both etypes, emb1 integrates edge features from
    # one etype, and emb2 does not integrate edge features
    assert not th.allclose(emb0['n1'], emb1['n1'], atol=0.001)
    assert not th.allclose(emb0['n1'], emb2['n1'], atol=0.001)
    assert not th.allclose(emb1['n1'], emb2['n1'], atol=0.001)

    # Test case 3: normal case, all 5 message passing ops
    # Test 3.1, "add" op
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
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

    # the value of emb31, emb32, emb33, and emb34 should be different
    assert not th.allclose(emb31['n1'], emb32['n1'], atol=0.001)
    assert not th.allclose(emb31['n1'], emb33['n1'], atol=0.001)
    assert not th.allclose(emb31['n1'], emb34['n1'], atol=0.001)
    assert not th.allclose(emb32['n1'], emb33['n1'], atol=0.001)
    assert not th.allclose(emb32['n1'], emb34['n1'], atol=0.001)
    assert not th.allclose(emb33['n1'], emb34['n1'], atol=0.001)

    # Test case 4: abnormal case, layer has no edge feature weights, but give edge features。
    #              this will trigger an assertion error to let users know that they need to use
    #              GATConvwithEdgeFeat.
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    with assert_raises(AssertionError):
        layer(block, node_feats, edge_feats)

    # Test case 5: abnormal case, layer has edge feature defined in initialization, but not give
    #              edge features in forward.
    #              This will trigger an assertion error of mismatch of the number of inputs.
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        activation=th.nn.ReLU(), self_loop=True,
        dropout=0.1)
    layer = layer.to(dev)

    with assert_raises(AssertionError):
        layer(block, node_feats)

    # Test case 6: normal case, checking forward results accuracy.
    #         we set all node and edge features to be 1s and all weights to be 1s.
    node_feats = {
        "n0": th.ones(src_idx.shape[0], input_dim).to(dev),
        "n1": th.ones(dst_idx.shape[0], input_dim).to(dev)
    }
    edge_feats = {
        ("n0", "r0", "n1"): th.ones(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.ones(r1_eid.shape[0], input_dim).to(dev)
    }

    # concat
    #     the output value for n1 should be: (input_sim * 2) * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    desired_emb6 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim * 2 + input_dim * 2)
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(), desired_emb6, decimal=5)

    # add
    #     the output value for n1 should be: (input_sim * 2) * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='add',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    desired_emb6 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim * 2 + input_dim * 2)
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(), desired_emb6, decimal=5)

    # sub
    #     the output value for n1 should be: 0s
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='sub',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    desired_emb6 = np.zeros([dst_idx.shape[0], output_dim])
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(), desired_emb6, decimal=5)

    # mul
    #     the output value for n1 should be: input_sim * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='mul',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    desired_emb6 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim + input_dim)
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(), desired_emb6, decimal=5)

    # div
    #     the output value for n1 should be: input_sim * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='div',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb6 = layer(block, node_feats, edge_feats)
    desired_emb6 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim + input_dim)
    assert_almost_equal(emb6['n1'].detach().cpu().numpy(), desired_emb6, decimal=5)

    # Test case 7: normal case, checking forward results accuracy.
    #         we set all node features to be 1s, all edge features to be 0s,
    #         and all weights to be 1s.
    node_feats = {
        "n0": th.ones(src_idx.shape[0], input_dim).to(dev),
        "n1": th.ones(dst_idx.shape[0], input_dim).to(dev)
    }
    edge_feats = {
        ("n0", "r0", "n1"): th.zeros(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.zeros(r1_eid.shape[0], input_dim).to(dev)
    }

    # concat
    #     the output value for n1 should be: input_sim * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb7 = layer(block, node_feats, edge_feats)
    desired_emb7 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim * 2)
    assert_almost_equal(emb7['n1'].detach().cpu().numpy(), desired_emb7, decimal=5)

    # add
    #     the output value for n1 should be: input_sim * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='add',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb7 = layer(block, node_feats, edge_feats)
    desired_emb7 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim * 2)
    assert_almost_equal(emb7['n1'].detach().cpu().numpy(), desired_emb7, decimal=5)

    # sub
    #     the output value for n1 should be: input_sim * num_etypes to 'n1'
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='sub',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb7 = layer(block, node_feats, edge_feats)
    desired_emb7 = np.ones([dst_idx.shape[0], output_dim]) * (input_dim * 2)
    assert_almost_equal(emb7['n1'].detach().cpu().numpy(), desired_emb7, decimal=5)

    # mul
    #     the output value for n1 should be: 0s
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='mul',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb7 = layer(block, node_feats, edge_feats)
    desired_emb7 = np.zeros([dst_idx.shape[0], output_dim])
    assert_almost_equal(emb7['n1'].detach().cpu().numpy(), desired_emb7, decimal=5)

    # div
    #     the output value for n1 should be: nan
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='div',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb7_n1 = layer(block, node_feats, edge_feats)['n1'].detach().cpu().numpy()
    assert emb7_n1.shape == (dst_idx.shape[0], output_dim)
    assert np.isnan(emb7_n1).all()

    # Test case 8: normal case, compare with using dgl.nn.GATConv without edge features
    #      sub-case 1.1: all edge features are 1s, 'mul' and 'div' make no difference, but
    #                    'add', 'sub', and 'concat' output differently.
    node_feats = {
        "n0": (th.ones(src_idx.shape[0], input_dim)*0.4).to(dev),
        "n1": (th.ones(src_idx.shape[0], input_dim)*0.7).to(dev)
    }
    edge_feats = {
        ("n0", "r0", "n1"): th.ones(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.ones(r1_eid.shape[0], input_dim).to(dev)
    }

    gat_layer_woefeat = RelationalAttLayer(
        (input_dim, input_dim), output_dim, etypes,
        num_heads=2,
        edge_feat_name=None,
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    gat_layer_woefeat = gat_layer_woefeat.to(dev)
    emb8_woefeat = gat_layer_woefeat(block, node_feats)

    # 'concat' operator, different outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='concat',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert np.not_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu().numpy()).any()

    # 'add' operator, different outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='add',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert np.not_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu().numpy()).any()

    # 'sub' operator, different outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='sub',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert np.not_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu().numpy()).any()

    # 'mul' operator, same outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='mul',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu(), decimal=3)

    # 'div' operator, same outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='div',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu(), decimal=3)

    #      sub-case 1.2: all edge features are 0s, 'add', 'sub', and 'concat' make no difference,
    #                    but 'mul' and 'div' output differently.
    edge_feats = {
        ("n0", "r0", "n1"): th.zeros(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.zeros(r1_eid.shape[0], input_dim).to(dev)
    }

    gat_layer_woefeat = RelationalAttLayer(
        (input_dim, input_dim), output_dim, etypes,
        num_heads=2,
        edge_feat_name=None,
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(gat_layer_woefeat.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    gat_layer_woefeat = gat_layer_woefeat.to(dev)
    emb8_woefeat = gat_layer_woefeat(block, node_feats)

    # 'concat' operator, same outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='concat',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu(), decimal=3)

    # 'add' operator, same outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='add',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu(), decimal=3)

    # 'sub' operator, same outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='sub',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert_almost_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu(), decimal=3)

    # 'mul' operator, different outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='mul',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert np.not_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu().numpy()).any()

    # 'div' operator, different outputs
    layer = RelationalAttLayer(
        input_dim, output_dim, etypes,
        num_heads=2,
        edge_feat_mp_op='div',
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        bias=False, activation=None, self_loop=False, dropout=0.0, norm=None)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r0', 'n1')).fc_dst.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_src.weight)
    th.nn.init.ones_(layer.conv._get_module(('n0', 'r1', 'n1')).fc_dst.weight)
    layer = layer.to(dev)

    emb8 = layer(block, node_feats, edge_feats)
    assert np.not_equal(emb8_woefeat['n1'].detach().cpu().numpy(),
                        emb8['n1'].detach().cpu().numpy()).any()

def init_hgtlayer(layer):
    """ Initialize an HGT layer to make it having all 1s weights, and all 0s biases.
    """
    for name, para in layer.named_parameters():
        if 'bias' in name or 'skip' in name:
            th.nn.init.constant_(para, 0)
        else:
            th.nn.init.constant_(para, 1)

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_hgt_with_edge_features(input_dim, output_dim, dev):
    """ Test the HGTLayerwithEdgeFeat module that supports edge features.

    Because HGT model is more complex than RGCN and it is hard to compute specific numeric
    values as outputs for layer testing, we use `HGTLayer` as the baseline to test the
    `HGTLayerwithEdgeFeat` class. The idea is:
        1. Preset the model parameters of `HGTLayer` and `HGTLayerwithEdgeFeat` to be the same,
           except for `edge_feat_mp_os==concat`, where `HGTLayerwithEdgeFeat` has one more set
           of parameters, `ef_linears`. Model weights will be all 1s, and biases will be all 0s.
        2. Set edge feature to be all 1s. With this setting, `mul` and `div` operators in
           `HGTLayerwithEdgeFeat` will have the same outputs as `HGTLayer`, meanwhile, `add`
           `sub` and `concat` will have different outputs.
        3. Set edge feature to be all 0s. With this setting, `add`, 'sub' and `concat` operators
           in `HGTLayerwithEdgeFeat` will have the same outputs as `HGTLayer`, meanwhile, `mul`
           and `div` will have different outputs.
    """
    # construct test block and input features
    heter_graph = generate_dummy_hetero_graph(size='tiny', gen_mask=False,
                                              add_reverse=False, is_random=False)

    # Test case 0: normal case, fix sub graph structure and input features.
    #              outputs are determistic for both HGTLayer and HGTLayerwithEdgeFeat
    seeds = {'n1': [0]}     # one dest node
    subg = dgl.sampling.sample_neighbors(heter_graph, seeds, 100)
    # subg has two edge types, r0 has 1 source node, hence one edge; r1 has 2 source nodes,
    # hence has two edges.

    block = dgl.to_block(subg, seeds).to(dev)

    ntypes = heter_graph.ntypes
    etypes = [("n0", "r0", "n1"), ("n0", "r1", "n1")]

    src1, dst1, r0_eid = subg.edges(form='all', etype='r0')
    src2, dst2, r1_eid = subg.edges(form='all', etype='r1')
    # r0 edge source nid: 66
    # r1 edge source nids: 13, 45

    src_idx = th.unique(th.concat([src1, src2]))
    dst_idx = th.unique(th.concat([dst1, dst2]))

    # set node feature input to be a fix value, all 1s
    node_feats = {
        "n0": th.ones(src_idx.shape[0], input_dim).to(dev),
        "n1": th.ones(dst_idx.shape[0], input_dim).to(dev)
    }

    hgt_layer = HGTLayer(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')                # MUST disable normalization. Pytorch v2.3 layernorm is diff v2.1
    # v2.3 output all 0s if input values are same in all dimensions
    init_hgtlayer(hgt_layer)
    hgt_layer = hgt_layer.to(dev)
    hgt_layer.eval()
    baseline_emb = hgt_layer(block, node_feats)
    # With all node features as 1s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5 after sigmoid, the output should be equal
    # to (in_dim * out_dim * 0.5 + in_dim * (1 - 0.5)) with N * out_dim shape. Here N=1 in this
    # case.
    target_val = input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    # set edge feature input to be a fix value, all 1s
    edge_feats = {
        ("n0", "r0", "n1"): th.ones(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.ones(r1_eid.shape[0], input_dim).to(dev)
    }

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='add',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    add_emb = layerwithef(block, node_feats, edge_feats)
    # With all node features as 1s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'add', the output should be equal to (2 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 2 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(add_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='concat',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    concat_emb = layerwithef(block, node_feats, edge_feats)
    # With all node features as 1s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'concat', the output should be equal to (2 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 2 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(concat_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='sub',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    sub_emb = layerwithef(block, node_feats, edge_feats)
    # With all node features as 1s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'sub', the output should be equal to (0 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 0 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(sub_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='mul',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    mul_emb = layerwithef(block, node_feats, edge_feats)
    # With all node features as 1s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'add', the output should be equal to (1 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 1 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(mul_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='div',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    div_emb = layerwithef(block, node_feats, edge_feats)
    # With all node features as 1s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'add', the output should be equal to (2 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 1 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(div_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    edge_feats_zeros = {
        ("n0", "r0", "n1"): th.zeros(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.zeros(r1_eid.shape[0], input_dim).to(dev)
    }

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='add',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    add_emb = layerwithef(block, node_feats, edge_feats_zeros)
    # With all node features as 1s, edge feature inputs as 0s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'add', the output should be equal to (1 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 1 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(add_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='concat',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    concat_emb = layerwithef(block, node_feats, edge_feats_zeros)
    # With all node features as 1s, edge feature inputs as 0s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'concat', the output should be equal to (1 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 1 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(concat_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='sub',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    sub_emb = layerwithef(block, node_feats, edge_feats_zeros)
    # With all node features as 1s, edge feature inputs as 0s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'sub', the output should be equal to (1 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 1 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(sub_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='mul',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    mul_emb = layerwithef(block, node_feats, edge_feats_zeros)
    # With all node features as 1s, edge feature inputs as 0s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'mul', the output should be equal to (0 * in_dim * out_dim * 0.5 + in_dim * (1 - 0.5))
    # with N * out_dim shape. Here N=1 in this case.
    target_val = 0 * input_dim * output_dim * 0.5 + input_dim * (1 - 0.5)
    assert_almost_equal(mul_emb['n1'].detach().cpu().numpy(),
                        np.ones([1, output_dim]) * target_val, decimal=5)

    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=1,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='div',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    div_emb = layerwithef(block, node_feats, edge_feats_zeros)
    # With all node features as 1s, edge feature inputs as 0s, and all weights are 1s in the shape of (in_dim, out_dim),
    # biases are 0s, skips are 0s, which will be 0.5s after sigmoid, and the 'edge_feat_mp_op'
    # is 'div', the output should be all nan with a shape of (1, output_dim)
    # with N * out_dim shape. Here N=1 in this case.

    output = div_emb['n1'].detach().cpu().numpy()
    assert output.shape == (1, output_dim)
    assert np.isnan(output).all()


    # Test case 1: normal case, have both node and edge feature on all node and edge types
    #      sub-case 1.1: all edge features are 1s, 'mul' and 'div' make no difference, but
    #                    'add', 'sub', and 'concat' output differently.
    seeds = {'n1': [0, 2]}
    subg = dgl.sampling.sample_neighbors(heter_graph, seeds, 100)
    block = dgl.to_block(subg, seeds).to(dev)

    ntypes = heter_graph.ntypes
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
        ("n0", "r0", "n1"): th.ones(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.ones(r1_eid.shape[0], input_dim).to(dev)
    }

    hgt_layer = HGTLayer(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')                # MUST disable normalization. Pytorch v2.3 layernorm is diff v2.1
    # v2.3 output all 0s if input values are same in all dimensions
    init_hgtlayer(hgt_layer)
    hgt_layer = hgt_layer.to(dev)
    hgt_layer.eval()
    baseline_emb = hgt_layer(block, node_feats)

    # 'mul' operator, same outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='mul',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    mul_emb = layerwithef(block, node_feats, edge_feats)
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        mul_emb['n1'].detach().cpu().numpy(), decimal=5)

    # 'div' operator, same outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='div',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    div_emb = layerwithef(block, node_feats, edge_feats)
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        div_emb['n1'].detach().cpu().numpy(), decimal=5)

    # 'add' operator, different outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='add',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    add_emb = layerwithef(block, node_feats, edge_feats)
    assert np.not_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        add_emb['n1'].detach().cpu().numpy()).any()

    # 'sub' operator, different outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='sub',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    sub_emb = layerwithef(block, node_feats, edge_feats)
    assert np.not_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        sub_emb['n1'].detach().cpu().numpy()).any()

    # 'concat' operator, different outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='concat',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    concat_emb = layerwithef(block, node_feats, edge_feats)
    assert np.not_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        concat_emb['n1'].detach().cpu().numpy()).any()

    #      sub-case 1.2: all edge features are 0s, 'add', 'sub', and 'concat' make no difference,
    #                    but 'mul' and 'div' output differently.
    edge_feats = {
        ("n0", "r0", "n1"): th.zeros(r0_eid.shape[0], input_dim).to(dev),
        ("n0", "r1", "n1"): th.zeros(r1_eid.shape[0], input_dim).to(dev)
    }

    hgt_layer = HGTLayer(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(hgt_layer)
    hgt_layer = hgt_layer.to(dev)
    hgt_layer.eval()
    baseline_emb = hgt_layer(block, node_feats)

    # 'concat' operator, same outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='concat',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    concat_emb = layerwithef(block, node_feats, edge_feats)
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        concat_emb['n1'].detach().cpu().numpy(), decimal=5)

    # 'add' operator, same outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='add',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    add_emb = layerwithef(block, node_feats, edge_feats)
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        add_emb['n1'].detach().cpu().numpy(), decimal=5)

    # 'sub' operator, same outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='sub',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    sub_emb = layerwithef(block, node_feats, edge_feats)
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        sub_emb['n1'].detach().cpu().numpy())

    # 'mul' operator, different outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='mul',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    mul_emb = layerwithef(block, node_feats, edge_feats)
    assert np.not_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        mul_emb['n1'].detach().cpu().numpy()).any()

    # 'div' operator, different outputs
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
        edge_feat_mp_op='div',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()
    div_emb = layerwithef(block, node_feats, edge_feats)
    assert np.not_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        div_emb['n1'].detach().cpu().numpy()).any()

    # test case 2: normal case, one edge type has features
    mp_ops = ['add', 'concat', 'sub', 'mul', 'div']
    for e_f_mp_op in mp_ops:
        # r0 case edge has feature, but r1 edge does not
        layerwithef = HGTLayerwithEdgeFeat(
            input_dim, output_dim,
            ntypes, etypes,
            num_heads=4,
            edge_feat_name={("n0", "r0", "n1"): ['feat']},
            edge_feat_mp_op=e_f_mp_op,
            activation=th.nn.ReLU(),
            dropout=0.0,
            norm='')
        init_hgtlayer(layerwithef)
        layerwithef = layerwithef.to(dev)
        layerwithef.eval()

        edge_feats = {
            ("n0", "r0", "n1"): th.rand(r0_eid.shape[0], input_dim).to(dev)
        }

        emb = layerwithef(block, node_feats, edge_feats)

        assert 'n0' not in emb
        assert emb['n1'].shape[0] == len(seeds['n1'])
        assert emb['n1'].shape[1] == output_dim

        # r1 case edge has features, but r0 edge does not.
        layerwithef = HGTLayerwithEdgeFeat(
            input_dim, output_dim,
            ntypes, etypes,
            num_heads=4,
            edge_feat_name={("n0", "r1", "n1"): ['feat']},
            edge_feat_mp_op=e_f_mp_op,
            activation=th.nn.ReLU(),
            dropout=0.0,
            norm='')
        init_hgtlayer(layerwithef)
        layerwithef = layerwithef.to(dev)
        layerwithef.eval()

        edge_feats = {
            ("n0", "r1", "n1"): th.rand(r1_eid.shape[0], input_dim).to(dev)
        }

        emb = layerwithef(block, node_feats, edge_feats)

        assert 'n0' not in emb
        assert emb['n1'].shape[0] == len(seeds['n1'])
        assert emb['n1'].shape[1] == output_dim

        # Test case 3: abnormal case, initialize HGTLayerwithEdgeFeat layer with no edge feature name.
        #               this will trigger an assertion error of empty edge
        with assert_raises(AssertionError):
            layerwithef = HGTLayerwithEdgeFeat(
                input_dim, output_dim,
                ntypes, etypes,
                num_heads=4,
                edge_feat_mp_op=e_f_mp_op,
                activation=th.nn.ReLU(),
                dropout=0.0,
                norm='')

        # Test case 4: abnormal case, initialize HGTLayer layer with edge feature name.
        #               this will trigger a type error of unexpected keyword argument
        with assert_raises(TypeError):
            layerwithef = HGTLayer(
                input_dim, output_dim,
                ntypes, etypes,
                num_heads=4,
                edge_feat_name={("n0", "r1", "n1"): ['feat']},
                edge_feat_mp_op=e_f_mp_op,
                activation=th.nn.ReLU(),
                dropout=0.0,
                norm='')

        # Test case 5: abnormal case, layer has edge feature weights, but give no edge features.
        #               this will trigger an assertion error of no edge features provided for message.
        layerwithef = HGTLayerwithEdgeFeat(
            input_dim, output_dim,
            ntypes, etypes,
            num_heads=4,
            edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
            edge_feat_mp_op=e_f_mp_op,
            activation=th.nn.ReLU(),
            dropout=0.0,
            norm='')
        init_hgtlayer(layerwithef)
        layerwithef = layerwithef.to(dev)
        layerwithef.eval()

        with assert_raises(AssertionError):
            layerwithef(block, node_feats)

        # Test case 6: abnormal case, initialize with 2 edge types but only provide 1 edge type in
        #               forward. Should work fine.
        layerwithef = HGTLayerwithEdgeFeat(
            input_dim, output_dim,
            ntypes, etypes,
            num_heads=4,
            edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
            edge_feat_mp_op=e_f_mp_op,
            activation=th.nn.ReLU(),
            dropout=0.0,
            norm='')
        init_hgtlayer(layerwithef)
        layerwithef = layerwithef.to(dev)
        layerwithef.eval()

        edge_feats = {
            ("n0", "r1", "n1"): th.rand(r1_eid.shape[0], input_dim).to(dev)
        }

        emb = layerwithef(block, node_feats, edge_feats)

        assert 'n0' not in emb
        assert emb['n1'].shape[0] == len(seeds['n1'])
        assert emb['n1'].shape[1] == output_dim

        # Test case 7: abnormal case, provide empty dict as edge feats. Should trigger an
        #               assertion error of no edge feature provided.
        layerwithef = HGTLayerwithEdgeFeat(
            input_dim, output_dim,
            ntypes, etypes,
            num_heads=4,
            edge_feat_name={("n0", "r0", "n1"): ['feat'], ("n0", "r1", "n1"): ['feat']},
            edge_feat_mp_op=e_f_mp_op,
            activation=th.nn.ReLU(),
            dropout=0.0,
            norm='')
        init_hgtlayer(layerwithef)
        layerwithef = layerwithef.to(dev)
        layerwithef.eval()

        edge_feats = {}

        with assert_raises(AssertionError):
            layerwithef(block, node_feats, edge_feats)

    # Test case 8: a corner case, layer was set with edge feature name, but the input block has 0
    #              number of edge types that should have edge features.
    #       8.1 abnormal case, set layer with  ('n0', 'r0', 'n1') edge feature name, but has >0
    #           input edges, but either don't provide edge features to forward or provide an emtpy
    #           dict. This will trigger an asserttion error.
    #       This has been tested in the Test Case 5 and the Test Case 7.

    #       8.2 normal case, set layer with ('n0', 'r0', 'n1') edge feature name, but 0 input edges
    #           by either not providing input edge feature or providing an empty dict

    # remove all edges in ('n0', 'r0', 'n1') edge type
    subg.remove_edges(th.tensor([0,1]), etype=('n0', 'r0', 'n1'))

    # collect new src and dst node ids
    src1, dst1, r0_eid = subg.edges(form='all', etype='r0')
    src2, dst2, r1_eid = subg.edges(form='all', etype='r1')

    src_idx = th.unique(th.concat([src1, src2]))
    dst_idx = th.unique(th.concat([dst1, dst2]))

    # set node feature input to be a fix value, all 1s
    node_feats = {
        "n0": th.ones(src_idx.shape[0], input_dim).to(dev),
        "n1": th.ones(dst_idx.shape[0], input_dim).to(dev)
    }

    # convert the new subgraph to a block with 0 number of edges in ('n0', 'r0', 'n1')
    block_zero_edge = dgl.to_block(subg, seeds).to(dev)

    # intialize hgt layer with ('n0', 'r0', 'n1') edge
    layerwithef = HGTLayerwithEdgeFeat(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        edge_feat_name={("n0", "r0", "n1"): ['feat']},
        edge_feat_mp_op='sub',
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(layerwithef)
    layerwithef = layerwithef.to(dev)
    layerwithef.eval()

    # method A: not provide input edge feature
    emb_a = layerwithef(block_zero_edge, node_feats)

    assert 'n0' not in emb_a
    assert emb_a['n1'].shape[0] == len(seeds['n1'])
    assert emb_a['n1'].shape[1] == output_dim

    # method B: provide an empty dict as input edge feature
    emb_b = layerwithef(block_zero_edge, node_feats, {})

    assert 'n0' not in emb_b
    assert emb_b['n1'].shape[0] == len(seeds['n1'])
    assert emb_b['n1'].shape[1] == output_dim

    # check the two inputs should generate the same output
    assert_almost_equal(emb_a['n1'].detach().cpu().numpy(),
                        emb_b['n1'].detach().cpu().numpy())

    # check this should be the same as HGTLayer withouth any edge features.
    hgt_layer = HGTLayer(
        input_dim, output_dim,
        ntypes, etypes,
        num_heads=4,
        activation=th.nn.ReLU(),
        dropout=0.0,
        norm='')
    init_hgtlayer(hgt_layer)
    hgt_layer = hgt_layer.to(dev)
    hgt_layer.eval()
    baseline_emb = hgt_layer(block_zero_edge, node_feats)

    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        emb_a['n1'].detach().cpu().numpy())
    assert_almost_equal(baseline_emb['n1'].detach().cpu().numpy(),
                        emb_b['n1'].detach().cpu().numpy())

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_gat_with_edge_features(input_dim, output_dim, dev):
    """ Test the GAT encoder that supports edge features """
    from dgl.distributed.constants import DEFAULT_NTYPE, DEFAULT_ETYPE
    import torch as th
    import dgl
    
    # Create a simple homogeneous graph
    num_src_nodes = 10
    num_dst_nodes = 5
    
    src_nodes = th.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    dst_nodes = th.tensor([0, 1, 2, 3, 0, 1, 2, 4])
    
    block = dgl.create_block((src_nodes, dst_nodes), 
                           num_src_nodes=num_src_nodes, 
                           num_dst_nodes=num_dst_nodes).to(dev)
    
    # Add required node IDs for GAT encoder
    block.nodes[DEFAULT_NTYPE].data[dgl.NID] = th.arange(num_src_nodes).to(dev)
    
    node_feats = {DEFAULT_NTYPE: th.rand(num_src_nodes, input_dim).to(dev)}
    edge_feats = {DEFAULT_ETYPE: th.rand(len(src_nodes), input_dim).to(dev)}

    # Test case 0: normal case, have both node and edge features
    edge_feat_name = {DEFAULT_ETYPE: ['feat']}
    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=2, 
                           num_hidden_layers=0, 
                           edge_feat_name=edge_feat_name)
    gat_encoder = gat_encoder.to(dev)

    blocks = [block]
    edge_feat_blocks = [edge_feats]
    
    emb0 = gat_encoder(blocks, node_feats, edge_feats=edge_feat_blocks)
    # check output numbers, dimensions and device
    assert emb0[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert emb0[DEFAULT_NTYPE].shape[1] == output_dim
    assert emb0[DEFAULT_NTYPE].get_device() == (-1 if dev == 'cpu' else 0)

    # Test case 1: normal case, no edge features as inputs
    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=2, num_hidden_layers=0)
    gat_encoder = gat_encoder.to(dev)

    emb1 = gat_encoder(blocks, node_feats)
    # check output numbers, and dimensions
    assert emb1[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert emb1[DEFAULT_NTYPE].shape[1] == output_dim

    # the value of emb0 and emb1 should be different,
    # as emb0 integrates edge features and emb1 does not
    assert not th.allclose(emb0[DEFAULT_NTYPE], emb1[DEFAULT_NTYPE], atol=0.001)

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_gat_different_head_counts(input_dim, output_dim, num_heads):
    """Test GAT with different numbers of attention heads"""
    from dgl.distributed.constants import DEFAULT_NTYPE
    import torch as th
    import dgl
    
    num_src_nodes = 256
    num_dst_nodes = 64
    
    src_nodes = th.randint(0, num_src_nodes, (200,))
    dst_nodes = th.randint(0, num_dst_nodes, (200,))
    
    block = dgl.create_block((src_nodes, dst_nodes), 
                           num_src_nodes=num_src_nodes, 
                           num_dst_nodes=num_dst_nodes)
    
    # Add required node IDs for GAT encoder
    block.nodes[DEFAULT_NTYPE].data[dgl.NID] = th.arange(num_src_nodes)
    
    inputs = {DEFAULT_NTYPE: th.randn((num_src_nodes, input_dim))}

    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=num_heads, num_hidden_layers=0)
    blocks = [block]
    
    outputs = gat_encoder(blocks, inputs)
    
    assert outputs[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert outputs[DEFAULT_NTYPE].shape[1] == output_dim
    # Check output is not NaN or Inf
    assert not th.isnan(outputs[DEFAULT_NTYPE]).any(), "Output contains NaN values"
    assert not th.isinf(outputs[DEFAULT_NTYPE]).any(), "Output contains Inf values"


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("num_hidden_layers", [0])  # Only test single layer for now
def test_gat_different_depths(input_dim, output_dim, num_hidden_layers):
    """Test GAT encoder with different numbers of hidden layers"""
    from dgl.distributed.constants import DEFAULT_NTYPE
    import torch as th
    import dgl
    
    num_src_nodes = 256
    num_dst_nodes = 64
    
    src_nodes = th.randint(0, num_src_nodes, (200,))
    dst_nodes = th.randint(0, num_dst_nodes, (200,))
    
    block = dgl.create_block((src_nodes, dst_nodes), 
                           num_src_nodes=num_src_nodes, 
                           num_dst_nodes=num_dst_nodes)
    
    # Add required node IDs for GAT encoder
    block.nodes[DEFAULT_NTYPE].data[dgl.NID] = th.arange(num_src_nodes)
    
    inputs = {DEFAULT_NTYPE: th.randn((num_src_nodes, input_dim))}

    gat_encoder = GATEncoder(input_dim, output_dim, num_heads=2, num_hidden_layers=num_hidden_layers)
    
    # Create appropriate number of blocks for the layers
    blocks = [block] * (num_hidden_layers + 1) if num_hidden_layers > 0 else [block]
    
    outputs = gat_encoder(blocks, inputs)
    
    assert outputs[DEFAULT_NTYPE].shape[0] == num_dst_nodes
    assert outputs[DEFAULT_NTYPE].shape[1] == output_dim
    # Check output is not NaN or Inf
    assert not th.isnan(outputs[DEFAULT_NTYPE]).any(), "Output contains NaN values"
    assert not th.isinf(outputs[DEFAULT_NTYPE]).any(), "Output contains Inf values"


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
def test_gat_edge_feature_support(input_dim, output_dim):
    """Test GAT encoder edge feature support detection"""
    # Test without edge features
    gat_encoder_no_edge = GATEncoder(input_dim, output_dim, num_heads=2, num_hidden_layers=1)
    assert not gat_encoder_no_edge.is_support_edge_feat(), "Should not support edge features without edge_feat_name"
    
    # Test with edge features
    from dgl.distributed.constants import DEFAULT_ETYPE
    edge_feat_name = {DEFAULT_ETYPE: ['feat']}
    gat_encoder_with_edge = GATEncoder(input_dim, output_dim, num_heads=2, 
                                     num_hidden_layers=1, edge_feat_name=edge_feat_name)
    assert gat_encoder_with_edge.is_support_edge_feat(), "Should support edge features with edge_feat_name"
