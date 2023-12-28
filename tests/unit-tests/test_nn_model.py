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

from graphstorm.model.rgat_encoder import RelationalAttLayer
from graphstorm.model.rgcn_encoder import RelGraphConvLayer
from graphstorm.model.hgt_encoder import HGTLayer

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

if __name__ == '__main__':
    test_rgcn_with_zero_input(32, 64)
    test_rgat_with_zero_input(32, 64)
    test_hgt_with_zero_input(32, 64)

    test_rgcn_with_no_indegree_dstnodes(32, 64)
    test_rgat_with_no_indegree_dstnodes(32, 64)
    test_hgt_with_no_indegree_dstnodes(32, 64)
