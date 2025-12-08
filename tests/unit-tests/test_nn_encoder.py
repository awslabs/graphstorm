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

    Test basic GNN encoders
"""
import pytest
import tempfile
import torch as th
import dgl
from numpy.testing import assert_raises

from graphstorm.dataloading import GSgnnData, GSgnnEdgeDataLoader, GSgnnNodeDataLoader
from graphstorm.model.rgcn_encoder import GraphConvwithEdgeFeat, RelationalGCNEncoder
from graphstorm.model.rgat_encoder import GATConvwithEdgeFeat, RelationalGATEncoder
from graphstorm.model.hgt_encoder import HGTEncoder, HGTLayer, HGTLayerwithEdgeFeat
from graphstorm.model.gat_encoder import GATEncoder

from data_utils import (generate_dummy_dist_graph)

def generate_dummy_features(input_feats, output_dim, feat_pattern='random', device='cpu'):
    """ generate dummy feature values to replace original feature values.
    
    Only support 2D features now.
    
    Parameters:
    ------------
    feat_pattern: str
        Options for feature patterns. Options are in ["random", "zeros". "ones"].
        Default is "random".
    """
    output_feats = {}
    for key, feats in input_feats.items():
        if feat_pattern == 'random':
            new_feats = th.rand(feats.shape[0], output_dim).to(device)
        elif feat_pattern == 'zeros':
            new_feats = th.zeros(feats.shape[0], output_dim).to(device)
        elif feat_pattern == 'ones':
            new_feats = th.ones(feats.shape[0], output_dim).to(device)
        else:   # do nothing
            new_feats = feats

        output_feats[key] = new_feats

    return output_feats

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_rgcn_encoder_with_edge_features(input_dim, output_dim, dev):
    """ Test the RelationalGCNEncoder that supports edge features
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                   dirname=tmpdirname, add_reverse=True,
                                                   is_random=False)

        # there will be three etypes:
        # ('n0', 'r1', 'n1'), ('n0', 'r0', 'n1'), ("n1", "r2", "n0")
        gdata = GSgnnData(part_config=part_config)

        # Test 1: normal case, two node types have features, two edge types have features,
        #         and one edge type ("n1", "r2", "n0") does not have features
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGCNEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 2
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['n0'].shape[-1] == output_dim
        assert emb1['n1'].shape[-1] == output_dim
        assert emb1['n0'].get_device() == (-1 if dev == 'cpu' else 0)
        assert emb1['n1'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 2: normal case, one edge type has features but one edge type does not
        #         have features.
        nfeat_fields = {'n0':['feat'], 'n1':['feat']}
        efeat_fields = {('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGCNEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        encoder = encoder.to(dev)
        emb2 = encoder(blocks, nfeats, efeats_list)
        assert emb2['n0'].shape[-1] == output_dim
        assert emb2['n1'].shape[-1] == output_dim

        # Test 3: normal case, two node types have features, no edge feature
        nfeat_fields = {'n0':['feat'], 'n1':['feat']}
        efeat_fields = None

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGCNEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r0', 'n1')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r1', 'n1')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        # no need of input edge features
        encoder = encoder.to(dev)
        emb3 = encoder(blocks, nfeats)
        assert emb3['n0'].shape[-1] == output_dim
        assert emb3['n1'].shape[-1] == output_dim

        # Test 4: abnormal case, input edge feature length is smaller than num. of blocks
        #         should trigger an assertion error
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats_list[0], input_dim, feat_pattern='random', device=dev)]
        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGCNEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        encoder = encoder.to(dev)
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, efeats_list)

        # Test 5: normal case, same as case 1, but one layer of GNN
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev) \
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGCNEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 1
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        encoder = encoder.to(dev)
        emb5 = encoder(blocks, nfeats, efeats_list)
        assert emb5['n0'].shape[-1] == output_dim
        assert emb5['n1'].shape[-1] == output_dim

        # Test 6: normal case, same as case 1, but 3 layers of GNN
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev) \
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGCNEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 3
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r0', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r1', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[1].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        assert isinstance(encoder.layers[2].conv._get_module(('n0', 'r0', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[2].conv._get_module(('n0', 'r1', 'n1')),
                          GraphConvwithEdgeFeat)
        assert isinstance(encoder.layers[2].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GraphConv)
        encoder = encoder.to(dev)
        emb5 = encoder(blocks, nfeats, efeats_list)
        assert emb5['n0'].shape[-1] == output_dim
        assert emb5['n1'].shape[-1] == output_dim

    # Test case 7: abnormal case, incorrect edge type string.
    #              Should trigger an assertion error
    efeat_fields = {'r0': ['feat'], 'r1': ['feat']}
    with assert_raises(AssertionError):
        encoder = RelationalGCNEncoder(gdata.g,
                                input_dim, output_dim,
                                num_hidden_layers=len(fanout)-1,
                                edge_feat_name=efeat_fields,
                                edge_feat_mp_op='concat')


    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_rgat_encoder_with_edge_features(input_dim, output_dim, dev):
    """ Test the RelationalGATEncoder that supports edge features
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                   dirname=tmpdirname, add_reverse=True,
                                                   is_random=False, add_reverse_efeat=True)

        # there will be three etypes:
        # ('n0', 'r1', 'n1'), ('n0', 'r0', 'n1'), ("n1", "r2", "n0")
        gdata = GSgnnData(part_config=part_config)
        # Test 0: normal case, two node types have features, three edge types have features
        nfeat_fields = {'n0': ['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat'],
                        ("n1", "r2", "n0"): ['feat']}
        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                         label_field='label',
                                         train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout) - 1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 2
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          GATConvwithEdgeFeat)
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['n0'].shape[-1] == output_dim
        assert emb1['n1'].shape[-1] == output_dim
        assert emb1['n0'].get_device() == (-1 if dev == 'cpu' else 0)
        assert emb1['n1'].get_device() == (-1 if dev == 'cpu' else 0)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                   dirname=tmpdirname, add_reverse=True,
                                                   is_random=False, add_reverse_efeat=False)

        # there will be three etypes:
        # ('n0', 'r1', 'n1'), ('n0', 'r0', 'n1'), ("n1", "r2", "n0")
        gdata = GSgnnData(part_config=part_config)

        # Test 1: normal case, two node types have features, two edge types have features,
        #         and one edge type ("n1", "r2", "n0") does not have features
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 2
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['n0'].shape[-1] == output_dim
        assert emb1['n1'].shape[-1] == output_dim
        assert emb1['n0'].get_device() == (-1 if dev == 'cpu' else 0)
        assert emb1['n1'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 2: normal case, one edge type has features but one edge type does not
        #         have features.
        nfeat_fields = {'n0': ['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                         label_field='label',
                                         train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout) - 1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        encoder = encoder.to(dev)
        emb2 = encoder(blocks, nfeats, efeats_list)
        assert emb2['n0'].shape[-1] == output_dim
        assert emb2['n1'].shape[-1] == output_dim

        # Test 3: normal case, two node types have features, no edge feature
        nfeat_fields = {'n0': ['feat'], 'n1': ['feat']}
        efeat_fields = None

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                         label_field='label',
                                         train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout) - 1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r0', 'n1')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r1', 'n1')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        # no need of input edge features
        encoder = encoder.to(dev)
        emb3 = encoder(blocks, nfeats)
        assert emb3['n0'].shape[-1] == output_dim
        assert emb3['n1'].shape[-1] == output_dim

        # Test 4: abnormal case, input edge feature length is smaller than num. of blocks
        #         should trigger an assertion error
        nfeat_fields = {'n0': ['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                         label_field='label',
                                         train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [
            generate_dummy_features(efeats_list[0], input_dim, feat_pattern='random', device=dev)]
        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout) - 1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        encoder = encoder.to(dev)
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, efeats_list)

        # Test 5: normal case, same as case 1, but one layer of GNN
        nfeat_fields = {'n0': ['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                         label_field='label',
                                         train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [
            generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev) \
            for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout) - 1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 1
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        encoder = encoder.to(dev)
        emb5 = encoder(blocks, nfeats, efeats_list)
        assert emb5['n0'].shape[-1] == output_dim
        assert emb5['n1'].shape[-1] == output_dim

        # Test 6: normal case, same as case 1, but 3 layer of GNN
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev) \
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = RelationalGATEncoder(gdata.g,
                                       input_dim, output_dim,
                                       num_heads=4,
                                       num_hidden_layers=len(fanout)-1,
                                       edge_feat_name=efeat_fields,
                                       edge_feat_mp_op='concat')
        assert len(encoder.layers) == 3
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r0', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[0].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r0', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[1].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[1].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        assert isinstance(encoder.layers[2].conv._get_module(('n0', 'r0', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[2].conv._get_module(('n0', 'r1', 'n1')),
                          GATConvwithEdgeFeat)
        assert isinstance(encoder.layers[2].conv._get_module(('n1', 'r2', 'n0')),
                          dgl.nn.GATConv)
        encoder = encoder.to(dev)
        emb5 = encoder(blocks, nfeats, efeats_list)
        assert emb5['n0'].shape[-1] == output_dim
        assert emb5['n1'].shape[-1] == output_dim

    # Test 7: abnormal case, incorrect edge type string.
    #              Should trigger an assertion error
    efeat_fields = {'r0': ['feat'], 'r1': ['feat']}
    with assert_raises(AssertionError):
        encoder = RelationalGATEncoder(gdata.g,
                                input_dim, output_dim,
                                num_heads=4,
                                num_hidden_layers=len(fanout)-1,
                                edge_feat_name=efeat_fields,
                                edge_feat_mp_op='concat')

    # after test pass, destroy all process group
    th.distributed.destroy_process_group()

@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32,64])
@pytest.mark.parametrize("dev", ['cpu','cuda:0'])
def test_hgt_encoder_with_edge_features(input_dim, output_dim, dev):
    """ Test the HGTEncoder that supports edge features
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy',
                                                   dirname=tmpdirname, add_reverse=True,
                                                   is_random=False)

        # there will be three etypes:
        # ('n0', 'r1', 'n1'), ('n0', 'r0', 'n1'), ("n1", "r2", "n0")
        gdata = GSgnnData(part_config=part_config)

        # Test 1: normal case, two node types have features, two edge types have features,
        #         and one edge type ("n1", "r2", "n0") does not have features
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')
        assert len(encoder.layers) == 2
        assert isinstance(encoder.layers[0],
                          HGTLayerwithEdgeFeat)
        assert isinstance(encoder.layers[1],
                          HGTLayerwithEdgeFeat)
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['n0'].shape[-1] == output_dim
        assert emb1['n1'].shape[-1] == output_dim
        assert emb1['n0'].get_device() == (-1 if dev == 'cpu' else 0)
        assert emb1['n1'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 2: normal case, one edge type has features but one edge type does not
        #         have features.
        nfeat_fields = {'n0':['feat'], 'n1':['feat']}
        efeat_fields = {('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]
        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')
        assert len(encoder.layers) == 2
        assert isinstance(encoder.layers[0],
                          HGTLayerwithEdgeFeat)
        assert isinstance(encoder.layers[1],
                          HGTLayerwithEdgeFeat)
        encoder = encoder.to(dev)
        emb2 = encoder(blocks, nfeats, efeats_list)
        assert emb2['n0'].shape[-1] == output_dim
        assert emb2['n1'].shape[-1] == output_dim

        # Test 3: normal case, two node types have features, no edge feature
        nfeat_fields = {'n0':['feat'], 'n1':['feat']}
        efeat_fields = None

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]
        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')
        assert len(encoder.layers) == 2
        assert isinstance(encoder.layers[0],
                          HGTLayer)
        assert isinstance(encoder.layers[1],
                          HGTLayer)
        # no need of input edge features
        encoder = encoder.to(dev)
        emb3 = encoder(blocks, nfeats)
        assert emb3['n0'].shape[-1] == output_dim
        assert emb3['n1'].shape[-1] == output_dim

        # Test 4: abnormal case, input edge feature length is smaller than num. of blocks
        #         should trigger an assertion error
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats_list[0], input_dim, feat_pattern='random', device=dev)]
        blocks = [block.to(dev) for block in blocks]
        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')
        encoder = encoder.to(dev)
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, efeats_list)

        # Test 5: normal case, same as case 1, but one layer of GNN
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev) \
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]
        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')
        assert len(encoder.layers) == 1
        assert isinstance(encoder.layers[0],
                          HGTLayerwithEdgeFeat)
        encoder = encoder.to(dev)
        emb5 = encoder(blocks, nfeats, efeats_list)
        assert emb5['n0'].shape[-1] == output_dim
        assert emb5['n1'].shape[-1] == output_dim

        # Test 6: normal case, same as case 1, but 3 layers of GNN
        nfeat_fields = {'n0':['feat'], 'n1': ['feat']}
        efeat_fields = {('n0', 'r0', 'n1'): ['feat'], ('n0', 'r1', 'n1'): ['feat']}

        fanout = [100, 100, 100]
        target_idx = {('n0', 'r1', 'n1'): [0, 1]}
        dataloader = GSgnnEdgeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False, remove_target_edge_type=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev) \
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')
        assert len(encoder.layers) == 3
        assert isinstance(encoder.layers[0],
                          HGTLayerwithEdgeFeat)
        assert isinstance(encoder.layers[1],
                          HGTLayerwithEdgeFeat)
        assert isinstance(encoder.layers[2],
                          HGTLayerwithEdgeFeat)
        encoder = encoder.to(dev)
        emb5 = encoder(blocks, nfeats, efeats_list)
        assert emb5['n0'].shape[-1] == output_dim
        assert emb5['n1'].shape[-1] == output_dim

    # Test case 7: abnormal case, incorrect edge type string.
    #              Should trigger an assertion error
    efeat_fields = {'r0': ['feat'], 'r1': ['feat']}
    with assert_raises(AssertionError):
        encoder = HGTEncoder(gdata.g,
                            input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            num_heads=1,
                            edge_feat_name=efeat_fields,
                            edge_feat_mp_op='concat',
                            dropout=0,
                            norm='')


    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32, 64])
@pytest.mark.parametrize("dev", ['cpu', 'cuda:0'])
def test_gat_encoder_homogeneous_with_edge_features(input_dim, output_dim, dev):
    """ Test the GATEncoder on homogeneous graphs with edge features
    """
    # initialize the torch distributed environment
    if not th.distributed.is_initialized():
        th.distributed.init_process_group(backend='gloo',
                                          init_method='tcp://127.0.0.1:23457',
                                          rank=0,
                                          world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy homogeneous distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_homo',
                                                   dirname=tmpdirname, 
                                                   is_homo=True,
                                                   is_random=False)

        # homogeneous graph has single node type '_N' and single edge type ('_N', '_E', '_N')
        gdata = GSgnnData(part_config=part_config)

        # Test 1: normal case, homogeneous graph with edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]


        blocks = [block.to(dev) for block in blocks]

        encoder = GATEncoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        assert len(encoder.layers) == 2
        assert encoder.is_support_edge_feat() == True
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['_N'].shape[-1] == output_dim
        assert emb1['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 2: normal case, homogeneous graph without edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = None

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = GATEncoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        assert len(encoder.layers) == 2
        assert encoder.is_support_edge_feat() == False
        # no need of input edge features
        encoder = encoder.to(dev)
        emb2 = encoder(blocks, nfeats)
        assert emb2['_N'].shape[-1] == output_dim
        assert emb2['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 3: normal case, single layer GNN with edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = GATEncoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        assert len(encoder.layers) == 1
        assert encoder.is_support_edge_feat() == True
        encoder = encoder.to(dev)
        emb3 = encoder(blocks, nfeats, efeats_list)
        assert emb3['_N'].shape[-1] == output_dim
        assert emb3['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 4: abnormal case, input edge feature length is smaller than num. of blocks
        #         should trigger an assertion error
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats_list[0], input_dim, feat_pattern='random', device=dev)]
        blocks = [block.to(dev) for block in blocks]

        encoder = GATEncoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        encoder = encoder.to(dev)
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, efeats_list)
    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32])
def test_gat_encoder_edge_feature_error_cases(input_dim, output_dim):
    """ Test GAT encoder error cases when edge features are expected but not provided
    """
    # Initialize distributed environment
    if not th.distributed.is_initialized():
        th.distributed.init_process_group(backend='gloo',
                                          init_method='tcp://127.0.0.1:23459',
                                          rank=0,
                                          world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test Case 1: Homogeneous graph - edge_feat_name is set but no edge features provided
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_homo',
                                                   dirname=tmpdirname, 
                                                   is_homo=True,
                                                   is_random=False)
        
        gdata = GSgnnData(part_config=part_config)
        
        # Set up encoder with edge feature name
        efeat_fields = {('_N', '_E', '_N'): ['feat']}
        encoder = GATEncoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=1,
                            edge_feat_name=efeat_fields)
        
        # Prepare node features but NO edge features
        nfeat_fields = {'_N': ['feat']}
        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device='cpu')
            break
        
        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device='cpu')
        blocks = [block.to('cpu') for block in blocks]
        
        # This should raise AssertionError because edge_feat_name is set but edge_feats is None
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, edge_feats=None)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test Case 2: Heterogeneous graph - should raise error with wrong edge type
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_hetero',
                                                   dirname=tmpdirname,
                                                   is_homo=False,
                                                   add_reverse=True,
                                                   is_random=False)
        
        gdata = GSgnnData(part_config=part_config)
        
        # Try to create GAT encoder with heterogeneous edge type (should fail)
        efeat_fields_hetero = {('n0', 'r0', 'n1'): ['feat']}
        
        # This should raise AssertionError because GAT only supports homogeneous graphs
        with assert_raises(AssertionError):
            encoder = GATEncoder(input_dim, output_dim,
                                num_heads=4,
                                num_hidden_layers=1,
                                edge_feat_name=efeat_fields_hetero)
    
    # Destroy process group after tests
    if th.distributed.is_initialized():
        th.distributed.destroy_process_group()


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32, 64])
@pytest.mark.parametrize("dev", ['cpu', 'cuda:0'])
def test_gatv2_encoder_homogeneous_with_edge_features(input_dim, output_dim, dev):
    """ Test the GATv2Encoder on homogeneous graphs with edge features
    """
    from graphstorm.model.gatv2_encoder import GATv2Encoder
    # initialize the torch distributed environment
    if not th.distributed.is_initialized():
        th.distributed.init_process_group(backend='gloo',
                                          init_method='tcp://127.0.0.1:23459',
                                          rank=0,
                                          world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy homogeneous distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_homo',
                                                   dirname=tmpdirname, 
                                                   is_homo=True,
                                                   is_random=False)

        # homogeneous graph has single node type '_N' and single edge type ('_N', '_E', '_N')
        gdata = GSgnnData(part_config=part_config)

        # Test 1: normal case, homogeneous graph with edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = GATv2Encoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        assert len(encoder.layers) == 2
        assert encoder.is_support_edge_feat() == True
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['_N'].shape[-1] == output_dim
        assert emb1['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 2: normal case, homogeneous graph without edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = None

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = GATv2Encoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        assert len(encoder.layers) == 2
        assert encoder.is_support_edge_feat() == False
        # no need of input edge features
        encoder = encoder.to(dev)
        emb2 = encoder(blocks, nfeats)
        assert emb2['_N'].shape[-1] == output_dim
        assert emb2['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 3: normal case, single layer GNN with edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = GATv2Encoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        assert len(encoder.layers) == 1
        assert encoder.is_support_edge_feat() == True
        encoder = encoder.to(dev)
        emb3 = encoder(blocks, nfeats, efeats_list)
        assert emb3['_N'].shape[-1] == output_dim
        assert emb3['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 4: abnormal case, input edge feature length is smaller than num. of blocks
        #         should trigger an assertion error
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats_list[0], input_dim, feat_pattern='random', device=dev)]
        blocks = [block.to(dev) for block in blocks]

        encoder = GATv2Encoder(input_dim, output_dim,
                            num_heads=4,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields)
        encoder = encoder.to(dev)
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, efeats_list)
    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32])
def test_gatv2_encoder_edge_feature_error_cases(input_dim, output_dim):
    """ Test GATv2 encoder error cases when edge features are expected but not provided
    """
    from graphstorm.model.gatv2_encoder import GATv2Encoder
    
    # Initialize distributed environment
    if not th.distributed.is_initialized():
        th.distributed.init_process_group(backend='gloo',
                                          init_method='tcp://127.0.0.1:23460',
                                          rank=0,
                                          world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test Case 1: Homogeneous graph - edge_feat_name is set but no edge features provided
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_homo',
                                                   dirname=tmpdirname, 
                                                   is_homo=True,
                                                   is_random=False)
        
        gdata = GSgnnData(part_config=part_config)
        
        # Set up encoder with edge feature name
        efeat_fields = {('_N', '_E', '_N'): ['feat']}
        encoder = GATv2Encoder(input_dim, output_dim,
                              num_heads=4,
                              num_hidden_layers=1,
                              edge_feat_name=efeat_fields)
        
        # Prepare node features but NO edge features
        nfeat_fields = {'_N': ['feat']}
        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device='cpu')
            break
        
        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device='cpu')
        blocks = [block.to('cpu') for block in blocks]
        
        # This should raise AssertionError because edge_feat_name is set but edge_feats is None
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, edge_feats=None)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test Case 2: Heterogeneous graph - should raise error with wrong edge type
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_hetero',
                                                   dirname=tmpdirname,
                                                   is_homo=False,
                                                   add_reverse=True,
                                                   is_random=False)
        
        gdata = GSgnnData(part_config=part_config)
        
        # Try to create GATv2 encoder with heterogeneous edge type (should fail)
        efeat_fields_hetero = {('n0', 'r0', 'n1'): ['feat']}
        
        # This should raise AssertionError because GATv2 only supports homogeneous graphs
        with assert_raises(AssertionError):
            encoder = GATv2Encoder(input_dim, output_dim,
                                  num_heads=4,
                                  num_hidden_layers=1,
                                  edge_feat_name=efeat_fields_hetero)
    
    # Destroy process group after tests
    if th.distributed.is_initialized():
        th.distributed.destroy_process_group()



@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32, 64])
@pytest.mark.parametrize("dev", ['cpu', 'cuda:0'])
@pytest.mark.parametrize("aggregator_type", ['mean', 'gcn', 'pool', 'lstm'])
def test_sage_encoder_homogeneous_with_edge_features(input_dim, output_dim, dev, aggregator_type):
    """ Test the SAGEEncoder on homogeneous graphs with edge features and different aggregation types
    """
    from graphstorm.model.sage_encoder import SAGEEncoder
    # initialize the torch distributed environment
    if not th.distributed.is_initialized():
        th.distributed.init_process_group(backend='gloo',
                                          init_method='tcp://127.0.0.1:23458',
                                          rank=0,
                                          world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy homogeneous distributed graph
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_homo',
                                                   dirname=tmpdirname, 
                                                   is_homo=True,
                                                   is_random=False)

        # homogeneous graph has single node type '_N' and single edge type ('_N', '_E', '_N')
        gdata = GSgnnData(part_config=part_config)

        # Test 1: normal case, homogeneous graph with edge features and specified aggregator
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = SAGEEncoder(input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields,
                            aggregator_type=aggregator_type)
        assert len(encoder.layers) == 2
        assert encoder.is_support_edge_feat() == True
        encoder = encoder.to(dev)
        emb1 = encoder(blocks, nfeats, efeats_list)
        assert emb1['_N'].shape[-1] == output_dim
        assert emb1['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 2: normal case, homogeneous graph without edge features
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = None

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = SAGEEncoder(input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields,
                            aggregator_type=aggregator_type)
        assert len(encoder.layers) == 2
        assert encoder.is_support_edge_feat() == False
        # no need of input edge features
        encoder = encoder.to(dev)
        emb2 = encoder(blocks, nfeats)
        assert emb2['_N'].shape[-1] == output_dim
        assert emb2['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 3: normal case, single layer GNN with edge features and specified aggregator
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats, input_dim, feat_pattern='random', device=dev)
                       for efeats in efeats_list]

        blocks = [block.to(dev) for block in blocks]

        encoder = SAGEEncoder(input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields,
                            aggregator_type=aggregator_type)
        assert len(encoder.layers) == 1
        assert encoder.is_support_edge_feat() == True
        encoder = encoder.to(dev)
        emb3 = encoder(blocks, nfeats, efeats_list)
        assert emb3['_N'].shape[-1] == output_dim
        assert emb3['_N'].get_device() == (-1 if dev == 'cpu' else 0)

        # Test 4: abnormal case, input edge feature length is smaller than num. of blocks
        #         should trigger an assertion error
        nfeat_fields = {'_N': ['feat']}
        efeat_fields = {('_N', '_E', '_N'): ['feat']}

        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device=dev)
            efeats_list = gdata.get_blocks_edge_feats(blocks, efeat_fields, device=dev)

        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device=dev)
        efeats_list = [generate_dummy_features(efeats_list[0], input_dim, feat_pattern='random', device=dev)]
        blocks = [block.to(dev) for block in blocks]

        encoder = SAGEEncoder(input_dim, output_dim,
                            num_hidden_layers=len(fanout)-1,
                            edge_feat_name=efeat_fields,
                            aggregator_type=aggregator_type)
        encoder = encoder.to(dev)
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, efeats_list)
    # after test pass, destroy all process group
    th.distributed.destroy_process_group()


@pytest.mark.parametrize("input_dim", [32])
@pytest.mark.parametrize("output_dim", [32])
def test_sage_encoder_edge_feature_error_cases(input_dim, output_dim):
    """ Test SAGE encoder error cases when edge features are expected but not provided
    """
    from graphstorm.model.sage_encoder import SAGEEncoder
    
    # Initialize distributed environment
    if not th.distributed.is_initialized():
        th.distributed.init_process_group(backend='gloo',
                                          init_method='tcp://127.0.0.1:23461',
                                          rank=0,
                                          world_size=1)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test Case 1: Homogeneous graph - edge_feat_name is set but no edge features provided
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_homo',
                                                   dirname=tmpdirname, 
                                                   is_homo=True,
                                                   is_random=False)
        
        gdata = GSgnnData(part_config=part_config)
        
        # Set up encoder with edge feature name
        efeat_fields = {('_N', '_E', '_N'): ['feat']}
        encoder = SAGEEncoder(input_dim, output_dim,
                             num_hidden_layers=1,
                             edge_feat_name=efeat_fields,
                             aggregator_type='mean')
        
        # Prepare node features but NO edge features
        nfeat_fields = {'_N': ['feat']}
        fanout = [100, 100]
        target_idx = {'_N': [0, 1]}
        dataloader = GSgnnNodeDataLoader(gdata, target_idx, fanout, 10,
                                        label_field='label',
                                        train_task=False)
        
        for input_nodes, _, blocks in dataloader:
            nfeats = gdata.get_node_feats(input_nodes, nfeat_fields, device='cpu')
            break
        
        nfeats = generate_dummy_features(nfeats, input_dim, feat_pattern='random', device='cpu')
        blocks = [block.to('cpu') for block in blocks]
        
        # This should raise AssertionError because edge_feat_name is set but edge_feats is None
        with assert_raises(AssertionError):
            encoder(blocks, nfeats, edge_feats=None)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Test Case 2: Heterogeneous graph - should raise error with wrong edge type
        _, part_config = generate_dummy_dist_graph(graph_name='dummy_hetero',
                                                   dirname=tmpdirname,
                                                   is_homo=False,
                                                   add_reverse=True,
                                                   is_random=False)
        
        gdata = GSgnnData(part_config=part_config)
        
        # Try to create SAGE encoder with heterogeneous edge type (should fail)
        efeat_fields_hetero = {('n0', 'r0', 'n1'): ['feat']}
        
        # This should raise AssertionError because SAGE only supports homogeneous graphs
        with assert_raises(AssertionError):
            encoder = SAGEEncoder(input_dim, output_dim,
                                 num_hidden_layers=1,
                                 edge_feat_name=efeat_fields_hetero,
                                 aggregator_type='mean')
    
    # Destroy process group after tests
    if th.distributed.is_initialized():
        th.distributed.destroy_process_group()
