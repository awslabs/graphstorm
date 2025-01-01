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

from graphstorm.dataloading import GSgnnData, GSgnnEdgeDataLoader
from graphstorm.model.rgcn_encoder import GraphConvwithEdgeFeat, RelationalGCNEncoder

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

        # Test 4: abnormal case, input edge feature lenght is smaller than num. of blocks
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

    # Test case 6: abnormal case, incorrect edge type string.
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


if __name__ == '__main__':
    test_rgcn_encoder_with_edge_features(32, 64, 'cpu')
    test_rgcn_encoder_with_edge_features(64, 64, 'cpu')
    test_rgcn_encoder_with_edge_features(32, 64, 'cuda:0')
