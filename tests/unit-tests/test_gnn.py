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
from pathlib import Path
import os
import yaml
import tempfile
import pytest
from argparse import Namespace
from types import MethodType

import torch as th
from torch import nn
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

import dgl

from graphstorm.config import GSConfig
from graphstorm.config import BUILTIN_LP_DOT_DECODER
from graphstorm.model import GSNodeEncoderInputLayer, RelationalGCNEncoder
from graphstorm.model import GSgnnNodeModel, GSgnnEdgeModel
from graphstorm.model import GSLMNodeEncoderInputLayer
from graphstorm.model import GSgnnLinkPredictionModel
from graphstorm.model.rgcn_encoder import RelationalGCNEncoder
from graphstorm.model.rgat_encoder import RelationalGATEncoder
from graphstorm.model.sage_encoder import SAGEEncoder
from graphstorm.model.edge_decoder import (DenseBiDecoder,
                                           MLPEdgeDecoder,
                                           MLPEFeatEdgeDecoder,
                                           LinkPredictDotDecoder,
                                           LinkPredictWeightedDotDecoder,
                                           LinkPredictWeightedDistMultDecoder)
from graphstorm.model.node_decoder import EntityRegression, EntityClassifier
from graphstorm.dataloading import GSgnnNodeTrainData, GSgnnEdgeTrainData
from graphstorm.dataloading import GSgnnNodeDataLoader, GSgnnEdgeDataLoader
from graphstorm import create_builtin_edge_gnn_model, create_builtin_node_gnn_model
from graphstorm import create_builtin_lp_gnn_model
from graphstorm import get_feat_size
from graphstorm.model.gnn import do_full_graph_inference
from graphstorm.model.node_gnn import node_mini_batch_predict, node_mini_batch_gnn_predict
from graphstorm.model.edge_gnn import edge_mini_batch_predict, edge_mini_batch_gnn_predict

from data_utils import generate_dummy_dist_graph
from data_utils import create_lm_graph

def is_int(a):
    if not th.is_floating_point(a) and not th.is_complex(a):
        return True
    return False

def create_rgcn_node_model(g):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGCNEncoder(g, 4, 4,
                                       num_bases=2,
                                       num_hidden_layers=1,
                                       dropout=0,
                                       use_self_loop=True)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_rgat_node_model(g):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGATEncoder(g, 4, 4,
                                       num_heads=2,
                                       num_hidden_layers=1,
                                       dropout=0,
                                       use_self_loop=True)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model
  
def create_sage_node_model(g):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = SAGEEncoder(4, 4,
                              num_hidden_layers=1,
                              dropout=0,
                              aggregator_type='mean')

    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model
  
def check_node_prediction(model, data, is_homo=False):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnNodeModel with GNN layers.

    Parameters
    ----------
    model: GSgnnNodeModel
        Node model
    data: GSgnnNodeTrainData
        Train data
    """
    g = data.g
    # do_full_graph_inference() runs differently if require_cache_embed()
    # returns different values. Here we simulate these two use cases and
    # triggers the different paths in do_full_graph_inference() to compute
    # embeddings. The embeddings computed by the two paths should be
    # numerically the same.
    assert not model.node_input_encoder.require_cache_embed()
    embs = do_full_graph_inference(model, data)
    def require_cache_embed(self):
        return True
    model.node_input_encoder.require_cache_embed = MethodType(require_cache_embed,
                                                              model.node_input_encoder)
    assert model.node_input_encoder.require_cache_embed()
    embs2 = do_full_graph_inference(model, data)
    assert len(embs) == len(embs2)
    for ntype in embs:
        assert ntype in embs2
        assert_almost_equal(embs[ntype][0:len(embs[ntype])].numpy(),
                            embs2[ntype][0:len(embs2[ntype])].numpy())

    embs3 = do_full_graph_inference(model, data, fanout=None)
    embs4 = do_full_graph_inference(model, data, fanout=[-1, -1])
    assert len(embs3) == len(embs4)
    for ntype in embs3:
        assert ntype in embs4
        assert_almost_equal(embs3[ntype][0:len(embs3[ntype])].numpy(),
                            embs4[ntype][0:len(embs4[ntype])].numpy())

    target_nidx = {"n1": th.arange(g.number_of_nodes("n0"))} \
        if not is_homo else {"_N": th.arange(g.number_of_nodes("_N"))}
    dataloader1 = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                      batch_size=10, device="cuda:0", train_task=False)
    pred1, labels1 = node_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnNodeDataLoader(data, target_nidx, fanout=[-1, -1],
                                      batch_size=10, device="cuda:0", train_task=False)
    pred2, _, labels2 = node_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    assert_almost_equal(pred1[0:len(pred1)].numpy(), pred2[0:len(pred2)].numpy(), decimal=5)
    assert_equal(labels1.numpy(), labels2.numpy())

    # Test the return_proba argument.
    pred3, labels3 = node_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    assert pred3.dim() == 2  # returns all predictions (2D tensor) when return_proba is true
    assert(th.is_floating_point(pred3))
    pred4, labels4 = node_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    assert(pred4.dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4))
    assert(th.equal(pred3.argmax(dim=1), pred4))

def check_mlp_node_prediction(model, data):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnNodeModel without GNN layers.

    Parameters
    ----------
    model: GSgnnNodeModel
        Node model
    data: GSgnnNodeTrainData
        Train data
    """
    g = data.g
    embs = do_full_graph_inference(model, data)
    target_nidx = {"n1": th.arange(g.number_of_nodes("n0"))}
    dataloader1 = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                      batch_size=10, device="cuda:0", train_task=False)
    pred1, labels1 = node_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                      batch_size=10, device="cuda:0", train_task=False)
    pred2, _, labels2 = node_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    assert_almost_equal(pred1[0:len(pred1)].numpy(), pred2[0:len(pred2)].numpy(), decimal=5)
    assert_equal(labels1.numpy(), labels2.numpy())

    # Test the return_proba argument.
    pred3, labels3 = node_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    assert pred3.dim() == 2  # returns all predictions (2D tensor) when return_proba is true
    assert(th.is_floating_point(pred3))
    pred4, labels4 = node_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    assert(pred4.dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4))
    assert(th.equal(pred3.argmax(dim=1), pred4))

def test_rgcn_node_prediction():
    """ Test edge prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + RGCNLayer + Decoder

        The test will compare the prediction results from full graph inference
        and mini-batch inference.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n1'], label_field='label',
                                     node_feat_field='feat')
    model = create_rgcn_node_model(np_data.g)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgat_node_prediction():
    """ Test edge prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + RGATLayer + Decoder

        The test will compare the prediction results from full graph inference
        and mini-batch inference.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['n1'], label_field='label',
                                     node_feat_field='feat')
    model = create_rgat_node_model(np_data.g)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_sage_node_prediction():
    """ Test edge prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + SAGELayer + Decoder

        The test will compare the prediction results from full graph inference
        and mini-batch inference.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname, is_homo=True)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                     train_ntypes=['_N'], label_field='label',
                                     node_feat_field='feat')
    model = create_sage_node_model(np_data.g)
    check_node_prediction(model, np_data, is_homo=True)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()
    
def create_rgcn_edge_model(g, num_ffn_layers):
    model = GSgnnEdgeModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGCNEncoder(g, 4, 4,
                                       num_bases=2,
                                       num_hidden_layers=1,
                                       dropout=0,
                                       use_self_loop=True)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(MLPEdgeDecoder(model.gnn_encoder.out_dims,
                                     3, multilabel=False, target_etype=("n0", "r1", "n1"),
                                     num_ffn_layers=num_ffn_layers))
    return model


def check_edge_prediction(model, data):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnEdgeModel with GNN layers.

    Parameters
    ----------
    model: GSgnnEdgeModel
        Node model
    data: GSgnnEdgeTrainData
        Train data
    """
    g = data.g
    embs = do_full_graph_inference(model, data)
    target_idx = {("n0", "r1", "n1"): th.arange(g.number_of_edges("r1"))}
    dataloader1 = GSgnnEdgeDataLoader(data, target_idx, fanout=[],
                                      batch_size=10, device="cuda:0", train_task=False,
                                      remove_target_edge_type=False)
    pred1, labels1 = edge_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnEdgeDataLoader(data, target_idx, fanout=[-1, -1],
                                      batch_size=10, device="cuda:0", train_task=False,
                                      remove_target_edge_type=False)
    pred2, labels2 = edge_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    assert_almost_equal(pred1[0:len(pred1)].numpy(), pred2[0:len(pred2)].numpy(), decimal=5)
    assert_equal(labels1.numpy(), labels2.numpy())

    # Test the return_proba argument.
    pred3, labels3 = edge_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    assert(th.is_floating_point(pred3))
    assert pred3.dim() == 2  # returns all predictions (2D tensor) when return_proba is true
    pred4, labels4 = edge_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    assert(pred4.dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4))
    assert(th.equal(pred3.argmax(dim=1), pred4))

def check_mlp_edge_prediction(model, data):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnEdgeModel without GNN layers.

    Parameters
    ----------
    model: GSgnnEdgeModel
        Node model
    data: GSgnnEdgeTrainData
        Train data
    """
    g = data.g
    embs = do_full_graph_inference(model, data)
    target_idx = {("n0", "r1", "n1"): th.arange(g.number_of_edges("r1"))}
    dataloader1 = GSgnnEdgeDataLoader(data, target_idx, fanout=[],
                                      batch_size=10, device="cuda:0", train_task=False,
                                      remove_target_edge_type=False)
    pred1, labels1 = edge_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnEdgeDataLoader(data, target_idx, fanout=[],
                                      batch_size=10, device="cuda:0", train_task=False,
                                      remove_target_edge_type=False)
    pred2, labels2 = edge_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    assert_almost_equal(pred1[0:len(pred1)].numpy(), pred2[0:len(pred2)].numpy(), decimal=5)
    assert_equal(labels1.numpy(), labels2.numpy())

    # Test the return_proba argument.
    pred3, labels3 = edge_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    assert pred3.dim() == 2  # returns all predictions (2D tensor) when return_proba is true
    assert(th.is_floating_point(pred3))
    pred4, labels4 = edge_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    assert(pred4.dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4))
    assert(th.equal(pred3.argmax(dim=1), pred4))

@pytest.mark.parametrize("num_ffn_layers", [0, 2])
def test_rgcn_edge_prediction(num_ffn_layers):
    """ Test edge prediction logic correctness with a edge prediction model
        composed of InputLayerEncoder + RGCNLayer + Decoder

        The test will compare the prediction results from full graph inference
        and mini-batch inference.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                     train_etypes=[('n0', 'r1', 'n1')], label_field='label',
                                     node_feat_field='feat')
    model = create_rgcn_edge_model(ep_data.g, num_ffn_layers=num_ffn_layers)
    check_edge_prediction(model, ep_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_mlp_edge_model(g, lm_config, num_ffn_layers):
    """ Create a GSgnnEdgeModel with only an input encoder and a decoder.

    Parameters
    ----------
    g: dgl.DistGraph
        Input graph.
    lm_config:
        Language model config

    Return
    ------
    GSgnnEdgeModel
    """
    model = GSgnnEdgeModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')

    encoder = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 2, num_train=0)
    model.set_node_input_encoder(encoder)

    model.set_decoder(MLPEdgeDecoder(model.node_input_encoder.out_dims,
                                     3, multilabel=False, target_etype=("n0", "r1", "n1"),
                                     num_ffn_layers=num_ffn_layers))
    return model

@pytest.mark.parametrize("num_ffn_layers", [0, 2])
def test_mlp_edge_prediction(num_ffn_layers):
    """ Test edge prediction logic correctness with a edge prediction model
        composed of InputLayerEncoder + Decoder

        The test will compare the prediction results from full graph inference
        and mini-batch inference.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, _, _, _, g, part_config = create_lm_graph(tmpdirname)
        ep_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                        train_etypes=[('n0', 'r1', 'n1')], label_field='label',
                                        node_feat_field='feat')
        g.edges['r1'].data['label']= ep_data.g.edges['r1'].data['label']
    model = create_mlp_edge_model(g, lm_config, num_ffn_layers=num_ffn_layers)
    assert model.gnn_encoder is None
    check_mlp_edge_prediction(model, ep_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_mlp_node_model(g, lm_config):
    """ Create a GSgnnNodeModel with only an input encoder and a decoder.

    Parameters
    ----------
    g: dgl.DistGraph
        Input graph.
    lm_config:
        Language model config

    Return
    ------
    GSgnnNodeModel
    """
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')

    encoder = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 2, num_train=0)
    model.set_node_input_encoder(encoder)

    model.set_decoder(EntityClassifier(model.node_input_encoder.out_dims, 3, False))
    return model

def test_mlp_node_prediction():
    """ Test node prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + Decoder

        The test will compare the prediction results from full graph inference
        and mini-batch inference.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, _, _, _, g, part_config = create_lm_graph(tmpdirname)
        np_data = GSgnnNodeTrainData(graph_name='dummy', part_config=part_config,
                                        train_ntypes=['n1'],
                                        label_field='label',
                                        node_feat_field='feat')
        g.nodes['n1'].data['label'] = np_data.g.nodes['n1'].data['label']
    model = create_mlp_node_model(g, lm_config)
    assert model.gnn_encoder is None
    check_mlp_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_mlp_lp_model(g, lm_config):
    """ Create a GSgnnLinkPredictionModel with only an input encoder and a decoder.

    Parameters
    ----------
    g: dgl.DistGraph
        Input graph.
    lm_config:
        Language model config

    Return
    ------
    GSgnnLinkPredictionModel
    """
    model = GSgnnLinkPredictionModel(alpha_l2norm=0)

    feat_size = get_feat_size(g, 'feat')
    encoder = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 2, num_train=0)
    model.set_node_input_encoder(encoder)

    model.set_decoder(LinkPredictDotDecoder(model.node_input_encoder.out_dims))
    return model


def test_mlp_link_prediction():
    """ Test full graph inference logic with a link prediciton model
        composed of InputLayerEncoder + Decoder
    """
    #  initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, _, _, _, g, part_config = create_lm_graph(tmpdirname)
        np_data = GSgnnEdgeTrainData(graph_name='dummy', part_config=part_config,
                                        train_etypes=[('n0', 'r1', 'n1')],
                                        eval_etypes=[('n0', 'r1', 'n1')],
                                        node_feat_field='feat')
    model = create_mlp_lp_model(g, lm_config)
    assert model.gnn_encoder is None
    embs = do_full_graph_inference(model, np_data)
    assert 'n0' in embs
    assert 'n1' in embs
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_ec_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "model_encoder_type": "rgcn",
                "lr": 0.001,
            },
            "input": {},
            "output": {},
            "rgcn": {
                "num_bases": 2,
            },
            "edge_classification": {
                "target_etype": ["n0,r0,n1"],
                "num_classes": 2,
                "decoder_type": "DenseBiDecoder",
                "multilabel": True,
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_edge_classification():
    """ Test logic of building a edge classification model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_ec_config(Path(tmpdirname), 'gnn_ec.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_ec.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_builtin_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGCNEncoder)
    assert isinstance(model.decoder, DenseBiDecoder)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_edge_classification_feat():
    """ Test logic of building a edge classification model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_ec_config(Path(tmpdirname), 'gnn_ec.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_ec.yaml'),
                         local_rank=0,
                         decoder_edge_feat=["feat"],
                         decoder_type="MLPEFeatEdgeDecoder")
        config = GSConfig(args)
    model = create_builtin_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGCNEncoder)
    assert isinstance(model.decoder, MLPEFeatEdgeDecoder)
    assert model.decoder.feat_dim == 2

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_ec_config(Path(tmpdirname), 'gnn_ec.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_ec.yaml'),
                         local_rank=0,
                         decoder_edge_feat=["n0,r0,n1:feat"],
                         decoder_type="MLPEFeatEdgeDecoder")
        config = GSConfig(args)
    model = create_builtin_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGCNEncoder)
    assert isinstance(model.decoder, MLPEFeatEdgeDecoder)
    assert model.decoder.feat_dim == 2

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        g.edges['r0'].data['feat1'] = g.edges['r0'].data['feat']
        create_ec_config(Path(tmpdirname), 'gnn_ec.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_ec.yaml'),
                         local_rank=0,
                         decoder_edge_feat=["n0,r0,n1:feat,feat1"],
                         decoder_type="MLPEFeatEdgeDecoder")
        config = GSConfig(args)
    model = create_builtin_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGCNEncoder)
    assert isinstance(model.decoder, MLPEFeatEdgeDecoder)
    assert model.decoder.feat_dim == 4
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_er_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "lr": 0.001,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "edge_regression": {
                "target_etype": ["n0,r0,n1"],
                "decoder_type": "DenseBiDecoder",
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_edge_regression():
    """ Test logic of building a edge regression model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_er_config(Path(tmpdirname), 'gnn_er.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_er.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_builtin_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, DenseBiDecoder)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_nr_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "lr": 0.001,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "node_regression": {
                "target_ntype": "n0",
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_node_regression():
    """ Test logic of building a node regression model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_nr_config(Path(tmpdirname), 'gnn_nr.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nr.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_builtin_node_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, EntityRegression)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_nc_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "lr": 0.001,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "node_classification": {
                "num_classes": 2,
                "target_ntype": "n0",
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_node_classification():
    """ Test logic of building a node classification model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_nc_config(Path(tmpdirname), 'gnn_nc.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_builtin_node_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, EntityClassifier)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_lp_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "lr": 0.001,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "link_prediction": {
                "train_etype": ["n0,r0,n1"],
                "lp_decoder_type": BUILTIN_LP_DOT_DECODER
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_link_prediction():
    """ Test logic of building a link prediction model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_lp_config(Path(tmpdirname), 'gnn_lp.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_builtin_lp_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, LinkPredictDotDecoder)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_link_prediction_weight():
    """ Test logic of building a link prediction model
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_lp_config(Path(tmpdirname), 'gnn_lp.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                         local_rank=0,
                         lp_edge_weight_for_loss=["weight"])
        config = GSConfig(args)
    model = create_builtin_lp_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, LinkPredictWeightedDotDecoder)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname)
        create_lp_config(Path(tmpdirname), 'gnn_lp.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                         local_rank=0,
                         lp_edge_weight_for_loss=["weight"],
                         lp_decoder_type="distmult",
                         train_etype=[("n0,r0,n1"), ("n0,r1,n1")])
        config = GSConfig(args)
    model = create_builtin_lp_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, LinkPredictWeightedDistMultDecoder)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

if __name__ == '__main__':
    test_rgcn_edge_prediction()
    test_rgcn_node_prediction()
    test_rgat_node_prediction()
    test_sage_node_prediction()
    test_edge_classification()
    test_edge_classification_feat()
    test_edge_regression()
    test_node_classification()
    test_node_regression()
    test_link_prediction()
    test_link_prediction_weight()

    test_mlp_edge_prediction()
    test_mlp_node_prediction()
    test_mlp_link_prediction()