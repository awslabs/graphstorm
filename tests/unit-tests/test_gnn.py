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
import copy
import os
import yaml
import tempfile
import pytest
from argparse import Namespace
from types import MethodType
from unittest.mock import patch
from dgl.distributed.constants import DEFAULT_NTYPE

import torch as th
from torch import nn
import numpy as np
import torch.nn.functional as F
from numpy.testing import assert_almost_equal, assert_equal

import dgl
from dgl.distributed import DistTensor

from graphstorm.config import GSConfig, TaskInfo
from graphstorm.config import BUILTIN_LP_DOT_DECODER
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION,
                                BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                                GRAPHSTORM_LP_EMB_L2_NORMALIZATION)
from graphstorm.model import GSNodeEncoderInputLayer, RelationalGCNEncoder
from graphstorm.model import GSgnnNodeModel, GSgnnEdgeModel
from graphstorm.model import GSLMNodeEncoderInputLayer, GSPureLMNodeInputLayer
from graphstorm.model import GSgnnLinkPredictionModel
from graphstorm.model.gnn_with_reconstruct import GNNEncoderWithReconstructedEmbed
from graphstorm.model.rgcn_encoder import RelationalGCNEncoder, RelGraphConvLayer
from graphstorm.model.rgat_encoder import RelationalGATEncoder
from graphstorm.model.sage_encoder import SAGEEncoder
from graphstorm.model.gat_encoder import GATEncoder
from graphstorm.model.gatv2_encoder import GATv2Encoder
from graphstorm.model.hgt_encoder import HGTEncoder
from graphstorm.model.edge_decoder import (DenseBiDecoder,
                                           MLPEdgeDecoder,
                                           MLPEFeatEdgeDecoder,
                                           LinkPredictDotDecoder,
                                           LinkPredictWeightedDotDecoder,
                                           LinkPredictWeightedDistMultDecoder)
from graphstorm.model.node_decoder import EntityRegression, EntityClassifier
from graphstorm.model.loss_func import RegressionLossFunc
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import GSgnnNodeDataLoader, GSgnnEdgeDataLoader
from graphstorm.dataloading.dataset import prepare_batch_input
from graphstorm import (create_builtin_edge_gnn_model,
                        create_builtin_node_gnn_model,
                        create_builtin_lp_gnn_model,
                        create_builtin_reconstruct_nfeat_decoder)
from graphstorm import get_node_feat_size
from graphstorm.gsf import get_rel_names_for_reconstruct
from graphstorm.model import do_full_graph_inference, do_mini_batch_inference
from graphstorm.model.node_gnn import (node_mini_batch_predict,
                                       run_node_mini_batch_predict,
                                       node_mini_batch_gnn_predict)
from graphstorm.model.node_gnn import GSgnnNodeModelInterface
from graphstorm.model.edge_gnn import (edge_mini_batch_predict,
                                       run_edge_mini_batch_predict,
                                       edge_mini_batch_gnn_predict)
from graphstorm.model.multitask_gnn import (multi_task_mini_batch_predict,
                                            gen_emb_for_nfeat_reconstruct)
from graphstorm.model.gnn_with_reconstruct import construct_node_feat, get_input_embeds_combined
from graphstorm.model.utils import load_model, save_model
from graphstorm.model import GSgnnMultiTaskSharedEncoderModel

from util import (DummyGSgnnNodeDataLoader,
                  DummyGSgnnEdgeDataLoader,
                  DummyGSgnnLinkPredictionDataLoader,
                  DummyGSgnnModel,
                  DummyGSgnnEncoderModel)

from data_utils import generate_dummy_dist_graph, generate_dummy_dist_graph_multi_target_ntypes
from data_utils import generate_dummy_dist_graph_reconstruct
from data_utils import create_lm_graph, create_lm_graph2

def is_int(a):
    if not th.is_floating_point(a) and not th.is_complex(a):
        return True
    return False

def create_rgcn_node_model(g, norm=None):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGCNEncoder(g, 4, 4,
                                       num_bases=2,
                                       num_hidden_layers=1,
                                       dropout=0,
                                       use_self_loop=True,
                                       norm=norm)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_rgcn_node_model_with_reconstruct(data, reconstructed_embed_ntype,
        cache_embed=False, lm_configs=None):
    model = GSgnnNodeModel(alpha_l2norm=0)

    g = data.g
    node_feats = data.node_feat_field
    feat_size = get_node_feat_size(g, node_feats)
    if lm_configs is None:
        encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                          dropout=0,
                                          use_node_embeddings=False,
                                          force_no_embeddings=reconstructed_embed_ntype,
                                          cache_embed=cache_embed)
    else:
        encoder = GSLMNodeEncoderInputLayer(g, lm_configs, feat_size, 4,
                                            num_train=0,
                                            force_no_embeddings=reconstructed_embed_ntype)
    feat_size = encoder.in_dims
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGCNEncoder(g, 4, 4,
                                       num_bases=2,
                                       num_hidden_layers=0,
                                       dropout=0,
                                       use_self_loop=True)
    rel_names = get_rel_names_for_reconstruct(g, reconstructed_embed_ntype, feat_size)
    assert len(rel_names) > 0
    dst_types = set([rel_name[2] for rel_name in rel_names])
    for ntype in reconstructed_embed_ntype:
        assert ntype in dst_types, \
                f"We cannot reconstruct features of node {ntype} " \
                + "probably because their neighbors don't have features."
    input_gnn = RelGraphConvLayer(4, 4,
                                  rel_names, len(rel_names),
                                  activation=F.relu,
                                  self_loop=False)
    gnn_encoder = GNNEncoderWithReconstructedEmbed(gnn_encoder, input_gnn, rel_names)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_rgat_node_model(g, norm=None):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGATEncoder(g, 4, 4,
                                       num_heads=2,
                                       num_hidden_layers=1,
                                       dropout=0,
                                       use_self_loop=True,
                                       norm=norm)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_hgt_node_model(g):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = HGTEncoder(g,
                            hid_dim=4,
                            out_dim=4,
                            num_hidden_layers=1,
                            num_heads=2,
                            dropout=0.0,
                            norm='layer',
                            num_ffn_layers_in_gnn=0)
    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_sage_node_model(g, norm=None):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = SAGEEncoder(4, 4,
                              num_hidden_layers=1,
                              dropout=0,
                              aggregator_type='mean',
                              norm=norm)

    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_gat_node_model(g):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 8,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = GATEncoder(8, 8, 4,
                             num_hidden_layers=1,
                             dropout=0)

    model.set_gnn_encoder(gnn_encoder)
    model.set_decoder(EntityClassifier(model.gnn_encoder.out_dims, 3, False))
    return model

def create_gatv2_node_model(g):
    model = GSgnnNodeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 8,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = GATv2Encoder(8, 8, 4,
                             num_hidden_layers=1,
                             dropout=0)

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
    data: GSgnnData
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
        if not is_homo else {DEFAULT_NTYPE: th.arange(g.number_of_nodes(DEFAULT_NTYPE))}
    dataloader1 = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                      batch_size=10, label_field='label',
                                      node_feats='feat', train_task=False)
    pred1, labels1 = node_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnNodeDataLoader(data, target_nidx, fanout=[-1, -1],
                                      batch_size=10, label_field='label',
                                      node_feats='feat', train_task=False)
    # Call GNN mini-batch inference
    pred2_gnn_pred, _, labels2_gnn_pred, = node_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    # Call last layer mini-batch inference with the GNN dataloader
    pred2_pred, labels2_pred = node_mini_batch_predict(model, embs, dataloader2, return_label=True)

    pred2_d_pred, labels2_d_pred = run_node_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_label=True)

    if isinstance(pred1,dict):
        assert len(pred1) == len(pred2_gnn_pred) and len(labels1) == len(labels2_gnn_pred)
        assert len(pred1) == len(pred2_pred) and len(labels1) == len(labels2_pred)
        assert len(pred1) == len(pred2_d_pred) and len(labels1) == len(labels2_gnn_pred)
        for ntype in pred1:
            assert_almost_equal(pred1[ntype][0:len(pred1)].numpy(),
                                pred2_gnn_pred[ntype][0:len(pred2_gnn_pred)].numpy(), decimal=5)
            assert_equal(labels1[ntype].numpy(), labels2_gnn_pred[ntype].numpy())
            assert_almost_equal(pred1[ntype][0:len(pred1)].numpy(),
                                pred2_pred[ntype][0:len(pred2_pred)].numpy(), decimal=5)
            assert_equal(labels1[ntype].numpy(), labels2_pred[ntype].numpy())
            assert_almost_equal(pred1[ntype][0:len(pred1)].numpy(),
                                pred2_d_pred[ntype][0:len(pred2_d_pred)].numpy())
            assert_equal(labels1[ntype].numpy(), labels2_d_pred[ntype].numpy())
    else:
        assert_almost_equal(pred1[0:len(pred1)].numpy(),
                            pred2_gnn_pred[0:len(pred2_gnn_pred)].numpy(), decimal=5)
        assert_equal(labels1.numpy(), labels2_gnn_pred.numpy())
        assert_almost_equal(pred1[0:len(pred1)].numpy(),
                            pred2_pred[0:len(pred2_pred)].numpy(), decimal=5)
        assert_equal(labels1.numpy(), labels2_pred.numpy())
        assert_almost_equal(pred1[0:len(pred1)].numpy(),
                            labels2_d_pred[0:len(labels2_d_pred)].numpy())
        assert_equal(labels1.numpy(), labels2_d_pred.numpy())

    # Test the return_proba argument.
    pred3, labels3 = node_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    pred3_d, labels3_d = run_node_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=True, return_label=True)

    pred4, labels4 = node_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    pred4_d, labels4_d = run_node_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=False, return_label=True)
    if isinstance(pred3, dict):
        assert len(pred3) == len(pred4) and len(labels3) == len(labels4)
        assert len(pred3) == len(pred3_d) and len(labels3) == len(labels3_d)
        assert len(pred4) == len(pred4_d) and len(labels4) == len(labels4_d)
        for key in pred3:
            assert pred3[key].dim() == 2  # returns all predictions (2D tensor) when return_proba is true
            assert(th.is_floating_point(pred3[key]))
            assert pred3_d[key].dim() == 2
            assert(th.is_floating_point(pred3_d[key]))
            assert(pred4[key].dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
            assert(is_int(pred4[key]))
            assert(th.equal(pred3[key].argmax(dim=1), pred4[key]))
            assert(pred4_d[key].dim() == 1)
            assert(is_int(pred4_d[key]))
            assert(th.equal(pred3[key].argmax(dim=1), pred4_d[key]))
    else:
        assert pred3.dim() == 2  # returns all predictions (2D tensor) when return_proba is true
        assert(th.is_floating_point(pred3))
        assert pred3_d.dim() == 2
        assert(th.is_floating_point(pred3_d))
        assert(pred4.dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
        assert(is_int(pred4))
        assert(th.equal(pred3.argmax(dim=1), pred4))
        assert(labels4_d.dim() == 1)
        assert(is_int(labels4_d))
        assert(th.equal(pred3.argmax(dim=1), labels4_d))

def check_node_prediction_with_reconstruct(model, data, construct_feat_ntype, train_ntypes, node_feat_field=None):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnNodeModel with GNN layers.

    Parameters
    ----------
    model: GSgnnNodeModel
        Node model
    data: GSgnnData
        Train data
    """
    target_ntype = train_ntypes[0]
    device = "cuda:0"
    g = data.g
    if data.node_feat_field is None:
        feat_ntype = []
    else:
        feat_ntype = list(data.node_feat_field.keys())
    model = model.to(device)
    def get_input_embeds(input_nodes):
        feats = prepare_batch_input(g, input_nodes, dev=device,
                                    feat_field=data.node_feat_field)
        return model.node_input_encoder(feats, input_nodes)

    # Verify the internal of full-graph inference.
    feat_size = model.node_input_encoder.in_dims
    rel_names = get_rel_names_for_reconstruct(g, construct_feat_ntype, feat_size)
    constructed = construct_node_feat(g, rel_names, model.gnn_encoder._input_gnn,
                                      get_input_embeds, 10, device=device)
    assert set(constructed.keys()) == set(construct_feat_ntype)

    input_nodes = {}
    for ntype in feat_ntype + construct_feat_ntype:
        input_nodes[ntype] = th.arange(g.number_of_nodes(ntype))
    combined_node_feats = get_input_embeds_combined(input_nodes, constructed,
                                                    get_input_embeds, device=device)
    assert set(combined_node_feats.keys()) == set(feat_ntype + construct_feat_ntype)
    for ntype in construct_feat_ntype:
        feat1 = combined_node_feats[ntype].detach().cpu()
        feat2 = constructed[ntype]
        assert np.all(feat1[0:len(feat1)].numpy() == feat2[0:len(feat2)].numpy())
    for ntype in feat_ntype:
        emb = get_input_embeds({ntype: input_nodes[ntype]})[ntype].detach().cpu()
        feat = combined_node_feats[ntype].detach().cpu()
        assert np.all(emb.numpy() == feat.numpy())

    # Run end-to-end full-graph inference.
    embs = do_full_graph_inference(model, data)
    embs = embs[target_ntype]
    embs = embs[0:len(embs)].numpy()

    # Verify the internal of mini-batch inference.
    target_nidx = {target_ntype: th.arange(g.number_of_nodes(target_ntype))}
    dataloader = GSgnnNodeDataLoader(data, target_nidx, fanout=[-1],
                                     batch_size=10, label_field='label',
                                     node_feats=node_feat_field, train_task=False,
                                     construct_feat_ntype=construct_feat_ntype)
    for input_nodes, seeds, blocks in dataloader:
        assert len(blocks) == 2
        blocks = [block.to(device) for block in blocks]
        input_feats = get_input_embeds(input_nodes)
        for ntype, feat in input_feats.items():
            assert model.gnn_encoder.in_dims == feat.shape[1]
        for ntype in blocks[0].srctypes:
            assert ntype in input_nodes
            assert blocks[0].num_src_nodes(ntype) == len(input_nodes[ntype])
        hs = model.gnn_encoder.construct_node_feat(blocks[0], input_feats)
        for ntype, h in hs.items():
            if ntype not in construct_feat_ntype and ntype in feat_ntype:
                assert np.all(h.detach().cpu().numpy()
                        == input_feats[ntype][0:len(h)].detach().cpu().numpy())

    # verify the end-to-end mini-batch inference.
    pred1, embs1, _ = node_mini_batch_gnn_predict(model, dataloader)

    embs1 = embs1[target_ntype]
    embs1 = embs1[0:len(embs1)].numpy()
    assert_almost_equal(embs1, embs, decimal=5)

def check_mlp_node_prediction(model, data):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnNodeModel without GNN layers.

    Parameters
    ----------
    model: GSgnnNodeModel
        Node model
    data: GSgnnData
        Train data
    """
    g = data.g
    embs = do_full_graph_inference(model, data)
    target_nidx = {"n1": th.arange(g.number_of_nodes("n0"))}
    dataloader1 = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                      batch_size=10, label_field='label',
                                      node_feats='feat', train_task=False)
    pred1, labels1 = node_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                      batch_size=10, label_field='label',
                                      node_feats='feat', train_task=False)
    pred2, _, labels2 = node_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    pred1_d, labels1_d = run_node_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_label=True)
    if isinstance(pred1, dict):
        assert len(pred1) == len(pred2) and len(labels1) == len(labels2)
        assert len(pred1) == len(pred1_d) and len(labels1) == len(labels1_d)
        for ntype in pred1:
            assert_almost_equal(pred1[ntype][0:len(pred1)].numpy(), pred2[ntype][0:len(pred2)].numpy(), decimal=5)
            assert_equal(labels1[ntype].numpy(), labels2[ntype].numpy())
            assert_almost_equal(pred1[ntype][0:len(pred1)].numpy(), pred1_d[ntype][0:len(pred1_d)].numpy())
            assert_equal(labels1[ntype].numpy(), labels1_d[ntype].numpy())
    else:
        assert_almost_equal(pred1[0:len(pred1)].numpy(), pred2[0:len(pred2)].numpy(), decimal=5)
        assert_equal(labels1.numpy(), labels2.numpy())
        assert_almost_equal(pred1[0:len(pred1)].numpy(), pred1_d[0:len(pred1_d)].numpy())
        assert_equal(labels1.numpy(), labels1_d.numpy())

    # Test the return_proba argument.
    pred3, _ = node_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    pred4, _ = node_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    pred3_d, _ = run_node_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=True, return_label=True)
    pred4_d, _ = run_node_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=False, return_label=True)
    if isinstance(pred3, dict):
        assert len(pred3) == len(pred4)
        assert len(pred3) == len(pred3_d)
        assert len(pred4) == len(pred4_d)
        for ntype in pred3:
            assert pred3[ntype].dim() == 2  # returns all predictions (2D tensor) when return_proba is true
            assert(th.is_floating_point(pred3[ntype]))
            assert pred3_d[ntype].dim() == 2
            assert(th.is_floating_point(pred3_d[ntype]))
            assert(pred4[ntype].dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
            assert(is_int(pred4[ntype]))
            assert(th.equal(pred3[ntype].argmax(dim=1), pred4[ntype]))
            assert(is_int(pred4_d[ntype]))
            assert(th.equal(pred3[ntype].argmax(dim=1), pred4_d[ntype]))
    else:
        assert pred3.dim() == 2  # returns all predictions (2D tensor) when return_proba is true
        assert(th.is_floating_point(pred3))
        assert pred3_d.dim() == 2
        assert(th.is_floating_point(pred3_d))
        assert(pred4.dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
        assert(is_int(pred4))
        assert(th.equal(pred3.argmax(dim=1), pred4))
        assert(pred4_d.dim() == 1)
        assert(is_int(pred4_d))
        assert(th.equal(pred3.argmax(dim=1), pred4_d))

@pytest.mark.parametrize("norm", [None, 'batch', 'layer'])
def test_rgcn_node_prediction(norm):
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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_rgcn_node_model(np_data.g, norm)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgcn_node_prediction_multi_target_ntypes():
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
        _, part_config = generate_dummy_dist_graph_multi_target_ntypes(tmpdirname)
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_rgcn_node_model(np_data.g, None)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("cache_embed", [True, False])
def test_rgcn_node_prediction_with_reconstruct(cache_embed):
    """ Test node prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + RGCNLayerWithReconstruct + Decoder

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
        _, part_config = generate_dummy_dist_graph_reconstruct(graph_name='dummy',
                                                               dirname=tmpdirname)
        node_feat_field = {'n0': ['feat'], 'n4': ['feat']}
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field=node_feat_field)
    model = create_rgcn_node_model_with_reconstruct(np_data, ['n2'], cache_embed=cache_embed)
    check_node_prediction_with_reconstruct(model, np_data, ['n2'], ['n0'], node_feat_field)

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_lm_rgcn_node_prediction_with_reconstruct():
    """ Test node prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + RGCNLayerWithReconstruct + Decoder

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
        lm_configs, _, _, _, g, part_config = create_lm_graph(tmpdirname, text_ntype='n1')
        np_data = GSgnnData(part_config=part_config,
                            lm_feat_ntypes=['n1'])
        np_data._g = g
    model = create_rgcn_node_model_with_reconstruct(np_data, ['n0'], lm_configs=lm_configs)
    check_node_prediction_with_reconstruct(model, np_data, ['n0'], ['n1'])

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("norm", [None, 'batch', 'layer'])
def test_rgat_node_prediction(norm):
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
        np_data = GSgnnData(part_config=part_config,

                                     node_feat_field='feat')
    model = create_rgat_node_model(np_data.g, norm)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_hgt_node_prediction():
    """ Test edge prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + HGTLayer + Decoder

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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model=create_hgt_node_model(np_data.g)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_rgat_node_prediction_multi_target_ntypes():
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
        _, part_config = generate_dummy_dist_graph_multi_target_ntypes(tmpdirname)
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_rgat_node_model(np_data.g)
    check_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("norm", [None, 'batch', 'layer'])
def test_sage_node_prediction(norm):
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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_sage_node_model(np_data.g, norm)
    check_node_prediction(model, np_data, is_homo=True)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("device", ['cpu', 'cuda:0'])
def test_gatv2_node_prediction(device):
    """ Test edge prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + GATv2Conv + Decoder

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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_gatv2_node_model(np_data.g)
    model = model.to(device)
    check_node_prediction(model, np_data, is_homo=True)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("device", ['cpu', 'cuda:0'])
def test_gat_node_prediction(device):
    """ Test edge prediction logic correctness with a node prediction model
        composed of InputLayerEncoder + GATConv + Decoder

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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_gat_node_model(np_data.g)
    model = model.to(device)
    check_node_prediction(model, np_data, is_homo=True)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_rgcn_edge_model(g, num_ffn_layers):
    model = GSgnnEdgeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
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

def create_hgt_edge_model(g, num_ffn_layers):
    model = GSgnnEdgeModel(alpha_l2norm=0)

    feat_size = get_node_feat_size(g, 'feat')
    encoder = GSNodeEncoderInputLayer(g, feat_size, 4,
                                      dropout=0,
                                      use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = HGTEncoder(g,
                             hid_dim=4,
                             out_dim=4,
                             num_hidden_layers=1,
                             num_heads=2,
                             dropout=0.0,
                             norm='layer',
                             num_ffn_layers_in_gnn=0)
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
    data: GSgnnData
        Train data
    """
    g = data.g
    embs = do_full_graph_inference(model, data)
    target_idx = {("n0", "r1", "n1"): th.arange(g.number_of_edges("r1"))}
    dataloader1 = GSgnnEdgeDataLoader(data, target_idx, fanout=[],
                                      batch_size=10,
                                      label_field='label',
                                      node_feats='feat',
                                      train_task=False,
                                      remove_target_edge_type=False)
    pred1, labels1 = edge_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnEdgeDataLoader(data, target_idx, fanout=[-1, -1],
                                      batch_size=10,
                                      label_field='label',
                                      node_feats='feat',
                                      train_task=False,
                                      remove_target_edge_type=False)
    pred2, labels2 = edge_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    assert_almost_equal(pred1[("n0", "r1", "n1")][0:len(pred1[("n0", "r1", "n1")])].numpy(),
                        pred2[("n0", "r1", "n1")][0:len(pred2[("n0", "r1", "n1")])].numpy(), decimal=5)
    assert_equal(labels1[("n0", "r1", "n1")].numpy(), labels2[("n0", "r1", "n1")].numpy())

    pred1_d, labels1_d = run_edge_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_label=True)
    assert_almost_equal(pred1[("n0", "r1", "n1")][0:len(pred1[("n0", "r1", "n1")])].numpy(),
                        pred1_d[("n0", "r1", "n1")][0:len(pred1_d[("n0", "r1", "n1")])].numpy())
    assert_equal(labels1[("n0", "r1", "n1")].numpy(), labels1_d[("n0", "r1", "n1")].numpy())


    # Test the return_proba argument.
    pred3, _ = edge_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    assert(th.is_floating_point(pred3[("n0", "r1", "n1")]))
    assert pred3[("n0", "r1", "n1")].dim() == 2  # returns all predictions (2D tensor) when return_proba is true

    pred3_d, _ = run_edge_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=True, return_label=True)
    assert(th.is_floating_point(pred3_d[("n0", "r1", "n1")]))
    assert pred3_d[("n0", "r1", "n1")].dim() == 2  # returns all predictions (2D tensor) when return_proba is true

    pred4, _ = edge_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    assert(pred4[("n0", "r1", "n1")].dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4[("n0", "r1", "n1")]))
    assert(th.equal(pred3[("n0", "r1", "n1")].argmax(dim=1), pred4[("n0", "r1", "n1")]))

    pred4_d, _ = run_edge_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=False, return_label=True)
    assert(pred4[("n0", "r1", "n1")].dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4_d[("n0", "r1", "n1")]))
    assert(th.equal(pred3[("n0", "r1", "n1")].argmax(dim=1), pred4_d[("n0", "r1", "n1")]))

def check_mlp_edge_prediction(model, data):
    """ Check whether full graph inference and mini batch inference generate the same
        prediction result for GSgnnEdgeModel without GNN layers.

    Parameters
    ----------
    model: GSgnnEdgeModel
        Node model
    data: GSgnnData
        Train data
    """
    g = data.g
    embs = do_full_graph_inference(model, data)
    target_idx = {("n0", "r1", "n1"): th.arange(g.number_of_edges("r1"))}
    dataloader1 = GSgnnEdgeDataLoader(data, target_idx, fanout=[],
                                      batch_size=10,
                                      label_field='label',
                                      node_feats='feat',
                                      train_task=False,
                                      remove_target_edge_type=False)
    pred1, labels1 = edge_mini_batch_predict(model, embs, dataloader1, return_label=True)
    dataloader2 = GSgnnEdgeDataLoader(data, target_idx, fanout=[],
                                      batch_size=10,
                                      label_field='label',
                                      node_feats='feat',
                                      train_task=False,
                                      remove_target_edge_type=False)
    pred2, labels2 = edge_mini_batch_gnn_predict(model, dataloader2, return_label=True)
    assert_almost_equal(pred1[("n0", "r1", "n1")][0:len(pred1[("n0", "r1", "n1")])].numpy(),
                        pred2[("n0", "r1", "n1")][0:len(pred2[("n0", "r1", "n1")])].numpy(), decimal=5)
    assert_equal(labels1[("n0", "r1", "n1")].numpy(), labels2[("n0", "r1", "n1")].numpy())

    pred1_d, labels1_d = run_edge_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_label=True)
    assert_almost_equal(pred1[("n0", "r1", "n1")][0:len(pred1[("n0", "r1", "n1")])].numpy(),
                        pred1_d[("n0", "r1", "n1")][0:len(pred1_d[("n0", "r1", "n1")])].numpy())
    assert_equal(labels1[("n0", "r1", "n1")].numpy(), labels1_d[("n0", "r1", "n1")].numpy())

    # Test the return_proba argument.
    pred3, _ = edge_mini_batch_predict(model, embs, dataloader1, return_proba=True, return_label=True)
    assert pred3[("n0", "r1", "n1")].dim() == 2  # returns all predictions (2D tensor) when return_proba is true
    assert(th.is_floating_point(pred3[("n0", "r1", "n1")]))

    pred3_d, _ = run_edge_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=True, return_label=True)
    assert(th.is_floating_point(pred3_d[("n0", "r1", "n1")]))
    assert pred3_d[("n0", "r1", "n1")].dim() == 2  # returns all predictions (2D tensor) when return_proba is true

    pred4, _ = edge_mini_batch_predict(model, embs, dataloader1, return_proba=False, return_label=True)
    assert(pred4[("n0", "r1", "n1")].dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4[("n0", "r1", "n1")]))
    assert(th.equal(pred3[("n0", "r1", "n1")].argmax(dim=1), pred4[("n0", "r1", "n1")]))

    pred4_d, _ = run_edge_mini_batch_predict(model.decoder, embs, dataloader1, model.device, return_proba=False, return_label=True)
    assert(pred4[("n0", "r1", "n1")].dim() == 1)  # returns maximum prediction (1D tensor) when return_proba is False
    assert(is_int(pred4_d[("n0", "r1", "n1")]))
    assert(th.equal(pred3[("n0", "r1", "n1")].argmax(dim=1), pred4_d[("n0", "r1", "n1")]))

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
        ep_data = GSgnnData(part_config=part_config, node_feat_field="feat")
    model = create_rgcn_edge_model(ep_data.g, num_ffn_layers=num_ffn_layers)
    check_edge_prediction(model, ep_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

@pytest.mark.parametrize("num_ffn_layers", [0, 2])
def test_hgt_edge_prediction(num_ffn_layers):
    """ Test edge prediction logic correctness with a edge prediction model
        composed of InputLayerEncoder + HGTLayer + Decoder

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
        ep_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_hgt_edge_model(ep_data.g, num_ffn_layers=num_ffn_layers)
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

    feat_size = get_node_feat_size(g, 'feat')

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
        ep_data = GSgnnData(part_config=part_config,
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

    feat_size = get_node_feat_size(g, 'feat')

    encoder = GSLMNodeEncoderInputLayer(g, lm_config, feat_size, 2, num_train=0)
    model.set_node_input_encoder(encoder)

    model.set_decoder(EntityClassifier(model.node_input_encoder.out_dims, 3, False))
    return model

def create_lm_model(g, lm_config):
    """ Create a GSgnnNodeModel with only an input encoder.

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

    feat_size = get_node_feat_size(g, 'feat')

    encoder = GSPureLMNodeInputLayer(g, lm_config, num_train=0)
    model.set_node_input_encoder(encoder)
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
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
        g.nodes['n1'].data['label'] = np_data.g.nodes['n1'].data['label']
    model = create_mlp_node_model(g, lm_config)
    assert model.gnn_encoder is None
    check_mlp_node_prediction(model, np_data)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_gnn_model_load_save():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        lm_config, _, _, _, g, part_config = create_lm_graph(tmpdirname)
        np_data = GSgnnData(part_config=part_config,
                            node_feat_field='feat')
    model = create_mlp_node_model(g, lm_config)
    dense_params = {name: param.data[:] for name, param in model.node_input_encoder.named_parameters()}
    sparse_params = [param[0:len(param)] for param in model.node_input_encoder.get_sparse_params()]
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_model(tmpdirname)
        model1 = copy.deepcopy(model)
        for param in model1.parameters():
            param.data[:] += 1
        for param in model1.node_input_encoder.get_sparse_params():
            param[0:len(param)] = param[0:len(param)] + 1
        for name, param in model1.node_input_encoder.named_parameters():
            assert np.all(dense_params[name].numpy() != param.data.numpy())
        for i, param in enumerate(model1.node_input_encoder.get_sparse_params()):
            assert np.all(sparse_params[i].numpy() != param.numpy())

        model1.restore_model(tmpdirname, "dense_embed")
        for name, param in model1.node_input_encoder.named_parameters():
            assert np.all(dense_params[name].numpy() == param.data.numpy())
        model1.restore_model(tmpdirname, "sparse_embed")
        for i, param in enumerate(model1.node_input_encoder.get_sparse_params()):
            assert np.all(sparse_params[i].numpy() == param.numpy())
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_lm_model_load_save():
    """ Test if we can load and save LM+GNN model correctly.
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        data = create_lm_graph2(tmpdirname)
        lm_config = data[0]
        g = data[6]
    # Test the case that two node types share the same BERT model.
    model = create_lm_model(g, lm_config)
    model2 = create_lm_model(g, lm_config)
    for param in model2.parameters():
        param.data[:] += 1
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_model(tmpdirname, embed_layer=model.node_input_encoder)
        load_model(tmpdirname, embed_layer=model2.node_input_encoder)
    params1 = {name: param for name, param in model.node_input_encoder.named_parameters()}
    params2 = {name: param for name, param in model2.node_input_encoder.named_parameters()}
    for key in params1:
        assert np.all(params1[key].data.numpy() == params2[key].data.numpy())

    # Test the case that two node types have different BERT models.
    lm_config = [copy.deepcopy(lm_config[0]), copy.deepcopy(lm_config[0])]
    lm_config[0]["node_types"] = ["n0"]
    lm_config[1]["node_types"] = ["n1"]

    lm_config2 = copy.deepcopy(lm_config)
    lm_config2[0]["node_types"] = ["n1"]
    lm_config2[1]["node_types"] = ["n0"]

    # Create models and make sure the two BERT models have different parameters.
    model = create_lm_model(g, lm_config)
    for i, ntype in enumerate(model.node_input_encoder._lm_models.ntypes):
        for param in model.node_input_encoder._lm_models.get_lm_model(ntype).parameters():
            param.data[:] += i
    model2 = create_lm_model(g, lm_config2)
    for i, ntype in enumerate(model2.node_input_encoder._lm_models.ntypes):
        for param in model2.node_input_encoder._lm_models.get_lm_model(ntype).parameters():
            param.data[:] += i + 2
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_model(tmpdirname, embed_layer=model.node_input_encoder)
        load_model(tmpdirname, embed_layer=model2.node_input_encoder)
    for ntype in model.node_input_encoder._lm_models.ntypes:
        params1 = model.node_input_encoder._lm_models.get_lm_model(ntype)
        params1 = {name: param for name, param in params1.named_parameters()}
        params2 = model2.node_input_encoder._lm_models.get_lm_model(ntype)
        params2 = {name: param for name, param in params2.named_parameters()}
        for key in params1:
            assert np.all(params1[key].data.numpy() == params2[key].data.numpy())

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

    feat_size = get_node_feat_size(g, 'feat')
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
        np_data = GSgnnData(part_config=part_config,
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
                "norm": "batch"
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
    # It is single float regression by default
    assert model.decoder.decoder.shape[1] == 1
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def test_node_feat_reconstruct():
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
        setattr(config, "_reconstruct_nfeat_name", "feat")
    decoder, loss_func = create_builtin_reconstruct_nfeat_decoder(
        g, decoder_input_dim=32, config=config, train_task=True)
    assert isinstance(decoder, EntityRegression)
    assert decoder.decoder.shape[1] == 2
    assert isinstance(loss_func, RegressionLossFunc)
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
                "norm": "layer"
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

@pytest.mark.parametrize("num_embs", [10, 10000])
def test_link_prediction(num_embs):
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
                         lp_embed_normalizer="l2_norm")
        config = GSConfig(args)
    model = create_builtin_lp_gnn_model(g, config, True)
    assert model.gnn_encoder.num_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, LinkPredictDotDecoder)

    embs = {
        "n0": th.rand((num_embs, 10)),
        "n1": th.rand((num_embs, 20))
    }
    embs_new = model.normalize_node_embs(embs)
    model.inplace_normalize_node_embs(embs)

    assert_equal(embs["n0"].numpy(), embs_new["n0"].numpy())
    assert_equal(embs["n1"].numpy(), embs_new["n1"].numpy())

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

@pytest.mark.parametrize("num_ffn_layers", [0, 2])
def test_mini_batch_full_graph_inference(num_ffn_layers):
    """ Test will compare embedings from layer-by-layer full graph inference
        and mini-batch full graph inference
    """
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        data = GSgnnData(part_config=part_config,
                         node_feat_field='feat')
    model = create_rgcn_edge_model(data.g, num_ffn_layers=num_ffn_layers)

    embs_layer = do_full_graph_inference(model, data, fanout=[-1, -1])
    embs_mini_batch = do_mini_batch_inference(model, data, fanout=[-1, -1])
    for ntype in data.g.ntypes:
        assert_almost_equal(embs_layer[ntype][0:data.g.num_nodes(ntype)].numpy(),
                            embs_mini_batch[ntype][0:data.g.num_nodes(ntype)].numpy())

    embs_layer = do_full_graph_inference(model, data)
    embs_mini_batch = do_mini_batch_inference(model, data)
    for ntype in data.g.ntypes:
        assert_almost_equal(embs_layer[ntype][0:data.g.num_nodes(ntype)].numpy(),
                            embs_mini_batch[ntype][0:data.g.num_nodes(ntype)].numpy())

    embs_mini_batch = do_mini_batch_inference(model, data, infer_ntypes=["n0"])
    assert len(embs_mini_batch) == 1
    assert_almost_equal(embs_layer["n0"][0:data.g.num_nodes("n0")].numpy(),
                            embs_mini_batch["n0"][0:data.g.num_nodes("n0")].numpy(), decimal=3)


    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

class Dummy_GSEdgeDecoder(MLPEdgeDecoder):
    def __init__(self):
        pass

    def predict(self, g, h, eh):
        return th.arange(10)

    def predict_proba(self, g, h, eh):
        return th.arange(10)

class Dummy_GSEdgeModel():
    def eval(self):
        pass

    def train(self):
        pass

    @property
    def device(self):
        return "cpu"

    @property
    def decoder(self):
        return Dummy_GSEdgeDecoder()

def test_edge_mini_batch_gnn_predict():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        target_type = ("n0", "r1", "n1")
        _, part_config = generate_dummy_dist_graph(tmpdirname)

        @patch.object(GSgnnData, 'get_edge_feats')
        def check_predict(mock_get_edge_feats):
            ep_data = GSgnnData(part_config=part_config)
            g = ep_data.g
            embs = {
                "n0": th.rand((g.number_of_nodes("n0"), 10)),
                "n1": th.rand((g.number_of_nodes("n1"), 10))
            }
            target_idx = {target_type: th.arange(g.number_of_edges("r1"))}
            mock_get_edge_feats.side_effect = \
                [{target_type: th.arange(10)}] * (ep_data.g.number_of_edges(target_type) // 10)
            model = Dummy_GSEdgeModel()
            dataloader = GSgnnEdgeDataLoader(ep_data, target_idx, fanout=[],
                                             batch_size=10,
                                             node_feats='feat',
                                             label_field='label',
                                             train_task=False,
                                             remove_target_edge_type=False)
            pred, labels = edge_mini_batch_predict(model, embs, dataloader, return_label=True)
            assert isinstance(pred, dict)
            assert isinstance(labels, dict)

            assert target_type in pred
            assert pred[target_type].shape[0] == \
                (ep_data.g.number_of_edges(target_type) // 10) * 10 # pred result is a dummy result
            assert labels[target_type].shape[0] == \
                (ep_data.g.number_of_edges(target_type) // 10) * 10
            for i in range(ep_data.g.number_of_edges(target_type) // 10):
                assert_equal(pred[target_type][i*10:(i+1)*10].numpy(),
                             np.arange(10))
                assert_equal(labels[target_type][i*10:(i+1)*10].numpy(),
                             np.arange(10))
        check_predict()
    th.distributed.destroy_process_group()

class Dummy_GSNodeModel(GSgnnNodeModelInterface):
    def __init__(self, return_dict=False):
        self._return_dict = return_dict

    def eval(self):
        pass

    def train(self):
        pass

    @property
    def device(self):
        return "cpu"

    def predict(self, blocks, node_feats, edge_feats, input_nodes, return_proba):
        if self._return_dict:
            return {"n1": th.arange(10)}, {"n1": th.rand((10,10))}
        else:
            return th.arange(10),  th.rand((10,10))

def test_node_mini_batch_gnn_predict():
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        _, part_config = generate_dummy_dist_graph(tmpdirname)
        data = GSgnnData(part_config=part_config)
        target_nidx = {"n1": th.arange(data.g.number_of_nodes("n0"))}
        dataloader = GSgnnNodeDataLoader(data, target_nidx, fanout=[],
                                        batch_size=10, label_field='label',
                                        node_feats=None, train_task=False)

        @patch.object(GSgnnData, 'get_node_feats')
        def check_predict(mock_get_labels, return_dict):
            model = Dummy_GSNodeModel(return_dict=return_dict)
            # Data.get_node_feats is called twice in
            # a minibatch, once for get node feature
            # and once for get labels
            mock_get_labels.side_effect = [{"n1": th.arange(10)}] * 20

            pred, embs, labels = node_mini_batch_gnn_predict(model, dataloader, return_label=True)
            assert isinstance(pred, dict)
            assert isinstance(embs, dict)
            assert isinstance(labels, dict)

            assert "n1" in pred
            assert pred["n1"].shape[0] == (data.g.number_of_nodes("n1") // 10) * 10 # pred result is a dummy result
            assert embs["n1"].shape[0] == (data.g.number_of_nodes("n1") // 10) * 10 # embs result is a dummy result
            assert labels["n1"].shape[0] == (data.g.number_of_nodes("n1") // 10) * 10
            for i in range(data.g.number_of_nodes("n1") // 10):
                assert_equal(pred["n1"][i*10:(i+1)*10].numpy(),
                             np.arange(10))
                assert_equal(labels["n1"][i*10:(i+1)*10].numpy(),
                             np.arange(10))

        check_predict(return_dict=True)
        check_predict(return_dict=False)

    th.distributed.destroy_process_group()

class DummyNCDecoder(nn.Module):

    def forward(self, inputs):
        return inputs

    def predict(self, inputs):
        return inputs

    def predict_proba(self, inputs):
        return inputs * 2

class DummyNRDecoder(nn.Module):

    def forward(self, inputs):
        return inputs

    def predict(self, inputs):
        return inputs

    def predict_proba(self, inputs):
        return inputs * 2

class DummyECDecoder(nn.Module):

    def forward(self, g, h, e_h):
        return h["n0"]

    def predict(self, g, h, e_h):
        return h["n0"]

    def predict_proba(self, g, h, e_h):
        return h["n0"] * 2

class DummyERDecoder(nn.Module):

    def forward(self, g, h, e_h):
        return h["n0"] * 2

    def predict(self, g, h, e_h):
        return h["n0"]

    def predict_proba(self, g, h, e_h):
        return h["n0"] * 2

class DummyLPDecoder(nn.Module):

    def forward(self, g, h, e_h=None):
        return h

class DummyPredLoss(nn.Module):
    def forward(self, logits, labels):
        return logits - labels

class DummyLPPredLoss(nn.Module):
    def forward(self, pos_score, neg_score):
        return pos_score["n0"] + neg_score["n0"]

def test_multi_task_norm_node_embs():
    mt_model = GSgnnMultiTaskSharedEncoderModel(0.1)
    mt_model.add_task("nc_task",
                    BUILTIN_TASK_NODE_CLASSIFICATION,
                    DummyNCDecoder(),
                    DummyPredLoss(),
                    "")
    mt_model.add_task("nr_task",
                    BUILTIN_TASK_NODE_REGRESSION,
                    DummyNRDecoder(),
                    DummyPredLoss(),
                    GRAPHSTORM_LP_EMB_L2_NORMALIZATION)

    mt_model.add_task("ec_task",
                    BUILTIN_TASK_EDGE_CLASSIFICATION,
                    DummyECDecoder(),
                    DummyPredLoss(),
                    "")

    mt_model.add_task("er_task",
                    BUILTIN_TASK_EDGE_REGRESSION,
                    DummyERDecoder(),
                    DummyPredLoss(),
                    GRAPHSTORM_LP_EMB_L2_NORMALIZATION)

    mt_model.add_task("lp_task",
                    BUILTIN_TASK_LINK_PREDICTION,
                    DummyLPDecoder(),
                    DummyLPPredLoss(),
                    "")

    mt_model.add_task("lp_task2",
                    BUILTIN_TASK_LINK_PREDICTION,
                    DummyLPDecoder(),
                    DummyLPPredLoss(),
                    GRAPHSTORM_LP_EMB_L2_NORMALIZATION)

    embs = {
        "n0": th.rand((10,16)),
        "n1": th.rand((20,16))
    }
    norm_embs = {
        "n0": F.normalize(embs["n0"]),
        "n1": F.normalize(embs["n1"])
    }

    new_embs = mt_model.normalize_task_node_embs("nc_task", embs, inplace=False)
    assert_equal(embs["n0"].numpy(), new_embs["n0"].numpy())
    assert_equal(embs["n1"].numpy(), new_embs["n1"].numpy())

    new_embs = mt_model.normalize_task_node_embs("nr_task", embs, inplace=False)
    assert_equal(norm_embs["n0"].numpy(), new_embs["n0"].numpy())
    assert_equal(norm_embs["n1"].numpy(), new_embs["n1"].numpy())

    new_embs = mt_model.normalize_task_node_embs("ec_task", embs, inplace=False)
    assert_equal(embs["n0"].numpy(), new_embs["n0"].numpy())
    assert_equal(embs["n1"].numpy(), new_embs["n1"].numpy())

    new_embs = mt_model.normalize_task_node_embs("er_task", embs, inplace=False)
    assert_equal(norm_embs["n0"].numpy(), new_embs["n0"].numpy())
    assert_equal(norm_embs["n1"].numpy(), new_embs["n1"].numpy())

    inplace_emb = {
        "n0": th.clone(embs["n0"]),
        "n1": th.clone(embs["n1"])
    }
    mt_model.normalize_task_node_embs("lp_task", inplace_emb, inplace=True)
    assert_equal(embs["n0"].numpy(), inplace_emb["n0"].numpy())
    assert_equal(embs["n1"].numpy(), inplace_emb["n1"].numpy())

    mt_model.normalize_task_node_embs("lp_task2", inplace_emb, inplace=True)
    assert_equal(norm_embs["n0"].numpy(), inplace_emb["n0"].numpy())
    assert_equal(norm_embs["n1"].numpy(), inplace_emb["n1"].numpy())

def test_multi_task_norm_node_embs_dist():
    mt_model = GSgnnMultiTaskSharedEncoderModel(0.1)
    mt_model.add_task("lp_task",
                    BUILTIN_TASK_LINK_PREDICTION,
                    DummyLPDecoder(),
                    DummyLPPredLoss(),
                    "")

    mt_model.add_task("lp_task2",
                    BUILTIN_TASK_LINK_PREDICTION,
                    DummyLPDecoder(),
                    DummyLPPredLoss(),
                    GRAPHSTORM_LP_EMB_L2_NORMALIZATION)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # get the test dummy distributed graph
        g, _ = generate_dummy_dist_graph(tmpdirname, size="tiny")

    embs = {}
    norm_embs = {}
    dist_embs = {}

    for ntype in g.ntypes:
        embs[ntype] = th.rand(g.number_of_nodes(ntype), 16)
        norm_embs[ntype] = F.normalize(embs[ntype])
        dist_embs[ntype] = DistTensor((g.number_of_nodes(ntype), 16),
            dtype=th.float32, name=f'ntype-{ntype}',
            part_policy=g.get_node_partition_policy(ntype))
        dist_embs[ntype][th.arange(g.number_of_nodes(ntype))] = embs[ntype][:]

    new_embs = mt_model.normalize_task_node_embs("lp_task", dist_embs, inplace=False)
    for ntype in g.ntypes:
        assert_equal(embs[ntype].numpy(), new_embs[ntype][th.arange(g.number_of_nodes(ntype))].numpy())

    new_embs = mt_model.normalize_task_node_embs("lp_task2", dist_embs, inplace=False)
    for ntype in g.ntypes:
        assert_equal(norm_embs[ntype].numpy(), new_embs[ntype][th.arange(g.number_of_nodes(ntype))].numpy())

    dgl.distributed.kvstore.close_kvstore()


def test_multi_task_forward():
    mt_model = GSgnnMultiTaskSharedEncoderModel(0.1)

    mt_model.add_task("nc_task",
                      BUILTIN_TASK_NODE_CLASSIFICATION,
                      DummyNCDecoder(),
                      DummyPredLoss())

    mt_model.add_task("nr_task",
                      BUILTIN_TASK_NODE_REGRESSION,
                      DummyNRDecoder(),
                      DummyPredLoss())

    mt_model.add_task("ec_task",
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      DummyECDecoder(),
                      DummyPredLoss())

    mt_model.add_task("er_task",
                      BUILTIN_TASK_EDGE_REGRESSION,
                      DummyERDecoder(),
                      DummyPredLoss())

    mt_model.add_task("lp_task",
                      BUILTIN_TASK_LINK_PREDICTION,
                      DummyLPDecoder(),
                      DummyLPPredLoss())

    @patch.object(GSgnnMultiTaskSharedEncoderModel, 'comput_input_embed')
    @patch.object(GSgnnMultiTaskSharedEncoderModel, 'compute_embed_step')
    @patch.object(GSgnnMultiTaskSharedEncoderModel, 'normalize_node_embs')
    def check_forward(mock_normalize_node_embs,
                      mock_compute_emb,
                      mock_input_embed):

        def normalize_size_effect_func(task_id, embs):
            return embs

        def compute_side_effect_func(blocks, node_feats, input_nodes):
            return input_nodes

        def input_embed_side_effect_func(input_nodes, node_feats):
            return input_nodes

        mock_normalize_node_embs.side_effect = normalize_size_effect_func
        mock_compute_emb.side_effect = compute_side_effect_func
        mock_input_embed.side_effect = input_embed_side_effect_func

        ### blocks is None (no GNN setting)
        # NC task
        task_id = "nc_task"
        blocks = None
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {"n0": th.randint(5, (10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), labels)
        assert_equal(loss.numpy(), (input_nodes["n0"]-labels["n0"]).numpy())

        # NR task
        task_id = "nr_task"
        blocks = None
        input_nodes = {"n0": th.rand((10,))}
        labels = {"n0": th.rand((10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), labels)
        assert_equal(loss.numpy(), (input_nodes["n0"]-labels["n0"]).numpy())

        # EC task
        task_id = "ec_task"
        blocks = None
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {("n0", "r1", "n1"): th.randint(5, (10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), (None, None, labels))
        assert_equal(loss.numpy(), (input_nodes["n0"]-labels[("n0", "r1", "n1")]).numpy())

        # ER task
        task_id = "er_task"
        blocks = None
        input_nodes = {"n0": th.rand((10,))}
        labels = {("n0", "r1", "n1"): th.rand((10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), (None, None, labels))
        assert_equal(loss.numpy(), (input_nodes["n0"]*2-labels[("n0", "r1", "n1")]).numpy())

        # LP task
        task_id = "lp_task"
        blocks = None
        input_nodes = {"n0": th.rand((10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), (None, None, None, None))
        assert_equal(loss.numpy(), (input_nodes["n0"]*2).numpy())

        ### blocks is a list (GNN setting)
        # NC task
        task_id = "nc_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {"n0": th.randint(5, (10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), labels)
        assert_equal(loss.numpy(), (input_nodes["n0"]-labels["n0"]).numpy())

        # NR task
        task_id = "nr_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.rand((10,))}
        labels = {"n0": th.rand((10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), labels)
        assert_equal(loss.numpy(), (input_nodes["n0"]-labels["n0"]).numpy())

        # EC task
        task_id = "ec_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {("n0", "r1", "n1"): th.randint(5, (10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), (None, None, labels))
        assert_equal(loss.numpy(), (input_nodes["n0"]-labels[("n0", "r1", "n1")]).numpy())

        # ER task
        task_id = "er_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.rand((10,))}
        labels = {("n0", "r1", "n1"): th.rand((10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), (None, None, labels))
        assert_equal(loss.numpy(), (input_nodes["n0"]*2-labels[("n0", "r1", "n1")]).numpy())

        # LP task
        task_id = "lp_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.rand((10,))}
        loss = mt_model._forward(task_id, (blocks, None, None, input_nodes), (None, None, None, None))
        assert_equal(loss.numpy(), (input_nodes["n0"]*2).numpy())


    check_forward()

def test_multi_task_predict():
    mt_model = GSgnnMultiTaskSharedEncoderModel(0.1)

    mt_model.add_task("nc_task",
                      BUILTIN_TASK_NODE_CLASSIFICATION,
                      DummyNCDecoder(),
                      DummyPredLoss())

    mt_model.add_task("nr_task",
                      BUILTIN_TASK_NODE_REGRESSION,
                      DummyNRDecoder(),
                      DummyPredLoss())

    mt_model.add_task("ec_task",
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      DummyECDecoder(),
                      DummyPredLoss())

    mt_model.add_task("er_task",
                      BUILTIN_TASK_EDGE_REGRESSION,
                      DummyERDecoder(),
                      DummyPredLoss())

    mt_model.add_task("lp_task",
                      BUILTIN_TASK_LINK_PREDICTION,
                      DummyLPDecoder(),
                      DummyLPPredLoss())

    @patch.object(GSgnnMultiTaskSharedEncoderModel, 'comput_input_embed')
    @patch.object(GSgnnMultiTaskSharedEncoderModel, 'compute_embed_step')
    @patch.object(GSgnnMultiTaskSharedEncoderModel, 'normalize_node_embs')
    def check_forward(mock_normalize_node_embs,
                      mock_compute_emb,
                      mock_input_embed):

        def normalize_size_effect_func(task_id, embs):
            return embs

        def compute_side_effect_func(blocks, node_feats, input_nodes):
            return input_nodes

        def input_embed_side_effect_func(input_nodes, node_feats):
            return input_nodes

        mock_normalize_node_embs.side_effect = normalize_size_effect_func
        mock_compute_emb.side_effect = compute_side_effect_func
        mock_input_embed.side_effect = input_embed_side_effect_func

        ### blocks is None (no GNN setting)
        # NC task
        task_id = "nc_task"
        blocks = None
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {"n0": th.randint(5, (10,))}
        mini_batch = ((blocks, None, None, input_nodes), labels)
        pred = mt_model.predict(task_id, mini_batch)
        assert_equal(pred["n0"].numpy(), (input_nodes["n0"]).numpy())

        # NR task
        task_id = "nr_task"
        blocks = None
        input_nodes = {"n0": th.rand((10,))}
        labels = {"n0": th.rand((10,))}
        mini_batch = ((blocks, None, None, input_nodes), labels)
        pred = mt_model.predict(task_id, mini_batch)
        assert_equal(pred["n0"].numpy(), (input_nodes["n0"]).numpy())

        # EC task
        task_id = "ec_task"
        blocks = None
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {("n0", "r1", "n1"): th.randint(5, (10,))}
        mini_batch = ((blocks, None, None, input_nodes), (None, None, labels))
        pred = mt_model.predict(task_id, mini_batch)
        assert_equal(pred.numpy(), (input_nodes["n0"]).numpy())

        # ER task
        task_id = "er_task"
        blocks = None
        input_nodes = {"n0": th.rand((10,))}
        labels = {("n0", "r1", "n1"): th.rand((10,))}
        mini_batch = ((blocks, None, None, input_nodes), (None, None, labels))
        pred = mt_model.predict(task_id, mini_batch)
        assert_equal(pred.numpy(), (input_nodes["n0"]).numpy())

        # LP task
        task_id = "lp_task"
        blocks = None
        input_nodes = {"n0": th.rand((10,))}
        mini_batch = mini_batch = ((blocks, None, None, input_nodes), (None, None, None, None))
        pred = mt_model.predict(task_id, mini_batch)
        assert pred is None

        ### blocks is a list (GNN setting) and call return_proba=True
        # NC task
        task_id = "nc_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {"n0": th.randint(5, (10,))}
        mini_batch = ((blocks, None, None, input_nodes), labels)
        pred = mt_model.predict(task_id, mini_batch, return_proba=True)
        assert_equal(pred["n0"].numpy(), (input_nodes["n0"]*2).numpy())

        # NR task
        task_id = "nr_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.rand((10,))}
        labels = {"n0": th.rand((10,))}
        mini_batch = ((blocks, None, None, input_nodes), labels)
        pred = mt_model.predict(task_id, mini_batch, return_proba=True)
        assert_equal(pred["n0"].numpy(), (input_nodes["n0"]*2).numpy())

        # EC task
        task_id = "ec_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.randint(5, (10,))}
        labels = {("n0", "r1", "n1"): th.randint(5, (10,))}
        mini_batch = ((blocks, None, None, input_nodes), (None, None, labels))
        pred = mt_model.predict(task_id, mini_batch, return_proba=True)
        assert_equal(pred.numpy(), (input_nodes["n0"]*2).numpy())

        # ER task
        task_id = "er_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.rand((10,))}
        labels = {("n0", "r1", "n1"): th.rand((10,))}
        mini_batch = ((blocks, None, None, input_nodes), (None, None, labels))
        pred = mt_model.predict(task_id, mini_batch, return_proba=True)
        assert_equal(pred.numpy(), (input_nodes["n0"]*2).numpy())

        # LP task
        task_id = "lp_task"
        blocks = [None, None] # trick mt_model there are two gnn layers.
        input_nodes = {"n0": th.rand((10,))}
        mini_batch = mini_batch = ((blocks, None, None, input_nodes), (None, None, None, None))
        pred = mt_model.predict(task_id, mini_batch, return_proba=True)
        assert pred is None

    check_forward()

def test_multi_task_mini_batch_predict():
    mt_model = GSgnnMultiTaskSharedEncoderModel(0.1)

    mt_model.add_task("nc_task",
                      BUILTIN_TASK_NODE_CLASSIFICATION,
                      DummyNCDecoder(),
                      DummyPredLoss())

    mt_model.add_task("nr_task",
                      BUILTIN_TASK_NODE_REGRESSION,
                      DummyNRDecoder(),
                      DummyPredLoss())

    mt_model.add_task("ec_task",
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      DummyECDecoder(),
                      DummyPredLoss())

    mt_model.add_task("er_task",
                      BUILTIN_TASK_EDGE_REGRESSION,
                      DummyERDecoder(),
                      DummyPredLoss())

    mt_model.add_task("lp_task",
                      BUILTIN_TASK_LINK_PREDICTION,
                      DummyLPDecoder(),
                      DummyLPPredLoss())

    mt_model.add_task("nfr_task",
                      BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                      DummyLPDecoder(),
                      DummyLPPredLoss())

    tast_info_nc = TaskInfo(task_type=BUILTIN_TASK_NODE_CLASSIFICATION,
                            task_id='nc_task',
                            task_config=None)
    nc_dataloader = DummyGSgnnNodeDataLoader()
    tast_info_nr = TaskInfo(task_type=BUILTIN_TASK_NODE_REGRESSION,
                            task_id='nr_task',
                            task_config=None)
    nr_dataloader = DummyGSgnnNodeDataLoader()
    tast_info_ec = TaskInfo(task_type=BUILTIN_TASK_EDGE_CLASSIFICATION,
                            task_id='ec_task',
                            task_config=None)
    ec_dataloader = DummyGSgnnEdgeDataLoader()
    tast_info_er = TaskInfo(task_type=BUILTIN_TASK_EDGE_REGRESSION,
                            task_id='er_task',
                            task_config=None)
    er_dataloader = DummyGSgnnEdgeDataLoader()
    tast_info_lp = TaskInfo(task_type=BUILTIN_TASK_LINK_PREDICTION,
                            task_id='lp_task',
                            task_config=None)
    lp_dataloader = DummyGSgnnLinkPredictionDataLoader()
    tast_info_nfr = TaskInfo(task_type=BUILTIN_TASK_RECONSTRUCT_NODE_FEAT,
                            task_id='nfr_task',
                            task_config=None)
    nfr_dataloader = DummyGSgnnNodeDataLoader()
    task_infos = [tast_info_nc, tast_info_nr, tast_info_ec,
                  tast_info_er, tast_info_lp, tast_info_nfr]
    dataloaders = [nc_dataloader, nr_dataloader, ec_dataloader,
                   er_dataloader, lp_dataloader, nfr_dataloader]

    node_pred = {"n0": th.arange(10)}
    node_prob = {"n0": th.arange(10)/10}
    node_label = {"n0": th.arange(10)}
    edge_pred = {("n0", "r0", "n1"): th.arange(5)}
    edge_prob = {("n0", "r0", "n1"): th.arange(5)/10}
    edge_label = {("n0", "r0", "n1"): th.arange(5)}
    lp_pred = {("n0", "r0", "n1"): th.arange(5)/10,
                ("n0", "r0", "n2"): th.arange(5)/20}

    def run_node_mini_batch_predict_side_func(decoder, emb, loader, device, return_prob, return_label):
        pred = node_pred
        label = None
        if return_prob:
            pred = node_prob
        if return_label:
            label = node_label

        return pred, label

    def run_edge_mini_batch_predict_side_func(decoder, emb, loader, device, return_prob, return_label):
        pred = edge_pred
        label = None
        if return_prob:
            pred = edge_prob
        if return_label:
            label = edge_label

        return pred, label

    def run_lpmini_batch_predict_side_func(decoder, emb, loader, device):
        return lp_pred

    @patch("graphstorm.model.multitask_gnn.run_node_mini_batch_predict", side_effect = run_node_mini_batch_predict_side_func)
    @patch("graphstorm.model.multitask_gnn.run_edge_mini_batch_predict", side_effect = run_edge_mini_batch_predict_side_func)
    @patch("graphstorm.model.multitask_gnn.run_lp_mini_batch_predict", side_effect = run_lpmini_batch_predict_side_func)
    def check_forward(mock_run_lp_mini_batch_predict,
                      mock_run_edge_mini_batch_predict,
                      mock_run_node_mini_batch_predict):

        res = multi_task_mini_batch_predict(mt_model,
                                            None,
                                            dataloaders,
                                            task_infos,
                                            device=th.device('cpu'),
                                            return_proba=False,
                                            return_label=False)
        assert len(res["nc_task"]) == 2
        assert_equal(res["nc_task"][0].numpy(), node_pred["n0"].numpy())
        assert res["nc_task"][1] is None
        assert len(res["nr_task"]) == 2
        assert_equal(res["nr_task"][0].numpy(), node_pred["n0"].numpy())
        assert res["nr_task"][1] is None
        assert len(res["ec_task"]) == 2
        assert_equal(res["ec_task"][0].numpy(), edge_pred[("n0", "r0", "n1")].numpy())
        assert res["ec_task"][1] is None
        assert len(res["er_task"]) == 2
        assert_equal(res["er_task"][0].numpy(), edge_pred[("n0", "r0", "n1")].numpy())
        assert res["er_task"][1] is None
        assert_equal(res["lp_task"][("n0", "r0", "n1")].numpy(), lp_pred[("n0", "r0", "n1")].numpy())
        assert_equal(res["lp_task"][("n0", "r0", "n2")].numpy(), lp_pred[("n0", "r0", "n2")].numpy())
        # node feature reconstruction also calls
        # run_node_mini_batch_predict
        assert len(res["nfr_task"]) == 2
        assert_equal(res["nfr_task"][0].numpy(), node_pred["n0"].numpy())

        res = multi_task_mini_batch_predict(mt_model,
                                            None,
                                            dataloaders,
                                            task_infos,
                                            device=th.device('cpu'),
                                            return_proba=True,
                                            return_label=False)
        assert len(res["nc_task"]) == 2
        assert_equal(res["nc_task"][0].numpy(), node_prob["n0"].numpy())
        assert res["nc_task"][1] is None
        assert len(res["nr_task"]) == 2
        assert_equal(res["nr_task"][0].numpy(), node_prob["n0"].numpy())
        assert res["nr_task"][1] is None
        assert len(res["ec_task"]) == 2
        assert_equal(res["ec_task"][0].numpy(), edge_prob[("n0", "r0", "n1")].numpy())
        assert res["ec_task"][1] is None
        assert len(res["er_task"]) == 2
        assert_equal(res["er_task"][0].numpy(), edge_prob[("n0", "r0", "n1")].numpy())
        assert res["er_task"][1] is None
        assert_equal(res["lp_task"][("n0", "r0", "n1")].numpy(), lp_pred[("n0", "r0", "n1")].numpy())
        assert_equal(res["lp_task"][("n0", "r0", "n2")].numpy(), lp_pred[("n0", "r0", "n2")].numpy())
        # node feature reconstruction also calls
        # run_node_mini_batch_predict
        assert len(res["nfr_task"]) == 2
        assert_equal(res["nfr_task"][0].numpy(), node_prob["n0"].numpy())

        res = multi_task_mini_batch_predict(mt_model,
                                            None,
                                            dataloaders,
                                            task_infos,
                                            device=th.device('cpu'),
                                            return_proba=False,
                                            return_label=True)
        assert len(res["nc_task"]) == 2
        assert_equal(res["nc_task"][0].numpy(), node_pred["n0"].numpy())
        assert_equal(res["nc_task"][1].numpy(), node_label["n0"].numpy())
        assert len(res["nr_task"]) == 2
        assert_equal(res["nr_task"][0].numpy(), node_pred["n0"].numpy())
        assert_equal(res["nr_task"][1].numpy(), node_label["n0"].numpy())
        assert len(res["ec_task"]) == 2
        assert_equal(res["ec_task"][0].numpy(), edge_pred[("n0", "r0", "n1")].numpy())
        assert_equal(res["ec_task"][0].numpy(), edge_label[("n0", "r0", "n1")].numpy())
        assert len(res["er_task"]) == 2
        assert_equal(res["er_task"][0].numpy(), edge_pred[("n0", "r0", "n1")].numpy())
        assert_equal(res["ec_task"][0].numpy(), edge_label[("n0", "r0", "n1")].numpy())
        assert_equal(res["lp_task"][("n0", "r0", "n1")].numpy(), lp_pred[("n0", "r0", "n1")].numpy())
        assert_equal(res["lp_task"][("n0", "r0", "n2")].numpy(), lp_pred[("n0", "r0", "n2")].numpy())
        # node feature reconstruction also calls
        # run_node_mini_batch_predict
        assert len(res["nfr_task"]) == 2
        assert_equal(res["nfr_task"][0].numpy(), node_pred["n0"].numpy())
        assert_equal(res["nfr_task"][1].numpy(), node_label["n0"].numpy())


        new_dataloaders = [nc_dataloader, None, ec_dataloader, None, None, None]
        res = multi_task_mini_batch_predict(mt_model,
                                            None,
                                            new_dataloaders,
                                            task_infos,
                                            device=th.device('cpu'),
                                            return_proba=False,
                                            return_label=False)
        assert len(res["nc_task"]) == 2
        assert_equal(res["nc_task"][0].numpy(), node_pred["n0"].numpy())
        assert res["nc_task"][1] is None
        assert len(res["nr_task"]) == 2
        assert res["nr_task"][0] is None
        assert res["nr_task"][1] is None
        assert len(res["ec_task"]) == 2
        assert_equal(res["ec_task"][0].numpy(), edge_pred[("n0", "r0", "n1")].numpy())
        assert res["ec_task"][1] is None
        assert len(res["er_task"]) == 2
        assert res["er_task"][0] is None
        assert res["er_task"][1] is None
        assert res["lp_task"] is None
        assert len(res["nfr_task"]) == 2
        assert res["nfr_task"][0] is None
        assert res["nfr_task"][1] is None

    check_forward()

def test_gen_emb_for_nfeat_recon():
    encoder_model = DummyGSgnnEncoderModel()
    def check_call_gen_embs(skip_last_self_loop):
        assert skip_last_self_loop == skip_self_loop

    model = DummyGSgnnModel(encoder_model, has_sparse=True)
    skip_self_loop = False
    gen_emb_for_nfeat_reconstruct(model, check_call_gen_embs)

    skip_self_loop = True
    model = DummyGSgnnModel(encoder_model, has_sparse=False)
    gen_emb_for_nfeat_reconstruct(model, check_call_gen_embs)

    model = DummyGSgnnModel(None)
    skip_self_loop = False
    gen_emb_for_nfeat_reconstruct(model, check_call_gen_embs)


if __name__ == '__main__':
    test_node_feat_reconstruct()

    test_multi_task_norm_node_embs()
    test_multi_task_norm_node_embs_dist()
    test_multi_task_forward()
    test_multi_task_predict()
    test_multi_task_mini_batch_predict()
    test_gen_emb_for_nfeat_recon()

    test_lm_rgcn_node_prediction_with_reconstruct()
    test_rgcn_node_prediction_with_reconstruct(True)
    test_rgcn_node_prediction_with_reconstruct(False)
    test_mini_batch_full_graph_inference(0)

    test_gnn_model_load_save()
    test_lm_model_load_save()
    test_node_mini_batch_gnn_predict()
    test_edge_mini_batch_gnn_predict()
    test_hgt_edge_prediction(0)
    test_hgt_edge_prediction(2)
    test_hgt_node_prediction()
    test_rgcn_edge_prediction(2)
    test_rgcn_node_prediction(None)
    test_rgat_node_prediction(None)
    test_sage_node_prediction(None)
    test_gat_node_prediction('cpu')
    test_gat_node_prediction('cuda:0')

    test_edge_classification()
    test_edge_classification_feat()
    test_edge_regression()
    test_node_classification()
    test_node_regression()
    test_link_prediction(10000)
    test_link_prediction_weight()

    test_mlp_edge_prediction(2)
    test_mlp_node_prediction()
    test_mlp_link_prediction()

    test_rgcn_node_prediction_multi_target_ntypes()
    test_rgat_node_prediction_multi_target_ntypes()
