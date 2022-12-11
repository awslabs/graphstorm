from pathlib import Path
import os
import yaml
import tempfile
from argparse import Namespace

import torch as th
from torch import nn
import numpy as np
from numpy.testing import assert_almost_equal

import dgl

from graphstorm.config import GSConfig
from graphstorm.model import GSNodeInputLayer, RelationalGCNEncoder, GSgnnModel
from graphstorm.model.rgcn_encoder import RelationalGCNEncoder
from graphstorm.model.rgat_encoder import RelationalGATEncoder
from graphstorm.model.edge_decoder import DenseBiDecoder, LinkPredictDotDecoder
from graphstorm.model.node_decoder import EntityRegression, EntityClassifier
from graphstorm.model import create_edge_gnn_model, create_node_gnn_model, create_lp_gnn_model
from graphstorm.model.utils import get_feat_size
from graphstorm.model.gnn import do_full_graph_inference, do_mini_batch_inference

from data_utils import generate_dummy_dist_graph

def create_model(g):
    model = GSgnnModel(g)

    feat_size = get_feat_size(g, 'feat')
    encoder = GSNodeInputLayer(g, feat_size, 4,
                               dropout=0,
                               use_node_embeddings=True)
    model.set_node_input_encoder(encoder)

    gnn_encoder = RelationalGCNEncoder(g, 4, 4,
                                       num_bases=2,
                                       num_hidden_layers=1,
                                       dropout=0,
                                       use_self_loop=True)
    model.set_gnn_encoder(gnn_encoder)
    return model

def test_compute_embed():
    # get the test dummy distributed graph
    g = generate_dummy_dist_graph()
    # initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    model = create_model(g)
    embs1 = do_full_graph_inference(g, model, feat_field='feat')
    target_nidx = {ntype: th.arange(g.number_of_nodes(ntype)) for ntype in g.ntypes}
    embs2 = do_mini_batch_inference(g, model, target_nidx,
            [-1, -1], batch_size=1024, feat_field='feat')
    assert len(embs1) == len(embs2)
    assert set(embs1.keys()) == set(embs2.keys())
    for ntype in embs1:
        emb1 = embs1[ntype]
        emb2 = embs2[ntype]
        assert_almost_equal(emb1[0:len(emb1)].numpy(), emb2[0:len(emb2)].numpy(),
                decimal=5)

    emb = model.compute_embeddings(g, 'feat', {'n1': th.arange(10)},
            fanout=-1, batch_size=10, mini_batch_infer=False)
    assert_almost_equal(embs1['n1'][th.arange(10)].numpy(), emb['n1'][0:len(emb['n1'])].numpy(),
            decimal=5)
    emb = model.compute_embeddings(g, 'feat', {'n1': th.arange(10)},
            fanout=[-1, -1], batch_size=10, mini_batch_infer=True)
    assert_almost_equal(embs1['n1'][th.arange(10)].numpy(), emb['n1'][0:len(emb['n1'])].numpy(),
            decimal=5)
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

def create_ec_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "feat_name": "feat",
            },
            "gnn": {
                "n_layers": 1,
                "n_hidden": 4,
                "model_encoder_type": "rgcn"
            },
            "input": {},
            "output": {},
            "rgcn": {
                "n_bases": 2,
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
    # get the test dummy distributed graph
    g = generate_dummy_dist_graph()

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_ec_config(Path(tmpdirname), 'gnn_ec.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_ec.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.n_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGCNEncoder)
    assert isinstance(model.decoder, DenseBiDecoder)
    assert model.task_name == "edge_classification"
    dgl.distributed.kvstore.close_kvstore()

def create_er_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "feat_name": "feat",
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "n_layers": 1,
                "n_hidden": 4,
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
    # get the test dummy distributed graph
    g = generate_dummy_dist_graph()

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_er_config(Path(tmpdirname), 'gnn_er.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_er.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_edge_gnn_model(g, config, True)
    assert model.gnn_encoder.n_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, DenseBiDecoder)
    assert model.task_name == "edge_regression"
    dgl.distributed.kvstore.close_kvstore()

def create_nr_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "feat_name": "feat",
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "n_layers": 1,
                "n_hidden": 4,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "node_regression": {
                "predict_ntype": "n0",
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_node_regression():
    # get the test dummy distributed graph
    g = generate_dummy_dist_graph()

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_nr_config(Path(tmpdirname), 'gnn_nr.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nr.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_node_gnn_model(g, config, True)
    assert model.gnn_encoder.n_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, EntityRegression)
    assert model.task_name == "node_regression"
    dgl.distributed.kvstore.close_kvstore()

def create_nc_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "feat_name": "feat",
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "n_layers": 1,
                "n_hidden": 4,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "node_classification": {
                "num_classes": 2,
                "predict_ntype": "n0",
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_node_classification():
    # get the test dummy distributed graph
    g = generate_dummy_dist_graph()

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_nc_config(Path(tmpdirname), 'gnn_nc.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_nc.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_node_gnn_model(g, config, True)
    assert model.gnn_encoder.n_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, EntityClassifier)
    assert model.task_name == "node_classification"
    dgl.distributed.kvstore.close_kvstore()

def create_lp_config(tmp_path, file_name):
    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "feat_name": "feat",
                "model_encoder_type": "rgat",
            },
            "gnn": {
                "n_layers": 1,
                "n_hidden": 4,
            },
            "input": {},
            "output": {},
            "rgat": {
            },
            "link_prediction": {
                "train_etype": ["n0,r0,n1"],
                "use_dot_product": True
            },
        }
    }
    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

def test_link_prediction():
    # get the test dummy distributed graph
    g = generate_dummy_dist_graph()

    with tempfile.TemporaryDirectory() as tmpdirname:
        create_lp_config(Path(tmpdirname), 'gnn_lp.yaml')
        args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'gnn_lp.yaml'),
                         local_rank=0)
        config = GSConfig(args)
    model = create_lp_gnn_model(g, config, True)
    assert model.gnn_encoder.n_layers == 1
    assert model.gnn_encoder.out_dims == 4
    assert isinstance(model.gnn_encoder, RelationalGATEncoder)
    assert isinstance(model.decoder, LinkPredictDotDecoder)
    assert model.task_name == "link_prediction"
    dgl.distributed.kvstore.close_kvstore()

if __name__ == '__main__':
    test_compute_embed()
    test_edge_classification()
    test_edge_regression()
    test_node_classification()
    test_node_regression()
    test_link_prediction()
