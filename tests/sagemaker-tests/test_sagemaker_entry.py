"""
    Copyright 2025 Contributors

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
import json
import yaml
import tempfile
from argparse import Namespace
from pathlib import Path

import pytest
from argparse import ArgumentTypeError

import dgl
from graphstorm.dataloading import (GSDglDistGraphFromMetadata,
                                    load_metadata_from_json)
from graphstorm.config import GSConfig
from graphstorm.gsf import (create_builtin_node_gnn_model)
from graphstorm.model import GSNodeEncoderInputLayer
from graphstorm.sagemaker import GSRealTimeInferenceResponseMessage as res_msg

from realtime_entry_points import node_prediction_entry as npe
from realtime_entry_points.node_prediction_entry import model_fn as np_model_fn
from realtime_entry_points.node_prediction_entry import input_fn as np_input_fn
from realtime_entry_points.node_prediction_entry import predict_fn as np_predict_fn


# get two input files
_ROOT = os.path.abspath(os.path.dirname(__file__))

json_payload_file_path = os.path.join(_ROOT, "../end2end-tests/"
                                            "data_gen/movielens_realtime_payload.json")
with open(json_payload_file_path, 'r', encoding="utf8") as json_file:
    json_data = json.load(json_file)


def create_dummy_model_artifacts(model_dir, files=None):
    """ create dummy test model artifacts
    
    These artifacts is in a model folder that includes
    1. a model.bin model parameter files,
    2. a JSON file from GraphStorm's graph construction steps.
    3. a YAML file from GraphStorm's model training steps.

    """
    if files is None:
        files = []

    if "model" in files:
        temp_model_file_path = os.path.join(model_dir, "model.bin")
        with open(temp_model_file_path, 'w') as f:
            f.write("This is temporary GraphStorm model file.")

    if "json" in files:
        temp_json_file_path = os.path.join(model_dir, "new_configuraton.json")
        with open(temp_json_file_path, 'w') as f:
            f.write("This is temporary GraphStorm JSON file.")

    if "yaml" in files:
        temp_yaml_file_path = os.path.join(model_dir, ".yaml")
        with open(temp_yaml_file_path, 'w') as f:
            f.write("This is temporary GraphStorm YAML file.")    

def create_realtime_yaml_object(tmpdirname):
    """ Create a model yaml object to build a GraphStorm model
    """
    yaml_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "model_encoder_type": "rgcn",
                "ip_config": os.path.join(tmpdirname, "ip.txt"),
                "part_config": os.path.join(tmpdirname, "part.json"),
            },
            "gnn": {
                "hidden_size": 128,
                "fanout": "4",
                "num_layers": 1,
            },
            "input": {
                "node_feat_name": ["user:feat", "movie:feat"]
                },
            "output": {},
            "hyperparam": {
                "lr": 0.01,
                "lm_tune_lr": 0.0001,
                "sparse_optimizer_lr": 0.0001
            },
            "node_classification": {
                "target_ntype": "movie",
                "label_field": "label",
                "multilabel": False,
                "num_classes": 19
                },
            "rgcn":{
                "num_bases": -1,
                "use_self_loop": True,
                "sparse_optimizer_lr": 1e-2,
                "use_node_embeddings": False
            }
        }
    }
    # create dummpy ip.txt
    with open(os.path.join(tmpdirname, "ip.txt"), "w") as f:
        f.write("127.0.0.1\n")
    # create dummpy part.json
    with open(os.path.join(tmpdirname, "part.json"), "w") as f:
        json.dump({
            "graph_name": "test"
        }, f)
    with open(os.path.join(tmpdirname, "ml.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname), 'ml.yaml'),
                    local_rank=0)
    gs_config = GSConfig(args)

    return gs_config

def create_realtime_json_oject(tmpdirname):
    """ Create a graph construction json file to build GraphFromMetadata
    """
    json_object = \
    {
        "version": "gconstruct-v0.1",
        "nodes": [
                {
                        "node_id_col":  "id",
                        "node_type":    "user",
                        "format":       {"name": "hdf5"},
                        "files":        "/data/ml-100k/user.hdf5",
                        "features":     [
                            {
                                    "feature_col":  "feat",
                                    "feature_name": "feat",
                                    "feature_dim": [
                                        2
                                    ]
                            }
                        ]
                },
                {
                        "node_id_col":  "id",
                        "node_type":    "movie",
                        "format":       {"name": "parquet"},
                        "files":        "/data/ml-100k/movie.parquet",
                        "features":     [
                            {
                                    "feature_col":  "feat",
                                    "feature_name": "feat",
                                    "feature_dim": [
                                        2
                                    ]
                            },
                            {
                                "feature_col":  "title",
                                "transform":    {
                                        "name": "bert_hf",
                                        "bert_model": "bert-base-uncased",
                                        "max_seq_length": 16
                                },
                                "feature_name": "title",
                                "feature_dim": [
                                    16
                                ]
                        }
                    ],
                        "labels":	[
                            {
                                "label_col":	"label",
                                "task_type":	"classification",
                                "split_pct":	[0.8, 0.1, 0.1]
                            }
                        ]
                }
        ],
        "edges": [
                {
                        "source_id_col":    "src_id",
                        "dest_id_col":      "dst_id",
                        "relation":         ["user", "rating", "movie"],
                        "format":           {"name": "parquet"},
                        "files":        "/data/ml-100k/edges.parquet",
                        "labels":	[
                            {
                                "label_col":	"rate",
                                "task_type":	"classification",
                                "split_pct":	[0.1, 0.1, 0.1]
                            }
                        ]
                }
        ],
        "is_homogeneous": "false"
    }
    with open(os.path.join(tmpdirname, "ml.json"), "w") as f:
        json.dump(json_object, f, indent=4)

    with open(os.path.join(tmpdirname, "ml.json"), "r") as f:
        gs_json = json.load(f)

    return gs_json

def create_realtime_np_model(tmpdirname, model_error=False):
    """ Create a node prediction model

    This basic configuration is a subset of the `training_scripts/gsgnn_np/ml_nc.yaml`.
    """
    gs_json = create_realtime_json_oject(tmpdirname)
    gs_config = create_realtime_yaml_object(tmpdirname)

    gs_metadata = load_metadata_from_json(gs_json)
    gs_distgraph = GSDglDistGraphFromMetadata(gs_metadata, device='cpu')

    model = create_builtin_node_gnn_model(gs_distgraph, gs_config, train_task=False)

    if model_error:
        node_encoder = GSNodeEncoderInputLayer(gs_distgraph, {'user': 128, 'movie': 128},
                                               gs_config.hidden_size)
        model.set_node_input_encoder(node_encoder)

    return model, gs_config


def test_np_model_fn():
    """ Locally test the model_fn for np

    Because the create_builtin_node_gnn_model function has been tested in the
    unit-tests/test_gsf.py, this test will only check the file existance parts.
    """
    artifact_files = ['model', 'json', 'yaml']

    # Test case 1, normal case, including all three dummy files, but fail at model restore
    with tempfile.TemporaryDirectory() as tempdir:
        create_dummy_model_artifacts(tempdir, artifact_files)

        with pytest.raises(Exception, match='Fail to restore trained .*'):
            np_model_fn(tempdir)

    # Test case 2, abnormal case, missing one artifact, trigerring an AssertionError
    for artifact in artifact_files:
        with tempfile.TemporaryDirectory() as tempdir:
            create_dummy_model_artifacts(tempdir, [artifact])

            with pytest.raises(AssertionError, match='Missing .*'):
                np_model_fn(tempdir)


def test_np_input_fn():
    """ Locally test the input_fn for np
    
    Because the content of test payload json has been tested in the test_gconstruct_json_payload.py
    file, this test focuses on the other fields:
    1. version
    2. task_type
    3. targets
    4. target_mapping_dict
    """
    # add target field input the test data
    with tempfile.TemporaryDirectory() as tmpdir:
        gs_json = create_realtime_json_oject(tmpdir)
        npe.CONFIG_JSON = gs_json
    json_data['targets'] = [{'node_type': 'movie', 'node_id': 'm1'},
                            {'node_type': 'movie', 'node_id': 'm2'}]
    json_payload = json.dumps(json_data)

    # Test case 1: normal case, success load and parse http payload
    res = np_input_fn(json_payload)
    assert res.status_code == 200
    # only test if the data is a DGLGraph, not contents of the test graph which has been tested
    assert isinstance(res.data['data'], dgl.DGLGraph)
    # check target node id mapping
    assert isinstance(res.data['targets'], dict)
    assert len(res.data['targets']['movie']) == 2
    assert res.data['targets']['movie'][0] == ['m1', 'm2']
    assert res.data['targets']['movie'][1] == [0, 1]

    # Test case 2: abnormal cases
    #       2.1 missing gml_task or mismatch
    json_data['gml_task'] = 'edge_classification'
    json_payload_wrong_gml_task = json.dumps(json_data)
    res = np_input_fn(json_payload_wrong_gml_task)
    assert res.status_code == 421

    json_data.pop('gml_task')
    json_payload_wt_gml_task = json.dumps(json_data)
    res = np_input_fn(json_payload_wt_gml_task)
    assert res.status_code == 421

    # resume the gml_task field
    json_data['gml_task'] = 'node_classification'

    #       2.2 missing targets field or it is empty
    json_data['targets'] = None
    json_payload_none_targets = json.dumps(json_data)
    res = np_input_fn(json_payload_none_targets)
    assert res.status_code == 401

    json_data['targets'] = []
    json_payload_empty_targets = json.dumps(json_data)
    res = np_input_fn(json_payload_empty_targets)
    assert res.status_code == 401

    json_data['targets'] = [['m1', 'm2']]
    json_payload_nondict_targets = json.dumps(json_data)
    res = np_input_fn(json_payload_nondict_targets)
    assert res.status_code == 400

    json_data['targets'] = [{'ntype': 'movie'}]
    json_payload_nondict_targets = json.dumps(json_data)
    res = np_input_fn(json_payload_nondict_targets)
    assert res.status_code == 400

    #       2.3 targets do not exist in the payload graph
    json_data['targets'] = [{'node_type': 'movie', 'node_id': 'm1'},
                            {'node_type': 'movie', 'node_id': 'm3'}]
    json_payload_mis_targets = json.dumps(json_data)
    res = np_input_fn(json_payload_mis_targets)
    assert res.status_code == 403


def test_np_predict_fn():
    """ Locally test the predict_fn for np

    predict_fn asks for two arguments:
    - input_data: the DGLGraph and the target id mapping from the input_fn
    - model: the restored GraphStorm model from the model_fn

    """
    # Test case 1, normal case, success prediction
    json_data['targets'] = [{'node_type': 'movie', 'node_id': 'm1'},
                            {'node_type': 'movie', 'node_id': 'm2'}]
    json_payload = json.dumps(json_data)

    input_data = np_input_fn(json_payload)

    with tempfile.TemporaryDirectory() as tmpdir:
        model, gs_config = create_realtime_np_model(tmpdir)
        npe.GS_CONFIG = gs_config
        res = np_predict_fn(input_data, model)
        assert res['status_code'] == 200
        results = res['data']['results']
        for i, result in enumerate(results):
            assert result['node_type'] == 'movie'
            assert result['node_id'] == 'm'+str(i+1)
            assert len(result['prediction']) == gs_config.num_classes

    # Test case 2, abnormal cases
    #       2.1 input_data has error
    with tempfile.TemporaryDirectory() as tmpdir:
        model, gs_config = create_realtime_np_model(tmpdir)
        npe.GS_CONFIG = gs_config
        input_data_wt_feat = res_msg.missing_required_field("feat")
        res = np_predict_fn(input_data_wt_feat, model)
        assert res == input_data_wt_feat.to_dict()

    #       2.2 target id mismatch
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data['targets'] = [{'node_type': 'movie', 'node_id': 'm1'},
                                {'node_type': 'movie', 'node_id': 'm3'}]
        json_payload = json.dumps(json_data)

        input_data = np_input_fn(json_payload)

        model, gs_config = create_realtime_np_model(tmpdir)
        res = np_predict_fn(input_data, model)
        assert res['status_code'] == 403

    #       2.3 internal server error
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data['targets'] = [{'node_type': 'movie', 'node_id': 'm1'},
                                {'node_type': 'movie', 'node_id': 'm2'}]
        json_payload = json.dumps(json_data)

        input_data = np_input_fn(json_payload)

        model, gs_config = create_realtime_np_model(tmpdir, model_error=True)
        res = np_predict_fn(input_data, model)
        assert res['status_code'] == 500
