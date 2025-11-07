"""
    Copyright Contributors

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
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest
from graphstorm.dataloading import (GSDglDistGraphFromMetadata,
                                    load_metadata_from_json)
from graphstorm.config import GSConfig
from graphstorm.gsf import (create_builtin_node_gnn_model)
from graphstorm.model import GSNodeEncoderInputLayer
from graphstorm.sagemaker import GSRealTimeInferenceResponseMessage as res_msg

from realtime_entry_points.node_prediction_entry import (
    model_fn as np_model_fn,
    transform_fn as np_transform_fn
)

from config_utils import create_graph_config_json_object

# ============ helper functions ==============

@pytest.fixture(scope='session')
def create_test_realtime_payload():
    # get a real-time payload file
    _ROOT = os.path.abspath(os.path.dirname(__file__))

    json_payload_file_path = os.path.join(_ROOT, "../end2end-tests/"
                                                "data_gen/movielens_realtime_payload.json")
    with open(json_payload_file_path, 'r', encoding="utf8") as json_file:
        json_data = json.load(json_file)

    resource = {'json_data': json_data}

    yield resource

def create_request_uid(payload):
    hash_obj = hashlib.sha256(payload.encode('utf-8'))
    request_uid = hash_obj.hexdigest()[-16: ]
    return request_uid

def create_dummy_model_artifacts(model_dir, files=None):
    """ create dummy test model artifacts
    
    These artifacts is in a model folder that includes
    1. a model.bin model parameter files,
    2. a JSON file from GraphStorm's graph construction steps.
       Default is GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml.
    3. a YAML file from GraphStorm's model training steps.
       Default is data_transform_new.json.
    """
    if files is None:
        files = []

    if "model" in files:
        temp_model_file_path = os.path.join(model_dir, "model.bin")
        with open(temp_model_file_path, 'w') as f:
            f.write("This is temporary GraphStorm model file.")

    if "json" in files:
        temp_json_file_path = os.path.join(model_dir, "data_transform_new.json")
        with open(temp_json_file_path, 'w') as f:
            f.write("This is temporary GraphStorm JSON file.")

    if "yaml" in files:
        temp_yaml_file_path = os.path.join(model_dir, "GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml")
        with open(temp_yaml_file_path, 'w') as f:
            f.write("This is temporary GraphStorm YAML file.")    

def create_realtime_yaml_object(tmpdirname):
    """ Create a model yaml object to build a GraphStorm model

    This yaml object works for the real-time test payload data, which is based on movielens.
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
    with open(os.path.join(tmpdirname,
                           "GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    args = Namespace(yaml_config_file=os.path.join(Path(tmpdirname),
                                            'GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml'),
                    local_rank=0)
    gs_config = GSConfig(args)

    return gs_config

def create_realtime_np_model(tmpdirname, model_error=False):
    """ Create a node prediction model

    This basic configuration is a subset of the `training_scripts/gsgnn_np/ml_nc.yaml`.
    """
    gs_json = create_graph_config_json_object(tmpdirname, has_tokenize=False)
    gs_config = create_realtime_yaml_object(tmpdirname)

    gs_metadata = load_metadata_from_json(gs_json)
    gs_distgraph = GSDglDistGraphFromMetadata(gs_metadata, device='cpu')

    model = create_builtin_node_gnn_model(gs_distgraph, gs_config, train_task=False)

    if model_error:
        node_encoder = GSNodeEncoderInputLayer(gs_distgraph, {'user': 128, 'movie': 128},
                                               gs_config.hidden_size)
        model.set_node_input_encoder(node_encoder)

    return model, gs_json, gs_config


# ============ test functions ==============

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


def test_np_transform_fn(create_test_realtime_payload):
    """ Locally test the transform_fn for np

    Because the content of test payload json has been tested in the test_gconstruct_json_payload.py
    file, this test focuses on the other fields for input contents:
    0. request_content_type
    1. version
    2. task_type
    3. targets
    4. target_mapping_dict
    5. prediction results
    """
    json_data = create_test_realtime_payload['json_data']

    # Test case 1: normal case, successful http payload parsing and graph building, model loading,
    #              and prediction
    with tempfile.TemporaryDirectory() as tmpdir:
        json_data['targets'] = [{'node_type': 'user', 'node_id': 'a1'},
                                {'node_type': 'movie', 'node_id': 'm2'}]

        json_payload = json.dumps(json_data)
        request_uid = create_request_uid(json_payload)

        model, gs_json, gs_config = create_realtime_np_model(tmpdir)
        # setattr(gs_config, '_target_ntype', ['user', 'movie'])
        setattr(gs_config, '_target_ntype', ['user'])
        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 200
        assert res['request_uid'] == request_uid
        assert res_type == 'application/json'
        results = res['data']['results']
        for result, target in zip(results, json_data['targets']):
            assert result['node_type'] == target['node_type']
            assert result['node_id'] == target['node_id']
            assert len(result['prediction']) == gs_config.num_classes
        print(results)
        # exit(-1)

    # Test case 2: abnormal cases of input payload
    #       2.1 missing gml_task or mismatch
        json_data['gml_task'] = 'edge_classification'
        json_payload_wrong_gml_task = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_wrong_gml_task)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_wrong_gml_task,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 421
        assert res['request_uid'] == request_uid

        json_data.pop('gml_task')
        json_payload_wt_gml_task = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_wt_gml_task)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_wt_gml_task,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 421
        assert res['request_uid'] == request_uid

        # resume the gml_task field
        json_data['gml_task'] = 'node_classification'

        #       2.2 missing targets field or it is empty
        json_data['targets'] = None
        json_payload_none_targets = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_none_targets)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_none_targets,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 401
        assert res['request_uid'] == request_uid

        json_data['targets'] = []
        json_payload_empty_targets = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_empty_targets)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_empty_targets,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 401
        assert res['request_uid'] == request_uid

        json_data['targets'] = [['m1', 'm2']]
        json_payload_nondict_targets = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_nondict_targets)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_nondict_targets,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 400
        assert res['request_uid'] == request_uid

        json_data['targets'] = [{'ntype': 'movie'}]
        json_payload_noids_targets = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_noids_targets)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_noids_targets,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 400
        assert res['request_uid'] == request_uid

        #       2.3 some targets, other_ntype or m3, do not exist in the payload graph
        json_data['targets'] = [{'node_type': 'other_ntype', 'node_id': 'a1'},
                                {'node_type': 'movie', 'node_id': 'm3'}]
        json_payload_mis_targets = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_mis_targets)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_mis_targets,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 403
        assert res['request_uid'] == request_uid

        json_data['targets'] = [{'node_type': 'user', 'node_id': 'a1'},
                                {'node_type': 'movie', 'node_id': 'm3'}]
        json_payload_mis_targets = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_mis_targets)

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_mis_targets,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 404
        assert res['request_uid'] == request_uid

        #       2.4 internal server error
        json_data['targets'] = [{'node_type': 'user', 'node_id': 'a1'},
                                {'node_type': 'movie', 'node_id': 'm2'}]
        json_payload = json.dumps(json_data)
        request_uid = create_request_uid(json_payload)

        model, gs_json, gs_config = create_realtime_np_model(tmpdir, model_error=True)
        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 500
        assert res['request_uid'] == request_uid

        #       2.5 missing node features
        json_data['targets'] = [{'node_type': 'user', 'node_id': 'a1'},
                                {'node_type': 'movie', 'node_id': 'm2'}]
        json_payload = json.dumps(json_data)
        request_uid = create_request_uid(json_payload)

        model, gs_json, gs_config = create_realtime_np_model(tmpdir)
        setattr(gs_config, "_node_feat_name", {'user:test_feat', 'movie:test_feat'})

        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 404
        assert res['request_uid'] == request_uid

        #       2.6 graph construction error with missing node or edge features
        json_data['graph']['nodes'][0].pop('features')
        json_data['targets'] = [{'node_type': 'user', 'node_id': 'a1'},
                                {'node_type': 'movie', 'node_id': 'm2'}]
        json_payload_miss_feat = json.dumps(json_data)
        request_uid = create_request_uid(json_payload_miss_feat)

        model, gs_json, gs_config = create_realtime_np_model(tmpdir)
        res, res_type = np_transform_fn(model=(model, gs_json, gs_config),
                                        request_body=json_payload_miss_feat,
                                        request_content_type='application/json')
        res = json.loads(res)
        assert res['status_code'] == 411
        assert res['request_uid'] == request_uid

