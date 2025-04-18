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

import os
import json
import tempfile
import yaml
from argparse import Namespace
from unittest.mock import patch, MagicMock, call

import dgl
import torch as th

from graphstorm.run.gsgnn_np.np_infer_gnn import main

def create_nc_config(tmp_path, file_name, infer_all=False, no_validation=False):
    """Create a node classification config file for testing."""
    # Create a dummy part.json file
    part_config_path = os.path.join(tmp_path, "part.json")
    with open(part_config_path, "w") as f:
        json.dump({"graph_name": "test"}, f)

    # Create a dummy ip.txt file
    ip_config_path = os.path.join(tmp_path, "ip.txt")
    with open(ip_config_path, "w") as f:
        f.write("127.0.0.1\n")

    conf_object = {
        "version": 1.0,
        "gsf": {
            "basic": {
                "node_feat_name": ["feat"],
                "model_encoder_type": "rgat",
                "no_validation": no_validation,
                "part_config": part_config_path,
                "ip_config": ip_config_path,
                "backend": "gloo",
                "task_type": "node_classification",
            },
            "gnn": {
                "num_layers": 1,
                "hidden_size": 4,
                "lr": 0.001,
                "norm": "layer"
            },
            "input": {
                "restore_model_path": os.path.join(tmp_path, "model")
            },
            "output": {},
            "node_classification": {
                "num_classes": 2,
                "target_ntype": "n0",
                "label_field": "label",
            },
        }
    }

    # Only add infer_all_targets if it's True
    if infer_all:
        conf_object["gsf"]["basic"]["infer_all_targets"] = True

    with open(os.path.join(tmp_path, file_name), "w") as f:
        yaml.dump(conf_object, f)

    # Create dummy model directory
    os.makedirs(os.path.join(tmp_path, "model"), exist_ok=True)

    return part_config_path, ip_config_path

def test_np_infer_gnn_infer_all_targets():
    """Test the infer_all_targets option in np_infer_gnn.py."""
    # Initialize the torch distributed environment
    th.distributed.init_process_group(backend='gloo',
                                      init_method='tcp://127.0.0.1:23456',
                                      rank=0,
                                      world_size=1)

    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create config files with different settings
        create_nc_config(tmpdirname, 'gnn_nc_default.yaml')
        create_nc_config(tmpdirname, 'gnn_nc_infer_all.yaml', infer_all=True)
        create_nc_config(tmpdirname, 'gnn_nc_no_validation.yaml', no_validation=True)

        # Mock the necessary functions and classes
        with patch('graphstorm.run.gsgnn_np.np_infer_gnn.GSgnnData') as mock_data, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.gs') as mock_gs, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.GSgnnNodeDataLoader') as _, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.GSgnnNodePredictionInferrer') as mock_inferrer, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.get_evaluator') as mock_get_evaluator, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.get_device') as mock_get_device, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.get_lm_ntypes') as mock_get_lm_ntypes, \
             patch('graphstorm.run.gsgnn_np.np_infer_gnn.use_wholegraph') as mock_use_wholegraph:

            # Setup mocks
            mock_model = MagicMock()
            mock_gs.create_builtin_node_gnn_model.return_value = mock_model
            mock_gs.create_builtin_task_tracker.return_value = MagicMock()
            mock_get_device.return_value = 'cpu'
            mock_get_lm_ntypes.return_value = None
            mock_use_wholegraph.return_value = False

            mock_data_instance = MagicMock()
            mock_data.return_value = mock_data_instance
            mock_data_instance.g = MagicMock()

            # Make get_node_infer_set return a non-empty dictionary
            mock_data_instance.get_node_infer_set.return_value = {'n0': th.tensor([0, 1, 2])}

            mock_inferrer_instance = MagicMock()
            mock_inferrer.return_value = mock_inferrer_instance

            mock_evaluator = MagicMock()
            mock_get_evaluator.return_value = mock_evaluator

            # Run all three tests
            # Test Case 1: Default behavior (infer_all_targets=False)
            args = Namespace(yaml_config_file=os.path.join(tmpdirname, 'gnn_nc_default.yaml'),
                            local_rank=0)
            main(args)

            # Test Case 2: infer_all_targets=True
            args = Namespace(yaml_config_file=os.path.join(tmpdirname, 'gnn_nc_infer_all.yaml'),
                            local_rank=0)
            main(args)

            # Test Case 3: no_validation=True
            args = Namespace(yaml_config_file=os.path.join(tmpdirname, 'gnn_nc_no_validation.yaml'),
                            local_rank=0)
            main(args)

            # Now check all the calls to get_node_infer_set
            calls = mock_data_instance.get_node_infer_set.call_args_list

            # There should be 3 calls
            assert len(calls) == 3, f"Expected 3 calls to get_node_infer_set, got {len(calls)}"

            # First call should be with test_mask
            assert calls[0] == call('n0', mask='test_mask'), \
                f"First call should be with test_mask, got {calls[0]}"

            # Second call should be with empty string mask
            assert calls[1] == call('n0', mask=''), \
                f"Second call should be with empty string mask, got {calls[1]}"

            # Third call should be without mask parameter
            assert calls[2] == call('n0'), \
                f"Third call should be without mask parameter, got {calls[2]}"

    # Clean up
    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()

if __name__ == '__main__':
    test_np_infer_gnn_infer_all_targets()
