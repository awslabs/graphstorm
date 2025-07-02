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

    Create dummy configurations for unit tests
"""

import os
import json
import yaml

from data_utils import generate_dummy_hetero_graph

def create_dummy_config_obj():
    yaml_object = { # dummy config, bypass checks by default
        "version": 1.0,
        "gsf": {
            "basic": {},
            "gnn": {
                "fanout": "4",
                "num_layers": 1,
            },
            "input": {},
            "output": {},
            "hyperparam": {
                "lr": 0.01,
                "lm_tune_lr": 0.0001,
                "sparse_optimizer_lr": 0.0001
            },
            "rgcn": {},
        }
    }
    return yaml_object

def create_basic_config(tmp_path, file_name):
    yaml_object = create_dummy_config_obj()
    yaml_object["gsf"]["basic"] = {
        "backend": "gloo",
        "ip_config": os.path.join(tmp_path, "ip.txt"),
        "part_config": os.path.join(tmp_path, "part.json"),
        "model_encoder_type": "rgat",
        "eval_frequency": 100,
        "no_validation": True,
    }
    # create dummpy ip.txt
    with open(os.path.join(tmp_path, "ip.txt"), "w") as f:
        f.write("127.0.0.1\n")
    # create dummpy part.json
    with open(os.path.join(tmp_path, "part.json"), "w") as f:
        json.dump({
            "graph_name": "test"
        }, f)
    with open(os.path.join(tmp_path, file_name+".yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for check default value
    yaml_object["gsf"]["basic"] = {
        "ip_config": os.path.join(tmp_path, "ip.txt"),
        "part_config": os.path.join(tmp_path, "part.json"),
    }

    with open(os.path.join(tmp_path, file_name+"_default.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for wrong values
    yaml_object["gsf"]["basic"] = {
        "backend": "error",
        "eval_frequency": 0,
        "model_encoder_type": "abc"
    }

    with open(os.path.join(tmp_path, file_name+"_fail.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

    # config for none exist ip config file and partition file
    yaml_object["gsf"]["basic"] = {
        "ip_config": "ip_missing.txt",
        "part_config": "part_missing.json",
    }

    with open(os.path.join(tmp_path, file_name+"_fail2.yaml"), "w") as f:
        yaml.dump(yaml_object, f)

