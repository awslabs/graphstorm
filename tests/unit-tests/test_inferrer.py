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
import dgl
from argparse import Namespace
import torch as th

from graphstorm.config import GSConfig
from graphstorm.tracker import GSSageMakerTaskTracker
from graphstorm import create_builtin_node_gnn_model
from graphstorm.inference.graphstorm_infer import GSInferrer
from graphstorm.eval import GSgnnAccEvaluator

from data_utils import generate_dummy_dist_graph


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


def test_inferrer_setup_evaluator():

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
    inferrer = GSInferrer(model)

    # case 1: by default trainer has no task_tracker
    assert inferrer.task_tracker is None

    evaluator = GSgnnAccEvaluator(config.eval_frequency,
                                  config.eval_metric,
                                  config.multilabel,
                                  config.use_early_stop)

    # case 2: evaluator has no task_tracker by default
    assert evaluator.task_tracker is None
    
    # case 3: when setup an evaluator that has no task_tracker and train has no task tracker
    #         eitehr, create a new task_tracker and set it to the evaluator.
    inferrer.setup_evaluator(evaluator)

    assert inferrer.task_tracker is not None
    assert evaluator.eval_frequency == inferrer.task_tracker.log_report_frequency
    assert evaluator.task_tracker == inferrer.task_tracker

    # case 4: when setup an evaluator that has no task_tracker, but train has a task tracker,
    #         use the trainer's task_tracker to setup the evaluator.
    inferrer.setup_task_tracker(GSSageMakerTaskTracker(10))
    evaluator.setup_task_tracker(None)
    inferrer.setup_evaluator(evaluator)

    assert evaluator.task_tracker == inferrer.task_tracker
    assert evaluator.task_tracker.log_report_frequency == 10

    # case 5: when setup an evaluator that has a task_tracker, no change of the evaluator.
    evaluator.setup_task_tracker(GSSageMakerTaskTracker(100))
    inferrer.setup_evaluator(evaluator)

    assert evaluator.task_tracker != inferrer.task_tracker
    assert evaluator.task_tracker.log_report_frequency == 100
    assert inferrer.task_tracker.log_report_frequency != 100

    th.distributed.destroy_process_group()
    dgl.distributed.kvstore.close_kvstore()


if __name__ == '__main__':
    test_inferrer_setup_evaluator()
