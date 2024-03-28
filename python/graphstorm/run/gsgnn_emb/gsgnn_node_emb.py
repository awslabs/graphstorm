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

    GSgnn pure gpu generate embeddings.
"""
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.dataloading import (GSgnnEdgeInferData, GSgnnNodeInferData)
from graphstorm.config import  (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION)
from graphstorm.inference import GSgnnEmbGenInferer
from graphstorm.utils import get_lm_ntypes

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    use_wg_feats = use_wholegraph(config.part_config)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank,
                  use_wholegraph=config.use_wholegraph_embed or use_wg_feats)
    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)

    if config.task_type == BUILTIN_TASK_LINK_PREDICTION:
        input_data = GSgnnEdgeInferData(config.graph_name,
                                        config.part_config,
                                        eval_etypes=config.eval_etype,
                                        node_feat_field=config.node_feat_name,
                                        lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    elif config.task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
        input_data = GSgnnNodeInferData(config.graph_name,
                                        config.part_config,
                                        eval_ntypes=config.target_ntype,
                                        node_feat_field=config.node_feat_name,
                                        lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    elif config.task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
        input_data = GSgnnEdgeInferData(config.graph_name,
                                        config.part_config,
                                        eval_etypes=config.target_etype,
                                        node_feat_field=config.node_feat_name,
                                        lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    else:
        raise TypeError("Not supported for task type: ", config.task_type)

    # assert the setting for the graphstorm embedding generation.
    assert config.save_embed_path is not None, \
        "save embeded path cannot be none for gs_gen_node_embeddings"
    assert config.restore_model_path is not None, \
        "restore model path cannot be none for gs_gen_node_embeddings"

    # load the model
    if config.task_type == BUILTIN_TASK_LINK_PREDICTION:
        model = gs.create_builtin_lp_gnn_model(input_data.g, config, train_task=False)
    elif config.task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
        model = gs.create_builtin_node_gnn_model(input_data.g, config, train_task=False)
    elif config.task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
        model = gs.create_builtin_edge_gnn_model(input_data.g, config, train_task=False)
    else:
        raise TypeError("Not supported for task type: ", config.task_type)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)

    # start to infer
    emb_generator = GSgnnEmbGenInferer(model)
    emb_generator.setup_device(device=get_device())

    emb_generator.infer(input_data, config.task_type,
                save_embed_path=config.save_embed_path,
                eval_fanout=config.eval_fanout,
                use_mini_batch_infer=config.use_mini_batch_infer,
                node_id_mapping_file=config.node_id_mapping_file,
                save_embed_format=config.save_embed_format)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser


if __name__ == '__main__':
    arg_parser = generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, _ = arg_parser.parse_known_args()
    main(gs_args)
