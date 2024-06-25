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
import logging

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.utils import rt_profiler, sys_tracker, get_device, use_wholegraph
from graphstorm.dataloading import GSgnnData
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_LINK_PREDICTION,
                               GRAPHSTORM_MODEL_ALL_LAYERS,
                               GRAPHSTORM_MODEL_EMBED_LAYER,
                               GRAPHSTORM_MODEL_GNN_LAYER,
                               GRAPHSTORM_MODEL_DECODER_LAYER)
from graphstorm.inference import GSgnnEmbGenInferer
from graphstorm.utils import get_lm_ntypes
from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel

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

    if config.multi_tasks is None:
        # if not multi-task, check task type
        assert config.task_type in [BUILTIN_TASK_LINK_PREDICTION,
                                    BUILTIN_TASK_NODE_REGRESSION,
                                    BUILTIN_TASK_NODE_CLASSIFICATION,
                                    BUILTIN_TASK_EDGE_CLASSIFICATION,
                                    BUILTIN_TASK_EDGE_REGRESSION], \
            f"Not supported for task type: {config.task_type}"

    input_data = GSgnnData(config.part_config,
                           node_feat_field=config.node_feat_name,
                           edge_feat_field=config.edge_feat_name,
                           lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))

    # assert the setting for the graphstorm embedding generation.
    assert config.save_embed_path is not None, \
        "save embeded path cannot be none for gs_gen_node_embeddings"
    assert config.restore_model_path is not None, \
        "restore model path cannot be none for gs_gen_node_embeddings"

    # load the model
    if config.multi_tasks:
        # Only support multi-task shared encoder model.
        model = GSgnnMultiTaskSharedEncoderModel(config.alpha_l2norm)
        gs.gsf.set_encoder(model, input_data.g, config, train_task=False)
        assert config.restore_model_layers is not GRAPHSTORM_MODEL_ALL_LAYERS, \
            "When computing node embeddings with GSgnnMultiTaskSharedEncoderModel, " \
            "please set --restore-model-layers to " \
            f"{GRAPHSTORM_MODEL_EMBED_LAYER}, {GRAPHSTORM_MODEL_GNN_LAYER}." \
            f"Please do not include {GRAPHSTORM_MODEL_DECODER_LAYER}, " \
            f"but we get {config.restore_model_layers}"
    else:
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

    if config.multi_tasks:
        # infer_ntypes = None means all node types.
        infer_ntypes = None
    else:
        task_type = config.task_type
        # infer ntypes must be sorted for node embedding saving
        if task_type == BUILTIN_TASK_LINK_PREDICTION:
            infer_ntypes = None
        elif task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
            infer_ntypes = [config.target_ntype]
        elif task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
            infer_ntypes = set()
            for etype in config.target_etype:
                infer_ntypes.add(etype[0])
                infer_ntypes.add(etype[2])
            infer_ntypes = sorted(list(infer_ntypes))
        else:
            raise TypeError("Not supported for task type: ", task_type)

    emb_generator.infer(input_data, infer_ntypes,
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
    gs_args, unknown_args = arg_parser.parse_known_args()
    logging.warning("Unknown arguments for "
                    "graphstorm.run.gs_gen_node_embedding: %s",
                    unknown_args)
    main(gs_args)
