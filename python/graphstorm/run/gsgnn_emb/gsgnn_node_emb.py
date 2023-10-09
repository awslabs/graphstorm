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
from graphstorm.dataloading import (GSgnnEdgeInferData, GSgnnNodeInferData,
                            GSgnnEdgeDataLoader, GSgnnNodeDataLoader,
                            GSgnnLinkPredictionTestDataLoader,
                            GSgnnLinkPredictionJointTestDataLoader)
from graphstorm.utils import rt_profiler, sys_tracker, setup_device, use_wholegraph
from graphstorm.config import  (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION)
from graphstorm.dataloading import (BUILTIN_LP_UNIFORM_NEG_SAMPLER,
                                    BUILTIN_LP_JOINT_NEG_SAMPLER)
from graphstorm.inference import GSgnnEmbGenInferer

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(True)

    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  use_wholegraph=use_wholegraph(config.part_config))
    rt_profiler.init(config.profile_path, rank=gs.get_rank())
    sys_tracker.init(config.verbose, rank=gs.get_rank())
    device = setup_device(config.local_rank)
    tracker = gs.create_builtin_task_tracker(config)
    if gs.get_rank() == 0:
        tracker.log_params(config.__dict__)

    if config.task_type == BUILTIN_TASK_LINK_PREDICTION:
        input_graph = GSgnnEdgeInferData(config.graph_name,
                                        config.part_config,
                                        eval_etypes=config.eval_etype,
                                        node_feat_field=config.node_feat_name,
                                        decoder_edge_feat=config.decoder_edge_feat)
    elif config.task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
        input_graph = GSgnnNodeInferData(config.graph_name,
                                    config.part_config,
                                    eval_ntypes=config.target_ntype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field)
    elif config.task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
        input_graph = GSgnnEdgeInferData(config.graph_name,
                                    config.part_config,
                                    eval_etypes=config.target_etype,
                                    node_feat_field=config.node_feat_name,
                                    label_field=config.label_field,
                                    decoder_edge_feat=config.decoder_edge_feat)
    else:
        raise TypeError("Not supported for task type: ", config.task_type)

    # assert the setting for the graphstorm embedding generation.
    assert config.save_embed_path is not None, \
        "save embeded path cannot be none for gs_gen_node_embeddings"
    assert config.restore_model_path is not None, \
        "restore model path cannot be none for gs_gen_node_embeddings"

    if config.task_type == BUILTIN_TASK_LINK_PREDICTION:
        model = gs.create_builtin_lp_gnn_model(input_graph.g, config, train_task=False)
    elif config.task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
        model = gs.create_builtin_node_gnn_model(input_graph.g, config, train_task=False)
    elif config.task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
        model = gs.create_builtin_edge_gnn_model(input_graph.g, config, train_task=False)
    else:
        raise TypeError("Not supported for task type: ", config.task_type)

    if config.task_type == BUILTIN_TASK_LINK_PREDICTION:
        if config.eval_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
            link_prediction_loader = GSgnnLinkPredictionTestDataLoader
        elif config.eval_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
            link_prediction_loader = GSgnnLinkPredictionJointTestDataLoader
        else:
            raise ValueError('Unknown test negative sampler.'
                             'Supported test negative samplers include '
                             f'[{BUILTIN_LP_UNIFORM_NEG_SAMPLER}, {BUILTIN_LP_JOINT_NEG_SAMPLER}]')

        dataloader = link_prediction_loader(input_graph, input_graph.test_idxs,
                                         batch_size=config.eval_batch_size,
                                         num_negative_edges=config.num_negative_edges_eval,
                                         fanout=config.eval_fanout)
    elif config.task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
        dataloader = GSgnnNodeDataLoader(input_graph, input_graph.infer_idxs,
                                         fanout=config.eval_fanout,
                                         batch_size=config.eval_batch_size, device=device,
                                         train_task=False,
                                         construct_feat_ntype=config.construct_feat_ntype,
                                         construct_feat_fanout=config.construct_feat_fanout)
    elif config.task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
        dataloader = GSgnnEdgeDataLoader(input_graph, input_graph.infer_idxs,
                                         fanout=config.eval_fanout,
                                         batch_size=config.eval_batch_size,
                                         device=device, train_task=False,
                                         reverse_edge_types_map=config.reverse_edge_types_map,
                                         remove_target_edge_type=config.remove_target_edge_type,
                                         construct_feat_ntype=config.construct_feat_ntype,
                                         construct_feat_fanout=config.construct_feat_fanout)
    else:
        raise TypeError("Not supported for task type: ", config.task_type)

    emb_generator = GSgnnEmbGenInferer(model)
    emb_generator.setup_device(device=device)

    emb_generator.infer(input_graph, config.task_type,
                save_embed_path=config.save_embed_path,
                loader=dataloader,
                use_mini_batch_infer=config.use_mini_batch_infer,
                node_id_mapping_file=config.node_id_mapping_file,
                return_proba=config.return_proba,
                save_embed_format=config.save_embed_format)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser


if __name__ == '__main__':
    arg_parser = generate_parser()

    args = arg_parser.parse_args()
    main(args)
