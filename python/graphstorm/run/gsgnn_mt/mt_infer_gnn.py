"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GSgnn multi-task inference entry point.
"""
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION,
                               BUILTIN_TASK_EDGE_CLASSIFICATION,
                               BUILTIN_TASK_EDGE_REGRESSION,
                               BUILTIN_TASK_LINK_PREDICTION)
from graphstorm.dataloading import GSgnnData
from graphstorm.dataloading import (GSgnnNodeDataLoader,
                                    GSgnnEdgeDataLoader,
                                    GSgnnMultiTaskDataLoader)
from graphstorm.dataloading import (GSgnnLinkPredictionTestDataLoader,
                                    GSgnnLinkPredictionJointTestDataLoader,
                                    GSgnnLinkPredictionPredefinedTestDataLoader)
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER

from graphstorm.model.multitask_gnn import GSgnnMultiTaskSharedEncoderModel
from graphstorm.inference import GSgnnLinkPredictionInferrer

from graphstorm.utils import get_device, use_wholegraph
from graphstorm.utils import get_lm_ntypes
from gsgnn_mt import create_evaluator

def create_task_infer_dataloader(task, config, infer_data):
    """ Create task specific dataloader for inference tasks

    Parameters
    ----------
    task: TaskInfo
        Task info.
    config: GSConfig
        Inference config.
    infer_data: GSgnnData
        Inference data.

    Return
    ------
    Task dataloader
    """
    task_config = task.task_config
    # All tasks share the same input encoder, so the node feats must be same.
    node_feats = config.node_feat_name
    if task.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
        eval_ntype = task_config.target_ntype
        if not config.no_validation:
            target_idxs = infer_data.get_edge_test_set(eval_ntype, mask=task_config.test_mask)
            assert len(target_idxs) > 0, \
                f"There is not test data for evaluation for task {task.task_id}. " \
                "You can use --no-validation true to avoid do testing"
        else:
            target_idxs = infer_data.get_node_infer_set(eval_ntype, mask=task_config.test_mask)
            assert len(target_idxs) > 0, \
                f"To do inference on {config.target_ntype} for {task.task_id} " \
                "task without doing evaluation, you should not define test_mask " \
                "as its node feature. GraphStorm will do inference on the whole node set. "
        # All tasks share the same GNN model, so the fanout should be the global fanout
        fanout = config.eval_fanout if config.use_mini_batch_infer else []
        return GSgnnNodeDataLoader(infer_data,
                                   target_idxs,
                                   fanout=fanout,
                                   batch_size=task_config.eval_batch_size,
                                   train_task=False,
                                   node_feats=node_feats,
                                   label_field=task_config.label_field)
    elif task.task_type in [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
        eval_etype = task_config.target_etype
        if not config.no_validation:
            target_idxs = infer_data.get_edge_test_set(eval_etype)
            assert len(target_idxs) > 0, \
                f"There is not test data for evaluation for task {task.task_id}. " \
                "You can use --no-validation true to avoid do testing"
        else:
            target_idxs = infer_data.get_edge_infer_set(eval_etype)
            assert len(target_idxs) > 0, \
                f"To do inference on {config.target_etype} for {task.task_id} " \
                "without doing evaluation, you should not define test_mask as its " \
                "edge feature. GraphStorm will do inference on the whole edge set. "
        # All tasks share the same GNN model, so the fanout should be the global fanout
        fanout = config.eval_fanout if task_config.use_mini_batch_infer else []
        return GSgnnEdgeDataLoader(infer_data,
                                   target_idxs,
                                   fanout=fanout,
                                   batch_size=task_config.eval_batch_size,
                                   node_feats=node_feats,
                                   label_field=task_config.label_field,
                                   decoder_edge_feats=task_config.decoder_edge_feat,
                                   train_task=False,
                                   reverse_edge_types_map=task_config.reverse_edge_types_map,
                                   remove_target_edge_type=task_config.remove_target_edge_type)
    elif task.task_type in [BUILTIN_TASK_LINK_PREDICTION]:
        eval_etype = task_config.eval_etype
        if not config.no_validation:
            infer_idxs = infer_data.get_edge_test_set(eval_etype)
            assert len(infer_idxs) > 0, \
                "There is not test data for evaluation for task {task.task_id}. "
        else:
            infer_idxs = infer_data.get_edge_infer_set(eval_etype)

        # We do not need fanout for full graph inference.
        # In full graph inference, the test data loader
        # is only used to provide test edges.
        fanout = config.eval_fanout if task_config.use_mini_batch_infer else []
        if task_config.eval_etypes_negative_dstnode is not None:
            return GSgnnLinkPredictionPredefinedTestDataLoader(
                infer_data, infer_idxs,
                batch_size=task_config.eval_batch_size,
                fixed_edge_dst_negative_field=task_config.eval_etypes_negative_dstnode,
                fanout=fanout,
                node_feats=node_feats)
        else:
            if config.eval_negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
                test_dataloader_cls = GSgnnLinkPredictionTestDataLoader
            elif config.eval_negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
                test_dataloader_cls = GSgnnLinkPredictionJointTestDataLoader
            else:
                raise ValueError('Unknown test negative sampler.'
                    'Supported test negative samplers include '
                    f'[{BUILTIN_LP_UNIFORM_NEG_SAMPLER}, {BUILTIN_LP_JOINT_NEG_SAMPLER}]')

            return test_dataloader_cls(infer_data, infer_idxs,
                batch_size=task_config.eval_batch_size,
                num_negative_edges=task_config.num_negative_edges_eval,
                fanout=fanout,
                node_feats=node_feats)
    return None

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    config.verify_arguments(False)

    use_wg_feats = use_wholegraph(config.part_config)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank,
                  use_wholegraph=config.use_wholegraph_embed or use_wg_feats)

    infer_data = GSgnnData(config.part_config,
                           node_feat_field=config.node_feat_name,
                           edge_feat_field=config.edge_feat_name,
                           lm_feat_ntypes=get_lm_ntypes(config.node_lm_configs))
    model = GSgnnMultiTaskSharedEncoderModel(config.alpha_l2norm)
    gs.gsf.set_encoder(model, infer_data.g, config, train_task=False)
    tasks = config.multi_tasks
    dataloaders = []
    task_evaluators = {}
    encoder_out_dims = model.gnn_encoder.out_dims \
        if model.gnn_encoder is not None \
            else model.node_input_encoder.out_dims

    for task in tasks:
        decoder, loss_func = gs.create_task_decoder(task, infer_data.g, encoder_out_dims, train_task=True)

        data_loader = create_task_infer_dataloader(task, config, infer_data)
        dataloaders.append(data_loader)
        if not config.no_validation:
            task_evaluators[task.task_id] = \
                create_evaluator(task)
        model.add_task(task.task_id, task.task_type, decoder, loss_func)


    infer_dataloader = GSgnnMultiTaskDataLoader(infer_data, tasks, dataloaders)
    model.restore_model(config.restore_model_path,
                        model_layer_to_load=config.restore_model_layers)
    infer = GSgnnLinkPredictionInferrer(model)
    infer.setup_device(device=get_device())
    infer.infer(infer_data,
                infer_dataloader,
                save_embed_path=config.save_embed_path,
                save_prediction_path=config.save_prediction_path,
                use_mini_batch_infer=config.use_mini_batch_infer,
                node_id_mapping_file=config.node_id_mapping_file,
                edge_id_mapping_file=config.edge_id_mapping_file,
                return_proba=config.return_proba,
                edge_mask_for_gnn_embeddings=None if config.no_validation else \
                    'train_mask', # if no validation,any edge can be used in message passing.
                save_embed_format=config.save_embed_format)

def generate_parser():
    """ Generate an argument parser
    """
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    arg_parser=generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    gs_args, _ = arg_parser.parse_known_args()
    main(gs_args)
