""" GSgnn pure gpu link prediction
"""

import torch as th
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnLinkPredictionTrainer
from graphstorm.dataloading import GSgnnEdgeTrainData
from graphstorm.dataloading import GSgnnLinkPredictionDataLoader
from graphstorm.dataloading import GSgnnLPJointNegDataLoader
from graphstorm.dataloading import GSgnnLPLocalUniformNegDataLoader
from graphstorm.dataloading import GSgnnAllEtypeLPJointNegDataLoader
from graphstorm.dataloading import GSgnnAllEtypeLinkPredictionDataLoader
from graphstorm.dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER
from graphstorm.dataloading import BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.model.utils import save_embeddings
from graphstorm.model import do_full_graph_inference

def main(args):
    config = GSConfig(args)

    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    train_data = GSgnnEdgeTrainData(config.graph_name,
                                    config.part_config,
                                    train_etypes=config.train_etype,
                                    eval_etypes=config.eval_etype,
                                    node_feat_field=config.feat_name)
    model = gs.create_builtin_lp_gnn_model(train_data.g, config, train_task=True)
    trainer = GSgnnLinkPredictionTrainer(model, gs.get_rank(),
                                         topk_model_to_save=config.topk_model_to_save)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path)
    trainer.setup_cuda(dev_id=config.local_rank)
    if not config.no_validation:
        # TODO(zhengda) we need to refactor the evaluator.
        trainer.setup_evaluator(GSgnnMrrLPEvaluator(train_data.g, config, train_data))
        assert len(train_data.val_idxs) > 0, "The training data do not have validation set."
        # TODO(zhengda) we need to compute the size of the entire validation set to make sure
        # we have validation data.
    tracker = gs.create_builtin_task_tracker(config, trainer.rank)
    if trainer.rank == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)

    if config.negative_sampler == BUILTIN_LP_UNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnLinkPredictionDataLoader
    elif config.negative_sampler == BUILTIN_LP_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnLPJointNegDataLoader
    elif config.negative_sampler == BUILTIN_LP_LOCALUNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnLPLocalUniformNegDataLoader
    elif config.negative_sampler == BUILTIN_LP_ALL_ETYPE_UNIFORM_NEG_SAMPLER:
        dataloader_cls = GSgnnAllEtypeLinkPredictionDataLoader
    elif config.negative_sampler == BUILTIN_LP_ALL_ETYPE_JOINT_NEG_SAMPLER:
        dataloader_cls = GSgnnAllEtypeLPJointNegDataLoader
    else:
        raise Exception('Unknown negative sampler')
    device = 'cuda:%d' % trainer.dev_id
    dataloader = dataloader_cls(train_data, train_data.train_idxs, config.fanout,
                                config.batch_size, config.num_negative_edges, device,
                                train_task=True,
                                reverse_edge_types_map=config.reverse_edge_types_map,
                                exclude_training_targets=config.exclude_training_targets)

    # TODO(zhengda) let's use full-graph inference for now.
    val_dataloader = None
    test_dataloader = None
    trainer.fit(train_loader=dataloader, val_loader=val_dataloader,
                test_loader=test_dataloader, n_epochs=config.n_epochs,
                save_model_path=config.save_model_path,
                mini_batch_infer=config.mini_batch_infer,
                save_model_per_iters=config.save_model_per_iters,
                save_perf_results_path=config.save_perf_results_path)

    if config.save_embed_path is not None:
        best_model = trainer.get_best_model().to(device)
        assert best_model is not None, "Cannot get the best model from the trainer."
        embeddings = do_full_graph_inference(best_model, train_data, task_tracker=tracker)
        save_embeddings(config.save_embed_path, embeddings, gs.get_rank(),
                        th.distributed.get_world_size())

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
