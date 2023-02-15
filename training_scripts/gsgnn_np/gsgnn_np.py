""" GSgnn node prediction
"""

import torch as th
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.trainer import GSgnnNodePredictionTrainer
from graphstorm.dataloading import GSgnnNodeTrainData, GSgnnNodeDataLoader
from graphstorm.eval import GSgnnAccEvaluator
from graphstorm.eval import GSgnnRegressionEvaluator
from graphstorm.model.utils import save_embeddings
from graphstorm.model import do_full_graph_inference

def get_eval_class(config):
    if config.task_type == "node_classification":
        return GSgnnAccEvaluator
    elif config.task_type == "node_regression":
        return GSgnnRegressionEvaluator
    else:
        raise ValueError("Unknown task type")

def main(args):
    config = GSConfig(args)

    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    train_data = GSgnnNodeTrainData(config.graph_name,
                                    config.part_config,
                                    train_ntypes=config.predict_ntype,
                                    node_feat_field=config.feat_name,
                                    label_field=config.label_field)
    model = gs.create_builtin_node_gnn_model(train_data.g, config, train_task=True)
    trainer = GSgnnNodePredictionTrainer(model, gs.get_rank(),
                                         topk_model_to_save=config.topk_model_to_save)
    if config.restore_model_path is not None:
        trainer.restore_model(model_path=config.restore_model_path)
    trainer.setup_cuda(dev_id=config.local_rank)
    if not config.no_validation:
        # TODO(zhengda) we need to refactor the evaluator.
        eval_cls = get_eval_class(config)
        trainer.setup_evaluator(eval_cls(config))
        assert len(train_data.val_idxs) > 0, "The training data do not have validation set."
        # TODO(zhengda) we need to compute the size of the entire validation set to make sure
        # we have validation data.
    tracker = gs.create_builtin_task_tracker(config, trainer.rank)
    if trainer.rank == 0:
        tracker.log_params(config.__dict__)
    trainer.setup_task_tracker(tracker)
    device = 'cuda:%d' % trainer.dev_id
    dataloader = GSgnnNodeDataLoader(train_data, train_data.train_idxs, fanout=config.fanout,
                                     batch_size=config.batch_size, device=device, train_task=True)
    val_dataloader = None
    test_dataloader = None
    # we don't need fanout for full-graph inference
    fanout = config.eval_fanout if config.mini_batch_infer else []
    if len(train_data.val_idxs) > 0:
        val_dataloader = GSgnnNodeDataLoader(train_data, train_data.val_idxs, fanout=fanout,
                                             batch_size=config.eval_batch_size,
                                             device=device, train_task=False)
    if len(train_data.test_idxs) > 0:
        test_dataloader = GSgnnNodeDataLoader(train_data, train_data.test_idxs, fanout=fanout,
                                              batch_size=config.eval_batch_size,
                                              device=device, train_task=False)
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
