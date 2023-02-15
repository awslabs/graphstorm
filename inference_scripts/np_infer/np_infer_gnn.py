""" Inference script for node classification/regression tasks with GNN only
"""

import torch as th
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnNodePredictionInfer
from graphstorm.eval import GSgnnAccEvaluator, GSgnnRegressionEvaluator
from graphstorm.dataloading import GSgnnNodeInferData, GSgnnNodeDataLoader

def get_eval_class(config): # pylint: disable=unused-argument
    """ Get evaluator class
    """
    if config.task_type == "node_regression":
        return GSgnnRegressionEvaluator
    elif config.task_type == 'node_classification':
        return GSgnnAccEvaluator
    else:
        raise AttributeError(config.task_type + ' is not supported.')

def main(args):
    config = GSConfig(args)
    gs.initialize(ip_config=config.ip_config, backend=config.backend)

    infer_data = GSgnnNodeInferData(config.graph_name,
                                    config.part_config,
                                    eval_ntypes=config.predict_ntype,
                                    node_feat_field=config.feat_name,
                                    label_field=config.label_field)
    model = gs.create_builtin_node_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path)
    # TODO(zhengda) we should use a different way to get rank.
    infer = GSgnnNodePredictionInfer(model, gs.get_rank())
    infer.setup_cuda(dev_id=config.local_rank)
    if not config.no_validation:
        eval_class = get_eval_class(config)
        infer.setup_evaluator(eval_class(config))
        assert len(infer_data.test_idxs) > 0, "There is not test data for evaluation."
    tracker = gs.create_builtin_task_tracker(config, infer.rank)
    infer.setup_task_tracker(tracker)
    device = 'cuda:%d' % infer.dev_id
    fanout = config.eval_fanout if config.mini_batch_infer else []
    dataloader = GSgnnNodeDataLoader(infer_data, infer_data.test_idxs, fanout=fanout,
                                     batch_size=config.eval_batch_size, device=device,
                                     train_task=False)

    infer.infer(dataloader, save_embed_path=config.save_embed_path,
                save_predict_path=config.save_predict_path,
                mini_batch_infer=config.mini_batch_infer)

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
