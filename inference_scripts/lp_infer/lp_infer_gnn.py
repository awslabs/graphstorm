""" Inference script for link prediction tasks with GNN only
"""

import torch as th
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.inference import GSgnnLinkPredictionInfer
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.dataloading import GSgnnEdgeInferData, GSgnnEdgeDataLoader
from graphstorm.dataloading import GSgnnLinkPredictionDataLoader

def main(args):
    config = GSConfig(args)
    gs.initialize(ip_config=config.ip_config, backend=config.backend)

    infer_data = GSgnnEdgeInferData(config.graph_name,
                                    config.part_config,
                                    eval_etypes=config.eval_etype,
                                    node_feat_field=config.feat_name)
    model = gs.create_builtin_lp_gnn_model(infer_data.g, config, train_task=False)
    model.restore_model(config.restore_model_path)
    # TODO(zhengda) we should use a different way to get rank.
    infer = GSgnnLinkPredictionInfer(model, gs.get_rank())
    infer.setup_cuda(dev_id=config.local_rank)
    if not config.no_validation:
        infer.setup_evaluator(GSgnnMrrLPEvaluator(infer_data.g, config, infer_data))
        assert len(infer_data.test_idxs) > 0, "There is not test data for evaluation."
    tracker = gs.create_builtin_task_tracker(config, infer.rank)
    infer.setup_task_tracker(tracker)
    device = 'cuda:%d' % infer.dev_id
    # We only support full-graph inference for now.
    fanout = []
    dataloader = GSgnnLinkPredictionDataLoader(infer_data, infer_data.test_idxs, fanout=fanout,
                                     batch_size=config.eval_batch_size,
                                     num_negative_edges=config.num_negative_edges,
                                     device=device, train_task=False)
    infer.infer(dataloader, save_embed_path=config.save_embed_path)

def generate_parser():
    parser = get_argument_parser()
    return parser

if __name__ == '__main__':
    parser=generate_parser()

    args = parser.parse_args()
    print(args)
    main(args)
