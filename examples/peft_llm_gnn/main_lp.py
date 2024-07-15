import torch as th
import time
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.dataloading import GSgnnLinkPredictionDataLoader, GSgnnLinkPredictionTestDataLoader
from graphstorm.eval import GSgnnMrrLPEvaluator
from graphstorm.dataloading import GSgnnData
from graphstorm.utils import get_device
from graphstorm.inference import GSgnnLinkPredictionInferrer
from graphstorm.trainer import GSgnnLinkPredictionTrainer
from graphstorm.tracker import GSSageMakerTaskTracker
from llm_gnn_model import GNNLLM_LP


def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank)
    # Define the training dataset
    train_data = GSgnnData(
        config.part_config,
        node_feat_field=config.node_feat_name,
    )
    train_etypes=config.train_etype
    eval_etypes=config.eval_etype

    model = GNNLLM_LP(
            g=train_data.g,
            node_lm_configs=config.node_lm_configs,
            h_dim=config.hidden_size,
            num_layers=config.num_layers,
            target_ntype=config.target_ntype,
            target_etype=config.train_etype,
            use_norm=True,
            alpha_l2norm=config.alpha_l2norm,
            lr=config.lr)

    # Create a trainer for LP tasks.
    trainer = GSgnnLinkPredictionTrainer(
        model, topk_model_to_save=1
    )

    if config.restore_model_path is not None:
        trainer.restore_model(
            model_path=config.restore_model_path,
            model_layer_to_load=["gnn", "embed"],
        )

    trainer.setup_device(device=get_device())

    # set evaluator
    evaluator = GSgnnMrrLPEvaluator(
        eval_frequency=config.eval_frequency,
        use_early_stop=config.use_early_stop,
        early_stop_burnin_rounds=config.early_stop_burnin_rounds,
        early_stop_rounds=config.early_stop_rounds,
        early_stop_strategy=config.early_stop_strategy
    )
    # disbale validation for efficiency
    # trainer.setup_evaluator(evaluator)
    tracker = GSSageMakerTaskTracker(config.eval_frequency)
    trainer.setup_task_tracker(tracker)

    # create train loader with uniform negative sampling
    train_idxs = train_data.get_edge_train_set(train_etypes)
    dataloader = GSgnnLinkPredictionDataLoader(
        train_data,
        train_idxs,
        fanout=config.fanout,
        batch_size=config.batch_size,
        num_negative_edges=config.num_negative_edges,
        node_feats=config.node_feat_name,
        train_task=True,
        reverse_edge_types_map=config.reverse_edge_types_map,
        exclude_training_targets=config.exclude_training_targets,
    )

    # create val loader
    val_idxs = train_data.get_edge_val_set(eval_etypes)
    val_dataloader = GSgnnLinkPredictionTestDataLoader(
        train_data,
        val_idxs,
        batch_size=config.eval_batch_size,
        num_negative_edges=config.num_negative_edges,
        node_feats=config.node_feat_name,
        fanout=config.fanout,
    )

    # Start the training process.
    model.prepare_input_encoder(train_data)
    trainer.fit(
        train_loader=dataloader,
        num_epochs=config.num_epochs,
        val_loader=val_dataloader,
        # disable testing during training
        test_loader=None,
        save_model_path=config.save_model_path,
        save_model_frequency=config.save_model_frequency,
        use_mini_batch_infer=True
    )

    # Load the best checkpoint
    best_model_path = trainer.get_best_model_path()
    model.restore_model(best_model_path)

    # Create an inference for a node task.
    infer = GSgnnLinkPredictionInferrer(model)
    infer.setup_device(device=get_device())
    infer.setup_evaluator(evaluator)
    infer.setup_task_tracker(tracker)
    # Create test loader
    infer_idxs = train_data.get_edge_infer_set(eval_etypes)
    test_dataloader = GSgnnLinkPredictionTestDataLoader(
        train_data,
        infer_idxs,
        batch_size=config.eval_batch_size,
        num_negative_edges=config.num_negative_edges_eval,
        node_feats=config.node_feat_name,
        fanout=config.fanout,
    )
    # Run inference on the inference dataset and save the GNN embeddings in the specified path.
    infer.infer(train_data, test_dataloader, save_embed_path=config.save_embed_path,
                edge_mask_for_gnn_embeddings='train_mask',
                use_mini_batch_infer=True, infer_batch_size=config.eval_batch_size)

def generate_parser():
    """Generate an argument parser"""
    parser = get_argument_parser()
    return parser

if __name__ == "__main__":
    arg_parser = generate_parser()

    args = arg_parser.parse_args()
    print(args)
    main(args)

