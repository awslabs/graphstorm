import torch as th
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.dataloading import GSgnnNodeDataLoader
from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.dataloading import GSgnnData
from graphstorm.utils import get_device
from graphstorm.inference import GSgnnNodePredictionInferrer
from graphstorm.trainer import GSgnnNodePredictionTrainer
from graphstorm.tracker import GSSageMakerTaskTracker
from llm_gnn_model import GNNLLM_NC

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank)
    # Define the training dataset
    train_data = GSgnnData(
        config.part_config)

    model = GNNLLM_NC(
            g=train_data.g,
            node_lm_configs=config.node_lm_configs,
            h_dim=config.hidden_size,
            out_dim=config.num_classes,
            num_layers=config.num_layers,
            target_ntype=config.target_ntype,
            use_norm=True,
            alpha_l2norm=config.alpha_l2norm,
            lr=config.lr)

    # Create a trainer for NC tasks.
    trainer = GSgnnNodePredictionTrainer(
        model,
    )

    if config.restore_model_path is not None:
        trainer.restore_model(
            model_path=config.restore_model_path,
            model_layer_to_load=["gnn", "embed"],
        )

    trainer.setup_device(device=get_device())

    # set evaluator
    evaluator = GSgnnClassificationEvaluator(
        config.eval_frequency,
        config.eval_metric,
        config.multilabel,
        config.use_early_stop,
        config.early_stop_burnin_rounds,
        config.early_stop_rounds,
        config.early_stop_strategy,
    )
    trainer.setup_evaluator(evaluator)
    tracker = GSSageMakerTaskTracker(config.eval_frequency)
    trainer.setup_task_tracker(tracker)

    # create train loader
    train_idxs = train_data.get_node_train_set(config.target_ntype)
    dataloader = GSgnnNodeDataLoader(
        train_data,
        train_idxs,
        fanout=config.fanout,
        batch_size=config.batch_size,
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        train_task=True,
    )

    # create val loader
    val_idxs = train_data.get_node_val_set(config.eval_target_ntype)
    val_dataloader = GSgnnNodeDataLoader(
        train_data,
        val_idxs,
        fanout=config.fanout,
        batch_size=config.eval_batch_size,
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        train_task=False,
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
    infer = GSgnnNodePredictionInferrer(model)
    infer.setup_device(device=get_device())
    infer.setup_evaluator(evaluator)
    infer.setup_task_tracker(tracker)
    # Create test loader
    test_idxs = train_data.get_node_test_set(config.eval_target_ntype)
    test_dataloader = GSgnnNodeDataLoader(
        train_data,
        test_idxs,
        fanout=config.fanout,
        batch_size=config.eval_batch_size,
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        train_task=False,
    )
    # Run inference on the inference dataset and save the GNN embeddings in the specified path.
    infer.infer(test_dataloader, save_embed_path=None,
                save_prediction_path=config.save_prediction_path,
                use_mini_batch_infer=True)


def generate_parser():
    """Generate an argument parser"""
    parser = get_argument_parser()
    return parser

if __name__ == "__main__":
    arg_parser = generate_parser()

    args = arg_parser.parse_args()
    print(args)
    main(args)

