import torch as th
import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.dataloading import GSgnnNodeDataLoader
from graphstorm.eval import GSgnnClassificationEvaluator
from graphstorm.dataloading import GSgnnData
from graphstorm.utils import get_device
from graphstorm.trainer import GSgnnNodePredictionTrainer

from model_nc import create_rgcn_model_for_nc

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    gs.initialize(ip_config=config.ip_config, backend=config.backend,
                  local_rank=config.local_rank)

    # Define the training dataset
    train_data = GSgnnData(config.part_config)
    # Define TGAT model
    model = create_rgcn_model_for_nc(train_data.g, config)
    print(model)

    # Create a trainer for NC tasks.
    trainer = GSgnnNodePredictionTrainer(
        model, topk_model_to_save=config.topk_model_to_save
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
    val_idxs = train_data.get_node_val_set(config.eval_target_ntypes)
    val_dataloader = GSgnnNodeDataLoader(
        train_data,
        val_idxs,
        fanout=config.fanout,
        batch_size=config.eval_batch_size,
        node_feats=config.node_feat_name,
        label_field=config.label_field,
        train_task=False,
    )

    # create test loader
    test_idxs = train_data.get_node_test_set(config.eval_target_ntypes)
    test_dataloader = GSgnnNodeDataLoader(
        train_data,
        test_idxs,
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
        val_loader=val_dataloader,
        test_loader=test_dataloader,
        num_epochs=config.num_epochs,
        save_model_path=config.save_model_path,
        save_model_frequency=config.save_model_frequency,
    )


def generate_parser():
    """Generate an argument parser"""
    parser = get_argument_parser()
    return parser

if __name__ == "__main__":
    arg_parser = generate_parser()

    # Ignore unknown args to make script more robust to input arguments
    args, _ = arg_parser.parse_known_args()
    main(args)


