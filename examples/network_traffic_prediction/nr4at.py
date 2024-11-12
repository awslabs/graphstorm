# Setup log level in Jupyter Notebook to show running information
import logging
logging.basicConfig(level=20)
from graphstorm.dataloading import (GSgnnData,
                                    GSgnnNodeDataLoader)
from graphstorm.eval import GSgnnRegressionEvaluator
from nr_models import RgcnNRModel4TS, NodePredictionTrainer4TS, NodePredictionInferrer4TS


NUM_DEMAND_DAYS = 31

import graphstorm as gs
gs.initialize()

# create a GraphStorm Dataset for the movie_lens graph data generated with GraphStorm test code
ml_data = GSgnnData(part_config='./gs_1p/air_traffic.json')

# define dataloaders for training, validation, and testing
nfeats_4_modeling = {'airport':['latitude','longitude', 'inventory_amounts']}
efeats_4_modeling = {('airport', 'demand', 'airport'): ['demands'], \
                     ('airport', 'traffic', 'airport'): ['capacity', 'traffics']}

fanout=[10, 10]
train_dataloader = GSgnnNodeDataLoader(
    dataset=ml_data,
    target_idx=ml_data.get_node_train_set(ntypes=['airport']),
    node_feats=nfeats_4_modeling,
    edge_feats=efeats_4_modeling,
    label_field='inventory_amounts',
    fanout=fanout,
    batch_size=64,
    train_task=True)
val_dataloader = GSgnnNodeDataLoader(
    dataset=ml_data,
    target_idx=ml_data.get_node_val_set(ntypes=['airport']),
    node_feats=nfeats_4_modeling,
    edge_feats=efeats_4_modeling,
    label_field='inventory_amounts',
    fanout=fanout,
    batch_size=64,
    train_task=False)
test_dataloader = GSgnnNodeDataLoader(
    dataset=ml_data,
    target_idx=ml_data.get_node_test_set(ntypes=['airport']),
    node_feats=nfeats_4_modeling,
    edge_feats=efeats_4_modeling,
    label_field='inventory_amounts',
    fanout=fanout,
    batch_size=64,
    train_task=False)

# initialize the model with specific time series related arugments
ts_nfeat_names = {'airport':['inventory_amounts']}
ts_efeat_names = {('airport', 'demand', 'airport'): ['demands'], \
                  ('airport', 'traffic', 'airport'): ['traffics']}

# import a simplified RGCN model for node classification
model = RgcnNRModel4TS(
    g=ml_data.g,
    num_hid_layers=len(fanout),
    node_feat_field=nfeats_4_modeling,
    edge_feat_field=efeats_4_modeling,
    edge_feat_mp_op='add',
    target_ntype='airport',
    ts_nfeat_names=ts_nfeat_names,
    ts_efeat_names=ts_efeat_names,
    hid_size=128,
    ts_size=NUM_DEMAND_DAYS,
    window_size=7)

# setup a classification evaluator for the trainer
evaluator = GSgnnRegressionEvaluator(eval_frequency=1000)

trainer = NodePredictionTrainer4TS(model)

trainer.setup_evaluator(evaluator)
trainer.setup_device(gs.utils.get_device())

# Train the model with the trainer using fit() function
trainer.fit(train_loader=train_dataloader,
            val_loader=val_dataloader,
            test_loader=test_dataloader,
            num_epochs=100,
            save_model_frequency=1000,
            save_model_path='a_save_path/')

# after training, the best model is saved to disk
best_model_path = trainer.get_best_model_path()
print('Best model path:', best_model_path)

# we can restore the model from the saved path using the model's restore_model() function.
model.restore_model(best_model_path)

# Setup dataloader for inference
print(f'========================== Do Inference ================================')
infer_dataloader = GSgnnNodeDataLoader(
    dataset=ml_data,
    target_idx=ml_data.get_node_infer_set(ntypes='airport', mask=None),
    node_feats=nfeats_4_modeling,
    edge_feats=efeats_4_modeling,
    label_field='inventory_amounts',
    fanout=fanout,
    batch_size=64,
    train_task=False)

infer = NodePredictionInferrer4TS(model)

# Run inference on the inference dataset
infer.infer(infer_dataloader,
            use_mini_batch_infer=True,
            save_embed_path=None,
            node_id_mapping_file='./gs_1p/node_mapping.pt',
            save_prediction_path='infer/predictions')
