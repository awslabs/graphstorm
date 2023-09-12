# GNN Distillation Tutorial
GraphStorm supports to distill well-trained GNN models to user specified Transformer-based student model. User can also specify the node types to be distilled, where each node type will have a corresponding student model. The distillation is conducted to minimize the embeddings between GNN checkpoint and student model. MSE is used to supervise the training.

## Input
* A well trained GNN checkpoint.
* Paritioned graph.
* Textual dataset for distillation.
* Node types and textual features for distillation.
* Student model name from HuggingFace.
* Pre-trained weight name to initialize the student model.
* Distillation related hyper-parameters (e.g., learning rate, saved_path.

## Output
* Distilled Student model.

## Example Yaml Config
See distill_debug.yaml

## Running Command
```
LAUNCH_PATH=/fsx-CoreModeling/zhanhouy/home/dependency/dgl/tools/launch.py
WORKSPACE=/fsx-CoreModeling/zhanhouy/home/workspace/graphstorm/training_scripts/gsgnn_dt
GRAPH_NAME=20230501_20230531-3clicks-002ctr-20maxa2a-enriched-100c2q-2c2aclicks-2c2apurchases-evalv5-debug-json-graph
NUM_TRAINERS=2
NUM_SERVERS=2
NUM_SAMPLERS=0
PART_CONFIG=/fsx-CoreModeling/zhanhouy/home/m5gnn_dataset/us_ads_sourcing/${GRAPH_NAME}-dgl-graph/dist_${NUM_SERVERS}/${GRAPH_NAME}.json
EXTRA_ENVS="LD_LIBRARY_PATH=/usr/local/cuda/lib64:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/home/deepspeed/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH"
IP_CONFIG=ip_list_debug.txt
RUN_CONFIG=distill_debug.yaml
cd ${WORKSPACE}
export PYTHONPATH=$PYTHONPATH:/fsx-CoreModeling/zhanhouy/home/workspace/graphstorm/python

python3 -m graphstorm.run.gs_gnn_distillation \
        --workspace ${WORKSPACE} \
        --num-trainers ${NUM_TRAINERS} \
        --num-servers ${NUM_SERVERS} \
        --num-samplers ${NUM_SAMPLERS} \
        --part-config ${PART_CONFIG} \
        --ip-config ${IP_CONFIG} \
        --ssh-port 2222 \
        --cf ${RUN_CONFIG}
```