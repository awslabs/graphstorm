LAUNCH_PATH=/fsx-CoreModeling/zhanhouy/home/dependency/dgl/tools/launch.py
# WORKSPACE=/fsx-CoreModeling/zhanhouy/home/workspace/graphstorm/python/graphstorm/run
WORKSPACE=/fsx-CoreModeling/zhanhouy/home/workspace/graphstorm/training_scripts/gsgnn_dt
GRAPH_NAME=20230501_20230531-3clicks-002ctr-20maxa2a-enriched-100c2q-2c2aclicks-2c2apurchases-evalv5-debug-json-graph
NUM_TRAINERS=1
NUM_SERVERS=1
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

# python3 /fsx-CoreModeling/zhanhouy/home/workspace/graphstorm/python/graphstorm/run/gs_gnn_distillation.py \
#         --workspace ${WORKSPACE} \
#         --num-trainers ${NUM_TRAINERS} \
#         --num-servers ${NUM_SERVERS} \
#         --num-samplers ${NUM_SAMPLERS} \
#         --part-config ${PART_CONFIG} \
#         --ip-config ${IP_CONFIG} \
#         --ssh-port 2222 \
#         --cf ${RUN_CONFIG}

