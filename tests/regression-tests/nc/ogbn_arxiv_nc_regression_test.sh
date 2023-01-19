#!/bin/bash

DGL_HOME=/root/dgl
GSF_HOME=/graph-storm
NUM_TRAINERS=1
export PYTHONPATH=$GSF_HOME/python/

REG_DATA_PATH=/regression-tests-data
PART_CONFIG=${REG_DATA_PATH}/ogb_arxiv_nc_train_val_1p_4t/ogbn-arxiv.json
TRAINING_CONFIG=${GSF_HOME}/tests/regression-tests/nc/arxiv_nc.yaml
LAUNCH_PATH=${DGL_HOME}/tools/launch.py
WORKSPACE=${GSF_HOME}/tests/regression-tests/nc

NUM_TRAINERS=4
NUM_SERVERS=1
NUM_SAMPLERS=0
echo "127.0.0.1" > ${WORKSPACE}/ip_list.txt
IP_CONFIG=${WORKSPACE}/ip_list.txt

echo "************** dataset: Arxiv NC regression test, *****************"
python3  ${LAUNCH_PATH} \
        --workspace ${WORKSPACE} \
        --num_trainers ${NUM_TRAINERS} \
        --num_servers ${NUM_SERVERS} \
        --num_samplers ${NUM_SAMPLERS} \
        --part_config ${PART_CONFIG} \
        --ip_config ${IP_CONFIG} \
        --ssh_port 2222 \
        "python3 ${GSF_HOME}/training_scripts/gsgnn_np/gsgnn_np.py --cf ${TRAINING_CONFIG}  \
        --ip-config ${IP_CONFIG} \
        --part-config ${PART_CONFIG} \
        --save-perf-results-path ${WORKSPACE}"

python3 ${GSF_HOME}/tools/regression_tests_utils.py --graph_name ogbn-arxiv\
                                                   --filepath ${WORKSPACE}/performance_results.json \
                                                   --task_type node_classification
