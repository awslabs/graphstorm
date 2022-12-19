#!/bin/bash
LAUNCH_PATH=~/dgl/tools/launch.py
WORKSPACE=/graph-storm/training_scripts/gsgnn_nc
NUM_TRAINERS=4
NUM_SERVERS=1
NUM_SAMPLERS=0
PART_CONFIG=/data/ogbn-papers100M-4p/ogbn-papers100M.json
IP_CONFIG=/data/ip_list_p4.txt
TRAINING_CONFIG=/graph-storm/tests/regression-tests/OGBN-papers100M/ogbn_papers100M_nc_p4.yaml
export PYTHONPATH=/graph-storm/python/

python3 -u  ${LAUNCH_PATH} \
        --workspace ${WORKSPACE} \
        --num_trainers ${NUM_TRAINERS} \
        --num_servers ${NUM_SERVERS} \
        --num_samplers ${NUM_SAMPLERS} \
        --part_config ${PART_CONFIG} \
        --ip_config ${IP_CONFIG} \
        --ssh_port 2222 \
        "python3 gsgnn_nc.py --cf ${TRAINING_CONFIG} --feat-name feat"
