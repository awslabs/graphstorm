GSF_HOME=/graph-storm
REG_DATA_PATH=/regression-tests-data/ogbn-mag-data
export PYTHONPATH=${GSF_HOME}/python/
PART_CONFIG=${REG_DATA_PATH}/ogb_mag_lp_train_val_1p_4t/ogbn-mag.json
TRAINING_CONFIG=${GSF_HOME}/tests/regression-tests/lp/mag_lp.yaml

mkdir ${REG_DATA_PATH}/model_path
SAVE_MODEL_PATH=${REG_DATA_PATH}/model_path

LAUNCH_PATH=~/dgl/tools/launch.py
WORKSPACE=${GSF_HOME}/tests/regression-tests/lp

NUM_TRAINERS=4
NUM_SERVERS=1
NUM_SAMPLERS=0
echo "127.0.0.1" > ${WORKSPACE}/ip_list.txt
IP_CONFIG=${WORKSPACE}/ip_list.txt


python3  ${LAUNCH_PATH} \
        --workspace ${WORKSPACE} \
        --num_trainers ${NUM_TRAINERS} \
        --num_servers ${NUM_SERVERS} \
        --num_samplers ${NUM_SAMPLERS} \
        --part_config ${PART_CONFIG} \
        --ip_config ${IP_CONFIG} \
        --ssh_port 2222 \
        "python3 ${GSF_HOME}/training_scripts/gsgnn_lp/gsgnn_lp.py --cf ${TRAINING_CONFIG} \
        --num-gpus ${NUM_TRAINERS}\
        --ip-config ${IP_CONFIG} \
        --part-config ${PART_CONFIG} \
        --feat-name paper:feat \
        --save-model-path ${SAVE_MODEL_PATH} \
        --save-perf-results-path ${WORKSPACE}"

python3  ${GSF_HOME}/tools/regression_tests_utils.py --graph_name ogbn-mag \
                                                     --filepath ${WORKSPACE}/performance_results.json \
                                                     --task_type link_prediction
