date
GS_HOME=$(pwd)
mkdir -p /regression-tests-data
REG_DATA_PATH=/regression-tests-data


# create data for ogb_arxiv nc regression tests
# 1. Construct the graph directly downloading from the OGB site and partition the graph for test
python3 tools/partition_graph.py --dataset ogbn-arxiv \
                                 --filepath ${REG_DATA_PATH}/ogbn-arxiv-nc/ \
                                 --num_parts 1 \
                                 --num_trainers_per_machine 4 \
                                 --output ${REG_DATA_PATH}/ogb_arxiv_nc_train_val_1p_4t

date
