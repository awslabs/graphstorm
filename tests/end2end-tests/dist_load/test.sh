#!/bin/bash

service ssh restart

DGL_HOME=/data/dgl
GS_HOME=$(pwd)
NUM_WORKERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/tools

echo "127.0.0.1" > ip_list.txt

cat ip_list.txt

rm -fr ml-output

# Run multi process processing first
python3 launch_dist_process.py --workspace $GS_HOME/tools --num_workers $NUM_WORKERS --ip_config ip_list.txt --ssh_port 2222 'python3 preprocess_dist_graph.py --name ml --filepath /data/ml-json --output ml-output --hf_bert_model "bert-base-uncased" --ntext_fields "movie:title" --nlabel_fields "movie:genre" --predict_ntype "movie" --ntask_types "movie:classify" --undirected  --generate_new_split true --ntypes "movie occupation user" --etypes "user,rating,movie user,has-occupation,occupation"'

cnt=$(ls ml-output | wc -l)
if test $cnt -ne 16
then
    echo "output file is not completed"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges0_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges1_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges2_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges3_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check node feature
if [ ! -d "ml-output/movie/genre" ]
then
    echo "missing genre"
    exit -1
fi
if [ ! -d "ml-output/movie/input_ids" ]
then
    echo "missing input_ids"
    exit -1
fi
if [ ! -d "ml-output/movie/test_mask" ]
then
    echo "missing test_mask"
    exit -1
fi
if [ ! -d "ml-output/movie/train_mask" ]
then
    echo "missing train_mask"
    exit -1
fi
if [ ! -d "ml-output/movie/val_mask" ]
then
    echo "missing val_mask"
    exit -1
fi
if [ ! -d "ml-output/movie/valid_len" ]
then
    echo "missing valid_len"
    exit -1
fi

cnt=$(ls ml-output/movie/genre | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature genre files does not equal to 2"
    exit -1
fi

cnt=$(ls ml-output/movie/input_ids | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature input_ids files does not equal to 2"
    exit -1
fi

cnt=$(ls ml-output/movie/test_mask | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature test_mask files does not equal to 2"
    exit -1
fi

cnt=$(ls ml-output/movie/train_mask | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature train_mask files does not equal to 2"
    exit -1
fi

cnt=$(ls ml-output/movie/val_mask | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature val_mask files does not equal to 2"
    exit -1
fi

cnt=$(ls ml-output/movie/valid_len | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature valid_len files does not equal to 2"
    exit -1
fi

# Now check the graph file
ls $GS_HOME/tests/end2end-tests/dist_load/
python3 $GS_HOME/tests/end2end-tests/dist_load/test_dist_load_nc.py --path "/data/ml-json" --undirected
status=$?
if test $status -ne 0
then
    exit -1
fi

rm -fr ml-output

# Directed graph
python3 launch_dist_process.py --workspace $GS_HOME/tools --num_workers $NUM_WORKERS --ip_config ip_list.txt --ssh_port 2222 'python3 preprocess_dist_graph.py --name ml --filepath /data/ml-json --output ml-output --hf_bert_model "bert-base-uncased" --ntext_fields "movie:title" --nlabel_fields "movie:genre" --predict_ntype "movie" --ntask_types "movie:classify" --generate_new_split true --ntypes "movie occupation user" --etypes "user,rating,movie user,has-occupation,occupation"'

ls ml-output
cnt=$(ls ml-output | wc -l)
if test $cnt -ne 12
then
    echo "output file is not completed"
    exit -1
fi

# Now check the graph file
python3 $GS_HOME/tests/end2end-tests/dist_load/test_dist_load_nc.py --path "/data/ml-json"
status=$?
if test $status -ne 0
then
    exit -1
fi

rm -fr ml-output

# treat movielens as link prediction task
# Run multi process processing first
python3 launch_dist_process.py --workspace $GS_HOME/tools --num_workers $NUM_WORKERS --ip_config ip_list.txt --ssh_port 2222 'python3 preprocess_dist_graph.py --name ml --filepath /data/ml-json --output ml-output --hf_bert_model "bert-base-uncased" --ntext_fields "movie:title" --undirected --ntypes "movie occupation user" --etypes "user,rating,movie user,has-occupation,occupation"'

cnt=$(ls ml-output | wc -l)
if test $cnt -ne 16
then
    echo "output file is not completed"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges0_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges1_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges2_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check ml-output
cnt=$(ls ml-output | grep ml_edges3_ | wc -l)
if test $cnt -ne 2
then
    echo "edge files does not equal to 2"
    exit -1
fi

# check node feature
if [ ! -d "ml-output/movie/input_ids" ]
then
    echo "missing input_ids"
    exit -1
fi

if [ ! -d "ml-output/movie/valid_len" ]
then
    echo "missing valid_len"
    exit -1
fi

cnt=$(ls ml-output/movie/input_ids | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature input_ids files does not equal to 2"
    exit -1
fi

cnt=$(ls ml-output/movie/valid_len | wc -l)
if test $cnt -ne 2
then
    echo "movie node feature valid_len files does not equal to 2"
    exit -1
fi

# Now check the graph file
ls $GS_HOME/tests/end2end-tests/dist_load/
python3 $GS_HOME/tests/end2end-tests/dist_load/test_dist_load_lp.py --path "/data/ml-json" --undirected
status=$?
if test $status -ne 0
then
    exit -1
fi
