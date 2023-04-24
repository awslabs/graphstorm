#!/bin/bash

service ssh restart

DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=1
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/tools

echo "127.0.0.1" > ip_list.txt

cat ip_list.txt

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

python3 $GS_HOME/tools/gsf_mem_est.py --root_path /data/movielen_100k_train_val_1p_4t --supervise_task edge --is_train 1

error_and_exit $?

python3 $GS_HOME/tools/gsf_mem_est.py --root_path /data/movielen_100k_train_val_1p_4t --is_train 0 --hidden_size 16 --num_layers 1 --graph_name movie-lens-100k

error_and_exit $?

