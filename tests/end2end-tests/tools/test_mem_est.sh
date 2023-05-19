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

python3 $GS_HOME/tools/gsf_mem_est.py --root-path /data/movielen_100k_train_val_1p_4t --supervise-task edge --is-train 1

error_and_exit $?

python3 $GS_HOME/tools/gsf_mem_est.py --root-path /data/movielen_100k_train_val_1p_4t --is-train 0 --hidden-size 16 --num-layers 1 --graph-name movie-lens-100k

error_and_exit $?

