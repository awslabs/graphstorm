#!/bin/bash

service ssh restart

GS_HOME=$(pwd)
NUM_TRAINERS=4
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_np
echo "127.0.0.1" > ip_list.txt
cd $GS_HOME/training_scripts/gsgnn_ep
echo "127.0.0.1" > ip_list.txt

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}


echo "********* Test Homogeneous Graph Optimization ********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_homogenous.json --num-processes 1 --output-dir /tmp/movielen_100k_train_val_1p_4t_homogeneous --graph-name movie-lens-100k
error_and_exit $?

echo "********* Test Node Classification on GConstruct Homogeneous Graph ********"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --target-ntype _N
error_and_exit $?

echo "********* Test Edge Classification on GConstruct Homogeneous Graph ********"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --target-etype _N,_E,_N
error_and_exit $?

echo "********* Test Homogeneous Graph Optimization on reverse edge********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_homogenous.json --num-processes 1 --output-dir /tmp/movielen_100k_train_val_1p_4t_homogeneous --graph-name movie-lens-100k --add-reverse-edges
error_and_exit $?

echo "********* Test Node Classification on GConstruct Homogeneous Graph on reverse edge********"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --target-ntype _N
error_and_exit $?

echo "********* Test Edge Classification on GConstruct Homogeneous Graph on reverse edge ********"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --target-etype _N,_E,_N
error_and_exit $?