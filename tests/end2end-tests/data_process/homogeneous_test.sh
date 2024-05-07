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
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_homogeneous.json --num-processes 1 --output-dir /tmp/movielen_100k_train_val_1p_4t_homogeneous --graph-name movie-lens-100k
error_and_exit $?

echo "********* Test Node Classification on GConstruct Homogeneous Graph ********"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --target-ntype _N --model-encoder-type sage
error_and_exit $?

echo "********* Test Node Classification on GConstruct Homogeneous Graph with gatv2 encoder********"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --target-ntype _N --model-encoder-type gatv2
error_and_exit $?

echo "********* Test Edge Classification on GConstruct Homogeneous Graph ********"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --target-etype _N,_E,_N --model-encoder-type sage
error_and_exit $?

echo "********* Test Edge Classification on GConstruct Homogeneous Graph with gatv2 encoder********"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --target-etype _N,_E,_N --model-encoder-type gatv2
error_and_exit $?

echo "********* Test Homogeneous Graph Optimization on reverse edge********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_homogeneous.json --num-processes 1 --output-dir /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev --graph-name movie-lens-100k --add-reverse-edges
error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/check_homogeneous.py
error_and_exit $?

echo "********* Test Node Classification on GConstruct Homogeneous Graph with reverse edge********"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc.yaml --target-ntype _N --model-encoder-type gat
error_and_exit $?

echo "********* Test Edge Classification on GConstruct Homogeneous Graph with reverse edge********"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec.yaml --target-etype _N,_E,_N --model-encoder-type gat
error_and_exit $?

echo "********* Test Node Classification with homogeneous graph optimization********"
python3 -m graphstorm.run.gs_node_classification --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_homogeneous.yaml --save-model-path /tmp/homogeneous_node_model
error_and_exit $?

echo "********* Test Node Classification with homogeneous graph optimization doing inference********"
python3 -m graphstorm.run.gs_node_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_np/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_nc_homogeneous.yaml --restore-model-path /tmp/homogeneous_node_model/epoch-2
error_and_exit $?

echo "********* Test Edge Classification with homogeneous graph optimization********"
python3 -m graphstorm.run.gs_edge_classification --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec_homogeneous.yaml --save-model-path /tmp/homogeneous_edge_model
error_and_exit $?

echo "********* Test Edge Classification with homogeneous graph optimization doing inference********"
python3 -m graphstorm.run.gs_edge_classification --inference --workspace $GS_HOME/training_scripts/gsgnn_ep/ --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /tmp/movielen_100k_train_val_1p_4t_homogeneous_rev/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_ec_homogeneous.yaml --restore-model-path /tmp/homogeneous_edge_model/epoch-2
error_and_exit $?

rm -rf /tmp/homogeneous_node_model
rm -rf /tmp/homogeneous_edge_model
