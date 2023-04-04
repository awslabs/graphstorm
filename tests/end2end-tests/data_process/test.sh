#!/bin/bash

GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/

error_and_exit () {
	# check exec status of launch.py
	status=$1
	echo $status

	if test $status -ne 0
	then
		exit -1
	fi
}

python3 $GS_HOME/tests/end2end-tests/data_process/data_gen.py

# Test the DGLGraph format.
echo "********* Test the DGLGraph format *********"
python3 -m graphstorm.gconstruct.construct_graph --conf_file /tmp/test_data/test_data_transform.conf --num_processes 4 --output_dir /tmp/test_out --graph_name test --output_format DGL

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph_dir /tmp/test_out --conf_file /tmp/test_data/test_data_transform.conf --graph_format DGL

error_and_exit $?

# Test the DistDGL graph format.
echo "********* Test the DistDGL graph format ********"
python3 -m graphstorm.gconstruct.construct_graph --conf_file /tmp/test_data/test_data_transform.conf --num_processes 4 --output_dir /tmp/test_partition2 --graph_name test

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph_format DistDGL --graph_dir /tmp/test_partition2 --conf_file /tmp/test_data/test_data_transform.conf

error_and_exit $?

# Test the DistDGL graph format with reverse edges.
echo "*********** Test the DistDGL graph format with reverse edges *********"
python3 -m graphstorm.gconstruct.construct_graph --conf_file /tmp/test_data/test_data_transform.conf --num_processes 4 --output_dir /tmp/test_out --graph_name test --add_reverse_edges

error_and_exit $?
