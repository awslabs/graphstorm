#!/bin/bash

DGL_HOME=/root/dgl
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

python3 $GS_HOME/tools/construct_graph.py --conf_file /tmp/test_data/test_data_transform.conf --num_processes 4 --output_dir /tmp/test_out --graph_name test

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py

error_and_exit $?
