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
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform.conf --num-processes 2 --output-dir /tmp/test_out --graph-name test --output-format DGL --output-conf-file /tmp/test_data/test_data_transform_new.conf

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph_dir /tmp/test_out --conf_file /tmp/test_data/test_data_transform_new.conf --graph-format DGL

error_and_exit $?

# Test the generated config.
echo "********* Test using the generated config *********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform_new.conf --num-processes 4 --output-dir /tmp/test_out1 --graph-name test --output-format DGL

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/compare_graphs.py --graph-path1 /tmp/test_out/test.dgl --graph-path2 /tmp/test_out1/test.dgl

error_and_exit $?

# Test the DistDGL graph format.
echo "********* Test the DistDGL graph format ********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform.conf --num-processes 2 --output-dir /tmp/test_partition2 --graph-name test --output-conf-file /tmp/test_data/test_data_transform_new.conf

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph-format DistDGL --graph_dir /tmp/test_partition2 --conf_file /tmp/test_data/test_data_transform_new.conf

error_and_exit $?

# Test the DistDGL graph format with external memory support.
echo "********* Test the DistDGL graph format with external memory support ********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform.conf --num-processes 2 --output-dir /tmp/test_partition2 --graph-name test --ext-mem-workspace /tmp --ext-mem-feat-size 2 --output-conf-file /tmp/test_data/test_data_transform_new.conf

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph-format DistDGL --graph_dir /tmp/test_partition2 --conf_file /tmp/test_data/test_data_transform_new.conf

error_and_exit $?

# Test the DistDGL graph format with reverse edges.
echo "*********** Test the DistDGL graph format with reverse edges *********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform.conf --num-processes 2 --output-dir /tmp/test_out --graph-name test --add-reverse-edges

error_and_exit $?

# Test create both DGL and DistDGL graph
echo "********* Test the DGLGraph format *********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform.conf --num-processes 2 --output-dir /tmp/test_out --graph-name test --output-format DGL DistDGL --output-conf-file /tmp/test_data/test_data_transform_new.conf

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph_dir /tmp/test_out --conf_file /tmp/test_data/test_data_transform_new.conf --graph-format DGL

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py --graph-format DistDGL --graph_dir /tmp/test_out --conf_file /tmp/test_data/test_data_transform_new.conf
