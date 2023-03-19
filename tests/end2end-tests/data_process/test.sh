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

python3 $GS_HOME/tools/preprocess.py --num_processes 3 --conf_file /tmp/test_data/test_data_transform.conf

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_data.py

error_and_exit $?

python3 $GS_HOME/tools/gen_id_map.py --data_file "/tmp/test_data/edge_data1_*.parquet" --id_cols src,dst --out_dir /tmp/test_ids --group_size 2 --num_processes 2

error_and_exit $?

python3 tests/end2end-tests/data_process/test_id_map.py

error_and_exit $?
