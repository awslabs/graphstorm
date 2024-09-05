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

echo "127.0.0.1" > /ip_list.txt

cat ip_list.txt

# Test the DGLGraph format.
echo "********* Test the GSPartition *********"
python3 -m graphstorm.gpartition.dist_partition_graph --metadata-filename updated_row_counts_metadata.json --input-path /gsp-output --output-path /gspartition-out --num-parts 2 --partition-algorithm random --ssh-port 2222 --ip-config /ip_file.txt