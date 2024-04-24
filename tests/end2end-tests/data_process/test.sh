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

echo "********* Test the remap edge predictions *********"
python3 $GS_HOME/tests/end2end-tests/data_process/gen_edge_predict_remap_test.py --output /tmp/ep_remap/

# Test customize mask name
echo "********* Test the DistDGL graph format with customize mask ********"
python3 -m graphstorm.gconstruct.construct_graph --conf-file /tmp/test_data/test_data_transform_custom_mask.conf --num-processes 2 --output-dir /tmp/test_partition3 --graph-name test --output-conf-file /tmp/test_data/test_data_transform_custom_mask_new.conf

error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/test_custom_mask_data.py --graph-format DistDGL --graph_dir /tmp/test_partition3 --conf_file /tmp/test_data/test_data_transform_custom_mask_new.conf

error_and_exit $?

# Test remap edge prediction results
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/ep_remap/id_mapping/ --logging-level debug --pred-etypes "n0,access,n1" "n1,access,n0" --preserve-input True --prediction-dir /tmp/ep_remap/pred/ --rank 1 --world-size 2
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/ep_remap/id_mapping/ --logging-level debug --pred-etypes "n0,access,n1" "n1,access,n0" --preserve-input True --prediction-dir /tmp/ep_remap/pred/ --rank 0 --world-size 2
error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/check_edge_predict_remap.py --remap-output /tmp/ep_remap/pred/

error_and_exit $?

cnt=$(ls /tmp/ep_remap/pred/src_nids-*.pt | wc -l)
if test $cnt == 2
then
    echo "src_nids-xxx.pt must exist."
    exit -1
fi

cnt=$(ls /tmp/ep_remap/pred/dst_nids-*.pt | wc -l)
if test $cnt == 2
then
    echo "dst_nids-xxx.pt must exist."
    exit -1
fi

cnt=$(ls /tmp/ep_remap/pred/predict-*.pt | wc -l)
if test $cnt == 2
then
    echo "predict-xxx.pt must exist."
    exit -1
fi

cp -r /tmp/ep_remap/pred/ /tmp/ep_remap/rename-pred/
# Test remap edge prediction results and rename col names
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/ep_remap/id_mapping/ --logging-level debug --pred-etypes "n0,access,n1" "n1,access,n0" --preserve-input True --prediction-dir /tmp/ep_remap/rename-pred/ --rank 1 --world-size 2 --column-names "src_nid,~from:STRING" "dst_nid,~to:STRING" "pred,pred:FLOAT"
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/ep_remap/id_mapping/ --logging-level debug --pred-etypes "n0,access,n1" "n1,access,n0" --preserve-input True --prediction-dir /tmp/ep_remap/rename-pred/ --rank 0 --world-size 2 --column-names "src_nid,~from:STRING" "dst_nid,~to:STRING" "pred,pred:FLOAT"
error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/check_edge_predict_remap.py --remap-output /tmp/ep_remap/rename-pred/ --column-names "src_nid,~from:STRING" "dst_nid,~to:STRING" "pred,pred:FLOAT"

cnt=$(ls /tmp/ep_remap/rename-pred/src_nids-*.pt | wc -l)
if test $cnt == 0
then
    echo "src_nids-xxx.pt should be removed."
    exit -1
fi

cnt=$(ls /tmp/ep_remap/rename-pred/dst_nids-*.pt | wc -l)
if test $cnt == 0
then
    echo "dst_nids-xxx.pt should be removed."
    exit -1
fi

cnt=$(ls /tmp/ep_remap/rename-pred/predict-*.pt | wc -l)
if test $cnt == 0
then
    echo "predict-xxx.pt should be removed."
    exit -1
fi

error_and_exit $?
rm -fr /tmp/ep_remap/rename-pred/

# Test without shared filesystem
mkdir /tmp/ep_remap/pred/0/
mkdir /tmp/ep_remap/pred/1/
mkdir /tmp/ep_remap/pred/0/n0_access_n1/
mkdir /tmp/ep_remap/pred/1/n0_access_n1/
mkdir /tmp/ep_remap/pred/0/n1_access_n0/
mkdir /tmp/ep_remap/pred/1/n1_access_n0/
cp -r /tmp/ep_remap/pred/n0_access_n1/*0.pt /tmp/ep_remap/pred/0/n0_access_n1/
cp -r /tmp/ep_remap/pred/n0_access_n1/*1.pt /tmp/ep_remap/pred/1/n0_access_n1/
cp -r /tmp/ep_remap/pred/n1_access_n0/*0.pt /tmp/ep_remap/pred/0/n1_access_n0/
cp -r /tmp/ep_remap/pred/n1_access_n0/*1.pt /tmp/ep_remap/pred/1/n1_access_n0/

# Test remap edge prediction results
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/ep_remap/id_mapping/ --logging-level debug --pred-etypes "n0,access,n1" "n1,access,n0" --preserve-input True --prediction-dir /tmp/ep_remap/pred/1/ --rank 1 --world-size 2 --with-shared-fs False
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/ep_remap/id_mapping/ --logging-level debug --pred-etypes "n0,access,n1" "n1,access,n0" --preserve-input True --prediction-dir /tmp/ep_remap/pred/0/ --rank 0 --world-size 2 --with-shared-fs False
error_and_exit $?

mkdir /tmp/ep_remap/pred/no-share/
cp -r /tmp/ep_remap/pred/0/* /tmp/ep_remap/pred/no-share/
cp -r /tmp/ep_remap/pred/1/* /tmp/ep_remap/pred/no-share/

python3 $GS_HOME/tests/end2end-tests/data_process/check_edge_predict_remap.py --remap-output /tmp/ep_remap/pred/no-share/
error_and_exit $?

rm -fr /tmp/ep_remap/

# Check node predict
echo "********* Test the remap node predictions *********"
python3 $GS_HOME/tests/end2end-tests/data_process/gen_node_predict_remap_test.py --output /tmp/np_remap/

# Test remap node prediction results
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/np_remap/id_mapping/ --logging-level debug --pred-ntypes "n0" "n1" --preserve-input True --prediction-dir /tmp/np_remap/pred/ --rank 1 --world-size 2
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/np_remap/id_mapping/ --logging-level debug --pred-ntypes "n0" "n1" --preserve-input True --prediction-dir /tmp/np_remap/pred/ --rank 0 --world-size 2
error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/check_node_predict_remap.py --remap-output /tmp/np_remap/pred/

error_and_exit $?

cnt=$(ls /tmp/np_remap/pred/predict_nids-*.pt | wc -l)
if test $cnt == 0
then
    echo "predict_nids-xxx.pt should be removed."
    exit -1
fi

cnt=$(ls /tmp/np_remap/pred/predict-*.pt | wc -l)
if test $cnt == 0
then
    echo "predict-xxx.pt should be removed."
    exit -1
fi

# Test without shared filesystem
echo "********* Test the remap node predictions without shared mem *********"
mkdir /tmp/np_remap/pred/0/
mkdir /tmp/np_remap/pred/1/
mkdir /tmp/np_remap/pred/0/n0/
mkdir /tmp/np_remap/pred/1/n1/
mkdir /tmp/np_remap/pred/0/n1/
mkdir /tmp/np_remap/pred/1/n0/


cp -r /tmp/np_remap/pred/n0/*0.pt /tmp/np_remap/pred/0/n0/
cp -r /tmp/np_remap/pred/n0/*1.pt /tmp/np_remap/pred/1/n0/
cp -r /tmp/np_remap/pred/n1/*0.pt /tmp/np_remap/pred/0/n1/
cp -r /tmp/np_remap/pred/n1/*1.pt /tmp/np_remap/pred/1/n1/

# Test remap edge prediction results
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/np_remap/id_mapping/ --logging-level debug --pred-ntypes "n0" "n1" --preserve-input True --prediction-dir /tmp/np_remap/pred/1/ --rank 1 --world-size 2 --with-shared-fs False
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/np_remap/id_mapping/ --logging-level debug --pred-ntypes "n0" "n1" --preserve-input True --prediction-dir /tmp/np_remap/pred/0/ --rank 0 --world-size 2 --with-shared-fs False
error_and_exit $?

mkdir /tmp/np_remap/pred/no-share/
cp -r /tmp/np_remap/pred/0/* /tmp/np_remap/pred/no-share/
cp -r /tmp/np_remap/pred/1/* /tmp/np_remap/pred/no-share/

python3 $GS_HOME/tests/end2end-tests/data_process/check_node_predict_remap.py --remap-output /tmp/np_remap/pred/no-share/

error_and_exit $?

rm -fr /tmp/np_remap/

# Check node embedding
echo "********* Test the remap node emb/partial emb *********"
python3 $GS_HOME/tests/end2end-tests/data_process/gen_emb_predict_remap_test.py --output /tmp/em_remap/

# Test remap emb results
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/em_remap/id_mapping/ --logging-level debug --node-emb-dir /tmp/em_remap/partial-emb/  --preserve-input True --rank 1 --world-size 2
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/em_remap/id_mapping/ --logging-level debug --node-emb-dir /tmp/em_remap/partial-emb/ --preserve-input True --rank 0 --world-size 2
error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/check_emb_remap.py --remap-output /tmp/em_remap/partial-emb/

error_and_exit $?

cnt=$(ls /tmp/em_remap/partial-emb/embed_nids-*.pt | wc -l)
if test $cnt == 2
then
    echo "embed_nids-xxx.pt must exist."
    exit -1
fi

cnt=$(ls /tmp/em_remap/partial-emb/embed-*.pt | wc -l)
if test $cnt == 2
then
    echo "embed-xxx.pt must exist."
    exit -1
fi


cp -r /tmp/em_remap/partial-emb/ /tmp/em_remap/partial-rename-emb/

# Test remap emb results and rename col names
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/em_remap/id_mapping/ --logging-level debug --node-emb-dir /tmp/em_remap/partial-rename-emb/  --rank 1 --world-size 2 --column-names "nid,~id:STRING" "emb,emb:FLOAT"
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/em_remap/id_mapping/ --logging-level debug --node-emb-dir /tmp/em_remap/partial-rename-emb/ --rank 0 --world-size 2 --column-names "nid,~id:STRING" "emb,emb:FLOAT"
error_and_exit $?

python3 $GS_HOME/tests/end2end-tests/data_process/check_emb_remap.py --remap-output /tmp/em_remap/partial-rename-emb/ --column-names "nid,~id:STRING" "emb,emb:FLOAT"

error_and_exit $?

cnt=$(ls /tmp/em_remap/partial-rename-emb/embed_nids-*.pt | wc -l)
if test $cnt == 0
then
    echo "embed_nids-xxx.pt should be removed."
    exit -1
fi

cnt=$(ls /tmp/em_remap/partial-rename-emb/embed-*.pt | wc -l)
if test $cnt == 0
then
    echo "embed-xxx.pt should be removed."
    exit -1
fi

# Test without shared filesystem
echo "********* Test the remap partial node embedding without shared mem *********"
mkdir /tmp/em_remap/partial-emb/0/
mkdir /tmp/em_remap/partial-emb/1/
mkdir /tmp/em_remap/partial-emb/0/n0/
mkdir /tmp/em_remap/partial-emb/1/n1/
mkdir /tmp/em_remap/partial-emb/0/n1/
mkdir /tmp/em_remap/partial-emb/1/n0/

cp -r /tmp/em_remap/partial-emb/n0/*0.pt /tmp/em_remap/partial-emb/0/n0/
cp -r /tmp/em_remap/partial-emb/n0/*1.pt /tmp/em_remap/partial-emb/1/n0/
cp -r /tmp/em_remap/partial-emb/n1/*0.pt /tmp/em_remap/partial-emb/0/n1/
cp -r /tmp/em_remap/partial-emb/n1/*1.pt /tmp/em_remap/partial-emb/1/n1/

# Test remap emb results
python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/em_remap/id_mapping/ --logging-level debug --node-emb-dir /tmp/em_remap/partial-emb/1/  --preserve-input True --rank 1 --world-size 2 --with-shared-fs False
error_and_exit $?

python3 -m graphstorm.gconstruct.remap_result --num-processes 16 --node-id-mapping /tmp/em_remap/id_mapping/ --logging-level debug --node-emb-dir /tmp/em_remap/partial-emb/0/ --preserve-input True --rank 0 --world-size 2 --with-shared-fs False
error_and_exit $?

mkdir /tmp/em_remap/partial-emb/no-share/
cp -r /tmp/em_remap/partial-emb/0/* /tmp/em_remap/partial-emb/no-share/
cp -r /tmp/em_remap/partial-emb/1/* /tmp/em_remap/partial-emb/no-share/

python3 $GS_HOME/tests/end2end-tests/data_process/check_emb_remap.py --remap-output /tmp/em_remap/partial-emb/no-share/

error_and_exit $?

rm -fr /tmp/em_remap/
