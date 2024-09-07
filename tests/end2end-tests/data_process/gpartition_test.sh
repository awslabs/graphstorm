#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

service ssh restart

GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/

INPUT_PATH=/data/ml-100k/
OUTPUT_PATH=/tmp/gpartition-e2e-tests

echo "127.0.0.1" > ip_list.txt


cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
  if [[ -d "${OUTPUT_PATH}" ]]; then
    echo "Cleaning up ${OUTPUT_PATH}"
    rm -rf "${OUTPUT_PATH}"
  fi
}

# Ensure test data have been generated
if [ ! -f "${INPUT_PATH}/chunked_graph_meta.json" ]; then
    python3 "$GS_HOME/tests/end2end-tests/data_gen/process_movielens.py"
fi

echo "********* Test partition creation with DistDGL graph format ********"

DIST_DGL_PATH="${OUTPUT_PATH}/dist-dgl-part"
python3 -m graphstorm.gpartition.dist_partition_graph \
    --input-path ${INPUT_PATH} \
    --ip-config ip_list.txt \
    --metadata-filename chunked_graph_meta.json \
    --num-parts 2 \
    --output-path ${DIST_DGL_PATH} \
    --ssh-port 2222

# Ensure expected files and directories were created
if [ ! -f "${DIST_DGL_PATH}/dist_graph/metadata.json" ];
    then
    echo "${DIST_DGL_PATH}/dist_graph/metadata.json does not exist"
fi

for i in $(seq 0 1); do
    if [ ! -f "${DIST_DGL_PATH}/dist_graph/part${i}/graph.dgl" ];
    then
        echo "${DIST_DGL_PATH}/dist_graph/part${i}/graph.dgl must exist"
        exit 1
    fi
done
