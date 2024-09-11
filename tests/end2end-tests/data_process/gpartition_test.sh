#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

service ssh restart

GS_HOME=$(pwd)
export PYTHONPATH=$GS_HOME/python/

INPUT_PATH=/data/ml-100k/
OUTPUT_PATH=/tmp/gpartition-e2e-tests

echo "127.0.0.1" > ip_list.txt


msg() {
  echo >&2 -e "${1-}"
}

die() {
  local msg=$1
  local code=${2-1} # default exit status 1
  msg "$msg"
  exit "$code"
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
  # script cleanup here
  if [[ -d "${OUTPUT_PATH}" ]]; then
    msg "Cleaning up ${OUTPUT_PATH}"
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
    die "${DIST_DGL_PATH}/dist_graph/metadata.json does not exist"
fi

for i in $(seq 0 1); do
    if [ ! -f "${DIST_DGL_PATH}/dist_graph/part${i}/graph.dgl" ];
    then
        die "${DIST_DGL_PATH}/dist_graph/part${i}/graph.dgl must exist"
    fi
done

echo "********* Test partition assignment only creation ********"

PART_ONLY_PATH="${OUTPUT_PATH}/part-assign-only"
python3 -m graphstorm.gpartition.dist_partition_graph \
    --input-path ${INPUT_PATH} \
    --ip-config ip_list.txt \
    --metadata-filename chunked_graph_meta.json \
    --num-parts 2 \
    --output-path ${PART_ONLY_PATH} \
    --partition-assignment-only \
    --ssh-port 2222

# Ensure expected files and directories were created
for NTYPE in user movie; do
    if [ ! -f "${PART_ONLY_PATH}/partition_assignment/$NTYPE.txt" ];
        then
        die "${PART_ONLY_PATH}/partition_assignment/$NTYPE.txt does not exist"
    fi
done

if [ -f "${PART_ONLY_PATH}/dist_graph/metadata.json" ];
    then
    die "${PART_ONLY_PATH}/dist_graph/metadata.json should not exist"
fi

msg "********* Distributed partitioning tests passed *********"
