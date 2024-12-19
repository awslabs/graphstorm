#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] [-i /path/to/ml-100k] [-o /path/to/output/data]

Run the GraphBolt graph construction integration tests.

Available options:

-h, --help          Print this help and exit
-x, --verbose       Print script debug info (set -x)
-i, --ml100k-path   Path to the processed ml-100k data, created by tests/end2end-tests/data_gen/process_movielens.py
-o, --output-path   Path under which the converted data will be created.

EOF
  exit
}

# Parse command-line arguments
parse_params() {
    # Default values for input and output paths
    INPUT_PATH="/data/ml-100k"
    OUTPUT_PATH="/tmp/gb-graphconstruction-e2e-tests"

    while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -x | --verbose) set -x ;;
    -i | --ml100k-path)
        INPUT_PATH="${2-}"
        shift
        ;;
    -o | --output-path)
        OUTPUT_PATH="${2-}"
        shift
        ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
    done

    return 0
}

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    # script cleanup here
    if [[ -d "${OUTPUT_PATH}" ]]; then
        echo "Cleaning up ${OUTPUT_PATH}"
        rm -rf "${OUTPUT_PATH}"
    fi
}

parse_params "$@"

GS_HOME=$(pwd)

mkdir -p "$OUTPUT_PATH"
cd "$OUTPUT_PATH" || exit
cp -R "$INPUT_PATH" "$OUTPUT_PATH"

# Ensure ip_list.txt exists and self-ssh works
echo "127.0.0.1" > ip_list.txt
ssh -o PreferredAuthentications=publickey StrictHostKeyChecking=no \
    -p 2222 127.0.0.1 /bin/true || service ssh restart


# Ensure test data have been generated
if [ ! -f "${INPUT_PATH}/chunked_graph_meta.json" ]; then
    python3 "$GS_HOME/tests/end2end-tests/data_gen/process_movielens.py"
fi

echo "********* Test GConstruct with GraphBolt graph format ********"

GCONS_GRAPHBOLT_PATH="${OUTPUT_PATH}/graphbolt-gconstruct-lp"
python3 -m graphstorm.gconstruct.construct_graph \
    --add-reverse-edges \
    --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_lp.json \
    --graph-name ml-lp \
    --num-parts 2 \
    --num-processes 1 \
    --output-dir "$GCONS_GRAPHBOLT_PATH" \
    --part-method random \
    --use-graphbolt "true"


# Ensure GraphBolt files were created by GConstruct
for i in $(seq 0 1); do
    if [ ! -f "$GCONS_GRAPHBOLT_PATH/part${i}/fused_csc_sampling_graph.pt" ]
    then
        echo "$GCONS_GRAPHBOLT_PATH/part${i}/fused_csc_sampling_graph.pt must exist"
        exit 1
    fi
done

echo "********* Test GraphBolt standalone conversion ********"
# We remove the previously generated GraphBolt files to test the standalone generation
for i in $(seq 0 1); do
    rm "$GCONS_GRAPHBOLT_PATH/part${i}/fused_csc_sampling_graph.pt"
done

python3 -m graphstorm.gpartition.convert_to_graphbolt \
    --metadata-filepath "${GCONS_GRAPHBOLT_PATH}/ml-lp.json"

# Ensure GraphBolt files were re-created by standalone script
for i in $(seq 0 1); do
    if [ ! -f "$GCONS_GRAPHBOLT_PATH/part${i}/fused_csc_sampling_graph.pt" ]
    then
        echo "$GCONS_GRAPHBOLT_PATH/part${i}/fused_csc_sampling_graph.pt must exist"
        exit 1
    fi
done


echo "********* Test GSPartition with GraphBolt graph format ********"

DIST_GRAPHBOLT_PATH="${OUTPUT_PATH}/graphbolt-gspartition-nc"
python3 -m graphstorm.gpartition.dist_partition_graph \
    --input-path "${INPUT_PATH}" \
    --ip-config ip_list.txt \
    --metadata-filename chunked_graph_meta.json \
    --num-parts 2 \
    --output-path "$DIST_GRAPHBOLT_PATH" \
    --ssh-port 2222 \
    --use-graphbolt "true" \
    --process-group-timeout 3600

# Ensure GraphBolt files were created by GSPartition
for i in $(seq 0 1); do
    if [ ! -f "$DIST_GRAPHBOLT_PATH/dist_graph/part${i}/fused_csc_sampling_graph.pt" ]
    then
        echo "$DIST_GRAPHBOLT_PATH/dist_graph/part${i}/fused_csc_sampling_graph.pt must exist"
        exit 1
    fi
done

echo "********* GraphBolt graph construction and partitioning tests passed *********"
