#!/usr/bin/env bash
set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-x] [-i /path/to/ml-100k] [-o /path/to/output/data]

Run the GraphBolt training and inference integration tests.

Available options:

-h, --help          Print this help and exit
-x, --verbose       Print script debug info (set -x)
-i, --ml100k-path   Path to the processed ml-100k data, created by tests/end2end-tests/data_gen/process_movielens.py
-o, --output-path   Path under which the converted data will be created.

EOF
  exit
}

msg() {
  echo >&2 -e "${1-}"
}

# Parse command-line arguments
parse_params() {
    # Default values for input and output paths
    INPUT_PATH="/data/ml-100k/"
    OUTPUT_PATH="/tmp/gb-training-e2e-tests"

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

fdir_exists() {
    # Take two args: first should be f or d, for file or directory
    # second is the path to check

    if [ "$1" == "f" ]
    then
        if [ ! -f "$2" ]
        then
            msg "$2 must exist"
            exit 1
        fi
    elif [ "$1" == "d" ]
    then
        if [ ! -d "$2" ]
        then
            msg "$2 must exist"
            exit 1
        fi
    else
        msg "First arg to fdir_exists must be f or d"
        exit 1
    fi
}

parse_params "$@"

GS_HOME=$(pwd)

mkdir -p "$OUTPUT_PATH"
cp -R "$INPUT_PATH" "$OUTPUT_PATH"

# Ensure ip_list.txt exists and self-ssh works
rm "$OUTPUT_PATH/ip_list.txt" &> /dev/null || true
echo "127.0.0.1" > "$OUTPUT_PATH/ip_list.txt"
ssh -o PreferredAuthentications=publickey -o StrictHostKeyChecking=no \
    -p 2222 127.0.0.1 /bin/true || service ssh restart

# Generate 1P LP data
msg "**************GraphBolt Link Prediction data generation **************"
LP_INPUT_1P="${OUTPUT_PATH}/graphbolt-gconstruct-lp-1p"
python3 -m graphstorm.gconstruct.construct_graph \
    --add-reverse-edges \
    --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_lp.json \
    --graph-name ml-lp \
    --num-parts 1 \
    --num-processes 1 \
    --output-dir "$LP_INPUT_1P" \
    --part-method random \
    --use-graphbolt "true"

LP_OUTPUT="$OUTPUT_PATH/gb-lp"
msg "**************GraphBolt Link Prediction training. dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch, negative_sampler: joint, exclude_training_targets: false"
python3 -m graphstorm.run.gs_link_prediction \
    --cf $GS_HOME/training_scripts/gsgnn_lp/ml_lp.yaml \
    --eval-frequency 300 \
    --ip-config "$OUTPUT_PATH/ip_list.txt" \
    --num-epochs 1 \
    --num-samplers 0 \
    --num-servers 1 \
    --num-trainers 1 \
    --part-config "$LP_INPUT_1P/ml-lp.json" \
    --save-model-path "$LP_OUTPUT/model" \
    --ssh-port 2222 \
    --use-graphbolt true

# Ensure model files were saved
fdir_exists f "$LP_OUTPUT/model/epoch-0/model.bin"
fdir_exists f "$LP_OUTPUT/model/epoch-0/optimizers.bin"

msg " **************GraphBolt Link Prediction embedding generation **************"

python3 -m graphstorm.run.gs_gen_node_embedding \
    --cf $GS_HOME/training_scripts/gsgnn_lp/ml_lp.yaml \
    --eval-frequency 300 \
    --inference \
    --ip-config "$OUTPUT_PATH/ip_list.txt" \
    --num-epochs 1 \
    --num-samplers 0 \
    --num-servers 1 \
    --num-trainers 1 \
    --part-config "$LP_INPUT_1P/ml-lp.json" \
    --restore-model-path "$LP_OUTPUT/model/epoch-0" \
    --save-embed-path "$LP_OUTPUT/embeddings" \
    --ssh-port 2222 \
    --use-graphbolt true

# Ensure embeddings were created
fdir_exists d "$LP_OUTPUT/embeddings/movie"
fdir_exists d "$LP_OUTPUT/embeddings/user"

LP_OUTPUT="$OUTPUT_PATH/gb-lp-inbatch_joint"
msg "**************GraphBolt Link Prediction training. dataset: Movielens, RGCN layer 1, inference: mini-batch, negative_sampler: inbatch_joint, exclude_training_targets: true"
python3 -m graphstorm.run.gs_link_prediction \
    --cf $GS_HOME/training_scripts/gsgnn_lp/ml_lp.yaml \
    --eval-frequency 300 \
    --exclude-training-targets True \
    --ip-config "$OUTPUT_PATH/ip_list.txt" \
    --num-epochs 1 \
    --num-samplers 0 \
    --num-servers 1 \
    --num-trainers 1 \
    --part-config "$LP_INPUT_1P/ml-lp.json" \
    --reverse-edge-types-map user,rating,rating-rev,movie \
    --save-model-path "$LP_OUTPUT/model" \
    --ssh-port 2222 \
    --train-negative-sampler inbatch_joint \
    --use-graphbolt true

# Ensure model files were saved
fdir_exists f "$LP_OUTPUT/model/epoch-0/model.bin"
fdir_exists f "$LP_OUTPUT/model/epoch-0/optimizers.bin"

LP_OUTPUT="$OUTPUT_PATH/gb-lp-all_etype_uniform"
msg "**************GraphBolt Link Prediction training. dataset: Movielens, RGCN layer 1, inference: mini-batch, negative_sampler: all_etype_uniform, exclude_training_targets: true"
python3 -m graphstorm.run.gs_link_prediction \
    --cf $GS_HOME/training_scripts/gsgnn_lp/ml_lp.yaml \
    --eval-frequency 300 \
    --exclude-training-targets True \
    --ip-config "$OUTPUT_PATH/ip_list.txt" \
    --num-epochs 1 \
    --num-samplers 0 \
    --num-servers 1 \
    --num-trainers 1 \
    --part-config "$LP_INPUT_1P/ml-lp.json" \
    --reverse-edge-types-map user,rating,rating-rev,movie \
    --save-model-path "$LP_OUTPUT/model" \
    --ssh-port 2222 \
    --train-negative-sampler all_etype_uniform \
    --use-graphbolt true

# Ensure model file were saved
fdir_exists f "$LP_OUTPUT/model/epoch-0/model.bin"
fdir_exists f "$LP_OUTPUT/model/epoch-0/optimizers.bin"


# Generate 1P NC data
msg "************** GraphBolt Node Classification data generation. **************"
NC_INPUT_1P="${OUTPUT_PATH}/graphbolt-gconstruct-nc-1p"
python3 -m graphstorm.gconstruct.construct_graph \
    --add-reverse-edges \
    --conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens.json \
    --graph-name ml-nc \
    --num-parts 1 \
    --num-processes 1 \
    --output-dir "$NC_INPUT_1P" \
    --part-method random \
    --use-graphbolt "true"


msg "************** GraphBolt Node Classification training. dataset: Movielens, RGCN layer 1, node feat: fixed HF BERT, BERT nodes: movie, inference: mini-batch"
NC_OUTPUT="$OUTPUT_PATH/gb-nc"
python3 -m graphstorm.run.gs_node_classification \
    --cf $GS_HOME/training_scripts/gsgnn_np/ml_nc.yaml \
    --eval-frequency 300 \
    --ip-config "$OUTPUT_PATH/ip_list.txt" \
    --num-epochs 1 \
    --num-samplers 0 \
    --num-servers 1 \
    --num-trainers 1 \
    --part-config "$NC_INPUT_1P/ml-nc.json" \
    --save-model-path "$NC_OUTPUT/model" \
    --ssh-port 2222 \
    --use-graphbolt true

# Ensure model files were saved
fdir_exists f "$NC_OUTPUT/model/epoch-0/model.bin"
fdir_exists f "$NC_OUTPUT/model/epoch-0/optimizers.bin"

msg "************** GraphBolt Node Classification inference. **************"
python3 -m graphstorm.run.gs_node_classification \
    --cf $GS_HOME/training_scripts/gsgnn_np/ml_nc.yaml \
    --eval-frequency 300 \
    --inference \
    --ip-config "$OUTPUT_PATH/ip_list.txt" \
    --no-validation true \
    --num-epochs 1 \
    --num-samplers 0 \
    --num-servers 1 \
    --num-trainers 1 \
    --part-config "$NC_INPUT_1P/ml-nc.json" \
    --restore-model-path "$NC_OUTPUT/model/epoch-0" \
    --save-embed-path "$NC_OUTPUT/embeddings" \
    --save-prediction-path "$NC_OUTPUT/predictions" \
    --ssh-port 2222 \
    --use-graphbolt true \
    --use-mini-batch-infer false

# Ensure embeddings and predictions were created
fdir_exists d "$NC_OUTPUT/embeddings"
fdir_exists d "$NC_OUTPUT/predictions"

msg "********* GraphBolt training and inference tests passed *********"
