#!/bin/env bash
set -euox pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd -P)

msg() {
    echo >&2 -e "${1-}"
}

die() {
    local msg=$1
    local code=${2-1} # default exit status 1
    msg "$msg"
    exit "$code"
}

parse_params() {
    # default values of variables set from params
    ACCOUNT=$(aws sts get-caller-identity --query Account --output text || true)
    REGION=$(aws configure get region || true)
    REGION=${REGION:-"us-east-1"}
    PIPELINE_NAME=""


    while :; do
        case "${1-}" in
        -h | --help) usage ;;
        -x | --verbose) set -x ;;
        -r | --execution-role)
            ROLE_ARN="${2-}"
            shift
            ;;
        -a | --account)
            ACCOUNT="${2-}"
            shift
            ;;
        -b | --bucket-name)
            BUCKET_NAME="${2-}"
            shift
            ;;
        -n | --pipeline-name)
            PIPELINE_NAME="${2-}"
            shift
            ;;
        -g | --use-graphbolt)
            USE_GRAPHBOLT="${2-}"
            shift
            ;;
        -?*) die "Unknown option: $1" ;;
        *) break ;;
        esac
        shift
    done

    # check required params and arguments
    [[ -z "${ACCOUNT-}" ]] && die "Missing required parameter: -a/--account <aws-account-id>"
    [[ -z "${BUCKET_NAME-}" ]] && die "Missing required parameter: -b/--bucket-name <s3-bucket>"
    [[ -z "${ROLE_ARN-}" ]] && die "Missing required parameter: -r/--execution-role <execution-role-arn>"
    [[ -z "${USE_GRAPHBOLT-}" ]] && die "Missing required parameter: -g/--use-graphbolt <true|false>"

    return 0
}

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    # script cleanup here
}

parse_params "$@"

DATASET_S3_PATH="s3://${BUCKET_NAME}/ogb-arxiv-input"
OUTPUT_PATH="s3://${BUCKET_NAME}/pipelines-output"
GRAPH_NAME="ogbn-arxiv"
INSTANCE_COUNT="2"
REGION="us-east-1"
NUM_TRAINERS=4

PARTITION_OUTPUT_JSON="$GRAPH_NAME.json"
PARTITION_ALGORITHM="metis"
GCONSTRUCT_INSTANCE="ml.r5.4xlarge"
GCONSTRUCT_CONFIG="gconstruct_config_arxiv.json"

TRAIN_CPU_INSTANCE="ml.m5.4xlarge"
TRAIN_YAML_S3="s3://$BUCKET_NAME/yaml/arxiv_nc_train.yaml"
INFERENCE_YAML_S3="s3://$BUCKET_NAME/yaml/arxiv_nc_inference.yaml"

TASK_TYPE="node_classification"
INFERENCE_MODEL_SNAPSHOT="epoch-9"

JOBS_TO_RUN="gconstruct train inference"
GSF_CPU_IMAGE_URI=${ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/graphstorm:sagemaker-cpu
GSF_GPU_IMAGE_URI=${ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/graphstorm:sagemaker-gpu
VOLUME_SIZE=50

if [[ -z "${PIPELINE_NAME-}" ]]; then
    if [[ $USE_GRAPHBOLT == "true" ]]; then
        PIPELINE_NAME="ogbn-arxiv-gs-graphbolt-pipeline"
    else
        PIPELINE_NAME="ogbn-arxiv-gs-pipeline"
    fi
fi

python3 $SCRIPT_DIR/../../sagemaker/pipeline/create_sm_pipeline.py \
    --cpu-instance-type ${TRAIN_CPU_INSTANCE} \
    --execution-role "${ROLE_ARN}" \
    --graph-construction-args "--num-processes 8" \
    --graph-construction-instance-type ${GCONSTRUCT_INSTANCE} \
    --graph-construction-config-filename ${GCONSTRUCT_CONFIG} \
    --graph-name ${GRAPH_NAME} \
    --graphstorm-pytorch-cpu-image-uri "${GSF_CPU_IMAGE_URI}" \
    --graphstorm-pytorch-gpu-image-uri "${GSF_GPU_IMAGE_URI}" \
    --inference-model-snapshot "${INFERENCE_MODEL_SNAPSHOT}" \
    --inference-yaml-s3 ${INFERENCE_YAML_S3} \
    --input-data-s3 ${DATASET_S3_PATH} \
    --instance-count ${INSTANCE_COUNT} \
    --jobs-to-run ${JOBS_TO_RUN} \
    --num-trainers ${NUM_TRAINERS} \
    --output-prefix-s3 "${OUTPUT_PATH}" \
    --pipeline-name "${PIPELINE_NAME}" \
    --partition-output-json ${PARTITION_OUTPUT_JSON} \
    --partition-algorithm ${PARTITION_ALGORITHM} \
    --region ${REGION} \
    --train-on-cpu \
    --train-inference-task ${TASK_TYPE} \
    --train-yaml-s3 "${TRAIN_YAML_S3}" \
    --save-embeddings \
    --save-predictions \
    --volume-size-gb ${VOLUME_SIZE} \
    --use-graphbolt "${USE_GRAPHBOLT}"
