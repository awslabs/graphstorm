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

if [[ ${USE_GRAPHBOLT} == "true" || ${USE_GRAPHBOLT} == "false" ]]; then
    : # Do nothing
else
    die "-g/--use-graphbolt parameter needs to be one of 'true' or 'false', got ${USE_GRAPHBOLT}"
fi


JOBS_TO_RUN="gconstruct train inference"

DATASET_S3_PATH="s3://${BUCKET_NAME}/papers-100M-input"
OUTPUT_PATH="s3://${BUCKET_NAME}/pipelines-output"
GRAPH_NAME="papers-100M"
INSTANCE_COUNT="4"

CPU_INSTANCE_TYPE="ml.r5.24xlarge"
TRAIN_GPU_INSTANCE="ml.g5.48xlarge"
GCONSTRUCT_INSTANCE="ml.r5.24xlarge"
NUM_TRAINERS=8

GSF_CPU_IMAGE_URI=${ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/graphstorm:sagemaker-cpu
GSF_GPU_IMAGE_URI=${ACCOUNT}.dkr.ecr.$REGION.amazonaws.com/graphstorm:sagemaker-gpu

GCONSTRUCT_CONFIG="gconstruct_config_papers100m.json"
GRAPH_CONSTRUCTION_ARGS="--num-processes 16"

PARTITION_OUTPUT_JSON="metadata.json"
PARTITION_OUTPUT_JSON="$GRAPH_NAME.json"
PARTITION_ALGORITHM="metis"
TRAIN_YAML_S3="s3://$BUCKET_NAME/yaml/papers100M_nc.yaml"
INFERENCE_YAML_S3="s3://$BUCKET_NAME/yaml/papers100M_nc.yaml"
TASK_TYPE="node_classification"
INFERENCE_MODEL_SNAPSHOT="epoch-14"
VOLUME_SIZE=400

if [[ -z "${PIPELINE_NAME-}" ]]; then
    if [[ $USE_GRAPHBOLT == "true" ]]; then
        PIPELINE_NAME="papers100M-gs-graphbolt-pipeline"
    else
        PIPELINE_NAME="papers100M-gs-pipeline"
    fi
fi

python3 $SCRIPT_DIR/../../sagemaker/pipeline/create_sm_pipeline.py \
    --execution-role "${ROLE_ARN}" \
    --cpu-instance-type ${CPU_INSTANCE_TYPE} \
    --gpu-instance-type ${TRAIN_GPU_INSTANCE} \
    --graph-construction-args "${GRAPH_CONSTRUCTION_ARGS}" \
    --graph-construction-instance-type ${GCONSTRUCT_INSTANCE} \
    --graph-construction-config-filename ${GCONSTRUCT_CONFIG} \
    --graph-name ${GRAPH_NAME} \
    --graphstorm-pytorch-cpu-image-uri "${GSF_CPU_IMAGE_URI}" \
    --graphstorm-pytorch-gpu-image-uri "${GSF_GPU_IMAGE_URI}" \
    --inference-model-snapshot "${INFERENCE_MODEL_SNAPSHOT}" \
    --inference-yaml-s3 "${INFERENCE_YAML_S3}" \
    --input-data-s3 "${DATASET_S3_PATH}" \
    --instance-count ${INSTANCE_COUNT} \
    --jobs-to-run ${JOBS_TO_RUN} \
    --num-trainers ${NUM_TRAINERS} \
    --output-prefix-s3 "${OUTPUT_PATH}" \
    --pipeline-name "${PIPELINE_NAME}" \
    --partition-output-json ${PARTITION_OUTPUT_JSON} \
    --partition-algorithm ${PARTITION_ALGORITHM} \
    --region ${REGION} \
    --train-inference-task ${TASK_TYPE} \
    --train-yaml-s3 "${TRAIN_YAML_S3}" \
    --save-embeddings \
    --save-predictions \
    --volume-size-gb ${VOLUME_SIZE} \
    --use-graphbolt ${USE_GRAPHBOLT}
