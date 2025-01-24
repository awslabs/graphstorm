#!/usr/bin/env bash
set -Eeuox pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    # script cleanup here
}

# Download and unzip data in parallel
TEMP_DATA_PATH=/tmp/raw-data
mkdir -p $TEMP_DATA_PATH
cd $TEMP_DATA_PATH || exit 1


echo "Will execute script $1 with output prefix $2"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): Downloading files using axel, this will take at least 10 minutes depending on network speed"
time axel -n 16 --quiet http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): Unzipping files using ripunzip this will take 10-20 minutes"
time ripunzip unzip-file papers100M-bin.zip
# npz files are zip files, so we can also unzip them in parallel
cd papers100M-bin/raw || exit 1
time ripunzip unzip-file data.npz && rm data.npz


# Run the processing script
echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ'): Processing data and uploading to S3, this will take around 20 minutes"
time python3 /opt/ml/code/"$1" \
    --input-dir "$TEMP_DATA_PATH/papers100M-bin/" \
    --output-prefix "$2"
