""" Launch SageMaker training task
"""
import os
import argparse
import json
import logging
import boto3 # pylint: disable=import-error

from sagemaker.pytorch.estimator import PyTorch
from sagemaker.s3 import S3Downloader
import sagemaker


INSTANCE_TYPE = "ml.m5.12xlarge"

def run_job(input_args, image):
    """ Run job using SageMaker estimator.PyTorch

        TODO: We may need to simplify the argument list. We can use a config object.

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    """
    sm_task_name = input_args.task_name # SageMaker task name
    role = input_args.role # SageMaker ARN role
    instance_type = input_args.instance_type # SageMaker instance type
    instance_count = input_args.instance_count # Number of infernece instances
    region = input_args.region # AWS region
    entry_point = input_args.entry_point # GraphStorm training entry_point
    num_parts = input_args.num_parts # Number of partitions
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing input graph data (unpartitioned)
    graph_data_s3 = graph_data_s3[:-1] if graph_data_s3[-1] == '/' \
        else graph_data_s3 # The input will be an S3 folder
    output_data_s3 = input_args.output_data_s3 # S3 location storing partitioned graph data
    output_data_s3 = output_data_s3[:-1] if output_data_s3[-1] == '/' \
        else output_data_s3 # The output will be an S3 folder
    metadata_filename = input_args.metadata_filename # graph metadata filename
    prefix = input_args.job_prefix

    sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))

    container_image_uri = image
    graph_data_s3_no_trailing = graph_data_s3[:-1] if graph_data_s3.endswith('/') else graph_data_s3

    metadata_s3_path = f"{graph_data_s3_no_trailing}/{metadata_filename}"
    metadata_local_path = os.path.join("/tmp", metadata_filename)

    if os.path.exists(metadata_local_path):
        os.remove(metadata_local_path)

    print(f"Downloading metadata file from {metadata_s3_path} into {metadata_local_path}")
    S3Downloader.download(metadata_s3_path, "/tmp/")
    with open(metadata_local_path, 'r') as meta_file: # pylint: disable=unspecified-encoding
        metadata_dict = json.load(meta_file)
        graph_name = metadata_dict["graph_name"]

    print(f"Graph name during launch: {graph_name}")
    skip_partitioning_str = "true" if input_args.skip_partitioning else "false"
    params = {"graph-data-s3": graph_data_s3,
              "metadata-filename": metadata_filename,
              "num-parts": num_parts,
              "output-data-s3": output_data_s3,
              "skip-partitioning": skip_partitioning_str,
              "log-level": input_args.log_level,
              "partition-algorithm": input_args.partition_algorithm,}

    print(f"Parameters {params}")

    est = PyTorch(
        disable_profiler=True,
        debugger_hook_config=False,
        entry_point=os.path.basename(entry_point),
        source_dir=os.path.dirname(entry_point),
        image_uri=container_image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        max_run=input_args.max_runtime,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        sagemaker_session=sagemaker_session,
        tags=[{"Key":"GraphStorm", "Value":"beta"},
              {"Key":"GraphStorm_Task", "Value":"Partition"}],
        container_log_level=logging.getLevelName(input_args.log_level),
        volume_size=args.volume_size
    )

    est.fit(job_name=sm_task_name, wait=input_args.wait_for_job)

def get_partition_parser():
    """ Add arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-url", type=str,
        required=True,
        help="GraphStorm ECR image URI")
    parser.add_argument("--role", type=str,
        required=True,
        help="SageMaker role")
    parser.add_argument("--instance-type", type=str,
        default=INSTANCE_TYPE,
        help="instance type used to train models")
    parser.add_argument("--instance-count", type=int,
        required=True,
        help="number of training instances")
    parser.add_argument("--region", type=str,
        required=True,
        help="Region to launch the task")
    parser.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/run/partition_entry.py",
        help="PATH-TO graphstorm/sagemaker/run/partition_entry.py")
    parser.add_argument("--task-name", type=str,
        default=None, help="User defined SageMaker task name")
    parser.add_argument("--wait-for-job", action='store_true',
        help="When set will wait for the job to finish before returning")


    # task specific
    parser.add_argument("--graph-data-s3", type=str,
        required=True,
        help="S3 location of input training graph in chunked format")
    parser.add_argument("--metadata-filename", type=str,
        default="updated_row_counts_metadata.json",
        help="File name of metadata config file for chunked format data")
    parser.add_argument("--num-parts", type=int, help="Number of partitions")
    parser.add_argument("--partition-algorithm", type=str, default='random',
        help="Partition algorithm to use.", choices=['metis', 'random'])
    parser.add_argument("--skip-partitioning", action='store_true',
        help="When set, we skip the partitioning step. "
             "Partition assignments for all node types need to exist on S3.")
    parser.add_argument("--output-data-s3", type=str,
        help="Output S3 location to store the partitioned graph")
    parser.add_argument("--volume-size", type=int, default=30,
        help="Volume size for each instance in GB.")
    parser.add_argument("--job-prefix", help="Job name prefix", type=str, default='partition')
    parser.add_argument("--max-runtime", help="Maximum runtime in seconds", type=int, default=86400)
    parser.add_argument('--log-level', default='INFO',
        type=str, choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'FATAL'])

    return parser.parse_args()

if __name__ == "__main__":
    args = get_partition_parser()
    print(args)

    # NOTE: Ensure no logging has been done before setting logging configuration
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format='%(asctime)s - %(levelname)s - %(message)s'
        )

    train_image = args.image_url
    run_job(args, train_image)
