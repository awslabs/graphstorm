""" Launch SageMaker training task
"""
import os
import argparse
import boto3 # pylint: disable=import-error

from sagemaker.pytorch.estimator import PyTorch
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
    graph_name = input_args.graph_name # Graph name
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing input graph data (unpartitioned)
    graph_data_s3 = graph_data_s3[:-1] if graph_data_s3[-1] == '/' \
        else graph_data_s3 # The input will be an S3 folder
    output_data_s3 = input_args.output_data_s3 # S3 location storing partitioned graph data
    output_data_s3 = output_data_s3[:-1] if output_data_s3[-1] == '/' \
        else output_data_s3 # The output will be an S3 folder
    metadata_filename = input_args.metadata_filename # graph metadata filename

    boto_session = boto3.session.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    sess = sagemaker.session.Session(boto_session=boto_session,
        sagemaker_client=sagemaker_client)

    container_image_uri = image

    prefix = f"parmetis-{graph_name}"

    params = {"graph-name": graph_name,
              "graph-data-s3": graph_data_s3,
              "metadata-filename": metadata_filename,
              "num-parts": num_parts,
              "output-data-s3": output_data_s3,}

    print(f"Parameters {params}")

    est = PyTorch(
        entry_point=os.path.basename(entry_point),
        source_dir=os.path.dirname(entry_point),
        image_uri=container_image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        max_run=3600,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        sagemaker_session=sess,
        tags=[{"Key":"GraphStorm", "Value":"beta"},
              {"Key":"GraphStorm_Task", "Value":"Partition"}],
    )

    est.fit(job_name=sm_task_name, wait=True)

def parse_args():
    """ Add arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-url", type=str,
        help="Training docker image")
    parser.add_argument("--role", type=str,
        help="SageMaker role")
    parser.add_argument("--instance-type", type=str,
        default=INSTANCE_TYPE,
        help="instance type used to train models")
    parser.add_argument("--instance-count", type=int,
        default=4,
        help="number of training instances")
    parser.add_argument("--region", type=str,
        default="us-east-1",
        help="Region to launch the task")
    parser.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/scripts/sagemaker_parmetis.py",
        help="PATH-TO graphstorm/sagemaker/scripts/sagemaker_parmetis.py")
    parser.add_argument("--task-name", type=str,
        default=None, help="User defined SageMaker task name")

    # task specific
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--metadata-filename", type=str,
        default="metadata.json", help="file name of metadata config file")
    parser.add_argument("--num-parts", type=int, help="Number of partitions")
    parser.add_argument("--output-data-s3", type=str,
        help="S3 location to store the partitioned graph")

    return parser

if __name__ == "__main__":
    arg_parser = parse_args()
    args = arg_parser.parse_args()
    print(args)

    train_image = args.image_url
    run_job(args, train_image)
