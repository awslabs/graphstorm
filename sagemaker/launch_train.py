""" Launch SageMaker training task
"""
import os
import argparse

import boto3 # pylint: disable=import-error

from graphstorm.config import SUPPORTED_TASKS

from sagemaker.pytorch.estimator import PyTorch
import sagemaker

INSTANCE_TYPE = "ml.g4dn.12xlarge"

def run_job(image, role, instance_type, region, entry_point,
    task_type, graph_name, graph_data_s3,
    train_yaml_s3, train_yaml_name, enable_bert,
    model_artifact_s3, unknowargs):
    """ Run job using SageMaker estimator.PyTorch

    Parameters
    ----------
    image: str
        ECR image uri
    role: str
        SageMaker ARN role
    instance_type: str
        SageMaker training instance type
    region: str
        AWS region
    entry_point: str
        GraphStorm trainer entry_point
    task_type: str
        Training task type
    graph_name: str
        Training graph name
    graph_data_s3: str
        S3 location storing partitioned graph data
    train_yaml_s3: str
        S3 location storing the yaml file
    train_yaml_name: str
        Yaml file name
    enable_bert: bool
        Whether enable bert contraining
    model_artifact_s3: str
        Where to store model artifacts
    unknowargs: dict
        GraphStorm parameters
    """
    boto_session = boto3.session.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    sess = sagemaker.session.Session(boto_session=boto_session,
        sagemaker_client=sagemaker_client)

    container_image_uri = image

    prefix = "script-mode-container"

    params = {"task-type": task_type,
              "graph-name": graph_name,
              "graph-data-s3": graph_data_s3,
              "train-yaml-s3": train_yaml_s3,
              "train-yaml-name": train_yaml_name,
              "enable-bert": enable_bert}
    for i in range(len(unknowargs)//2):
        # trim --
        params[unknowargs[i*2][2:]] = unknowargs[i*2+1]

    print(f"Parameters {params}")
    print(f"GraphStorm Parameters {unknowargs}")

    est = PyTorch(
        entry_point=os.path.basename(entry_point),
        source_dir=os.path.dirname(entry_point),
        image_uri=container_image_uri,
        role=role,
        instance_count=2,
        instance_type=instance_type,
        output_path=model_artifact_s3,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        sagemaker_session=sess,
    )

    est.fit({"train": train_yaml_s3}, wait=True)

def parse_args():
    """ Add arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--version-tag", type=str,
        default="sagemaker_v3",
        help="Training image tag")
    parser.add_argument("--training-ecr-repository", type=str,
        default="graphlytics-pytorch-training-dev",
        help="ECR repository")
    parser.add_argument("--account-id", type=str,
        help="AWS account number")
    parser.add_argument("--role", type=str,
        help="SageMaker role")
    parser.add_argument("--instance-type", type=str,
        default=INSTANCE_TYPE,
        help="instance type used to train models")
    parser.add_argument("--region", type=str,
        default="us-east-1",
        help="Region")
    parser.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/scripts/sagemaker_train.py",
        help="PATH-TO graphstorm/sagemaker/scripts/sagemaker_train.py")

    # task specific
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}")
    parser.add_argument("--train-yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--train-yaml-name", type=str,
        help="Training yaml config file name")
    parser.add_argument("--enable-bert", type=bool, default=False,
        help="Whether enable cotraining Bert with GNN")
    parser.add_argument("--model-artifact-s3", type=str, default=None,
        help="S3 bucket to save model artifacts")

    return parser

if __name__ == "__main__":
    arg_parser = parse_args()
    args, unknownargs = arg_parser.parse_known_args()
    print(args)

    train_image = f"{args.account_id}.dkr.ecr.{args.region}.amazonaws.com/" \
        f"{args.training_ecr_repository}:{args.version_tag}"

    run_job(train_image, args.role, args.instance_type, args.region,
        args.entry_point, args.task_type,
        args.graph_name, args.graph_data_s3,
        args.train_yaml_s3, args.train_yaml_name,
        args.enable_bert, args.model_artifact_s3, unknownargs)
