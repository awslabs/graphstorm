"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Launch SageMaker training task
"""
import os
import argparse

import boto3 # pylint: disable=import-error
from sagemaker.pytorch.estimator import PyTorch
import sagemaker

INSTANCE_TYPE = "ml.g4dn.12xlarge"
SUPPORTED_TASKS = {
    "node_classification",
    "node_regression",
    "edge_classification",
    "edge_regression",
    "link_prediction"
}

def run_job(input_args, image, unknowargs):
    """ Run job using SageMaker estimator.PyTorch

        TODO: We may need to simplify the argument list. We can use a config object.

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    unknowargs: dict
        GraphStorm parameters
    """
    sm_task_name = input_args.task_name # SageMaker task name
    role = input_args.role # SageMaker ARN role
    instance_type = input_args.instance_type # SageMaker instance type
    instance_count = input_args.instance_count # Number of infernece instances
    region = input_args.region # AWS region
    entry_point = input_args.entry_point # GraphStorm training entry_point
    task_type = input_args.task_type # Training task type
    graph_name = input_args.graph_name # Training graph name
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing partitioned graph data
    train_yaml_s3 = input_args.yaml_s3 # S3 location storing the yaml file
    model_artifact_s3 = input_args.model_artifact_s3 # Where to store model artifacts
    custom_script = input_args.custom_script # custom_script if any

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
              "model-artifact-s3": model_artifact_s3}
    if custom_script is not None:
        params["custom-script"] = custom_script
    # We must handle cases like
    # --target-etype query,clicks,asin query,search,asin
    # --feat-name ntype0:feat0 ntype1:feat1
    unknow_idx = 0
    while unknow_idx < len(unknowargs):
        assert unknowargs[unknow_idx].startswith("--")
        sub_params = []
        for i in range(unknow_idx+1, len(unknowargs)+1):
            # end of loop or stand with --
            if i == len(unknowargs) or \
                unknowargs[i].startswith("--"):
                break
            sub_params.append(unknowargs[i])
        params[unknowargs[unknow_idx][2:]] = ' '.join(sub_params)
        unknow_idx = i

    print(f"Parameters {params}")
    print(f"GraphStorm Parameters {unknowargs}")

    est = PyTorch(
        entry_point=os.path.basename(entry_point),
        source_dir=os.path.dirname(entry_point),
        image_uri=container_image_uri,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        output_path=model_artifact_s3,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        sagemaker_session=sess,
        tags=[{"Key":"GraphStorm", "Value":"oss"},
              {"Key":"GraphStorm_Task", "Value":"Training"}],
    )

    est.fit({"train": train_yaml_s3}, job_name=sm_task_name, wait=True)

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
        default=2,
        help="number of infernece instances")
    parser.add_argument("--region", type=str,
        default="us-east-1",
        help="Region")
    parser.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/run/train_entry.py",
        help="PATH-TO graphstorm/sagemaker/scripts/sagemaker_train.py")
    parser.add_argument("--task-name", type=str,
        default=None, help="User defined SageMaker task name")

    # task specific
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input training graph")
    parser.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}")
    parser.add_argument("--yaml-s3", type=str,
        help="S3 location of training yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--model-artifact-s3", type=str, default=None,
        help="S3 bucket to save model artifacts")
    parser.add_argument("--custom-script", type=str, default=None,
        help="Custom training script provided by a customer to run customer training logic. \
            Please provide the path of the script within the docker image")

    return parser

if __name__ == "__main__":
    arg_parser = parse_args()
    args, unknownargs = arg_parser.parse_known_args()
    print(args)

    train_image = args.image_url

    run_job(args, train_image, unknownargs)
