""" Launch SageMaker training task
"""
import os
import sagemaker
from sagemaker.pytorch.estimator import PyTorch
import argparse
import boto3 # pylint: disable=import-error

from graphstorm.config import SUPPORTED_TASKS

from sagemaker.pytorch.estimator import PyTorch
import sagemaker

INSTANCE_TYPE = "ml.g4dn.12xlarge"

def run_job(input_args, image, unknowargs):
    """ Run job using SageMaker estimator.PyTorch

        We use SageMaker training task to run offline inference.

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
    entry_point = input_args.entry_point # GraphStorm inference entry_point
    task_type = input_args.task_type # Inference task type
    graph_name = input_args.graph_name # Inference graph name
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing partitioned graph data
    infer_yaml_s3 = input_args.infer_yaml_s3 # S3 location storing the yaml file
    infer_yaml_name = input_args.infer_yaml_name # Yaml file name
    emb_s3_path = input_args.emb_s3_path # S3 location to save node embeddings
    enable_bert = input_args.enable_bert # Whether enable bert contraining
    model_artifact_s3 = input_args.model_artifact_s3 # S3 location of saved model artifacts
    model_sub_path = input_args.model_sub_path # Relative path to the trained
                                               # model under <model_artifact_s3>

    boto_session = boto3.session.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    # need to skip s3://
    assert model_artifact_s3.startswith('s3://'), \
        "Saved model artifact should be stored in S3"
    sess = sagemaker.session.Session(boto_session=boto_session,
        sagemaker_client=sagemaker_client)

    container_image_uri = image

    prefix = "script-mode-container"

    params = {"task-type": task_type,
              "graph-name": graph_name,
              "graph-data-s3": graph_data_s3,
              "infer-yaml-s3": infer_yaml_s3,
              "infer-yaml-name": infer_yaml_name,
              "emb-s3-path": emb_s3_path,
              "model-artifact-s3": model_artifact_s3,
              "model-sub-path": model_sub_path,
              "enable-bert": enable_bert}
    # We must handle cases like
    # --target-etype query,clicks,asin query,search,asin
    # --feat-name ntype0:feat0 ntype1:feat1
    unknow_idx = 0
    while unknow_idx < len(unknowargs):
        print(unknowargs[unknow_idx])
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
        model_uri=model_artifact_s3,
        py_version="py3",
        base_job_name=prefix,
        hyperparameters=params,
        sagemaker_session=sess,
        tags=[{"Key":"GraphStorm", "Value":"beta"},
              {"Key":"GraphStorm_Task", "Value":"Inference"}],
    )

    est.fit(inputs={"train": infer_yaml_s3}, job_name=sm_task_name, wait=True)

def parse_args():
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
        default="graphstorm/sagemaker/scripts/sagemaker_infer.py",
        help="PATH-TO graphstorm/sagemaker/scripts/sagemaker_infer.py")
    parser.add_argument("--task-name", type=str,
        default=None, help="User defined SageMaker task name")

    # task specific
    parser.add_argument("--graph-name", type=str, help="Graph name")
    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input inference graph")
    parser.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}")
    parser.add_argument("--infer-yaml-s3", type=str,
        help="S3 location of inference yaml file. "
             "Do not store it with partitioned graph")
    parser.add_argument("--infer-yaml-name", type=str,
        help="Training yaml config file name")
    parser.add_argument("--enable-bert",
        type=lambda x: (str(x).lower() in ['true', '1']), default=False,
        help="Whether enable cotraining Bert with GNN")
    parser.add_argument("--emb-s3-path", type=str,
        help="S3 location to save node embeddings")
    parser.add_argument("--model-artifact-s3", type=str,
        help="S3 bucket to load the saved model artifacts")
    parser.add_argument("--model-sub-path", type=str, default=None,
        help="Relative path to the trained model under <model_artifact_s3>."
             "There can be multiple model checkpoints under"
             "<model_artifact_s3>, this argument is used to choose one.")

    return parser

if __name__ == "__main__":
    arg_parser = parse_args()
    args, unknownargs = arg_parser.parse_known_args()
    print(args)

    infer_image = args.image_url

    run_job(args, infer_image, unknownargs)
