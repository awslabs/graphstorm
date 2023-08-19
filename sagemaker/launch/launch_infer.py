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

    Launch SageMaker inference task
"""
import os

import boto3 # pylint: disable=import-error
import sagemaker
from sagemaker.pytorch.estimator import PyTorch

from common_parser import get_common_parser, parse_estimator_kwargs, SUPPORTED_TASKS

INSTANCE_TYPE = "ml.g4dn.12xlarge"

def run_job(input_args, image, unknownargs):
    """ Run job using SageMaker estimator.PyTorch

        We use SageMaker training task to run offline inference.

        TODO: We may need to simplify the argument list. We can use a config object.

    Parameters
    ----------
    input_args:
        Input arguments
    image: str
        ECR image uri
    unknownargs: dict
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
    infer_yaml_s3 = input_args.yaml_s3 # S3 location storing the yaml file
    output_emb_s3_path = input_args.output_emb_s3 # S3 location to save node embeddings
    output_predict_s3_path = input_args.output_prediction_s3 # S3 location to save prediction results
    model_artifact_s3 = input_args.model_artifact_s3 # S3 location of saved model artifacts

    boto_session = boto3.session.Session(region_name=region)
    sagemaker_client = boto_session.client(service_name="sagemaker", region_name=region)
    # need to skip s3://
    assert model_artifact_s3.startswith('s3://'), \
        "Saved model artifact should be stored in S3"
    sess = sagemaker.session.Session(boto_session=boto_session,
        sagemaker_client=sagemaker_client)

    container_image_uri = image

    prefix = f"gs-infer-{graph_name}"

    # In Link Prediction, no prediction outputs
    if task_type == "link_prediction":
        params = {"task-type": task_type,
                  "graph-name": graph_name,
                  "graph-data-s3": graph_data_s3,
                  "infer-yaml-s3": infer_yaml_s3,
                  "output-emb-s3": output_emb_s3_path,
                  "model-artifact-s3": model_artifact_s3}
    else:
        params = {"task-type": task_type,
                  "graph-name": graph_name,
                  "graph-data-s3": graph_data_s3,
                  "infer-yaml-s3": infer_yaml_s3,
                  "output-emb-s3": output_emb_s3_path,
                  "output-prediction-s3": output_predict_s3_path,
                  "model-artifact-s3": model_artifact_s3}
    # We must handle cases like
    # --target-etype query,clicks,asin query,search,asin
    # --feat-name ntype0:feat0 ntype1:feat1
    unknow_idx = 0
    while unknow_idx < len(unknownargs):
        print(unknownargs[unknow_idx])
        assert unknownargs[unknow_idx].startswith("--")
        sub_params = []
        for i in range(unknow_idx+1, len(unknownargs)+1):
            # end of loop or stand with --
            if i == len(unknownargs) or \
                unknownargs[i].startswith("--"):
                break
            sub_params.append(unknownargs[i])
        params[unknownargs[unknow_idx][2:]] = ' '.join(sub_params)
        unknow_idx = i

    print(f"Parameters {params}")
    print(f"GraphStorm Parameters {unknownargs}")
    if input_args.sm_estimator_parameters:
        print(f"SageMaker Estimator parameters: '{input_args.sm_estimator_parameters}'")

    estimator_kwargs = parse_estimator_kwargs(input_args.sm_estimator_parameters)

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
        **estimator_kwargs
    )

    est.fit(inputs={"train": infer_yaml_s3}, job_name=sm_task_name, wait=not input_args.async_execution)

def get_inference_parser():
    """
    Get GraphStorm inference task parser.
    """
    parser = get_common_parser()

    inference_args = parser.add_argument_group("GraphStorm Inference Args")

    # task specific
    inference_args.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/run/infer_entry.py",
        help="PATH-TO graphstorm/sagemaker/run/infer_entry.py")
    inference_args.add_argument("--graph-name", type=str, help="Graph name",
        required=True)
    inference_args.add_argument("--graph-data-s3", type=str,
        help="S3 location of input inference graph",
        required=True)
    inference_args.add_argument("--task-type", type=str,
        help=f"Task type in {SUPPORTED_TASKS}",
        required=True)
    inference_args.add_argument("--yaml-s3", type=str,
        help="S3 location of inference yaml file. "
             "Do not store it with partitioned graph",
             required=True)
    inference_args.add_argument("--model-artifact-s3", type=str,
        help="S3 bucket to load the saved model artifacts",
        required=True)
    inference_args.add_argument("--output-emb-s3", type=str,
        help="S3 location to store GraphStorm generated node embeddings.",
        default=None)
    inference_args.add_argument("--output-prediction-s3", type=str,
        help="S3 location to store prediction results. " \
             "(Only works with node classification/regression " \
             "and edge classification/regression tasks)",
        default=None)
    inference_args.add_argument("--model-sub-path", type=str, default=None,
        help="Relative path to the trained model under <model_artifact_s3>."
             "There can be multiple model checkpoints under"
             "<model_artifact_s3>, this argument is used to choose one.")

    return parser

if __name__ == "__main__":
    arg_parser = get_inference_parser()
    args, unknownargs = arg_parser.parse_known_args()
    print(args)

    infer_image = args.image_url
    if not args.instance_type:
        args.instance_type = INSTANCE_TYPE


    run_job(args, infer_image, unknownargs)
