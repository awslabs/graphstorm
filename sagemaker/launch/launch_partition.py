""" Launch SageMaker training task
"""
import logging
from time import strftime, gmtime

import boto3 # pylint: disable=import-error
from sagemaker.processing import ScriptProcessor
import sagemaker

from common_parser import ( # pylint: disable=wrong-import-order
    get_common_parser, parse_estimator_kwargs)

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
    timestamp = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    # SageMaker base job name
    sm_task_name = input_args.task_name if input_args.task_name else timestamp
    role = input_args.role # SageMaker ARN role
    instance_type = input_args.instance_type # SageMaker instance type
    instance_count = input_args.instance_count # Number of partition instances
    region = input_args.region # AWS region
    entry_point = input_args.entry_point # GraphStorm partition entry_point
    num_parts = input_args.num_parts # Number of partitions
    graph_data_s3 = input_args.graph_data_s3 # S3 location storing input graph data (unpartitioned)
    graph_data_s3 = graph_data_s3[:-1] if graph_data_s3[-1] == '/' \
        else graph_data_s3 # The input will be an S3 folder
    output_data_s3 = input_args.output_data_s3 # S3 location storing partitioned graph data
    output_data_s3 = output_data_s3[:-1] if output_data_s3[-1] == '/' \
        else output_data_s3 # The output will be an S3 folder
    metadata_filename = input_args.metadata_filename # graph metadata filename

    sagemaker_session = sagemaker.Session(boto3.Session(region_name=region))

    skip_partitioning_str = "true" if input_args.skip_partitioning else "false"
    arguments = [
        "--graph-data-s3", graph_data_s3,
        "--metadata-filename", metadata_filename,
        "--num-parts", num_parts,
        "--output-data-s3", output_data_s3,
        "--skip-partitioning", skip_partitioning_str,
        "--log-level", input_args.log_level,
        "--partition-algorithm", input_args.partition_algorithm,
    ]
    arguments = [str(x) for x in arguments]

    print(f"Parameters {arguments}")
    if input_args.sm_estimator_parameters:
        print(f"SageMaker Estimator parameters: '{input_args.sm_estimator_parameters}'")

    estimator_kwargs = parse_estimator_kwargs(input_args.sm_estimator_parameters)

    script_processor = ScriptProcessor(
        image_uri=image,
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        command=["python3"],
        base_job_name=f"gs-partition-{sm_task_name}",
        sagemaker_session=sagemaker_session,
        tags=[{"Key":"GraphStorm", "Value":"beta"},
              {"Key":"GraphStorm_Task", "Value":"Partition"}],
        **estimator_kwargs
    )

    script_processor.run(
        code=entry_point,
        arguments=arguments,
        inputs=[],
        outputs=[],
        wait=not input_args.async_execution
    )

def get_partition_parser():
    """
    Get GraphStorm partition task parser.
    """
    parser = get_common_parser()

    partition_args = parser.add_argument_group("GraphStorm Partition Arguments")

    partition_args.add_argument("--output-data-s3", type=str,
        required=True,
        help="Output S3 location to store the partitioned graph")
    partition_args.add_argument("--num-parts", type=int, help="Number of partitions",
                                required=True)

    partition_args.add_argument("--entry-point", type=str,
        default="graphstorm/sagemaker/run/partition_entry.py",
        help="PATH-TO graphstorm/sagemaker/run/partition_entry.py")

    partition_args.add_argument("--metadata-filename", type=str,
        default="updated_row_counts_metadata.json",
        help="File name of metadata config file for chunked format data")

    partition_args.add_argument("--partition-algorithm", type=str, default='random',
        help="Partition algorithm to use.", choices=['random'])

    partition_args.add_argument("--skip-partitioning", action='store_true',
        help="When set, we skip the partitioning step. "
             "Partition assignments for all node types need to exist on S3.")

    partition_args.add_argument('--log-level', default='INFO',
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

    partition_image = args.image_url
    if not args.instance_type:
        args.instance_type = INSTANCE_TYPE

    run_job(args, partition_image)
