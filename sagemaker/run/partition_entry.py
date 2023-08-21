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

    SageMaker training entry point.
"""
import argparse

from graphstorm.sagemaker.sagemaker_partition import run_partition, PartitionJobConfig

def partition_arg_parser():
    """ Add arguments for graph partition
    """
    parser = argparse.ArgumentParser(description='GSF SageMaker Partitioning')

    parser.add_argument("--graph-data-s3", type=str,
        help="S3 location of input graph")
    parser.add_argument("--num-parts", type=int, help="Number of partitions")
    parser.add_argument("--output-data-s3", type=str,
        help="S3 location to store the partitioned graph")
    parser.add_argument("--metadata-filename", type=str,
        default="metadata.json", help="file name of metadata config file")
    parser.add_argument("--partition-algorithm", type=str, default='random',
        choices=['random'],
        help="Partition algorithm to use.")
    parser.add_argument("--skip-partitioning", type=str, default='false',
        choices=['true', 'false'],
        help="When 'true', assume partition assignments already exists on S3 at the "
             "output location, just do the data_dispatch step.")
    parser.add_argument('--log-level', default='INFO',
        type=str, choices=['DEBUG', 'INFO', 'WARNING', 'CRITICAL', 'FATAL'])

    return parser

if __name__ =='__main__':
    args =  partition_arg_parser().parse_args()

    job_config = PartitionJobConfig(
        graph_data_s3=args.graph_data_s3,
        num_parts=args.num_parts,
        log_level=args.log_level,
        output_data_s3=args.output_data_s3,
        partition_algorithm=args.partition_algorithm,
        skip_partitioning=args.skip_partitioning=='true',
        metadata_filename=args.metadata_filename
    )

    run_partition(job_config)
