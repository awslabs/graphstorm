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

    Do random partition for distributed data processing.
    This script only works with graphstorm-processing (Distirbuted graph processing).
"""
import argparse
import time

from graphstorm.sagemaker.partition_algorithm import (PartitionerConfig,
                                                      LocalRandomPartitioner)


def main(args):
    partition_config = PartitionerConfig(
        metadata_file=args.meta_info_path,
        local_output_path=args.output_path,
        rank=0)

    partitioner = LocalRandomPartitioner(partition_config)
    local_partition_path = partitioner.create_partitions(args.num_parts)
    print(local_partition_path)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition DGL graphs for node and edge classification "
                                        + "or regression tasks")
    argparser.add_argument("--meta-info-path", type=str, required=True,
                           help="Path to meta-info")
    argparser.add_argument("--output-path", type=str, required=True,
                           help="Path to store the partitioned data")
    argparser.add_argument("--num-parts", type=int, required=True,
                           help="Number of partitions to generate")

    args = argparser.parse_args()
    start = time.time()

    main(args)
    print(f'Partition takes {time.time() - start:.3f} seconds')
