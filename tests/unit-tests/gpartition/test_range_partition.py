"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from tempfile import TemporaryDirectory

from graphstorm.gpartition import RangePartitionAlgorithm

from conftest import simple_test_partition


def test_create_range_partition(chunked_metadata_dict):
    range_partitioner = RangePartitionAlgorithm(chunked_metadata_dict)
    # TODO: DGL only supports random and metis as a name downstream
    simple_test_partition(range_partitioner, "random", chunked_metadata_dict)


def test_range_partition_ordered(chunked_metadata_dict):
    with TemporaryDirectory() as tmpdir:
        num_parts = 8
        range_partitioner = RangePartitionAlgorithm(chunked_metadata_dict)
        range_partitioner.create_partitions(num_parts, tmpdir)
        for _, node_type in enumerate(chunked_metadata_dict["node_type"]):
            with open(
                os.path.join(tmpdir, f"{node_type}.txt"), "r", encoding="utf-8"
            ) as f:
                ntype_partitions = [int(x) for x in f.read().splitlines()]
                # Ensure the partition assignments are in increasing order
                assert sorted(ntype_partitions) == ntype_partitions
