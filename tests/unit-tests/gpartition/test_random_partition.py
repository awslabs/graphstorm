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
import pytest
from tempfile import TemporaryDirectory

from graphstorm.gpartition import RandomPartitionAlgorithm

from conftest import simple_test_partition

def test_create_random_partition_basic():
    chunked_metadata_dict = {
        "num_nodes_per_type": [10, 20, 1024],
        "node_type": ["a", "b", "c"],
    }
    rand_partitioner = RandomPartitionAlgorithm(chunked_metadata_dict)
    simple_test_partition(
        rand_partitioner,
        "random",
        num_parts=4,
        chunked_metadata_dict=chunked_metadata_dict
    )


@pytest.mark.parametrize("num_nodes_parts", [1, 2, 8, 256, 512])
def test_random_partition_equal_nodes_parts(num_nodes_parts: int):
    """Test random partition when num_nodes==num_partitions"""
    chunked_meta_dict = {
        "num_nodes_per_type": [num_nodes_parts],
        "node_type": ["a"],
    }

    rand_partitioner = RandomPartitionAlgorithm(chunked_meta_dict)
    # Repeat assignment ten times to get random results
    for _ in range(10):
        simple_test_partition(
            rand_partitioner, "random", num_nodes_parts, chunked_meta_dict)

def test_random_partition_invalid_num_parts():
    """Test random partition when requested num_parts is not valid"""
    num_nodes = 4
    chunked_meta_dict = {
        "num_nodes_per_type": [num_nodes],
        "node_type": ["a"],
    }

    rand_partitioner = RandomPartitionAlgorithm(chunked_meta_dict)
    with TemporaryDirectory() as tmpdirname:
        # Ask for one more partition than num_nodes
        with pytest.raises(RuntimeError, match="Number of nodes for node type "):
            rand_partitioner.create_partitions(
                num_partitions=num_nodes+1,
                partition_assignment_dir=tmpdirname)

        # Ask for zero partitions
        with pytest.raises(AssertionError, match="Number of partitions must be greater than 0"):
            rand_partitioner.create_partitions(
                num_partitions=0,
                partition_assignment_dir=tmpdirname)
