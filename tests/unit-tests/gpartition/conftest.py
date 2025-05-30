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
import json
from tempfile import TemporaryDirectory
from typing import Dict

import numpy as np
import pytest

from graphstorm.gpartition import LocalPartitionAlgorithm

def simple_test_partition(
    partition_algorithm: LocalPartitionAlgorithm,
    algorithm_name: str,
    num_parts: int,
    chunked_metadata_dict: Dict):
    """Ensures that the provided algorithm will create the correct number of partitions,
    that the output contains at least one node per partition,
    and that the partition assignment and metadata files are correctly created.

    Parameters
    ----------
    partition_algorithm : LocalPartitionAlgorithm
        An implementation of the LocalPartitionAlgorithm base class.
    algorithm_name : str
        The expected name for the partition algorithm in the metadata
    num_parts: int
        Number of partitions to assign nodes to.
    chunked_metadata_dict : Dict
        The metadata dictionary that was passed to the algorithm.
    """

    with TemporaryDirectory() as tmpdir:
        assert partition_algorithm.metadata_dict == chunked_metadata_dict
        partition_algorithm.create_partitions(num_parts, tmpdir)

        for ntype in chunked_metadata_dict["node_type"]:
            assert os.path.exists(os.path.join(tmpdir, f"{ntype}.txt"))
        assert os.path.exists(os.path.join(tmpdir, "partition_meta.json"))

        # Ensure contents of partition_meta.json are correct
        with open(os.path.join(tmpdir, "partition_meta.json"), 'r', encoding="utf-8") as f:
            part_meta = json.load(f)
            assert part_meta["num_parts"] == num_parts
            assert part_meta["algo_name"] ==  algorithm_name

        # Ensure contents of partition assignment files are correct
        for i, node_type in enumerate(chunked_metadata_dict["node_type"]):
            with open(os.path.join(tmpdir, f"{node_type}.txt"), "r", encoding="utf-8") as f:
                node_partitions = f.read().splitlines()
                partition_array = np.array(node_partitions)
                assert len(np.unique(partition_array)) == num_parts, \
                    "Every partition needs to contain at least one node"
                assert len(node_partitions) == chunked_metadata_dict["num_nodes_per_type"][i]
                for part_id in node_partitions:
                    assert part_id.isdigit()
                    assert int(part_id) < num_parts
