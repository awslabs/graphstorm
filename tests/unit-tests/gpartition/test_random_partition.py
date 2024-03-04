import json
import os
from typing import Dict
from tempfile import TemporaryDirectory

import pytest

from graphstorm.gpartition import RandomPartitionAlgorithm

@pytest.fixture(scope="module")
def metadata_dict() -> Dict:
    return {
        "num_nodes_per_type": [10, 20],
        "node_type": ["a", "b"],
    }


def test_create_random_partition(metadata_dict):
    with TemporaryDirectory() as tmpdir:
        num_parts = 4
        rand_partitioner = RandomPartitionAlgorithm(metadata_dict)
        rand_partitioner.create_partitions(num_parts, tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "a.txt"))
        assert os.path.exists(os.path.join(tmpdir, "b.txt"))
        assert os.path.exists(os.path.join(tmpdir, "partition_meta.json"))

        with open(os.path.join(tmpdir, "partition_meta.json"), 'r', encoding="utf-8") as f:
            part_meta = json.load(f)
            assert part_meta["num_parts"] == num_parts
            assert part_meta["algo_name"] ==  "random"

        for i, node_type in enumerate(metadata_dict["node_type"]):
            with open(os.path.join(tmpdir, f"{node_type}.txt"), "r") as f:
                node_partitions = f.read().splitlines()
                assert len(node_partitions) == metadata_dict["num_nodes_per_type"][i]
                for part_id in node_partitions:
                    assert part_id.isdigit()
                    assert int(part_id) < num_parts


if __name__ == '__main__':
    test_create_random_partition(metadata_dict())
