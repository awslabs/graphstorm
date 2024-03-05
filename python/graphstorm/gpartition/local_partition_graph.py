from abc import ABC, abstractmethod
from typing import Dict, List
import os


class LocalPartitionAlgorithm(ABC):
    def __init__(self, metadata_dict: Dict) -> None:
        self.metadata_dict = metadata_dict

    def create_partitions(self, num_partitions: int, partition_assignment_dir: str):
        """Creates partition assignment files for each node type and a partition metadata file.

        Each partition assignment file is a text file named <node_type>.txt where each row
        is the assigned partition for the node with ID equal to the row number.

        The partition metadata file is a JSON file named partition_meta.json with the following format:
        {
            "algo_name": <algorithm_name>,
            "num_parts": <num_partitions>,
            "version": "1.0.0"
        }

        Parameters
        ----------
        num_parts : int
            Number of partitions to create
        partition_assignment_dir : str
            Local directory under which to create the partition assignment files and partition
            metadata file.
        """

        # Creates one partition assignment file per node type
        self._assign_partitions(num_partitions, partition_assignment_dir)
        ntypes = self.metadata_dict["node_type"]  # type: List[str]
        for ntype in ntypes:
            assert os.path.exists(os.path.join(partition_assignment_dir, f"{ntype}.txt")), \
                f"Missing partition assignment for node type {ntype}"

        # Creates the partition metadata file
        self._create_metadata(num_partitions, partition_assignment_dir)
        assert os.path.exists(os.path.join(partition_assignment_dir, "partition_meta.json")), \
            f"Missing partition partition_meta.json file under {partition_assignment_dir}"

    @abstractmethod
    def _assign_partitions(self, num_partitions: int, partition_dir: str):
        """Assigns each node in the data to a partition from 0 to `num_partitions-1`,
        and creates one "{ntype}".json partition assignment file per node type.

        Parameters
        ----------
        num_partitions: int
            Number of target partitions.
        partition_dir: str
            Directory under which to create the partition assignment files.
        """

    @abstractmethod
    def _create_metadata(self, num_partitions: int, partition_dir: str):
        """Creates the partition_meta.json file expected by the partitioning pipeline.

        https://github.com/dmlc/dgl/blob/29e666152390c272e0115ce8455da1adb5fcacb1/tools/partition_algo/base.py#L8
        defines the partition_meta.json format

        Parameters
        ----------
        num_partitions: int
            Number of target partitions.
        partition_dir: str
            Directory under which to create the partition metadata file.
        """
