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

    Random partition algorithm
"""
import os
import logging
import json

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

class RandomPartitionAlgorithm():
    """
    Single-instance random partitioning algorithm.

    The partition algorithm accepts the intermediate output from GraphStorm
    gs-processing which matches the requirements of the DGL distributed
    partitioning pipeline. It does random node assignments for each node
    during partition and output the node assignment result to the `output_path`.

    Parameters
    ----------
    output_path: str
        Local output path.
    metadata: dict
        Partition metadata JSON.
    rank: int
        Rank of the current worker process.
    """
    def __init__(self, output_path, metadata, rank=0):
        self.local_output_path = output_path
        self.metadata = metadata
        self.rank = rank

    def run_partitioning(self, num_partitions):
        """ Run partition

        Parameters
        ----------
        num_partitions: int
            Number of target partitions.
        """
        partition_dir = os.path.join(self.local_output_path, "partition")
        os.makedirs(partition_dir, exist_ok=True)

        # Random partitioning is done on the leader node only
        if self.rank != 0:
            return partition_dir

        num_nodes_per_type = self.metadata["num_nodes_per_type"]  # type: list [int]
        ntypes = self.metadata["node_type"]  # type: list [str]
        # Note: This assumes that the order of node_type is the same as the order num_nodes_per_type
        for ntype, num_nodes_for_type in zip(ntypes, num_nodes_per_type):
            logging.info("Generating random partition for node type %s", ntype)
            ntype_output = os.path.join(partition_dir, f"{ntype}.txt")

            partition_assignment = np.random.randint(0, num_partitions, (num_nodes_for_type,))

            arrow_partitions = pa.Table.from_arrays(
                [pa.array(partition_assignment, type=pa.int64())],
                names=["partition_id"])
            options = pa_csv.WriteOptions(include_header=False, delimiter=' ')
            pa_csv.write_csv(arrow_partitions, ntype_output, write_options=options)

        self._create_metadata(num_partitions, partition_dir)

        return partition_dir

    @staticmethod
    def _create_metadata(num_partitions, partition_dir):
        """
        Creates the metadata file expected by the partitioning pipeline.

        https://github.com/dmlc/dgl/blob/29e666152390c272e0115ce8455da1adb5fcacb1/tools/partition_algo/base.py#L8
        defines the partition_meta.json format
        """
        partition_meta = {
            "algo_name": "random",
            "num_parts": num_partitions,
            "version": "1.0.0"
        }
        partition_meta_filepath = os.path.join(partition_dir, "partition_meta.json")
        with open(partition_meta_filepath, "w", encoding='utf-8') as metafile:
            json.dump(partition_meta, metafile)

class LocalRandomPartitioner(): # pylint: disable=too-few-public-methods
    """
    Single-instance random partitioning algorithm running on a Linux instance.

    Parameters
    ----------
    metadata_file : str
        Path to the metadata file describing the graph.
    local_output_path : str
        Path to the local directory where the partitioning results should be stored.
    """
    def __init__(self, metadata_file, local_output_path):
        self.metadata_file = metadata_file
        self.local_output_path = local_output_path

        with open(self.metadata_file, 'r', encoding='utf-8') as metafile:
            self.metadata = json.load(metafile)

        self.graph_name = self.metadata["graph_name"]

        os.makedirs(self.local_output_path, exist_ok=True)

    def create_partitions(self, num_partitions):
        """
        Creates a partitioning

        Expected

        Parameters
        ----------
        num_partitions : int
            Number of partitions to create.
        """
        random_part = RandomPartitionAlgorithm(self.local_output_path,
                                               self.metadata)
        partition_dir = random_part.run_partitioning(num_partitions)

        return partition_dir
