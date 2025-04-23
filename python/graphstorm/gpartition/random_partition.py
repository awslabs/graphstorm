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

    Single-instance random partition assignment
"""
import os
import logging
import json

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from .partition_algo_base import LocalPartitionAlgorithm

class RandomPartitionAlgorithm(LocalPartitionAlgorithm):
    """
    Single-instance random partitioning algorithm.

    The partition algorithm accepts the intermediate output from GraphStorm
    gs-processing which matches the requirements of the DGL distributed
    partitioning pipeline. It does random node assignments for each node
    during partition and outputs the node assignment results and partition
    metadata file to the provided output directory.


    Parameters
    ----------
    metadata: dict
        DGL "Chunked graph data" JSON, as defined in
        https://docs.dgl.ai/guide/distributed-preprocessing.html#specification
    """
    def _assign_partitions(self, num_partitions: int, partition_dir: str):
        num_nodes_per_type = self.metadata_dict["num_nodes_per_type"]
        ntypes = self.metadata_dict["node_type"]
        # Note: This assumes that the order of node_type is the same as the order num_nodes_per_type
        assert num_partitions > 0, "Number of partitions must be greater than 0"

        for ntype, num_nodes_for_type in zip(ntypes, num_nodes_per_type):
            logging.info("Generating random partition for node type %s", ntype)
            ntype_output = os.path.join(partition_dir, f"{ntype}.txt")

            assert num_partitions < 2**16, \
                f"Number of partitions must be smaller than 65536, got {num_partitions}"
            partition_dtype = np.uint8 if num_partitions < 256 else np.uint16

            if num_nodes_for_type < num_partitions:
                # If we have fewer nodes than partitions, raise an error
                raise RuntimeError(
                    f"Number of nodes for node type {ntype} ({num_nodes_for_type}) "
                    f"is less than the number of partitions ({num_partitions})"
                    f"To fix the error, reduce the number of partitions or drop the '{ntype}' nodes"
                )

            # Ensure at least one node per partition
            partition_assignment = np.empty(num_nodes_for_type, dtype=partition_dtype)
            # Assign first num_partitions nodes to each one of the partitions
            all_parts = np.arange(num_partitions)
            np.random.shuffle(all_parts)
            partition_assignment[:num_partitions] = all_parts

            # Then randomly assign the remaining nodes
            partition_assignment[num_partitions:] = np.random.randint(
                0,
                num_partitions,
                (num_nodes_for_type - num_partitions,),
                dtype=partition_dtype
            )

            arrow_partitions = pa.Table.from_arrays(
                [pa.array(partition_assignment)],
                names=["partition_id"])
            options = pa_csv.WriteOptions(include_header=False, delimiter=' ')
            pa_csv.write_csv(arrow_partitions, ntype_output, write_options=options)

    def _create_metadata(self, num_partitions: int, partition_dir: str):
        partition_meta = {
            "algo_name": "random",
            "num_parts": num_partitions,
            "version": "1.0.0"
        }
        partition_meta_filepath = os.path.join(partition_dir, "partition_meta.json")
        with open(partition_meta_filepath, "w", encoding='utf-8') as metafile:
            json.dump(partition_meta, metafile)
