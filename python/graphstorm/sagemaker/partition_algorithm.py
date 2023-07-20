"""
    Copyright Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    A module to gather partitioning algorithm implementations to be executed in SageMaker.
"""
from typing import List, Tuple
import abc
from dataclasses import dataclass
import json
import logging
import os
import socket

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
from sagemaker import Session

from .s3_utils import upload_file_to_s3

DGL_TOOL_PATH = "/root/dgl/tools"

@dataclass()
class PartitionerConfig:
    """
    Dataclass for holding the configuration for a partitioning algorithm.

    Parameters
    ----------
    metadata_file : str
        Path to the metadata file describing the graph.
    local_output_path : str
        Path to the local directory where the partitioning results should be stored.
    rank : int
        Rank of the current worker process.
    sagemaker_session : sagemaker.Session
        The SageMaker session used for training.
    """
    metadata_file: str
    local_output_path: str
    rank: int
    sagemaker_session: Session


class PartitionAlgorithm(abc.ABC):
    """
    Base class for partition algorithm implementations.

    Parameters
    ----------
    partition_config : PartitionConfig
        The configuration for the partition algorithm.
        See `PartitionConfig` for detailed argument list.
    """
    def __init__(self,
        partition_config: PartitionerConfig
    ):
        self.metadata_file = partition_config.metadata_file
        self.local_output_path = partition_config.local_output_path
        self.rank = partition_config.rank
        self.sagemaker_session = partition_config.sagemaker_session

        with open(self.metadata_file, 'r', encoding='utf-8') as metafile:
            self.metadata = json.load(metafile)

        self.graph_name = self.metadata["graph_name"]

        os.makedirs(self.local_output_path, exist_ok=True)

    def create_partitions(self, output_s3_path: str, num_partitions: int) -> Tuple[str, str]:
        """
        Creates a partitioning and uploads the results to the provided S3 location.

        Expected

        Parameters
        ----------
        output_s3_path : str
            S3 prefix to upload the partitioning results to.
        num_partitions : int
            Number of partitions to create.

        Returns
        -------
        local_partition_path, s3_partition_path : Tuple[str, str]
            Paths to the local partitioning directory and S3 URI to the uploaded partition data.
        """
        local_partition_path = self._run_partitioning(num_partitions)

        # At this point the leader will need to have partition assignments and metadata
        # available locally
        if self.rank == 0:
            if not os.path.isfile(os.path.join(local_partition_path, "partition_meta.json")):
                raise RuntimeError("Expected partition_meta.json to be present in "
                    f"{local_partition_path} got contents: {os.listdir(local_partition_path)}")

        s3_partition_path = os.path.join(output_s3_path, "partition")
        self._upload_results_to_s3(local_partition_path, s3_partition_path)

        return local_partition_path, s3_partition_path

    def broadcast_partition_done(self, client_list, world_size, success=True):
        """ Notify each worker process the partition assignment process is done

        Parameters
        ----------
        client_list: list
            List of socket clients
        world_size: int
            Size of the distributed training/inference cluster
        success: bool
            True if preprocess success
        """
        if self.rank != 0:
            raise RuntimeError("broadcast_partition_done should only be called by the leader")
        msg = b"PartitionDone" if success else b"PartitionFail"
        for rank in range(1, world_size):
            client_list[rank].sendall(msg)

    def wait_for_partition_done(self, master_sock):
        """ Waiting for partition to be done

        Parameters
        ----------
        master_sock: socket
            Socket connecting master
        """
        if self.rank == 0:
            raise RuntimeError("wait_for_partition_done should only be called by a worker")
        msg = master_sock.recv(13, socket.MSG_WAITALL)
        msg = msg.decode()
        if msg != "PartitionDone":
            raise RuntimeError(f"Wait for partition Error detected, msg: {msg}")

    @abc.abstractmethod
    def _run_partitioning(self, num_partitions: int) -> str:
        """
        Runs the partitioning algorithm.

        Side-effect contract: At the end of this call the partition assignment files, as defined in
        https://docs.dgl.ai/guide/distributed-preprocessing.html#step-1-graph-partitioning
        and a partitioning metadata JSON file, as defined in
        https://github.com/dmlc/dgl/blob/29e666152390c272e0115ce8455da1adb5fcacb1/tools/partition_algo/base.py#L8
        should exist on the leader instance (rank 0), under the returned partition_dir.

        Parameters
        ----------
        num_partitions : int
            Number of partition assignments to create.

        Returns
        -------
        partition_dir : str
            Path to the partitioning directory.
            On the leader this must contain the partition assignment data
            and a partition_meta.json file.
        """


    # TODO: Because the locations are entangled is it better if we don't take arguments here?
    @abc.abstractmethod
    def _upload_results_to_s3(self, local_partition_directory: str, output_s3_path: str) -> None:
        """
        Uploads the partitioning results to S3 once they become available on the local filesystem.

        Parameters
        ----------
        local_partition_directory : str
            Path to the partitioning directory.
        output_s3_path : str
            S3 prefix to upload the partitioning results to.
        """

class RandomPartitioner(PartitionAlgorithm): # pylint: disable=too-few-public-methods
    """
    Single-instance random partitioning algorithm.
    """
    def _run_partitioning(self, num_partitions: int) -> str:
        partition_dir = os.path.join(self.local_output_path, "partition")
        os.makedirs(partition_dir, exist_ok=True)

        # Random partitioning is done on the leader node only
        if self.rank != 0:
            return partition_dir

        num_nodes_per_type = self.metadata["num_nodes_per_type"]  # type: List[int]
        ntypes = self.metadata["node_type"]  # type: List[str]
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
    def _create_metadata(num_partitions: int, partition_dir: str) -> None:
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


    def _upload_results_to_s3(self, local_partition_directory: str, output_s3_path: str) -> None:
        if self.rank == 0:
            logging.debug(
                "Uploading partition files to %s, local_partition_directory: %s",
                output_s3_path,
                local_partition_directory)
            upload_file_to_s3(output_s3_path, local_partition_directory, self.sagemaker_session)
        else:
            # Workers do not hold any partitioning information locally
            pass
