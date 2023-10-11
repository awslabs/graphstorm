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

try:
    from sagemaker import Session
    from .s3_utils import upload_file_to_s3
except:
    print("Can not run with SageMaker")

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

try:
    @dataclass()
    class SageMakerPartitionerConfig(PartitionerConfig):
        """
        Dataclass for holding the configuration for a partitioning algorithm run on SageMaker.

        Parameters
        ----------
        sagemaker_session : sagemaker.Session
            The SageMaker session used for training.
        """
        sagemaker_session: Session
except:
    print("Can not run with SageMaker")

class Partitioner(abc.ABC):
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

        with open(self.metadata_file, 'r', encoding='utf-8') as metafile:
            self.metadata = json.load(metafile)

        self.graph_name = self.metadata["graph_name"]

        os.makedirs(self.local_output_path, exist_ok=True)

    def create_partitions(self, num_partitions: int, output_path: str = None) -> Tuple[str, str]:
        """
        Creates a partitioning

        Expected

        Parameters
        ----------
        output_path : str
            Path prefix to upload the partitioning results to. It can be local, S3 or others.
        num_partitions : int
            Number of partitions to create.

        Returns
        -------
        local_partition_path, partition_path : Tuple[str, str]
            Paths to the local partitioning directory and the
            partitioning directory to the output path.
        """
        local_partition_path = self._run_partitioning(num_partitions)

        # At this point the leader will need to have partition assignments and metadata
        # available locally
        if self.rank == 0:
            if not os.path.isfile(os.path.join(local_partition_path, "partition_meta.json")):
                raise RuntimeError("Expected partition_meta.json to be present in "
                    f"{local_partition_path} got contents: {os.listdir(local_partition_path)}")

        if output_path is not None:
            partition_path = os.path.join(output_path, "partition")
            self._copy_results_to_output(local_partition_path, partition_path)
        else:
            partition_path = local_partition_path

        return local_partition_path, partition_path

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

    @abc.abstractmethod
    def _copy_results_to_output(self, local_partition_directory: str, output_path: str) -> None:
        """
        Copy the partitioning results to the output path.

        Parameters
        ----------
        local_partition_directory : str
            Path to the partitioning directory.
        output_path : str
            The output path to copy the partitioning results to.
        """

class SageMakerPartitioner(Partitioner):
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
        self.sagemaker_session = partition_config.sagemaker_session
        super(SageMakerPartitioner, self).__init__(partition_config)

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

class RandomPartitionAlgorithm(): # pylint: disable=too-few-public-methods
    """
    Single-instance random partitioning algorithm.

    Parameters
    ----------
    local_output_path: str
        Local output path
    metadata: dict
        partitioning metadata JSON
    rank: int
        Rank of the current worker process.
    """
    def __init__(self, local_output_path, metadata, rank=0) -> None:
        self.local_output_path = local_output_path
        self.metadata = metadata
        self.rank = rank

    def run_partitioning(self, num_partitions: int) -> str:
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

class SageMakerRandomPartitioner(SageMakerPartitioner): # pylint: disable=too-few-public-methods
    """
    Single-instance random partitioning algorithm running on SageMaker
    """
    def _run_partitioning(self, num_partitions: int) -> str:
        random_part = RandomPartitionAlgorithm(self.local_output_path,
                                               self.metadata,
                                               self.rank)
        partition_dir = random_part.run_partitioning(num_partitions)
        return partition_dir

    def _copy_results_to_output(self, local_partition_directory: str, output_path: str) -> None:
        """ Copy data to S3

        Parameters
        ----------
        local_partition_directory: str
            Path to local partition files.
        output_path: str
            S3 path to upload the local partition files.
        """
        if self.rank == 0:
            logging.debug(
                "Uploading partition files to %s, local_partition_directory: %s",
                output_path,
                local_partition_directory)
            upload_file_to_s3(output_path, local_partition_directory, self.sagemaker_session)
        else:
            # Workers do not hold any partitioning information locally
            pass

class LocalRandomPartitioner(Partitioner): # pylint: disable=too-few-public-methods
    """
    Single-instance random partitioning algorithm running on a Linux instance.
    """
    def _run_partitioning(self, num_partitions: int) -> str:
        random_part = RandomPartitionAlgorithm(self.local_output_path,
                                               self.metadata,
                                               self.rank)
        partition_dir = random_part.run_partitioning(num_partitions)

        return partition_dir

    def _copy_results_to_output(self, local_partition_directory: str, output_path: str) -> None:
        # Do nothing as we assume the partitions are stored in a shared file system.
        pass
