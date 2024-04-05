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

    Single-instance metis partition assignment
"""
import os
import logging
import json

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv

from .partition_algo_base import LocalPartitionAlgorithm


class MetisPartitionAlgorithm(LocalPartitionAlgorithm):
    """
    Multiple-instance metis partitioning algorithm.

    The partition algorithm accepts the intermediate output from GraphStorm
    gs-processing which matches the requirements of the DGL distributed
    partitioning pipeline.


    Parameters
    ----------
    metadata: dict
        DGL "Chunked graph data" JSON, as defined in
        https://docs.dgl.ai/guide/distributed-preprocessing.html#specification
    """

    def _launch_preprocess(self, num_parts, input_data_path, metadata_filename, output_path):
        return "Not Implmentation error"

    def _launch_parmetis(
            self,
            num_parts,
            net_ifname,
            input_data_path,
            metis_input_path):
        """ Launch parmetis script

        Parameters
        ----------
        num_parts: int
            Number of graph partitions
        net_ifname: str
            Network interface used by MPI
        input_data_path: str
            Path to the input graph data
        metis_input_path: str
            Path to metis input
        """
        return "Not Implmentation error"

    def _launch_postprocess(self, meta_data_config, parmetis_output_file, partitions_dir):
        """ Launch postprocess which translates nid-partid mapping into
            Per-node-type partid mappings.

        Parameters
        ----------
        meta_data_config: str
            Path to the meta data configuration.
        parmetis_output_file: str
            Path to ParMetis output.
        partitions_dir: str
            Output path
        """
        return "Not Implmentation error"