"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import abc
from abc import abstractmethod
from typing import Any

from .meta_configuration import NodeConfig, EdgeConfig


class ConfigConverter(abc.ABC):
    """Base class for configuration converters.

    We use these converters to convert input configuration files
    into a format that is compatible with GConstruct.

    This allows us to decouple GConstruct from specific
    configuration formats that can be used as input.
    """

    @staticmethod
    @abstractmethod
    def convert_nodes(nodes_entries: list[dict]) -> list[NodeConfig]:
        """Convert a list of node attributes into a list of `NodeConfig` objects.

        Parameters
        ----------
        nodes_entries : list[dict]
            List of node entry dictionaries.

        Returns
        -------
        list[NodeConfig]
            List of node entry configuration objects.
        """

    @staticmethod
    @abstractmethod
    def convert_edges(edges_entries: list[dict]) -> list[EdgeConfig]:
        """Convert a list of edge attributes into a list of `EdgeConfig` objects.

        Parameters
        ----------
        edges_entries : list[dict]
            List of edges entry dictionaries.

        Returns
        -------
        list[EdgeConfig]
            List of edges entry configuration objects.
        """

    def convert_to_gconstruct(self, input_dictionary: dict) -> dict:
        """Take a graph configuration input dictionary and convert it to a GConstruct-compatible
        dictionary.

        Parameters
        ----------
        input_dictionary : dict
            Input graph configuration dictionary. Needs to provide two top-level
            lists-of-dicts for the keys 'nodes' and 'edges'.

        Returns
        -------
        dict
            A graph description dictionary compatible with GSProcessing
        """
        # deal with corner case
        if input_dictionary == {}:
            return {"version": "gconstruct-v0.1", "nodes": [], "edges": []}
        nodes_entries: list[dict] = input_dictionary["nodes"]
        edges_entries: list[dict] = input_dictionary["edges"]

        node_configs: list[NodeConfig] = self.convert_nodes(nodes_entries)
        edge_configs: list[EdgeConfig] = self.convert_edges(edges_entries)

        gconstruct_dict: dict[str, Any] = {}

        gconstruct_dict["version"] = "gconstruct-v0.1"

        # deal with nodes
        gconstruct_dict["nodes"] = []
        for node_conf in node_configs:
            tmp_node: dict[str, Any] = {}
            # file attribute
            tmp_node["format"] = {"name": node_conf.file_format}
            tmp_node["files"] = node_conf.files

            # node attribute
            tmp_node["node_type"] = node_conf.node_type
            tmp_node["node_id_col"] = node_conf.column

            # features
            if node_conf.features is not None:
                tmp_node["features"] = node_conf.features

            # labels
            if node_conf.labels is not None:
                tmp_node["labels"] = node_conf.labels

            gconstruct_dict["nodes"].append(tmp_node)

        # deal with edges
        gconstruct_dict["edges"] = []
        for edge_conf in edge_configs:
            tmp_edge: dict[str, Any] = {}
            # file attribute
            tmp_edge["format"] = {"name": edge_conf.file_format}
            tmp_edge["files"] = edge_conf.files

            # edge attribute
            tmp_edge["source"] = {}
            tmp_edge["source_id_col"] = edge_conf.source_col
            tmp_edge["dest_id_col"] = edge_conf.dest_col
            tmp_edge["relation"] = [
                edge_conf.source_type,
                edge_conf.relation,
                edge_conf.dest_type,
            ]

            # features
            if edge_conf.features is not None:
                tmp_edge["features"] = edge_conf.features

            # labels
            if edge_conf.labels is not None:
                tmp_edge["labels"] = edge_conf.labels

            gconstruct_dict["edges"].append(tmp_edge)

        return gconstruct_dict
