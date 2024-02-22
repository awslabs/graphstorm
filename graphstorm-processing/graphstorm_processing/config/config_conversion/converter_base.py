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
    into a format that is compatible with GSProcessing.

    This allows us to decouple GSProcessing from specific
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

    def convert_to_gsprocessing(self, input_dictionary: dict) -> dict:
        """Take a graph configuration input dictionary and convert it to a GSProcessing-compatible
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
            return {"version": "gsprocessing-v1.0", "graph": {"nodes": [], "edges": []}}

        nodes_entries: list[dict] = input_dictionary["nodes"]
        edges_entries: list[dict] = input_dictionary["edges"]

        node_configs: list[NodeConfig] = self.convert_nodes(nodes_entries)
        edge_configs: list[EdgeConfig] = self.convert_edges(edges_entries)

        gsprocessing_dict: dict[str, Any] = {}

        gsprocessing_dict["version"] = "gsprocessing-v1.0"
        gsprocessing_dict["graph"] = {}

        # deal with nodes
        gsprocessing_dict["graph"]["nodes"] = []
        for node_conf in node_configs:
            tmp_node: dict[str, Any] = {}
            # data
            tmp_node["data"] = {}
            tmp_node["data"]["format"] = node_conf.file_format
            tmp_node["data"]["files"] = node_conf.files

            # separator
            if node_conf.separator is not None:
                tmp_node["data"]["separator"] = node_conf.separator

            # node type
            tmp_node["type"] = node_conf.node_type

            # column
            tmp_node["column"] = node_conf.column

            # features
            if node_conf.features is not None:
                tmp_node["features"] = node_conf.features

            # labels
            if node_conf.labels is not None:
                tmp_node["labels"] = node_conf.labels

            gsprocessing_dict["graph"]["nodes"].append(tmp_node)

        # deal with edges
        gsprocessing_dict["graph"]["edges"] = []
        for edge_conf in edge_configs:
            tmp_edge: dict[str, Any] = {}
            # data
            tmp_edge["data"] = {}
            tmp_edge["data"]["format"] = edge_conf.file_format
            tmp_edge["data"]["files"] = edge_conf.files

            # separator
            if edge_conf.separator is not None:
                tmp_edge["data"]["separator"] = edge_conf.separator

            # source
            tmp_edge["source"] = {}
            tmp_edge["source"]["column"], tmp_edge["source"]["type"] = (
                edge_conf.source_col,
                edge_conf.source_type,
            )

            # dest
            tmp_edge["dest"] = {}
            tmp_edge["dest"]["column"], tmp_edge["dest"]["type"] = (
                edge_conf.dest_col,
                edge_conf.dest_type,
            )

            # edge relation
            tmp_edge["relation"] = {"type": edge_conf.relation}

            # features
            if edge_conf.features is not None:
                tmp_edge["features"] = edge_conf.features

            # labels
            if edge_conf.labels is not None:
                tmp_edge["labels"] = edge_conf.labels

            gsprocessing_dict["graph"]["edges"].append(tmp_edge)

        return gsprocessing_dict
