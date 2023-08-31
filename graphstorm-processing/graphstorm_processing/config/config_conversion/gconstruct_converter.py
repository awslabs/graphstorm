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
from .converter_base import ConfigConverter
from .meta_configuration import NodeConfig, EdgeConfig


class GConstructConfigConverter(ConfigConverter):
    """The Config Converter from GConstruct to GSProcessing

    Parameters
    ----------
    data: valid GConstruct Config including nodes and edge config
    ----------
    """

    @staticmethod
    def _convert_label(labels: list[dict]) -> list[dict]:
        """Convert labels config

        Parameters
        ----------
        labels: list[dict]
            The label information in the GConstruct format

        Returns
        -------
        list[dict]
            The label information in the GSProcessing format
        """
        labels_list = []
        if labels in [[], [{}]]:
            return []
        for label in labels:
            try:
                label_column = label["label_col"] if "label_col" in label else ""
                label_type = label["task_type"]
                label_dict = {"column": label_column, "type": label_type}
                if "split_pct" in label:
                    label_splitrate = label["split_pct"]
                    # check if split_pct is valid
                    assert (
                        sum(label_splitrate) <= 1.0
                    ), "sum of the label split rate should be <=1.0"
                    label_dict["split_rate"] = {
                        "train": label_splitrate[0],
                        "val": label_splitrate[1],
                        "test": label_splitrate[2],
                    }
                if "separator" in label:
                    label_sep = label["separator"]
                    label_dict["separator"] = label_sep
                labels_list.append(label_dict)
            except KeyError as exc:
                raise KeyError(f"A required key was missing from label input {label}") from exc
        return labels_list

    @staticmethod
    def _convert_feature(feats: list[dict]) -> list[dict]:
        """Convert the feature config
        Parameters
        ----------
        feats: list[dict]
            The feature information in the GConstruct format

        Returns
        -------
        list[dict]
            The feature information in the GSProcessing format
        """
        feats_list = []
        if feats in [[], [{}]]:
            return []
        for ele in feats:
            if "transform" in ele:
                raise ValueError(
                    "Currently only support no-op operation, "
                    "we do not support any other no-op operation"
                )
            feat_dict = {}
            kwargs = {"name": "no-op"}
            for col in ele["feature_col"]:
                feat_dict = {"column": col, "transform": kwargs}
                feats_list.append(feat_dict)
            if "out_dtype" in ele:
                assert (
                    ele["out_dtype"] == "float32"
                ), "GSProcessing currently only supports float32 features"
            if "feature_name" in ele:
                feat_dict["name"] = ele["feature_name"]
        return feats_list

    @staticmethod
    def convert_nodes(nodes_entries):
        res = []
        for n in nodes_entries:
            # type, column id
            node_type, node_col = n["node_type"], n["node_id_col"]
            # format
            node_format = n["format"]["name"]
            if "separator" not in n["format"]:
                node_separator = None
            else:
                node_separator = n["format"]["separator"]

            # files
            node_files = n["files"] if isinstance(n["files"], list) else [n["files"]]
            for file_name in node_files:
                if "*" in file_name or "?" in file_name:
                    raise ValueError(
                        f"We do not currently support wildcards in node file names got: {file_name}"
                    )

            # features
            if "features" not in n:
                features = None
            else:
                features = GConstructConfigConverter._convert_feature(n["features"])

            # labels
            if "labels" not in n:
                labels = None
            else:
                labels = GConstructConfigConverter._convert_label(n["labels"])

            cur_node_config = NodeConfig(
                node_type, node_format, node_files, node_col, node_separator, features, labels
            )
            res.append(cur_node_config)
        return res

    @staticmethod
    def convert_edges(edges_entries):
        res = []
        for e in edges_entries:
            # column name
            source_col, dest_col = e["source_id_col"], e["dest_id_col"]

            # relation
            source_type, relation, dest_type = e["relation"][0], e["relation"][1], e["relation"][2]

            # files
            edge_files = e["files"] if isinstance(e["files"], list) else [e["files"]]
            for file_name in edge_files:
                if "*" in file_name or "?" in file_name:
                    raise ValueError("Not Support for wildcard in edge file name")

            # format
            edge_format = e["format"]["name"]
            if "separator" not in e["format"]:
                edge_separator = None
            else:
                edge_separator = e["format"]["separator"]

            # features
            if "features" not in e:
                features = None
            else:
                features = GConstructConfigConverter._convert_feature(e["features"])

            # labels
            if "labels" not in e:
                labels = None
            else:
                labels = GConstructConfigConverter._convert_label(e["labels"])

            cur_edge_config = EdgeConfig(
                source_col,
                source_type,
                dest_col,
                dest_type,
                edge_format,
                edge_files,
                relation,
                edge_separator,
                features,
                labels,
            )

            res.append(cur_edge_config)
        return res
