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

import math
from typing import Any
from collections.abc import Mapping

from graphstorm_processing.constants import SUPPORTED_FILE_TYPES, VALID_OUTDTYPE

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
                if "custom_split_filenames" not in label:
                    if "split_pct" in label:
                        label_splitrate = label["split_pct"]
                        # check if split_pct is valid
                        assert (
                            math.fsum(label_splitrate) == 1.0
                        ), "sum of the label split rate should be ==1.0"
                        label_dict["split_rate"] = {
                            "train": label_splitrate[0],
                            "val": label_splitrate[1],
                            "test": label_splitrate[2],
                        }
                else:
                    label_custom_split_filenames = label["custom_split_filenames"]
                    if isinstance(label_custom_split_filenames["column"], list):
                        assert len(label_custom_split_filenames["column"]) <= 2, (
                            "Custom split filenames should have one column for node labels, "
                            "and two columns for edges labels exactly"
                        )
                    label_dict["custom_split_filenames"] = {
                        "train": label_custom_split_filenames["train"],
                        "valid": label_custom_split_filenames["valid"],
                        "test": label_custom_split_filenames["test"],
                        "column": label_custom_split_filenames["column"],
                    }
                if "separator" in label:
                    label_sep = label["separator"]
                    label_dict["separator"] = label_sep
                # Not supported for multi-task config for GSProcessing
                assert "mask_field_names" not in label, (
                    "GSProcessing currently cannot " "construct labels for multi-task learning"
                )
                labels_list.append(label_dict)
            except KeyError as exc:
                raise KeyError(f"A required key was missing from label input {label}") from exc
        return labels_list

    @staticmethod
    def _convert_feature(feats: list[Mapping[str, Any]]) -> list[dict]:
        """Convert the feature config
        Parameters
        ----------
        feats: list[Mapping[str, Any]]
            The feature information in the GConstruct format

        Returns
        -------
        list[dict]
            The feature information in the GSProcessing format
        """
        gsp_feats_list = []
        if feats in [[], [{}]]:
            return []
        for gconstruct_feat_dict in feats:
            gsp_feat_dict = {}
            if isinstance(gconstruct_feat_dict["feature_col"], str):
                gsp_feat_dict["column"] = gconstruct_feat_dict["feature_col"]
            elif isinstance(gconstruct_feat_dict["feature_col"], list):
                gsp_feat_dict["column"] = gconstruct_feat_dict["feature_col"][0]
                if len(gconstruct_feat_dict["feature_col"]) >= 2:
                    assert "feature_name" in gconstruct_feat_dict, (
                        "feature_name should be in the gconstruct "
                        "feature field when feature_col is a list"
                    )
            if "feature_name" in gconstruct_feat_dict:
                gsp_feat_dict["name"] = gconstruct_feat_dict["feature_name"]

            gsp_transformation_dict: dict[str, Any] = {}
            if "transform" in gconstruct_feat_dict:
                gconstruct_transform_dict = gconstruct_feat_dict["transform"]

                if gconstruct_transform_dict["name"] == "max_min_norm":
                    gsp_transformation_dict["name"] = "numerical"
                    gsp_transformation_dict["kwargs"] = {
                        "normalizer": "min-max",
                        "imputer": "none",
                    }

                    if gconstruct_transform_dict.get("out_dtype") in VALID_OUTDTYPE:
                        gsp_transformation_dict["kwargs"]["out_dtype"] = gconstruct_transform_dict[
                            "out_dtype"
                        ]
                elif gconstruct_transform_dict["name"] == "bucket_numerical":
                    gsp_transformation_dict["name"] = "bucket-numerical"
                    assert (
                        "bucket_cnt" in gconstruct_transform_dict
                    ), "bucket_cnt should be in the gconstruct bucket feature transform field"
                    assert (
                        "range" in gconstruct_transform_dict
                    ), "range should be in the gconstruct bucket feature transform field"
                    gsp_transformation_dict["kwargs"] = {
                        "bucket_cnt": gconstruct_transform_dict["bucket_cnt"],
                        "range": gconstruct_transform_dict["range"],
                        "slide_window_size": gconstruct_transform_dict["slide_window_size"],
                        "imputer": "none",
                    }
                elif gconstruct_transform_dict["name"] == "rank_gauss":
                    gsp_transformation_dict["name"] = "numerical"
                    gsp_transformation_dict["kwargs"] = {
                        "normalizer": "rank-gauss",
                        "imputer": "none",
                    }

                    if "epsilon" in gconstruct_transform_dict:
                        gsp_transformation_dict["kwargs"]["epsilon"] = gconstruct_transform_dict[
                            "epsilon"
                        ]
                    if gconstruct_transform_dict.get("out_dtype") in ["float32", "float64"]:
                        gsp_transformation_dict["kwargs"]["out_dtype"] = gconstruct_transform_dict[
                            "out_dtype"
                        ]
                elif gconstruct_transform_dict["name"] == "to_categorical":
                    if "separator" in gconstruct_transform_dict:
                        gsp_transformation_dict["name"] = "multi-categorical"
                        gsp_transformation_dict["kwargs"] = {
                            "separator": gconstruct_transform_dict["separator"]
                        }
                    else:
                        gsp_transformation_dict["name"] = "categorical"
                        gsp_transformation_dict["kwargs"] = {}
                elif gconstruct_transform_dict["name"] == "tokenize_hf":
                    gsp_transformation_dict["name"] = "huggingface"
                    gsp_transformation_dict["kwargs"] = {
                        "action": "tokenize_hf",
                        "hf_model": gconstruct_transform_dict["bert_model"],
                        "max_seq_length": gconstruct_transform_dict["max_seq_length"],
                    }
                elif gconstruct_transform_dict["name"] == "bert_hf":
                    gsp_transformation_dict["name"] = "huggingface"
                    gsp_transformation_dict["kwargs"] = {
                        "action": "embedding_hf",
                        "hf_model": gconstruct_transform_dict["bert_model"],
                        "max_seq_length": gconstruct_transform_dict["max_seq_length"],
                    }
                # TODO: Add support for other common transformations here
                else:
                    raise ValueError(
                        "Unsupported GConstruct transformation name: "
                        f"{gconstruct_transform_dict['name']}"
                    )
            else:
                gsp_transformation_dict["name"] = "no-op"

            if "out_dtype" in gconstruct_feat_dict:
                assert (
                    gconstruct_feat_dict["out_dtype"] in VALID_OUTDTYPE
                ), "GSProcessing currently only supports float32 or float64 features"

            gsp_feat_dict["transformation"] = gsp_transformation_dict
            gsp_feats_list.append(gsp_feat_dict)

        return gsp_feats_list

    @staticmethod
    def convert_nodes(nodes_entries):
        res = []
        for n in nodes_entries:
            # type, column id
            node_type, node_col = n["node_type"], n["node_id_col"]
            # format
            node_format = n["format"]["name"]
            assert (
                node_format in SUPPORTED_FILE_TYPES
            ), "GSProcessing only supports parquet files and csv files."
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
            assert (
                edge_format in SUPPORTED_FILE_TYPES
            ), "GSProcessing only supports parquet files and csv files."
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
