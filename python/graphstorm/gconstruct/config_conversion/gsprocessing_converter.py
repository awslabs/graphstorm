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

from .converter_base import ConfigConverter
from .meta_configuration import NodeConfig, EdgeConfig


class GSProcessingConfigConverter(ConfigConverter):
    """The Config Converter from GSProcessing to GConstruct

    Parameters
    ----------
    data: valid GSProcessing Config including nodes and edge config
    ----------
    """

    @staticmethod
    def _convert_label(labels: list[dict]) -> list[dict]:
        """Convert labels config

        Parameters
        ----------
        labels: list[dict]
            The label information in the GSProcessing format

        Returns
        -------
        list[dict]
            The label information in the GConstruct format
        """
        labels_list = []
        if labels in [[], [{}]]:
            return []
        for label in labels:  # pylint: disable=too-many-nested-blocks
            try:
                label_column = label["column"]
                label_type = label["type"]
                label_dict = {"label_col": label_column, "task_type": label_type}
                if "custom_split_filenames" not in label:
                    if "split_rate" in label:
                        label_splitrate = label["split_rate"]
                        # check if split_pct is valid
                        assert (
                            math.fsum(
                                [
                                    label_splitrate["train"],
                                    label_splitrate["val"],
                                    label_splitrate["test"],
                                ]
                            )
                            == 1.0
                        ), f"sum of the label split rate should be == 1.0, got {label_splitrate}"
                        label_dict["split_pct"] = [
                            label_splitrate["train"],
                            label_splitrate["val"],
                            label_splitrate["test"],
                        ]
                else:
                    label_custom_split_filenames = label["custom_split_filenames"]
                    # Ensure at least one of ["train", "valid", "test"] is in the keys
                    assert any(
                        x in label_custom_split_filenames.keys()
                        for x in ["train", "valid", "test"]
                    ), (
                        "At least one of ['train', 'valid', 'test'] "
                        "needs to exist in custom split configs."
                    )

                    # Fill in missing values if needed
                    for entry in ["train", "valid", "test", "column"]:
                        entry_val = label_custom_split_filenames.get(entry, None)
                        if entry_val:
                            if isinstance(entry_val, str):
                                label_custom_split_filenames[entry] = [entry_val]
                            else:
                                assert isinstance(
                                    entry_val, list
                                ), "Custom split filenames should be a string or a list of strings"
                        else:
                            label_custom_split_filenames[entry] = []
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
                if "mask_field_names" in label:
                    label_dict["mask_field_names"] = label["mask_field_names"]

                labels_list.append(label_dict)
            except KeyError as exc:
                raise KeyError(
                    f"A required key was missing from label input {label}"
                ) from exc
        return labels_list

    @staticmethod
    def _convert_feature(feats: list[Mapping[str, Any]]) -> list[dict]:
        """Convert the feature config
        Parameters
        ----------
        feats: list[Mapping[str, Any]]
            The feature information in the GSProcessing format

        Returns
        -------
        list[dict]
            The feature information in the GConstruct format
        """
        gconstruct_feats_list = []
        if feats in [[], [{}]]:
            return []
        for gsp_feat_dict in feats:
            gconstruct_feat_dict = {}
            gconstruct_feat_dict["feature_col"] = gsp_feat_dict["column"]
            if "name" in gsp_feat_dict:
                gconstruct_feat_dict["feature_name"] = gsp_feat_dict["name"]

            gconstruct_transformation_dict: dict[str, Any] = {}
            if "transformation" in gsp_feat_dict:
                gsp_transformation_dict = gsp_feat_dict["transformation"]

                ### START TRANSFORMATION ALTERNATIVES ###
                if gsp_transformation_dict["name"] == "numerical":
                    if gsp_transformation_dict["kwargs"]["normalizer"] == "min-max":
                        gconstruct_transformation_dict["name"] = "max_min_norm"
                    elif gsp_transformation_dict["kwargs"]["normalizer"] == "standard":
                        gconstruct_transformation_dict["name"] = "standard"
                    elif (
                        gsp_transformation_dict["kwargs"]["normalizer"] == "rank-gauss"
                    ):
                        gconstruct_transformation_dict["name"] = "rank_gauss"
                        if gsp_transformation_dict.get("kwargs").get("epsilon"):
                            gconstruct_transformation_dict["epsilon"] = (
                                gsp_transformation_dict["kwargs"]["epsilon"]
                            )
                    else:
                        raise ValueError(
                            f"Unexpected numerical transformation "
                            f"{gsp_transformation_dict['kwargs']['normalizer']}"
                        )
                    if "out_dtype" in gsp_transformation_dict["kwargs"]:
                        gconstruct_transformation_dict["out_dtype"] = (
                            gsp_transformation_dict["kwargs"]["out_dtype"]
                        )
                elif gsp_transformation_dict["name"] == "bucket_numerical":
                    assert (
                        "bucket_cnt" in gsp_transformation_dict["kwargs"]
                    ), "bucket_cnt should be in the gconstruct bucket feature transform field"
                    assert (
                        "range" in gsp_transformation_dict["kwargs"]
                    ), "range should be in the gconstruct bucket feature transform field"
                    gconstruct_transformation_dict = {
                        "name": "bucket_numerical",
                        "bucket_cnt": gsp_transformation_dict["kwargs"]["bucket_cnt"],
                        "range": gsp_transformation_dict["kwargs"]["range"],
                        "slide_window_size": gsp_transformation_dict["kwargs"][
                            "slide_window_size"
                        ],
                    }
                elif gsp_transformation_dict["name"] == "categorical":
                    gconstruct_transformation_dict["name"] = "to_categorical"
                elif gsp_transformation_dict["name"] == "multi-categorical":
                    gconstruct_transformation_dict["name"] = "to_categorical"
                    gconstruct_transformation_dict["separator"] = (
                        gsp_transformation_dict["kwargs"]["separator"]
                    )
                elif gsp_transformation_dict["name"] == "huggingface":
                    if gsp_transformation_dict["kwargs"]["action"] == "embedding_hf":
                        action_name = "bert_hf"
                    elif gsp_transformation_dict["kwargs"]["action"] == "tokenize_hf":
                        action_name = "tokenize_hf"
                    else:
                        raise ValueError("Unexpected huggingface action {}"
                                         .format(gsp_transformation_dict["kwargs"]["action"]))
                    gconstruct_transformation_dict = {
                        "name": action_name,
                        "bert_model": gsp_transformation_dict["kwargs"]["hf_model"],
                        "max_seq_length": gsp_transformation_dict["kwargs"][
                            "max_seq_length"
                        ],
                    }
                elif gsp_transformation_dict["name"] == "edge_dst_hard_negative":
                    gconstruct_transformation_dict["name"] = "edge_dst_hard_negative"
                    if "separator" in gsp_transformation_dict["kwargs"]:
                        gconstruct_transformation_dict["separator"] = (
                            gsp_transformation_dict["kwargs"]["separator"]
                        )
                elif gsp_transformation_dict["name"] == "no-op":
                    gconstruct_transformation_dict["name"] = "no-op"
                    kwargs = gsp_transformation_dict.get("kwargs", {})

                    if "truncate_dim" in kwargs:
                        gconstruct_transformation_dict["truncate_dim"] = kwargs["truncate_dim"]

                    if "separator" in kwargs:
                        gconstruct_transformation_dict["separator"] = kwargs["separator"]
                else:
                    raise ValueError(
                        "Unsupported GSProcessing transformation name: "
                        f"{gsp_transformation_dict['name']}"
                    )
                ### END TRANSFORMATION ALTERNATIVES ###
            else:
                gconstruct_transformation_dict["name"] = "no-op"

            gconstruct_feat_dict["transform"] = gconstruct_transformation_dict
            gconstruct_feats_list.append(gconstruct_feat_dict)

        return gconstruct_feats_list

    @staticmethod
    def convert_nodes(nodes_entries) -> list[NodeConfig]:
        res = []
        for n in nodes_entries:
            # type, column id
            node_type, node_col = n["type"], n["column"]
            # format
            # Assume the format is already valid in GSProcessing
            node_format = n["data"]["format"]
            if "separator" not in n["data"]:
                node_separator = None
            else:
                node_separator = n["data"]["separator"]

            # files
            node_files = (
                n["data"]["files"]
                if isinstance(n["data"]["files"], list)
                else [n["data"]["files"]]
            )

            # features
            if "features" not in n:
                features = None
            else:
                features = GSProcessingConfigConverter._convert_feature(n["features"])

            # labels
            if "labels" not in n:
                labels = None
            else:
                labels = GSProcessingConfigConverter._convert_label(n["labels"])

            cur_node_config = NodeConfig(
                node_type,
                node_format,
                node_files,
                node_col,
                node_separator,
                features,
                labels,
            )
            res.append(cur_node_config)
        return res

    @staticmethod
    def convert_edges(edges_entries):
        res = []
        for e in edges_entries:
            # column name
            source_col, dest_col = e["source"]["column"], e["dest"]["column"]

            # relation
            source_type, relation, dest_type = (
                e["source"]["type"],
                e["relation"]["type"],
                e["dest"]["type"],
            )

            # files
            edge_files = (
                e["data"]["files"]
                if isinstance(e["data"]["files"], list)
                else [e["data"]["files"]]
            )

            # format
            edge_format = e["data"]["format"]
            if "separator" not in e["data"]:
                edge_separator = None
            else:
                edge_separator = e["data"]["separator"]

            # features
            if "features" not in e:
                features = None
            else:
                features = GSProcessingConfigConverter._convert_feature(e["features"])

            # labels
            if "labels" not in e:
                labels = None
            else:
                labels = GSProcessingConfigConverter._convert_label(e["labels"])

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
