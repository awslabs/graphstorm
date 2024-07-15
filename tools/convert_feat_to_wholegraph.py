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

    Tool to convert node features from distDGL format to WholeGraph format.
"""
import argparse

import json
import logging
import os

import torch

from graphstorm.wholegraph import convert_feat_to_wholegraph

def get_node_feat_info(node_feat_names):
    """Process node feature names

    Parameters
    ----------
    node_feat_names : str or dict of list of str
        The name of the node features. It's a dict if different node types have
        different feature names.

    Returns
    -------
        dict of list : dict of names of the node features of different node types
    """
    fname_dict = {}
    for feat_name in node_feat_names:
        feat_info = feat_name.split(":")
        assert len(feat_info) == 2, (
            f"Unknown format of the feature name: {feat_name}, " \
            "must be NODE_TYPE:FEAT_NAME"
        )
        ntype = feat_info[0]
        assert ntype not in fname_dict, (
            f"You already specify the feature names of {ntype} " \
            f"as {fname_dict[ntype]}"
        )
        # multiple features separated by ','
        fname_dict[ntype] = feat_info[1].split(",")

    return fname_dict


def get_edge_feat_info(edge_feat_names):
    """Process edge feature names

    Parameters
    ----------
    edge_feat_names : str or dict of list of str
        The name of the edge features. It's a dict if different edge types have
        different feature names.

    Returns
    -------
        dict of list : dict of names of the edge features of different edge types
    """

    fname_dict = {}
    for feat_name in edge_feat_names:
        feat_info = feat_name.split(":")
        assert len(feat_info) == 2, (
            f"Unknown format of the feature name: {feat_name}, " \
            "must be EDGE_TYPE:FEAT_NAME"
        )
        etype = feat_info[0].split(",")
        assert len(etype) == 3, (
            f"EDGE_TYPE should have 3 strings: {etype}, " \
            "must be NODE_TYPE,EDGE_TYPE,NODE_TYPE:"
        )
        etype = ":".join(etype)
        assert etype not in fname_dict, (
            f"You already specify the feature names of {etype} " \
            f"as {fname_dict[etype]}"
        )
        # multiple features separated by ','
        fname_dict[etype] = feat_info[1].split(",")
    return fname_dict

def main(folder, node_feat_names, edge_feat_names, use_low_mem=False):
    """Convert features from distDGL tensor format to WholeGraph format"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    torch.distributed.init_process_group("gloo", world_size=1, rank=0)
    wg_folder = os.path.join(folder, "wholegraph")
    if not os.path.exists(wg_folder):
        os.makedirs(wg_folder)

    logging.info(
        "node features: %s and edge features: %s will be converted to WholeGraph"
        "format.", node_feat_names, edge_feat_names
    )

    metadata = {}
    # Process node features
    if node_feat_names:
        fname_dict = get_node_feat_info(node_feat_names)
        convert_feat_to_wholegraph(
            fname_dict, "node_feat.dgl", metadata, folder, use_low_mem
        )

    # Process edge features
    if edge_feat_names:
        fname_dict = get_edge_feat_info(edge_feat_names)
        convert_feat_to_wholegraph(
            fname_dict, "edge_feat.dgl", metadata, folder, use_low_mem
        )

    # Save metatada
    with open(os.path.join(wg_folder, "metadata.json"), "w", encoding="utf8") as fp: # pylint: disable=invalid-name
        json.dump(metadata, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", action="store", type=str, required=True)
    parser.add_argument("--node-feat-names", nargs="+", action="store", type=str)
    parser.add_argument("--edge-feat-names", nargs="+", action="store", type=str)
    parser.add_argument(
        "--low-mem",
        action="store_true",
        help="Whether to use less memory consuming conversion method. Recommended to use "
        "this argument for very large dataset. Please Note, this method is slower than "
        "regular conversion method.",
    )
    args = parser.parse_args()
    main(args.dataset_path, args.node_feat_names, args.edge_feat_names, args.low_mem)
