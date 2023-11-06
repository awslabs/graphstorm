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
import gc
import json
import logging
import os
import re

import dgl
import pylibwholegraph.torch as wgth
import torch


def get_node_feat_info(node_feat_names):
    """Process node feature names"""
    fname_dict = {}
    for feat_name in node_feat_names:
        feat_info = feat_name.split(":")
        assert len(feat_info) == 2, (
            f"Unknown format of the feature name: {feat_name}, "
            + "must be NODE_TYPE:FEAT_NAME"
        )
        ntype = feat_info[0]
        assert ntype not in fname_dict, (
            f"You already specify the feature names of {ntype} "
            f"as {fname_dict[ntype]}"
        )
        assert isinstance(
            feat_info[1], str
        ), f"Feature name of {ntype} should be a string not {feat_info[1]}"
        # multiple features separated by ','
        fname_dict[ntype] = feat_info[1].split(",")

    return fname_dict


def get_edge_feat_info(edge_feat_names):
    """Process edge feature names"""
    fname_dict = {}
    for feat_name in edge_feat_names:
        feat_info = feat_name.split(":")
        assert len(feat_info) == 2, (
            f"Unknown format of the feature name: {feat_name}, "
            + "must be EDGE_TYPE:FEAT_NAME"
        )
        etype = feat_info[0].split(",")
        assert len(etype) == 3, (
            f"EDGE_TYPE should have 3 strings: {etype}, "
            + "must be NODE_TYPE,EDGE_TYPE,NODE_TYPE:"
        )
        etype = ":".join(etype)
        assert etype not in fname_dict, (
            f"You already specify the feature names of {etype} "
            f"as {fname_dict[etype]}"
        )
        assert isinstance(
            feat_info[1], str
        ), f"Feature name of {etype} should be a string not {feat_info[1]}"
        # multiple features separated by ','
        fname_dict[etype] = feat_info[1].split(",")
    return fname_dict


def wholegraph_processing(
    whole_feat_tensor, metadata, local_comm, feat, wg_folder, num_parts
):
    """Convert DGL tensors to wholememory tensor"""
    metadata[feat] = {
        "shape": list(whole_feat_tensor.shape),
        "dtype": str(whole_feat_tensor.dtype),
    }
    # Round up the integer division to match WholeGraph partitioning scheme
    subpart_size = -(whole_feat_tensor.shape[0] // -num_parts)

    for part_num in range(num_parts):
        st = part_num * subpart_size
        end = (
            (part_num + 1) * subpart_size
            if part_num != (num_parts - 1)
            else whole_feat_tensor.shape[0]
        )
        wg_tensor = wgth.create_wholememory_tensor(
            local_comm,
            "continuous",
            "cpu",
            (end - st, *whole_feat_tensor.shape[1:]),
            whole_feat_tensor.dtype,
            None,
        )
        local_tensor, _ = wg_tensor.get_local_tensor(host_view=True)
        local_tensor.copy_(whole_feat_tensor[st:end])
        filename = wgth.utils.get_part_file_name(
            feat.replace("/", "~"), part_num, num_parts
        )
        wg_tensor.local_to_file(os.path.join(wg_folder, filename))
        wgth.destroy_wholememory_tensor(wg_tensor)


def convert_feat_to_wholegraph(fname_dict, file_name, metadata, folder, use_low_mem):
    """Convert features from distDGL tensor format to WholeGraph format"""
    local_comm = wgth.comm.get_local_device_communicator()
    wg_folder = os.path.join(folder, "wholegraph")
    folder_pattern = re.compile(r"^part[0-9]+$")
    part_files = [
        f
        for f in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, f)) and folder_pattern.match(f)
    ]
    part_files = sorted(part_files, key=lambda x: int(x.split("part")[1]))
    feats_data = []

    # When 'use_low_mem' is not enabled, this code loads and appends different features from
    # individual partitions. Then each feature is converted into the WholeGraph format.
    if not use_low_mem:
        # Read features from file
        for path in (os.path.join(folder, name) for name in part_files):
            feats_data.append(dgl.data.utils.load_tensors(f"{path}/{file_name}"))
        num_parts = len(feats_data)
        for type_name, feats in fname_dict.items():
            for feat in feats:
                feat = type_name + "/" + feat
                if feat not in feats_data[0]:
                    raise RuntimeError(
                        f"Error: Unknown feature '{feat}'. Files contain \
                                       the following features: {feats_data[0].keys()}."
                    )
                logging.info(f"Processing '{feat}' features...")
                whole_feat_tensor = torch.concat(
                    tuple(t[feat] for t in feats_data), dim=0
                )
                # Delete processed feature from memory
                for t in feats_data:
                    del t[feat]
                wholegraph_processing(
                    whole_feat_tensor, metadata, local_comm, feat, wg_folder, num_parts
                )

    # This version is less memory-consuming. For each feature, it iterates through all the
    # partitions, loading the available features, but only stores (concatenating) only the
    # current feature to be converted to WholeGraph.
    else:  # low-mem
        for ntype, feats in fname_dict.items():
            for feat in feats:
                feat = ntype + "/" + feat
                node_feats_data = {feat: None}
                num_parts = 0
                # Read features from file
                for path in (os.path.join(folder, name) for name in part_files):
                    nfeat = dgl.data.utils.load_tensors(f"{path}/{file_name}")
                    if feat not in nfeat:
                        raise RuntimeError(
                            f"Error: Unknown feature '{feat}'. Files contain \
                                       the following features: {nfeat.keys()}."
                        )
                    if node_feats_data[feat] is None:
                        node_feats_data[feat] = nfeat[feat]
                    else:
                        node_feats_data[feat] = torch.concat(
                            (node_feats_data[feat], nfeat[feat]), dim=0
                        )
                    num_parts += 1
                del nfeat
                gc.collect()
                wholegraph_processing(
                    node_feats_data[feat],
                    metadata,
                    local_comm,
                    feat,
                    wg_folder,
                    num_parts,
                )

        for path in (os.path.join(folder, name) for name in part_files):
            feats_data.append(dgl.data.utils.load_tensors(f"{path}/{file_name}"))

        for type_name, feats in fname_dict.items():
            for feat in feats:
                feat = type_name + "/" + feat
                # Delete processed feature from memory
                for t in feats_data:
                    del t[feat]

    return feats_data


def trim_feat_files(trimmed_feats, folder, file_name):
    """Save new truncated distDGL tensors"""
    num_parts = len(trimmed_feats)
    for part in range(num_parts):
        dgl.data.utils.save_tensors(
            os.path.join(folder, f"part{part}", "new_" + file_name), trimmed_feats[part]
        )
    # swap 'node_feat.dgl' files
    for part in range(num_parts):
        os.rename(
            os.path.join(folder, f"part{part}", file_name),
            os.path.join(folder, f"part{part}", file_name + ".bak"),
        )
        os.rename(
            os.path.join(folder, f"part{part}", "new_" + file_name),
            os.path.join(folder, f"part{part}", file_name),
        )


def main(folder, node_feat_names, edge_feat_names, use_low_mem=False):
    """Convert node features from distDGL format to WholeGraph format"""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "1234"
    torch.distributed.init_process_group("gloo", world_size=1, rank=0)
    wg_folder = os.path.join(folder, "wholegraph")
    if not os.path.exists(wg_folder):
        os.makedirs(wg_folder)

    logging.info(
        "node features:{} and edge features: {} will be converted to WholeGraph"
        "format.".format(node_feat_names, edge_feat_names)
    )

    metadata = {}
    # Process node features
    if node_feat_names:
        fname_dict = get_node_feat_info(node_feat_names)
        trimmed_feats = convert_feat_to_wholegraph(
            fname_dict, "node_feat.dgl", metadata, folder, use_low_mem
        )
        trim_feat_files(trimmed_feats, folder, "node_feat.dgl")

    # Process edge features
    if edge_feat_names:
        fname_dict = get_edge_feat_info(edge_feat_names)
        trimmed_feats = convert_feat_to_wholegraph(
            fname_dict, "edge_feat.dgl", metadata, folder, use_low_mem
        )
        trim_feat_files(trimmed_feats, folder, "edge_feat.dgl")

    # Save metatada
    with open(os.path.join(wg_folder, "metadata.json"), "w") as fp:
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
