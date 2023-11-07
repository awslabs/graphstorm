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


def wholegraph_processing(
    whole_feat_tensor, metadata, feat, wg_folder, num_parts
):
    """Convert DGL tensors to wholememory tensor

    Parameters
    ----------
    whole_feat_tensor : Tensor
        The concatenated feature tensor of different partitions
    metadata : Tensor
        Metadata of the feature tensor
    feat : str
        Name of the feature to be converted
    wg_folder : str
        Name of the folder to store the converted files
    num_parts : int
        Number of partitions of the input features
    """
    metadata[feat] = {
        "shape": list(whole_feat_tensor.shape),
        "dtype": str(whole_feat_tensor.dtype),
    }
    local_comm = wgth.comm.get_local_device_communicator()
    # Round up the integer division to match WholeGraph partitioning scheme
    subpart_size = -(whole_feat_tensor.shape[0] // -num_parts)

    for part_num in range(num_parts):
        st = part_num * subpart_size
        end = (part_num + 1) * subpart_size \
            if part_num != (num_parts - 1) \
            else whole_feat_tensor.shape[0]

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
    """Convert features from distDGL tensor format to WholeGraph format

    Parameters
    ----------
    fname_dict: dict of list
        Dict of names of the edge features of different edge types
    file_name:
        Name of the feature file, either node_feat.dgl or edge_feat.dgl
    metadata : Tensor
        Metadata of the feature tensor
    folder: str
        Name of the folder of the input feature files
    use_low_mem: bool
        Whether to use low memory version for conversion
    """
    wg_folder = os.path.join(folder, "wholegraph")
    folder_pattern = re.compile(r"^part[0-9]+$")
    part_files = [
        f
        for f in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, f)) and folder_pattern.match(f)
    ]
    part_files = sorted(part_files, key=lambda x: int(x.split("part")[1]))
    feats_data = []

    # When 'use_low_mem' is not enabled, this code loads and appends features from individual
    # partitions. Then features are concatenated and converted into the WholeGraph format one
    # by one. The minimum memory requirement for this approach is 2X the size of the input
    # nodes or edges features in the graph.
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
                logging.info("Processing %s features...", feat)
                whole_feat_tensor = torch.concat(
                    tuple(t[feat] for t in feats_data), dim=0
                )
                # Delete processed feature from memory
                for t in feats_data:
                    del t[feat]
                wholegraph_processing(
                    whole_feat_tensor, metadata, feat, wg_folder, num_parts
                )
        # Trim the original distDGL tensors
        for part in range(num_parts):
            trim_feat_files(feats_data, folder, file_name, part)

    # This low-memory version loads one partition at a time. It processes features one by one,
    # iterating through all the partitions and appending only the current feature, converting
    # it to a WholeGraph. The minimum memory requirement for this approach is 2X the size of
    # the largest node or edge feature in the graph.
    else:  # low-mem
        for ntype, feats in fname_dict.items():
            for feat in feats:
                feat = ntype + "/" + feat
                node_feats_data = None
                num_parts = 0
                # Read features from file
                for path in (os.path.join(folder, name) for name in part_files):
                    nfeat = dgl.data.utils.load_tensors(f"{path}/{file_name}")
                    if feat not in nfeat:
                        raise RuntimeError(
                            f"Error: Unknown feature '{feat}'. Files contain \
                                       the following features: {nfeat.keys()}."
                        )
                    if node_feats_data is None:
                        node_feats_data = nfeat[feat]
                    else:
                        node_feats_data = torch.concat((node_feats_data, nfeat[feat]), dim=0)
                    num_parts += 1
                del nfeat
                gc.collect()
                wholegraph_processing(
                    node_feats_data,
                    metadata,
                    feat,
                    wg_folder,
                    num_parts,
                )
        num_parts = 0
        for path in (os.path.join(folder, name) for name in part_files):
            feats_data = dgl.data.utils.load_tensors(f"{path}/{file_name}")
            for type_name, feats in fname_dict.items():
                for feat in feats:
                    feat = type_name + "/" + feat
                    # Delete processed feature from memory
                    del feats_data[feat]
            num_parts += 1
            trim_feat_files(feats_data, folder, file_name, num_parts)


def trim_feat_files(trimmed_feats, folder, file_name, part):
    """Save new truncated distDGL tensors
    Parameters
    ----------
    trimmed_feats : list of tensors
        distDGL tensors after trimming out the processed features
    folder : str
        Name of the folder of the input feature files
    file_name : str
        Name of the feature file, either node_feat.dgl or edge_feat.dgl
    part : int
        Partition number of the input feature files

    """
    dgl.data.utils.save_tensors(
        os.path.join(folder, f"part{part}", "new_" + file_name), trimmed_feats[part]
    )
    os.rename(
        os.path.join(folder, f"part{part}", file_name),
        os.path.join(folder, f"part{part}", file_name + ".bak"),
    )
    os.rename(
        os.path.join(folder, f"part{part}", "new_" + file_name),
        os.path.join(folder, f"part{part}", file_name),
    )


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
    with open(os.path.join(wg_folder, "metadata.json"), "w", encoding="utf8") as fp:
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
