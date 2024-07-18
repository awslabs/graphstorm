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

    Varify the correctness of graph, including whether there are NaN in
    the node or edge features and whether the train/val/test masks are
    loadable by GraphStorm.
"""
import os
import re
import argparse
import logging

import dgl
import dgl.backend as F
import torch as th

def get_node_feat_info(node_feat_names):
    """ Process node feature names

    Parameters
    ----------
    node_feat_names : list of str
        The name of the node features. It's a dict if different node types have
        different feature names.
        The format will be ["ntype0:feat0,feat1", "ntype1:feat0,feat1", ...]

    Note: This is similar to the implementation in convert_feat_to_wholegraph.py.
          But GraphStorm keeps each tool self-contained.

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
    edge_feat_names : list of str
        The name of the edge features. It's a dict if different edge types have
        different feature names.
        The format will be ["src_type0,rel_type0,dst_type0:feat0,feat1",
                            "src_type1,rel_type1,dst_type1:feat0,feat1", ...]

    Note: This is similar to the implementation in convert_feat_to_wholegraph.py.
          But GraphStorm keeps each tool self-contained.

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

def gen_node_error_prefix(partition, ntype, fname):
    """ Create an error prefix for reporting node feature/masks.

        The Prefix will be in the following format nypte-fname-partition.
        So when errors are reported we can easily sort it and the errors from the
        same node type and same feature will be grouped.

        Parameters
        ----------
        partition: int
            Partition ID.
        ntype: str
            Node type.
        fname: str
            Feature name.

        Return
        ------
            str: the error prefix
    """
    return f"[Node type: {ntype}][Feature Name: {fname}][Part {partition}]:"

def gen_edge_error_prefix(partition, etype, fname):
    """ Create an error prefix for reporting eode feature/masks.

        The Prefix will be in the following format eypte-fname-partition.
        So when errors are reported we can easily sort it and the errors from the
        same eode type and same feature will be grouped.

        Parameters
        ----------
        partition: int
            Partition ID.
        etype: tuple
            Edge type.
        fname: str
            Feature name.

        Return
        ------
            str: the error prefix
    """
    return f"[Edge type: {etype}][Feature Name: {fname}][Part {partition}]:"

def verify_masks(mask, error_prefix):
    """ Validate the data of a mask

        Parameters
        ----------
        mask: tensor
            Mask
        error_prefix: str
            Error report prefix

        Return
        ------
            str: the error message if the mask is invalid otherwise None
    """
    try:
        # DGL use nonzero_1d to translate a mask into indices of nodes/edges.
        # so we check the mask using dgl.backend.nonzero_1d
        mask = F.nonzero_1d(mask)
    except:
        return f"ERROR: {error_prefix} The mask is not loadable for loading. " \
                 "Ideally, the mask should be a 1D bool tensor."
    return None

def verify_feats(feat, error_prefix):
    """ Validate the data of a feat

        Parameters
        ----------
        feat: tensor
            Features
        error_prefix: str
            Error report prefix

        Return
        ------
            str: the error message if the feature has NaNs
            str: the warning message if the feature is not normalized
    """
    # check whether feat has Nan
    has_nan = th.isnan(feat).any().item()
    err_msg = None
    warn_msg = None
    if has_nan:
        err_msg = f"ERROR: {error_prefix} There are NaN values in the feature, please check."

    # check whether feat has inf
    has_inf = th.isinf(feat).any().item()
    if has_inf:
        inf_err_msg = f"ERROR: {error_prefix} There are infinite values in the feature, please check."

        err_msg = f"{err_msg}\n{inf_err_msg}" if err_msg is not None else inf_err_msg

    # check whether the value is normalized between -1 to 1
    larger_1 = (feat > 1.0).any().item()
    small_minus_1 = (feat < -1.0).any().item()
    if larger_1 or small_minus_1:
        max_val = th.max(feat)
        min_val = th.min(feat)
        warn_msg = f"WARNING: {error_prefix} There are some value out of the range of [-1, 1]. " \
                f"We get [{max_val}: {min_val}]" \
                "It won't cause any error, but it is recommended to normalize the feature."
    return err_msg, warn_msg

def report_errors(error_messages):
    for message in sorted(error_messages):
        print(message)

def main(data_path, node_masks, edge_masks):
    # Process node features
    node_mask_info = get_node_feat_info(node_masks) if node_masks is not None else {}
    edge_mask_info = get_node_feat_info(edge_masks) if edge_masks is not None else {}

    partition_path_pattern = re.compile(r"^part[0-9]+$")
    partitions = [
        f for f in os.listdir(data_path) \
            if os.path.isdir(os.path.join(data_path, f)) and partition_path_pattern.match(f)
    ]
    partitions = sorted(partitions, key=lambda x: int(x.split("part")[1]))
    nmask_errors = []
    nfeat_errors = []
    nfeat_warns = []
    emask_errors = []
    efeat_errors = []
    efeat_warns = []
    for partition in partitions:
        partition_path = os.path.join(data_path, partition)
        node_feat_path = os.path.join(partition_path, "node_feat.dgl")
        edge_feat_path = os.path.join(partition_path, "edge_feat.dgl")

        if os.path.exists(node_feat_path):
            # check node feature
            node_feats = dgl.data.utils.load_tensors(node_feat_path)
            for fname, feat in node_feats.items():
                # dgl stores node feature in format of
                # "node_type/feat_name": Tensor
                ntype, fname = fname.split("/")
                if ntype in node_mask_info and feat in node_mask_info[ntype]:
                    # The feature is mask, varify the mask data
                    mask_error_info = verify_masks(feat, gen_node_error_prefix(partition, ntype, fname))
                    if mask_error_info is not None:
                        nmask_errors.append(mask_error_info)
                else:
                    nfeat_error_info, nfeat_warn_info = verify_feats(feat, gen_node_error_prefix(partition, ntype, fname))
                    if nfeat_error_info is not None:
                        nfeat_errors.append(nfeat_error_info)
                    if nfeat_warn_info is not None:
                        nfeat_warns.append(nfeat_warn_info)

        if os.path.exists(edge_feat_path):
            edge_feats = dgl.data.utils.load_tensors(edge_feat_path)
            for fname, feat in edge_feats.items():
                # dgl stores edge feature in format of
                # "edge_type/feat_name": Tensor
                etype, fname = fname.split("/")
                if etype in edge_mask_info and feat in edge_mask_info[etype]:
                    # The feature is mask, varify the mask data
                    mask_error_info = verify_masks(feat, gen_edge_error_prefix(partition, etype, fname))
                    if mask_error_info is not None:
                        emask_errors.append(mask_error_info)
                else:
                    efeat_error_info, efeat_warn_info = verify_feats(feat, gen_edge_error_prefix(partition, etype, fname))
                    if efeat_error_info is not None:
                        efeat_errors.append(efeat_error_info)
                    if efeat_warn_info is not None:
                        efeat_warns.append(efeat_warn_info)
    report_errors(nmask_errors)
    report_errors(emask_errors)
    report_errors(nfeat_errors)
    report_errors(efeat_errors)
    report_errors(nfeat_warns)
    report_errors(efeat_warns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", action="store", type=str, required=True)
    parser.add_argument("--node-masks", nargs="+", action="store", type=str,
                        help="The node masks to be checked. For example,"
                        "--node-masks ntype0:mask0,mask1 ntype1:mask0."
                        "The mask0 and mask1 of ntype0 and mask0 of ntype1 will be checked"
                        "If not specified, masks are not checked")
    parser.add_argument("--edge-masks", nargs="+", action="store", type=str,
                        help="The edge masks to be checked. For example,"
                        "--edge-masks src_type0,rel_type0,dst_type0:mask0,mask1 "
                        "src_type1,rel_type1,dst_type1:mask0."
                        "The mask0 and mask1 of edge type (src_type0, rel_type0, dst_type0) "
                        "and mask0 of edge type (src_type1, rel_type1, dst_type1) will be checked")

    args = parser.parse_args()
    main(args.dataset_path, args.node_masks, args.edge_masks)