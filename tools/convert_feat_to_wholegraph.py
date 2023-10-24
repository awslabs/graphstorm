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
import os
import re
import torch
import dgl
import pylibwholegraph.torch as wgth


def process_node_data(folder, node_feat_names):
    # per node type feature
    fname_dict = {}
    for feat_name in node_feat_names:
        feat_info = feat_name.split(":")
        assert len(feat_info) == 2, \
                f"Unknown format of the feature name: {feat_name}, " + \
                "must be NODE_TYPE:FEAT_NAME"
        ntype = feat_info[0]
        assert ntype not in fname_dict, \
                f"You already specify the feature names of {ntype} " \
                f"as {fname_dict[ntype]}"
        assert isinstance(feat_info[1], str), \
            f"Feature name of {ntype} should be a string not {feat_info[1]}"
        # multiple features separated by ','
        fname_dict[ntype] = feat_info[1].split(",")

    return fname_dict


def process_edge_data(folder, edge_feat_names):
    # per edge type feature
    fname_dict = {}
    for feat_name in edge_feat_names:
        feat_info = feat_name.split(":")
        assert len(feat_info) == 2, \
                f"Unknown format of the feature name: {feat_name}, " + \
                "must be EDGE_TYPE:FEAT_NAME"
        etype = feat_info[0].split(',')
        assert len(etype) == 3, \
                f"EDGE_TYPE should have 3 strings: {etype}, " + \
                "must be NODE_TYPE:EDGE_TYPE:NODE_TYPE:"
        etype = ":".join(etype)
        assert etype not in fname_dict, \
                f"You already specify the feature names of {etype} " \
                f"as {fname_dict[etype]}"
        assert isinstance(feat_info[1], str), \
            f"Feature name of {etype} should be a string not {feat_info[1]}"
        # multiple features separated by ','
        fname_dict[etype] = feat_info[1].split(",")

    return fname_dict

def convert_feat_to_wholegraph(fname_dict, file_name, metadata, local_comm, folder):

    feats_data = []
    wg_folder = os.path.join(folder, 'wholegraph')
    folder_pattern = re.compile(r"^part[0-9]+$")
    for path in (os.path.join(folder, name) for name in sorted( \
        (f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) \
        and folder_pattern.match(f)), key=lambda x: int(x.split("part")[1]))):
        feats_data.append(dgl.data.utils.load_tensors(f'{path}/{file_name}'))

    num_parts = len(feats_data)
    for ntype, feats in fname_dict.items():
        for feat in feats:
            feat = ntype + "/" + feat
            if feat not in feats_data[0]:
                raise RuntimeError(f"Error: Unknown feature '{feat}'. Files contain \
                                   the following features: {feats_data[0].keys()}.")
            print(f"Processing '{feat}' features...")
            whole_feat_tensor = torch.concat(tuple(t[feat] for t in feats_data), dim=0)
            metadata[feat] = {'shape': list(whole_feat_tensor.shape),
                              'dtype': str(whole_feat_tensor.dtype)}

            # Round up the integer division to match WholeGraph partitioning scheme
            subpart_size = -(whole_feat_tensor.shape[0] // -num_parts)

            for part_num in range(num_parts):
                st = part_num * subpart_size
                end = (part_num + 1) * subpart_size if part_num != (num_parts - 1) \
                    else whole_feat_tensor.shape[0]
                wg_tensor = wgth.create_wholememory_tensor(local_comm, 'continuous', 'cpu',
                                                          (end - st, *whole_feat_tensor.shape[1:]),
                                                          whole_feat_tensor.dtype, None)
                local_tensor, _ = wg_tensor.get_local_tensor(host_view=True)
                local_tensor.copy_(whole_feat_tensor[st:end])
                filename = wgth.utils.get_part_file_name(feat.replace('/', '~'),
                                                         part_num, num_parts)
                wg_tensor.local_to_file(os.path.join(wg_folder, filename))
                wgth.destroy_wholememory_tensor(wg_tensor)

            # Delete processed feature from memory
            for t in feats_data:
                del t[feat]
    return feats_data


def main(folder, node_feat_names, edge_feat_names):
    """ Convert node features from distDGL format to WholeGraph format"""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'
    torch.distributed.init_process_group('gloo', world_size=1, rank=0)
    local_comm = wgth.comm.get_local_device_communicator()
    wg_folder = os.path.join(folder, 'wholegraph')
    if not os.path.exists(wg_folder):
        os.makedirs(wg_folder)

    print("node features:", node_feat_names, " and edge features: ", edge_feat_names, " will be converted to WholeGraph format.")

    metadata = {}
    # Process node features
    if node_feat_names:
        fname_dict = process_node_data(folder, node_feat_names)
        trimmed_feats = convert_feat_to_wholegraph(fname_dict, "node_feat.dgl", metadata, local_comm, folder)
        num_parts = len(trimmed_feats)

        # Save new truncated distDGL tensors
        for part in range(num_parts):
            dgl.data.utils.save_tensors(os.path.join(folder, f'part{part}', 'new_node_feat.dgl'),
                                        trimmed_feats[part])
        # swap 'node_feat.dgl' files
        for part in range(num_parts):
            os.rename(os.path.join(folder, f'part{part}', 'node_feat.dgl'),
                      os.path.join(folder, f'part{part}', 'node_feat.dgl.bak'))
            os.rename(os.path.join(folder, f'part{part}', 'new_node_feat.dgl'),
                      os.path.join(folder, f'part{part}', 'node_feat.dgl'))

    # Process edge features
    if edge_feat_names:
        fname_dict = process_edge_data(folder, edge_feat_names)
        trimmed_feats = convert_feat_to_wholegraph(fname_dict, "edge_feat.dgl", metadata, local_comm, folder)
        num_parts = len(trimmed_feats)

        # Save new truncated distDGL tensors
        for part in range(num_parts):
            dgl.data.utils.save_tensors(os.path.join(folder, f'part{part}', 'new_edge_feat.dgl'),
                                        trimmed_feats[part])
        # swap 'edge_feat.dgl' files
        for part in range(num_parts):
            os.rename(os.path.join(folder, f'part{part}', 'edge_feat.dgl'),
                      os.path.join(folder, f'part{part}', 'edge_feat.dgl.bak'))
            os.rename(os.path.join(folder, f'part{part}', 'new_edge_feat.dgl'),
                      os.path.join(folder, f'part{part}', 'edge_feat.dgl'))

    # Save metatada
    with open(os.path.join(wg_folder, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', action='store', type=str, required=True)
    parser.add_argument('--node-feat-names', nargs='+', action='store', type=str)
    parser.add_argument('--edge-feat-names', nargs='+', action='store', type=str)
    args = parser.parse_args()
    main(args.dataset_path, args.node_feat_names, args.edge_feat_names)
