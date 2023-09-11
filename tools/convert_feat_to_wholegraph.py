import argparse
import dgl
import json
import os
import re
import sys
import torch
import pylibwholegraph.torch as wgth


def main(folder, feat_names):
    # Wholegraph setup
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '1234'
    torch.distributed.init_process_group('gloo', world_size=1, rank=0)
    local_comm = wgth.comm.get_local_device_communicator()
    wg_folder = os.path.join(folder, 'wholegraph')
    if not os.path.exists(wg_folder):
        os.makedirs(wg_folder)

    feats = feat_names.split(',')
    node_feats_data = []
    folder_pattern = re.compile(r"^part[0-9]+$")
    for path in (os.path.join(folder, name) for name in sorted((f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and folder_pattern.match(f)),
                                                                       key=lambda x: int(x.split("part")[1]))):
        node_feats_data.append(dgl.data.utils.load_tensors(f'{path}/node_feat.dgl'))

    num_parts = len(node_feats_data)
    metadata = {}
    for f in feats:
        print(f"Processing '{f}' features...")
        if f not in node_feats_data[0]:
            print(f"Error: Unknown feature '{f}'. Files contain the following features: {node_feats_data[0].keys()}.")
            sys.exit(1)
        whole_feat_tensor = torch.concat(tuple(t[f] for t in node_feats_data), dim=0)
        metadata[f] = {'shape': list(whole_feat_tensor.shape), 'dtype': str(whole_feat_tensor.dtype)}
        # Round up the integer division to match WholeGraph partitioning scheme
        subpart_size = -(whole_feat_tensor.shape[0] // -num_parts)
        for part in range(num_parts):
            l = part*subpart_size
            u = (part+1)*subpart_size if part != (num_parts - 1) else whole_feat_tensor.shape[0]
            wg_tensor = wgth.create_wholememory_tensor(local_comm, 'continuous', 'cpu', (u-l, *whole_feat_tensor.shape[1:]), whole_feat_tensor.dtype, None)
            local_tensor, _ = wg_tensor.get_local_tensor(host_view=True)
            local_tensor.copy_(whole_feat_tensor[l:u])
            filename = wgth.utils.get_part_file_name(f.replace('/', '~'), part, num_parts)
            wg_tensor.local_to_file(os.path.join(wg_folder, filename))
            wgth.destroy_wholememory_tensor(wg_tensor)

        # Delete processed feature from memory
        for t in node_feats_data:
            del t[f]

    # Save metatada
    with open(os.path.join(wg_folder, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)

    # Save new truncated distDGL tensors
    for part in range(num_parts):
        dgl.data.utils.save_tensors(os.path.join(folder, f'part{part}', 'new_node_feat.dgl'), node_feats_data[part])

    # swap 'node_feat.dgl' files
    for part in range(num_parts):
        os.rename(os.path.join(folder, f'part{part}', 'node_feat.dgl'), os.path.join(folder, f'part{part}', 'node_feat.dgl.bak'))
        os.rename(os.path.join(folder, f'part{part}', 'new_node_feat.dgl'), os.path.join(folder, f'part{part}', 'node_feat.dgl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', action='store', type=str, required=True)
    parser.add_argument('--feat-names', action='store', type=str, required=True)
    args = parser.parse_args()
    main(args.dataset_path, args.feat_names)
