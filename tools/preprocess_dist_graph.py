import os
import numpy as np
import torch as th
import argparse
import time, datetime
import json
import psutil
from m5_dataloaders.datasets.constants import REGRESSION_TASK, CLASSIFICATION_TASK

from graphstorm.data import StandardM5gnnDataset
from graphstorm.data.constants import TOKEN_IDX, VALID_LEN_IDX

def etype2path(etype):
    assert isinstance(etype, tuple) and len(etype) == 3, "etype should be a tuple with 3 elements"
    etype = "_".join(etype)

    return etype

def save_nodes(path, graph_name, start, end, ntype_id, num_ntypes, part_id):
    """ Save information of nodes of a specific node type into xxx_nodes.txt

    Data format: <node_type> <weight1>.... <orig_type_node_id>.
      * <node_type> is an integer. For homogeneous graphs it is always 0 and otherwise it indicates the type of the node.
      * <weight1>, <weight2>...etc. are integers indicating node weights used by ParMETIS to balance graph partitions
      * <orig_type_node_id> is an integer representing node id within its own note type.

    Note: Ndoe info is saved by rank0 process to simplify the implementation.
    TODO(xiangsx): Make saving node info distributed.

    Parameters
    ----------
    path: str
        Path to save feature
    graph_name: str
        Graph name
    start:
        nid start
    end:
        nid end
    ntype_id:
        Node type id
    num_ntypes:
        Total number of node types
    part_id:
        Partition id
    """
    nid_range_info = {}
    weight = ['0'] * num_ntypes
    weight[ntype_id] = '1'
    weight = " ".join(weight)
    basic_info = str(ntype_id) + " " + weight
    file_name = os.path.join(path, f"{graph_name}_nodes{ntype_id}_{part_id}.txt")
    with open(file_name, 'a+') as f:
        for i in range(start, end):
            f.write("{} {}\n".format(basic_info, i))
        nid_range_info[file_name] = (start, end)
    return nid_range_info

def save_edges(path, graph_name, etype_id, edges, part_id):
    """ Save information of edges of a specific edge type into xxx_edges_<part_id>.txt

    Data format: <src_id> <dst_id>
      * <src_id> is node ID of the source node.
      * <dst_id> is node ID of the destination node.

    Node: Edge info is saved in mutliple files.

    Parameters
    ----------
    path: str
        Path to save feature
    graph_name: str
        Graph name
    etype_id: int
        Edge type id
    edges: tuple of two tensors
        Edges stored as (src_tensor, dst_tensor)
    part_id:
        Partition id
    """
    total_edges = edges[0].shape[0]
    src_ids = edges[0].tolist()
    dst_ids = edges[1].tolist()

    with open(os.path.join(path, f"{graph_name}_edges{etype_id}_{part_id}.txt"), 'w') as f:
        for i, (src, dst) in enumerate(zip(src_ids, dst_ids)):
            f.write("{} {}\n".format(src, dst))
            if part_id == 0 and i % 10000000 == 0:
                print(f"[{part_id}:{etype_id}][{i}/{total_edges}] saved")
                print(f"[{part_id}:{psutil.virtual_memory()}")
    print(f"[{part_id}] Finish saving {etype_id}")

def create_path(path, subdir):
    path = os.path.join(path, subdir)
    os.makedirs(path, exist_ok=True)
    print(f"build path {path}")

    return path

def save_feat(path, type, feat_name, data, part_id):
    """ Save node or edge features into disk

    Parameters
    ----------
    path: str
        Path to save feature
    type: str
        Node or edge type
    feat_name: str
        Feature name
    data: torch.Tensor
        Tensor to save
    num_parts: int
        Whether we need to split a tensor into num_parts
    part_id: int
        Specific partition id.
    """
    dir = os.path.join(path, type)
    dir = os.path.join(dir, feat_name)

    assert part_id >= 0
    filename = os.path.join(dir, '{}.npy'.format(part_id))
    np.save(filename, data.cpu().numpy())

def save_efeat(path, etype, feat_name, data, part_id):
    """ Save edge features into disk

    Parameters
    ----------
    path: str
        Path to save feature
    etype: str
        Edge type
    feat_name: str
        Feature name
    data: torch.Tensor
        Tensor to save
    part_id: int
        Specific partition id.
    """
    if isinstance(etype, str) is False:
        etype = etype2path(etype)
    assert len(etype) > 0
    save_feat(path, etype, feat_name, data, part_id)

def save_type_map(path, ntype_map, etype_map):
    """ Save node type to ID and edge type to ID mapping

    Parameters
    ----------
    ntype_map: dict
        node type mapping
    etype_map: dict
        edge type mapping
    """
    with open(os.path.join(path, "map_info.json"), 'w') as f:
        json.dump({"ntype_map": ntype_map, "etype_map": {"{}_{}_{}".format(etype[0], etype[1], etype[2]): edges for etype, edges in etype_map.items()}}, f)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Preprocess graphs")
    argparser.add_argument("--name", type=str, required=True, help="The name of the dataset")
    argparser.add_argument("--filepath", type=str, default=None, help='The path of the dataset.')
    argparser.add_argument('--output', type=str, default='data',
                           help='The output directory to store the processed graph.')
    argparser.add_argument('--local_rank', type=int, default=-1)

    # Options for constructing the graph.
    argparser.add_argument("--num_dataset_workers", type=int, default=8,
                           help='The number of workers to process the dataset.')
    argparser.add_argument('--nid_fields', type=str, default=None,
                           help='The field that stores the node ID on node data files.')
    argparser.add_argument('--src_field', type=str, default='src_id',
                           help='The field that stores the source node ID in the edge data files.')
    argparser.add_argument('--dst_field', type=str, default='dst_id',
                           help='The field that stores the destination node ID in the edge data files.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')

    # Options for BERT computation.
    argparser.add_argument('--hf_bert_model', type=str, help='The name of the HuggingFace BERT model.')
    argparser.add_argument('--m5_vocab', type=str, help='The vocabulary file of M5 model.')
    argparser.add_argument('--m5_model', type=str, help='The file of the M5 model.')
    argparser.add_argument('--compute_bert_emb',  type=lambda x: (str(x).lower() in ['true', '1']),
                           default=False, help= "Whether or not compute BERT embeddings.")
    argparser.add_argument('--remove_text_tokens',  type=lambda x: (str(x).lower() in ['true', '1']),
                           default=False, help= "Whether or not remove text tokens after computing BERT embeddings.")
    argparser.add_argument('--ntext_fields', type=str, help='The fields that stores text data on nodes. '
                           + 'The format is "ntype1:text1,text2 ntype2:text3". e.g., "review:text".')
    argparser.add_argument("--max_seq_length", type=int, default=128)
    argparser.add_argument("--device", type=int, default=-1)

    # Options for prediction task.
    argparser.add_argument('--nlabel_fields', type=str, help='The fields that stores the labels on nodes. '
                           + 'The format is "ntype1:label1 ntype2:label2". e.g., "business:stars".')
    argparser.add_argument('--ntask_types', type=str, default=None,
            help='The prediction tasks on nodes. '
                           + 'The format is "ntype1:task1 ntype2:task2". The possible values of tasks are "classify", "regression".')
    argparser.add_argument('--predict_ntypes', type=str, help='The node types for making prediction. '
                           + 'Multiple node types can be separated by ",".')
    argparser.add_argument("--generate_new_split", type=lambda x: (str(x).lower() in ['true', '1']), default=False,
                           help="If we are splitting the data from scatch we should not do it by default.")
    argparser.add_argument("--split_ntypes", type=str, default=None,
                           help="split_ntypes : The format is ntype1 ntype2 list of strings The node types where we split data.")
    argparser.add_argument('--ntypes', type=str, help='The list of all node types' +
                           'The format is "ntype1 ntype2 ntype3"')
    argparser.add_argument('--etypes', type=str, help='The list of all edge types' +
                           'The format is "src_type1,rel_type1,dst_type1 src_type2,rel_type2,dst_type2"')
    # Options for edge prediction task.
    argparser.add_argument('--elabel_fields', type=str, help='The fields that stores the labels on edges. '
                           + 'The format is "srcntype1,etype1,dstntype1:label1 srcntype2,etype2,dstntype2:label2". e.g., "customer,review,movie:stars".')
    argparser.add_argument('--etask_types', type=str, help='The prediction tasks on edges. '
                           + 'The format is "srcntype1,etype1,dstntype1:task1 srcntype2,etype2,dstntype2:task2". The possible values of tasks are "classify", "regression".')
    argparser.add_argument('--predict_etypes', type=str, help='The edge types for making prediction. '
                           + 'The format is srcntype1,etype1,dstntype1 srcntype2,etype2,dstntype2 Multiple edge types can be separated by " ".')
    argparser.add_argument("--split_etypes", type=str, default=None,
                           help="split_etypes :  The format is srcntype1,etype1,dstntype1 srcntype2,etype2,dstntype2 list of strings The edge types where we split data.")


    # options for node feature loading.
    argparser.add_argument('--nfeat_format', type=str, default=None,
                           help='Specify the format of node feature files. Currently support "hdf5" and "npy".')
    argparser.add_argument('--efeat_format', type=str, default=None,
                           help='Specify the format of edge feature files. Currently support "hdf5" and "npy".')

    th.distributed.init_process_group(backend='gloo', timeout=datetime.timedelta(seconds=14000)) # no need to use GPU
    start = time.time()
    args = argparser.parse_args()
    global_rank = th.distributed.get_rank()
    if args.split_ntypes is not None:
        split_ntypes = args.split_ntypes.split(' ')
    else:
        split_ntypes = []

    if args.split_etypes is not None:
        split_etypes_str = args.split_etypes.split(' ')
        split_etypes = []
        for etype in split_etypes_str:
            split_etypes.append(tuple(etype.split(',')))
    else:
        split_etypes = []

    nid_fields = {}
    if args.nid_fields is not None:
        for label in args.nid_fields.split(' '):
            ntype, label = label.split(':')
            nid_fields[ntype] = label
        print('node id fields:', nid_fields)

    nlabel_fields = {}
    if args.nlabel_fields is not None:
        for label in args.nlabel_fields.split(' '):
            ntype, label = label.split(':')
            nlabel_fields[ntype] = label
        print('node label fields:', nlabel_fields)

    elabel_fields = {}
    if args.elabel_fields is not None:
        for label in args.elabel_fields.split(' '):
            etype, label = label.split(':')
            etype = tuple(etype.split(','))
            elabel_fields[etype] = label
        print('edge label fields:', elabel_fields)

    # For link prediction tasks, users don't specify node types. We need to allow the case
    # that users don't specify any node types.
    ntask_types = {}
    if args.ntask_types is not None:
        for text in args.ntask_types.split(' '):
            ntype, task_type = text.split(':')
            if task_type == 'classify':
                ntask_types[ntype] = CLASSIFICATION_TASK
            elif task_type == 'regression':
                ntask_types[ntype] = REGRESSION_TASK
            else:
                print('Unknown task {} on node type {}'.format(task_type, ntype))
                print('The possible values of tasks are "classify", "regression".')

    etask_types = {}
    if args.etask_types is not None:
        for text in args.etask_types.split(' '):
            etype, task_type = text.split(':')
            etype = tuple(etype.split(','))
            if task_type == 'classify':
                etask_types[etype] = CLASSIFICATION_TASK
            elif task_type == 'regression':
                etask_types[etype] = REGRESSION_TASK
            else:
                print('Unknown task {} on edge type {}'.format(task_type, etype))
                print('The possible values of tasks are "classify", "regression".')

    ntext_fields = {}
    if args.ntext_fields is not None:
        for text in args.ntext_fields.split(' '):
            ntype, fields = text.split(':')
            fields = fields.split(',')
            ntext_fields[ntype] = fields
        print('node text fields:', ntext_fields)

    assert args.hf_bert_model is not None or args.m5_vocab is not None
    if global_rank == 0:
        # TODO(xiangsx): We assume the output is in FSX
        # Support S3 later.
        os.makedirs(args.output, exist_ok = True)
    th.distributed.barrier()

    # load graph data
    # Note: args.compute_bert_emb should be passed-in
    dataset = StandardM5gnnDataset(args.filepath, args.name,
                                    rank=global_rank,
                                    m5_vocab=args.m5_vocab,
                                    hf_bert_model=args.hf_bert_model,
                                    nid_fields=nid_fields, src_field=args.src_field, dst_field=args.dst_field,
                                    nlabel_fields=nlabel_fields,
                                    ntask_types=ntask_types,
                                    split_ntypes=split_ntypes,
                                    elabel_fields=elabel_fields,
                                    etask_types=etask_types,
                                    split_etypes=split_etypes,
                                    ntext_fields=ntext_fields,
                                    # TODO(zhengda) right now all fields have the same max text length.
                                    max_node_seq_length={ntype: args.max_seq_length for ntype in ntext_fields},
                                    num_worker=args.num_dataset_workers,
                                    nfeat_format=args.nfeat_format,
                                    efeat_format=args.efeat_format)

    n_categories = []
    if args.predict_ntypes is not None:
        n_categories = args.predict_ntypes.split(' ')

    e_categories = []
    if args.predict_etypes is not None:
        e_categories_str = args.predict_etypes.split(' ')
        for etype in e_categories_str:
            e_categories.append(tuple(etype.split(',')))

    if len(n_categories) == 0:
        try:
            n_categories = [dataset.predict_category]
        except:
            pass

    graph_data = dataset[0]
    world_size = th.distributed.get_world_size() \
                if th.distributed.is_initialized() else 1

    print(graph_data)
    print("graph data")
    if args.generate_new_split:
        # only for node classification
        assert args.ntask_types is not None
        for category in n_categories:
            # Regenerate new ndata mask
            num_global_nodes = graph_data["number_of_nodes"][category]
            num_nodes = (num_global_nodes + world_size - 1) // world_size
            num_nodes = num_nodes if global_rank + 1 < world_size \
                else num_global_nodes - num_nodes * global_rank

            test_idx = np.random.choice(num_nodes, num_nodes // 10, replace=False)
            train_idx = np.setdiff1d(np.arange(num_nodes), test_idx)
            val_idx = np.random.choice(train_idx, num_nodes // 10, replace=False)
            train_idx = np.setdiff1d(train_idx, val_idx)
            train_mask = th.zeros((num_nodes,), dtype=th.bool)
            train_mask[train_idx] = True
            val_mask = th.zeros((num_nodes,), dtype=th.bool)
            val_mask[val_idx] = True
            test_mask = th.zeros((num_nodes,), dtype=th.bool)
            test_mask[test_idx] = True

            graph_data["ndata"][category]['train_mask'] = train_mask
            graph_data["ndata"][category]['val_mask'] = val_mask
            graph_data["ndata"][category]['test_mask'] = test_mask

    ntypes = args.ntypes.split(' ')
    assert len(ntypes) == len(graph_data["number_of_nodes"].keys())
    ntype_offset = {}
    etype_offset = {}
    last_noffset = 0
    last_eoffset = 0
    for ntype in ntypes:
        num_nodes = graph_data["number_of_nodes"][ntype]
        ntype_offset[ntype] = (last_noffset, last_noffset + num_nodes)
        last_noffset += num_nodes

    node_info = {}
    ntype_map = {}
    for ntype in ntypes:
        assert ntype in graph_data["num_local_nodes"]
        start = 0 if global_rank == 0 else \
            sum(graph_data["num_local_nodes"][ntype][:global_rank-1])
        end = start + graph_data["num_local_nodes"][ntype][global_rank]
        ntype_map[ntype] = len(ntype_map)
        nid_range_info = save_nodes(args.output, args.name, start, end, ntype_map[ntype], len(ntypes), part_id=global_rank)

        node_info[ntype] = nid_range_info

    th.distributed.barrier()

    # save edges
    edges = graph_data["edges"]
    etype_map = {}
    edge_info = []
    print(args.etypes)
    for etype in args.etypes.split(' '):
        etype = tuple(etype.split(','))
        etype_map[etype] = len(etype_map)
        save_edges(args.output, args.name, etype_map[etype],
            edges[etype], part_id=global_rank)

        # save graph stats
        num_edges = graph_data["number_of_edges"][etype]
        edge_info.append([f"{args.name}_edges{etype_map[etype]}_{part}.txt" for part in range(world_size)])
        etype_offset[etype] = (last_eoffset, last_eoffset + num_edges)
        last_eoffset += num_edges

        if args.undirected:
            rev_etype = (etype[2], etype[1] + '-rev', etype[0])
            etype_map[rev_etype] = len(etype_map)
            rev_edges = (edges[etype][1], edges[etype][0])
            save_edges(args.output, args.name, etype_map[rev_etype],
                rev_edges, part_id=global_rank)

            edge_info.append([f"{args.name}_edges{etype_map[rev_etype]}_{part}.txt" for part in range(world_size)])
            etype_offset[rev_etype] = (last_eoffset, last_eoffset + num_edges)
            last_eoffset += num_edges
    print(f"{global_rank} Finish save ndata")
    th.distributed.barrier()

    # clean edge data to save memory
    del edges
    del graph_data["edges"]

    print("Start save ndata")
    # build feature folders first
    if global_rank == 0:
        for ntype, nfeats in graph_data["ndata"].items():
            for feat_name, _ in nfeats.items():
                dir = create_path(args.output, ntype)
                create_path(dir, feat_name)
        for etype, efeats in graph_data["edata"].items():
            for feat_name, _ in efeats.items():
                dir = create_path(args.output, etype2path(etype))
                create_path(dir, feat_name)
    th.distributed.barrier()

    # output node list and edge list
    for ntype, nfeats in graph_data["ndata"].items():
        for feat_name, feat in nfeats.items():
            print(psutil.virtual_memory())
            print(f"Save feat {feat_name}")
            save_feat(args.output, ntype, feat_name, feat, part_id=global_rank)

    for etype, efeats in graph_data["edata"].items():
        for feat_name, feat in efeats.items():
            save_efeat(args.output, etype, feat_name, feat, part_id=global_rank)

    if global_rank == 0:
        # save config
        # node_info: node info (which files store ndoes)
        # edge_info: edge info (which files store edges)
        # ntype_offset: global range for nodes (debug info)
        # etype_offset: global range for edges (debug info)
        # ntype_map: node type map
        # etype_map: edge type map
        node_types = []
        edge_types = []
        num_nodes_per_chunk = []
        num_edges_per_chunk = []
        nid_info = {}
        for ntype, _ in ntype_offset.items():
            node_types.append(ntype)
            num_nodes_per_chunk.append(graph_data["num_local_nodes"][ntype])

        for etype, _ in etype_offset.items():
            edge_types.append(etype)
            if etype not in graph_data["num_local_edges"]:
                if args.undirected:
                    orig_etype = (etype[2], etype[1][:-4], etype[0])
                    num_edges_per_chunk.append(graph_data["num_local_edges"][orig_etype])
            else:
                num_edges_per_chunk.append(graph_data["num_local_edges"][etype])

        nfeat_info = {}
        for ntype, feats in graph_data["ndata"].items():
            info = graph_data["nfeat_split"][ntype]
            if len(feats) == 0:
                continue
            nfeat_info[ntype] = {}
            for feat_name, _ in feats.items():
                dir = os.path.join(args.output, ntype)
                dir = os.path.join(dir, feat_name)
                feat_data = []
                offset = 0
                part_size = info[1]
                num_nodes = info[2]
                for part_id in range(info[0]):
                    filename = os.path.join(dir, '{}.npy'.format(part_id))
                    feat_data.append([filename, offset, offset + part_size \
                        if offset + part_size < num_nodes else num_nodes])
                    offset += part_size
                nfeat_info[ntype][feat_name] = {
                    "format": {"name": "numpy"},
                    "data" : feat_data,
                }

        eid_info = {}
        for etype, info in graph_data["num_local_edges"].items():
            data = []
            for part_id, num_edges in enumerate(info):
                filename = os.path.join(args.output, f"{args.name}_edges{etype_map[etype]}_{part_id}.txt")
                data.append(filename)
            eid_info[":".join(etype)] = {
                "format" : {"name": "csv", "delimiter": " "},
                "data": data
            }

            if args.undirected:
                rev_etype = (etype[2], etype[1] + '-rev', etype[0])
                data = []
                offset = 0
                for part_id, num_edges in enumerate(info):
                    filename = os.path.join(args.output, f"{args.name}_edges{etype_map[rev_etype]}_{part_id}.txt")
                    data.append(filename)
                eid_info[":".join(rev_etype)] = {
                    "format" : {"name": "csv", "delimiter": " "},
                    "data": data
                }

        efeat_info = {}
        for etype, efeats in graph_data["edata"].items():
            if len(efeats) == 0:
                continue
            info = graph_data["num_local_edges"][etype]
            efeat_info[":".join(etype)] = {}
            offset = 0
            for feat_name, _ in efeats.items():
                feat_data = []
                dir = os.path.join(args.output, ":".join(etype))
                dir = os.path.join(dir, feat_name)
                for part_id, num_edges in enumerate(info):
                    filename = os.path.join(dir, '{}.npy'.format(part_id))
                    feat_data.append(
                        [filename, offset, offset + num_edges])
                    offset += num_edges

                efeat_info[":".join(etype)][feat_name] = {
                    "format": {"name": "numpy"},
                    "data": feat_data,
                }

        meta_info = {
            "graph_name": args.name,
            "node_type": node_types,
            "edge_type": [":".join(e) for e in edge_types],
            "num_nodes_per_chunk": num_nodes_per_chunk,
            "num_edges_per_chunk": num_edges_per_chunk,
            "node_data" : nfeat_info,
            "edge_data": efeat_info,
            "edges": eid_info,
        }

        print(meta_info)

        with open(os.path.join(args.output, 'metadata.json'), 'w') as f:
            json.dump(meta_info, f, sort_keys=True, indent=4, separators=(',', ': '))

    print('load {} takes {:.3f} seconds'.format(args.name, time.time() - start))
