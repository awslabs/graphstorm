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

    Generate example graph data using built-in datasets for link prediction.
"""

import os
import dgl
import numpy as np
import torch as th
import argparse
import time

from graphstorm.data import OGBTextFeatDataset
from graphstorm.data import MovieLens100kNCDataset
from graphstorm.data import ConstructedGraphDataset
from graphstorm.data import MAGLSCDataset
from graphstorm.utils import sys_tracker

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition DGL graphs for link prediction "
                                        + "tasks only.")
    # dataset and file arguments
    argparser.add_argument("-d", "--dataset", type=str, required=True,
                           help="dataset to use")
    argparser.add_argument("--filepath", type=str, default=None)
    # link prediction arguments
    argparser.add_argument('--target-etypes', type=str, help='The canonical edge types for making'
                           + ' prediction. Multiple edge types can be separated by " ". '
                           + 'For example, "EntA,Links,EntB EntC,Links,EntD"')
    # label split arguments
    argparser.add_argument('--train-pct', type=float, default=0.8,
                           help='The pct of train nodes/edges. Should be > 0 and < 1.')
    argparser.add_argument('--val-pct', type=float, default=0.1,
                           help='The pct of validation nodes/edges. Should be > 0 and < 1.')
    argparser.add_argument('--inductive-split', action='store_true',
                           help='split links for inductive settings: no overlapping nodes across '
                           + 'splits.')
    argparser.add_argument('--seed', type=int, default=42,
                           help='random seed for splitting links')    
    # graph modification arguments
    argparser.add_argument('--add-reverse-edges', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--train-graph-only', action='store_true',
                           help='Only partition the training graph.')
    argparser.add_argument('--retain-original-features',  type=lambda x: (str(x).lower() in ['true', '1']),
                           default=True, help= "whether to use the original features or use the paper title or abstract"
                                                "for the ogbn-arxiv dataset")
    argparser.add_argument('--retain-etypes', nargs='+', type=str, default=[],
        help='The list of canonical etype that will be retained before partitioning the graph. '
              + 'This might be helpfull to remove noise edges in this application. Format example: '
              + '--retain_etypes query,clicks,asin query,adds,asin query,purchases,asin '
              + 'asin,rev-clicks,query asin,rev-adds,query asin,rev-purchases,query')
    # partition arguments
    argparser.add_argument('--num-parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part-method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance-train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--balance-edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num-trainers-per-machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    # output arguments
    argparser.add_argument('--output', type=str, default='data',
                           help='The output directory to store the partitioned results.')
    # bert model name if any
    argparser.add_argument('--lm-model-name', type=str, default='bert-base-uncased',
                           help='lm model use to encode text feature if any')
    argparser.add_argument('--max-seq-length', type=int, default=128,
                           help="maximum sequence length when tokenizing text data")

    args = argparser.parse_args()
    print(args)
    start = time.time()
    np.random.seed(args.seed)

    constructed_graph = False

    # arugment sanity check
    assert (args.train_pct + args.val_pct) <= 1, \
        "The sum of train and validation percentages should NOT larger than 1."
    edge_pct = args.train_pct + args.val_pct

    # load graph data
    if args.dataset == 'ogbn-arxiv':
        dataset = OGBTextFeatDataset(args.filepath, args.dataset, edge_pct=edge_pct,
                                     retain_original_features=args.retain_original_features,
                                     max_sequence_length=args.max_seq_length,
                                     lm_model_name=args.lm_model_name)
    elif args.dataset == 'ogbn-products':
        dataset = OGBTextFeatDataset(args.filepath, args.dataset, edge_pct=edge_pct,
                                     retain_original_features=args.retain_original_features,
                                     max_sequence_length=args.max_seq_length,
                                     lm_model_name=args.lm_model_name)
    elif args.dataset == 'movie-lens-100k':
        dataset = MovieLens100kNCDataset(args.filepath, edge_pct=edge_pct)
    elif args.dataset == 'movie-lens-100k-text':
        dataset = MovieLens100kNCDataset(args.filepath,
                                         edge_pct=edge_pct, use_text_feat=True)
    elif args.dataset == 'ogbn-papers100M':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset, edge_pct=edge_pct,
                                     retain_original_features=args.retain_original_features,
                                     max_sequence_length=args.max_seq_length,
                                     lm_model_name=args.lm_model_name)
    elif args.dataset == 'mag-lsc':
        dataset = MAGLSCDataset(args.filepath, edge_pct=edge_pct)
    else:
        constructed_graph = True
        print("Loading user defined dataset " + str(args.dataset))
        dataset = ConstructedGraphDataset(args.dataset, args.filepath)
        assert args.target_etypes is not None, "For user defined dataset, you must provide target_etypes"

    g = dataset[0]

    if constructed_graph:
        if args.add_reverse_edges:
            print("Creating reverse edges ...")
            edges = {}
            for src_ntype, etype, dst_ntype in g.canonical_etypes:
                src, dst = g.edges(etype=(src_ntype, etype, dst_ntype))
                edges[(src_ntype, etype, dst_ntype)] = (src, dst)
                edges[(dst_ntype, etype + '-rev', src_ntype)] = (dst, src)
            num_nodes_dict = {}
            for ntype in g.ntypes:
                num_nodes_dict[ntype] = g.num_nodes(ntype)
            new_g = dgl.heterograph(edges, num_nodes_dict)
            # Copy the node data and edge data to the new graph. The reverse edges will not have data.
            for ntype in g.ntypes:
                for name in g.nodes[ntype].data:
                    new_g.nodes[ntype].data[name] = g.nodes[ntype].data[name]
            for etype in g.canonical_etypes:
                for name in g.edges[etype].data:
                    new_g.edges[etype].data[name] = g.edges[etype].data[name]
            g = new_g
            new_g = None

    target_etypes = dataset.target_etype if not constructed_graph else \
        [tuple(pred_etype.split(',')) for pred_etype in args.target_etypes.split(' ')]

    if not isinstance(target_etypes, list):
        target_etypes = [target_etypes]

    if constructed_graph:
        d_shuffled_nids = {} # to store shuffled nids by ntype to avoid different orders for the same ntype
        for target_e in target_etypes:
            num_edges = g.num_edges(target_e)
            g.edges[target_e].data['train_mask'] = th.full((num_edges,), False, dtype=th.bool)
            g.edges[target_e].data['val_mask'] = th.full((num_edges,), False, dtype=th.bool)
            g.edges[target_e].data['test_mask'] = th.full((num_edges,), False, dtype=th.bool)
            if not args.inductive_split:
                # Randomly split links
                g.edges[target_e].data['train_mask'][: int(num_edges * args.train_pct)] = True
                g.edges[target_e].data['val_mask'][int(num_edges * args.train_pct): \
                                                int(num_edges * (args.train_pct + args.val_pct))] = True
                g.edges[target_e].data['test_mask'][int(num_edges * (args.train_pct + args.val_pct)): ] = True
            else:
                # Inductive split for link prediction
                # 1. split the head nodes u into three disjoint sets (train/val/test)
                # such that model will be evaluted to predict links for unseen nodes
                utype, _, vtype = target_e
                num_nodes = g.number_of_nodes(utype)
                shuffled_index = d_shuffled_nids.get(utype,
                                                     np.random.permutation(np.arange(num_nodes)))
                if utype not in d_shuffled_nids:
                    d_shuffled_nids[utype] = shuffled_index
                train_u = shuffled_index[: int(num_nodes * args.train_pct)]
                val_u = shuffled_index[int(num_nodes * args.train_pct): \
                                        int(num_nodes * (args.train_pct + args.val_pct))]
                test_u = shuffled_index[int(num_nodes * (args.train_pct + args.val_pct)): ]
                # 2. find all out-edges for the 3 sets of head nodes:
                _, train_v, train_eids = g.out_edges(train_u, form='all', etype=target_e)
                _, val_v, val_eids = g.out_edges(val_u, form='all', etype=target_e)
                _, test_v, test_eids = g.out_edges(test_u, form='all', etype=target_e)
                if utype == vtype:
                    # we remove edges with tail nodes outside of the training set
                    # this isn't necessary if head and tail are different types
                    train_eids = train_eids[np.in1d(train_v, train_u)]
                    # remove overlaps between val and test
                    val_eids = val_eids[~np.in1d(val_v, test_u)]
                    test_eids = test_eids[~np.in1d(test_v, val_u)]
                # 3. build boolean edge masks: the edge mask prevents message-passing
                # flow graphs from fetching edges outside of the splits
                g.edges[target_e].data['train_mask'][train_eids] = True
                g.edges[target_e].data['val_mask'][val_eids] = True
                g.edges[target_e].data['test_mask'][test_eids] = True

    print(f'load {args.dataset} takes {time.time() - start:.3f} seconds')
    print(f'\n|V|={g.number_of_nodes()}, |E|={g.number_of_edges()}\n')
    for target_e in target_etypes:
        train_total = th.sum(g.edges[target_e].data['train_mask']) \
                      if 'train_mask' in g.edges[target_e].data else 0
        val_total = th.sum(g.edges[target_e].data['val_mask']) \
                    if 'val_mask' in g.edges[target_e].data else 0
        test_total = th.sum(g.edges[target_e].data['test_mask']) \
                     if 'test_mask' in g.edges[target_e].data else 0
        print(f'Edge type {target_e} :train: {train_total}, '
              +f'valid: {val_total}, test: {test_total}')

    # Get the train graph.
    if args.train_graph_only:
        sub_edges = {}
        for etype in g.canonical_etypes:
            sub_edges[etype] = g.edges[etype].data['train_mask'].bool() if 'train_mask' in g.edges[etype].data \
                    else th.ones(g.number_of_edges(etype), dtype=th.bool)
        g = dgl.edge_subgraph(g, sub_edges, relabel_nodes=False, store_ids=False)

    retain_etypes = [tuple(retain_etype.split(',')) for retain_etype in args.retain_etypes]

    if len(retain_etypes)>0:
        g = dgl.edge_type_subgraph(g, retain_etypes)
    sys_tracker.check("Finish processing the final graph")
    print(g)
    if args.balance_train and not args.train_graph_only:
        balance_etypes = {target_et: g.edges[target_et].data['train_mask'] for target_et in target_etypes}
    else:
        balance_etypes = None

    new_node_mapping, new_edge_mapping = dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                                                         part_method=args.part_method,
                                                                         balance_edges=args.balance_edges,
                                                                         num_trainers_per_machine=args.num_trainers_per_machine,
                                                                         return_mapping=True)
    sys_tracker.check('partition the graph')

    # the new_node_mapping contains per entity type on the ith row the original node id for the ith node.
    th.save(new_node_mapping, os.path.join(args.output, "node_mapping.pt"))
    # the new_edge_mapping contains per edge type on the ith row the original edge id for the ith edge.
    th.save(new_edge_mapping, os.path.join(args.output, "edge_mapping.pt"))
