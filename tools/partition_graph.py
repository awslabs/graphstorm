import dgl
import numpy as np
import torch as th
import argparse
import time
from graphstorm.data import OGBTextFeatDataset
from graphstorm.data import MovieLens100kNCDataset
from graphstorm.data import ConstructedGraphDataset

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument("-d", "--dataset", type=str, required=True,
                           help="dataset to use")
    argparser.add_argument("--filepath", type=str, default=None)
    argparser.add_argument('--retain_original_features',  type=lambda x: (str(x).lower() in ['true', '1']),
                           default=False, help= "whether to use the original features or use the paper title or abstract"
                                                "for the ogbn-arxiv dataset")
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--predict_ntypes', type=str, help='The node types for making prediction. '
                           + 'Multiple node types can be separated by ",".')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    argparser.add_argument('--output', type=str, default='data',
                           help='The output directory to store the partitioned results.')
    argparser.add_argument("--generate_new_split", type=lambda x: (str(x).lower() in ['true', '1']), default=False,
            help="If we are splitting the data from scatch we should not do it by default.")
    args = argparser.parse_args()

    start = time.time()

    # load graph data
    if args.dataset == 'ogbn-arxiv':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset)
    elif args.dataset == 'ogbn-products':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset)
    elif args.dataset == 'ogbn-papers100m':
        dataset = OGBTextFeatDataset(args.filepath, dataset=args.dataset)
    elif args.dataset == 'movie-lens-100k':
        dataset = MovieLens100kNCDataset(args.filepath)
    else:
        print("Loading user defined dataset " + str(args.dataset))
        dataset = ConstructedGraphDataset(args.dataset, args.filepath)
        assert args.predict_ntypes is not None, "For user defined dataset, you must provide predict_ntypes"

    categories = args.predict_ntypes.split(',') if args.predict_ntypes is not None else None
    if categories is None:
        try:
            categories = [dataset.predict_category]
        except:
            pass

    g = dataset[0]
    if args.generate_new_split:
        for category in categories:
            num_nodes = g.number_of_nodes(category)
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
            g.nodes[category].data['train_mask'] = train_mask
            g.nodes[category].data['val_mask'] = val_mask
            g.nodes[category].data['test_mask'] = test_mask

    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('node types:', g.ntypes)
    for ntype in g.ntypes:
        for name in g.nodes[ntype].data:
            data = g.nodes[ntype].data[name]
            print('node {} has data {} of type {}'.format(ntype, name, data.dtype))
    print('edge types:', g.canonical_etypes)
    for etype in g.canonical_etypes:
        for name in g.edges[etype].data:
            data = g.edges[etype].data[name]
            print('edge {} has data {} of type {}'.format(etype, name, data.dtype))
    for category in categories:
        print('training target node type: {}, train: {}, valid: {}, test: {}'.format(category,
            th.sum(g.nodes[category].data['train_mask']),
            th.sum(g.nodes[category].data['val_mask']) if 'val_mask' in g.nodes[category].data else 0,
            th.sum(g.nodes[category].data['test_mask']) if 'test_mask' in g.nodes[category].data else 0))

    if args.balance_train:
        balance_ntypes = {category: g.nodes[category].data['train_mask'] for category in categories}
    else:
        balance_ntypes = None

    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges,
                                    num_trainers_per_machine=args.num_trainers_per_machine)
