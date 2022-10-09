import dgl

import torch as th
import argparse
import time
from graphstorm.data import OGBTextFeatDataset
from graphstorm.data import MovieLens100kNCDataset
from graphstorm.data import ConstructedGraphDataset

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Create edge split graphs")
    argparser.add_argument("-d", "--dataset", type=str, required=True,
                           help="dataset to use")
    argparser.add_argument("--filepath", type=str, default=None)
    argparser.add_argument('--output', type=str, default=None,
                           help='The output directory to store the results.')
    argparser.add_argument('--train_pct', type=float, default=None,
                           help='The pct of train edges.')
    argparser.add_argument('--test_pct', type=float, default=None,
                           help='The pct of train edges.')
    argparser.add_argument('--target_etypes', nargs='+', type=str, default=[],
        help="The list of canonical etype that will be mapped to the random split for testing and training data before partitioning the graph. for example "
              "--target_etypes query,clicks,asin query,adds,asin query,purchases,asin asin,rev-clicks,query asin,rev-adds,query asin,rev-purchases,query")

    args = argparser.parse_args()

    start = time.time()

    # load graph data
    if args.dataset == 'ogbn-arxiv':
        dataset = OGBTextFeatDataset(args.filepath, args.dataset)
    elif args.dataset == 'ogbn-products':
        dataset = OGBTextFeatDataset(args.filepath, args.dataset)
    elif args.dataset == 'movie-lens-100k':
        dataset = MovieLens100kNCDataset(args.filepath)
    else:
        constructed_graph = True
        print("Loading user defined dataset " + str(args.dataset))
        dataset = ConstructedGraphDataset(args.dataset, args.filepath)

    g = dataset[0]
    print(g)
    target_etypes = [tuple(target_etype.split(',')) for target_etype in args.target_etypes]
    assert 1 > args.train_pct + args.test_pct

    for etype in target_etypes:
        etype = etype[1]
        int_edges = g.num_edges(etype)
        val_pct = 1 - (args.train_pct + args.test_pct)
        train_pct = args.train_pct
        # the test is 1 - the rest
        g.edges[etype].data['train_mask'] = th.full((int_edges,), False, dtype=th.bool)
        g.edges[etype].data['val_mask'] = th.full((int_edges,), False, dtype=th.bool)
        g.edges[etype].data['test_mask'] = th.full((int_edges,), False, dtype=th.bool)

        g.edges[etype].data['train_mask'][: int(int_edges * train_pct)] = True
        g.edges[etype].data['val_mask'][int(int_edges * train_pct):int(int_edges * (train_pct+ val_pct))] = True
        g.edges[etype].data['test_mask'][int(int_edges * (train_pct+ val_pct)):] = True
    if args.dataset == 'esci_classification':
        dgl.save_graphs(args.output, [g])
    else:
        dataset.save_graph(args.output)

# /opt/conda/bin/python3 create_edge_random_split.py --dataset esci_classification --filepath /fsx-dev/ivasilei/home/esci_public_reduced_graph/esci_classification.dgl --output /fsx-dev/ivasilei/home/esci_public_reduced_graph_0.6/esci_classification.dgl --train_pct 0.6 --test_pct 0.3 --target_etypes query,exactmatch,asin asin,exactmatch-rev,query
