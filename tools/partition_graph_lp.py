import dgl
import torch as th
import argparse
import time
from graphstorm.data import QueryASINSemanticMatchDataset, OGBTextFeatDataset
from graphstorm.data import MovieLens100kNCDataset
from graphstorm.data import ConstructedGraphDataset
from graphstorm.data.utils import save_maps

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument("-d", "--dataset", type=str, required=True,
                           help="dataset to use")
    argparser.add_argument("--filepath", type=str, default=None)
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--save_mappings', action='store_true',
                           help='Store the mappings for the edges and nodes after partition.')
    argparser.add_argument('--predict_etypes', type=str, default=None, help='The edge types for making prediction. '
                                                                            + 'Multiple edge types can be separated by " ". For example, "EntA,Links,EntB EntC,Links,EntD"')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    argparser.add_argument('--output', type=str, default='data',
                           help='The output directory to store the partitioned results.')
    argparser.add_argument('--train_graph_only', action='store_true',
                           help='Only partition the training graph.')
    argparser.add_argument('--retain_etypes', nargs='+', type=str, default=[],
        help="The list of canonical etype that will be retained before partitioning the graph. This might be"
             "helpfull to remove noise edges in this application for example "
              "--retain_etypes query,clicks,asin query,adds,asin query,purchases,asin asin,rev-clicks,query asin,rev-adds,query asin,rev-purchases,query"
              "then no aditional training target will "
              "be considered")
    args = argparser.parse_args()

    start = time.time()
    constructed_graph = False

    # load graph data
    if args.dataset == 'query_asin_match':
        dataset = QueryASINSemanticMatchDataset(args.filepath)
    elif args.dataset == 'ogbn-arxiv':
        dataset = OGBTextFeatDataset(args.filepath, args.dataset)
    elif args.dataset == 'ogbn-products':
        dataset = OGBTextFeatDataset(args.filepath, args.dataset)
    elif args.dataset == 'movie-lens-100k':
        dataset = MovieLens100kNCDataset(args.filepath)
    else:
        constructed_graph = True
        print("Loading user defined dataset " + str(args.dataset))
        dataset = ConstructedGraphDataset(args.dataset, args.filepath)
        assert args.predict_etypes is not None, "For user defined dataset, you must provide predict_etypes"

    g = dataset[0]
    print(g)
    target_etype = dataset.target_etype if not constructed_graph else [tuple(predict_etype.split(',')) for predict_etype in args.predict_etypes.split(' ')]
    retain_etypes = [tuple(retain_etype.split(',')) for retain_etype in args.retain_etypes]

    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    for target_e in target_etype:
        print('Edge type {} :train: {}, valid: {}, test: {}'.format(target_e, th.sum(g.edges[target_e].data['train_mask'])
                                                  if 'train_mask' in g.edges[target_e].data else 0,
                                                  th.sum(g.edges[target_e].data['val_mask'])
                                                  if 'val_mask' in g.edges[target_e].data else 0,
                                                  th.sum(g.edges[target_e].data['test_mask'])
                                                  if 'test_mask' in g.edges[target_e].data else 0))
    # Get the train graph.
    if args.train_graph_only:
        sub_edges = {}
        for etype in g.canonical_etypes:
            sub_edges[etype] = g.edges[etype].data['train_mask'].bool() if 'train_mask' in g.edges[etype].data \
                    else th.ones(g.number_of_edges(etype), dtype=th.bool)
        g = dgl.edge_subgraph(g, sub_edges, relabel_nodes=False, store_ids=False)
    if len(retain_etypes)>0:
        g = dgl.edge_type_subgraph(g, retain_etypes)
    print("The final graph after processing.")
    print(g)
    if args.balance_train and not args.train_graph_only:
        balance_etypes = {target_et: g.edges[target_et].data['train_mask'] for target_et in target_etype}
    else:
        balance_etypes = None

    new_node_mapping, new_edge_mapping = dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                                                         part_method=args.part_method,
                                                                         balance_edges=args.balance_edges,
                                                                         num_trainers_per_machine=args.num_trainers_per_machine,
                                                                         return_mapping=True)
    if args.save_mappings:
        # TODO add something that is more scalable here as a saving method and not pickle.

        # the new_node_mapping contains per entity type on the ith row the original node id for the ith node.
        save_maps(args.output, "new_node_mapping", new_node_mapping)
        # the new_edge_mapping contains per edge type on the ith row the original edge id for the ith edge.
        save_maps(args.output, "new_edge_mapping", new_edge_mapping)
