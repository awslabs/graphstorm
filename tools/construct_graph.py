import dgl
import os
import numpy as np
import torch as th
from torch import nn
import argparse
import time
from m5_dataloaders.datasets.constants import REGRESSION_TASK, CLASSIFICATION_TASK

from graphstorm.data import StandardM5gnnDataset
from graphstorm.data.constants import TOKEN_IDX, VALID_LEN_IDX
from graphstorm.data.utils import save_maps

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument("--name", type=str, required=True, help="The name of the dataset")
    argparser.add_argument("--filepath", type=str, default=None, help='The path of the dataset.')
    argparser.add_argument('--output', type=str, default='data',
                           help='The output directory to store the constructed graph.')
    argparser.add_argument('--dist_output', type=str, default='dist_data',
                           help='The output directory to store the partitioned results.')

    # Options for constructing the graph.
    argparser.add_argument("--num_dataset_workers", type=int, default=8,
                           help='The number of workers to process the dataset.')
    argparser.add_argument('--nid_fields', type=str,
                           help='The field that stores the node ID on node data files.'
                           + 'The format is "ntype1:id_field1 ntype2:id_field2')
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
    argparser.add_argument('--random_seed', type=int, default=None, help='the random seed for splitting.')
    argparser.add_argument('--nlabel_fields', type=str, help='The fields that stores the labels on nodes. '
                           + 'The format is "ntype1:label1 ntype2:label2". e.g., "business:stars".')
    argparser.add_argument('--ntask_types', type=str, help='The prediction tasks on nodes. '
                           + 'The format is "ntype1:task1 ntype2:task2". The possible values of tasks are "classify", "regression".')
    argparser.add_argument('--predict_ntypes', type=str, help='The node types for making prediction. '
                           + 'Multiple node types can be separated by " ".')
    argparser.add_argument("--generate_new_split", type=lambda x: (str(x).lower() in ['true', '1']), default=False,
                           help="Split the node. If we are splitting the data from scatch we should not do it by default.")
    argparser.add_argument("--split_ntypes", type=str, default=None,
                           help="split_ntypes : The format is ntype1 ntype2 list of strings The node types where we split data.")
    argparser.add_argument('--nfeat_fields', type=str, help='The fields that stores node features. '
                           + 'The format is "ntype1:feat1,feat2". e.g., "customer:f1,f2".')

    # Options for edge prediction task.
    argparser.add_argument('--elabel_fields', type=str, help='The fields that stores the labels on edges. '
                           + 'The format is "srcntype1,etype1,dstntype1:label1 srcntype2,etype2,dstntype2:label2". e.g., "customer,review,movie:stars".')
    argparser.add_argument('--efeat_fields', type=str, help='The fields that stores edge features. '
                           + 'The format is "srcntype1,etype1,dstntype1:feat1,feat2 srcntype2,etype2,dstntype2:feat3". e.g., "customer,review,movie:f1,f2".')

    argparser.add_argument('--etask_types', type=str, help='The prediction tasks on edges. '
                           + 'The format is "srcntype1,etype1,dstntype1:task1 srcntype2,etype2,dstntype2:task2". The possible values of tasks are "classify", "regression".')
    argparser.add_argument('--predict_etypes', type=str, help='The edge types for making prediction. '
                           + 'The format is srcntype1,etype1,dstntype1 srcntype2,etype2,dstntype2 Multiple edge types can be separated by " ".')
    argparser.add_argument("--generate_new_edge_split", type=lambda x: (str(x).lower() in ['true', '1']), default=False,
                           help="If we are splitting the data from scatch we should not do it by default.")
    argparser.add_argument("--split_etypes", type=str, default=None,
                           help="split_etypes :  The format is srcntype1,etype1,dstntype1 srcntype2,etype2,dstntype2:label2 list of strings The edge types where we split data.")
    argparser.add_argument("--edge_name_delimiter", type=str, default='_',
                           help="the token(s) that connect node names forming edge file name. e.g., '_' is for edge-query_clicks_asin, '::' is for edge-buyer::has_positive_feedback:asin")

    # Options for graph partitioning.
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')

    # options for node feature loading.
    argparser.add_argument('--nfeat_format', type=str, default=None,
                           help='Specify the format of node feature files. Currently support "hdf5" and "npy".')

    # options for edge feature loading.
    argparser.add_argument('--efeat_format', type=str, default=None,
                           help='Specify the format of edge feature files. Currently support "hdf5" and "npy".')

    argparser.add_argument('--save_mappings', action='store_true',
                           help='Store the mappings for the edges and nodes after partition.')

    args = argparser.parse_args()

    start = time.time()

    nid_fields = {}
    if args.nid_fields is not None:
        for nid in args.nid_fields.split(' '):
            ntype, id = nid.split(':')
            nid_fields[ntype] = id
        print('node id fields:', nid_fields)


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

    efeat_fields = {}
    if args.efeat_fields is not None:
        for feat in args.efeat_fields.split(' '):
            etype, fields = feat.split(':')
            fields = fields.split(',')
            etype = tuple(etype.split(','))
            efeat_fields[etype] = fields
        print('edge feature fields:', efeat_fields)

    nfeat_fields = {}
    if args.nfeat_fields is not None:
        for feat in args.nfeat_fields.split(' '):
            ntype, fields = feat.split(':')
            fields = fields.split(',')
            nfeat_fields[ntype] = fields
        print('node feature fields:', nfeat_fields)

    # load graph data
    assert args.hf_bert_model is not None or args.m5_vocab is not None
    dataset = StandardM5gnnDataset(args.filepath, args.name,
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
                                    efeat_fields=efeat_fields,
                                    nfeat_fields=nfeat_fields,
                                    # TODO(zhengda) right now all fields have the same max text length.
                                    max_node_seq_length={ntype: args.max_seq_length for ntype in ntext_fields},
                                    num_worker=args.num_dataset_workers,
                                    nfeat_format=args.nfeat_format,
                                    efeat_format=args.efeat_format,
                                    edge_name_delimiter=args.edge_name_delimiter)

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

    g = dataset[0]
    if args.undirected:
        print("Creating reverse edges ...")
        edges = {}
        for src_ntype, etype, dst_ntype in g.canonical_etypes:
            src, dst = g.edges(etype=(src_ntype, etype, dst_ntype))
            edges[(src_ntype, etype, dst_ntype)] = (src, dst)
            edges[(dst_ntype, etype + '-rev', src_ntype)] = (dst, src)
        new_g = dgl.heterograph(edges, num_nodes_dict={name: len(nid_map) for name, nid_map in dataset.nid_maps.items()})
        # Copy the node data and edge data to the new graph. The reverse edges will
        # not have data.
        for ntype in g.ntypes:
            for name in g.nodes[ntype].data:
                new_g.nodes[ntype].data[name] = g.nodes[ntype].data[name]
        for etype in g.canonical_etypes:
            for name in g.edges[etype].data:
                new_g.edges[etype].data[name] = g.edges[etype].data[name]
        g = new_g
        new_g = None

    assert not (args.generate_new_split and args.generate_new_edge_split), "You should not generate new split for both nodes and edges."

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    if args.generate_new_split:
        print("Generating new node split ...")
        for category in n_categories:
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

    if args.generate_new_edge_split:
        print("Generating new edge split ...")
        for category in e_categories:
            num_edges = g.number_of_edges(category)
            test_idx = np.random.choice(num_edges, num_edges // 10, replace=False)
            train_idx = np.setdiff1d(np.arange(num_edges), test_idx)
            val_idx = np.random.choice(train_idx, num_edges // 10, replace=False)
            train_idx = np.setdiff1d(train_idx, val_idx)
            train_mask = th.zeros((num_edges,), dtype=th.bool)
            train_mask[train_idx] = True
            val_mask = th.zeros((num_edges,), dtype=th.bool)
            val_mask[val_idx] = True
            test_mask = th.zeros((num_edges,), dtype=th.bool)
            test_mask[test_idx] = True
            g.edges[category].data['train_mask'] = train_mask
            g.edges[category].data['val_mask'] = val_mask
            g.edges[category].data['test_mask'] = test_mask

    if args.compute_bert_emb:
        print("Computing bert embedding ...")
        device = 'cpu' if args.device < 0 else 'cuda:' + str(args.device)
        assert args.m5_model is not None or args.hf_bert_model is not None
        if args.m5_model is not None:
            from graphstorm.model.m5 import load_m5_bert
            bert_model = load_m5_bert(args.m5_model, device=None)
        elif args.hf_bert_model is not None:
            from graphstorm.model.hbert import run_bert, init_bert
            bert_model = init_bert(bert_model_name=args.hf_bert_model)
        if th.cuda.device_count() > 1 and args.hf_bert_model is not None:
            print('use {} GPUs for computing BERT embeddings'.format(th.cuda.device_count()))
            bert_model = nn.DataParallel(bert_model)
        bert_model = bert_model.to(device)
        bert_model.eval()
        for ntype in g.ntypes:
            if TOKEN_IDX in g.nodes[ntype].data:
                t1 = time.time()
                token_list = th.split(g.nodes[ntype].data[TOKEN_IDX], 1000)
                valid_len_list = th.split(g.nodes[ntype].data[VALID_LEN_IDX], 1000)
                emb_list = []
                for tokens, valid_len in zip(token_list, valid_len_list):
                    with th.no_grad():
                        embs, _ = run_bert(bert_model, tokens.to(device),
                                           valid_len.to(device),
                                           token_type_ids=None,
                                           dev=device)
                        emb_list.append(embs.cpu())
                if 'feat' in g.nodes[ntype].data:
                    feat = th.cat(emb_list)
                    assert len(g.nodes[ntype].data['feat'].shape) == 2
                    feat = th.cat([g.nodes[ntype].data['feat'], feat], dim=1)
                    g.nodes[ntype].data['feat'] = feat
                else:
                    g.nodes[ntype].data['feat'] = th.cat(emb_list)
                t2 = time.time()
                print('Computing BERT embeddings on node {}: {:.3f} seconds'.format(ntype, t2 - t1))
                if args.remove_text_tokens:
                    del g.nodes[ntype].data[TOKEN_IDX]
                    del g.nodes[ntype].data[VALID_LEN_IDX]

    print('load {} takes {:.3f} seconds'.format(args.name, time.time() - start))
    print(g)
    print('node types:', g.ntypes)
    for ntype in g.ntypes:
        print('node {}:'.format(ntype), {name: g.nodes[ntype].data[name].shape for name in g.nodes[ntype].data})

    print('edge types:', g.canonical_etypes)
    for etype in g.canonical_etypes:
        print('edge {}:'.format(etype), {name: g.edges[etype].data[name].shape for name in g.edges[etype].data})

    for category in e_categories:
        print('training target edge type: {}, train: {}, valid: {}, test: {}'.format(category,
            th.sum(g.edges[category].data['train_mask']) \
                if 'train_mask' in g.edges[category].data else 0,
            th.sum(g.edges[category].data['val_mask']) \
                if 'val_mask' in g.edges[category].data else 0,
            th.sum(g.edges[category].data['test_mask']) \
                if 'test_mask' in g.edges[category].data else 0))

    for category in n_categories:
        print('training target node type: {}, train: {}, valid: {}, test: {}'.format(category,
            th.sum(g.nodes[category].data['train_mask']) \
                if 'train_mask' in g.nodes[category].data else 0,
            th.sum(g.nodes[category].data['val_mask']) \
                if 'val_mask' in g.nodes[category].data else 0,
            th.sum(g.nodes[category].data['test_mask']) \
                if 'test_mask' in g.nodes[category].data else 0))

    # If users don't provide the n_categories for prediction and it doesn't exist in the input dataset,
    # that means there is no data split on nodes and we don't need to balance them.
    if args.balance_train and len(n_categories) > 0:
        balance_ntypes = {category: g.nodes[category].data['train_mask'] for category in n_categories}
    else:
        balance_ntypes = None

    os.makedirs(args.output, exist_ok = True)

    for retry_cnt in range(20):
        try:
            print ("Saving graph for {}th time ...".format(retry_cnt+1))
            dgl.save_graphs(os.path.join(args.output, args.name + '.dgl'), [g])
            break
        except:
            if retry_cnt == 19:
                print ("Job failed for saving single graph. Jump to graph partition")
                break
            print ("{}th time failed, waiting 2 hours for retry. Please free up more space.".format(retry_cnt+1))
            time.sleep(7200)


    new_node_mapping, new_edge_mapping = dgl.distributed.partition_graph(g, args.name, args.num_parts, args.dist_output,
                                                                         part_method=args.part_method,
                                                                         balance_ntypes=balance_ntypes,
                                                                         balance_edges=args.balance_edges,
                                                                         num_trainers_per_machine=args.num_trainers_per_machine,
                                                                         return_mapping=True)
    if args.save_mappings:
        # TODO add something that is more scalable here as a saving method and not pickle.

        # the new_node_mapping contains per entity type on the ith row the original node id for the ith node.
        save_maps(args.dist_output, "new_node_mapping", new_node_mapping)
        # the new_edge_mapping contains per edge type on the ith row the original edge id for the ith edge.
        save_maps(args.dist_output, "new_edge_mapping", new_edge_mapping)
