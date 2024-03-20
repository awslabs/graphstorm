import torch as th
import time
import graphstorm as gs
from graphstorm.utils import is_distributed
import faiss
import dgl
import numpy as np
from collections import defaultdict
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.dataloading import GSgnnNodeDataLoader
from graphstorm.dataloading import GSgnnNodeTrainData
from graphstorm.utils import setup_device
from graphstorm.model.utils import load_gsgnn_embeddings

def calculate_recall(pred, ground_truth):
    # Convert list_data to a set if it's not already a set
    if not isinstance(pred, set):
        pred = set(pred)
    
    overlap = len(pred & ground_truth)
    #if overlap > 0:
    #    return 1
    #else:
    #    return 0
    return overlap / len(ground_truth)

def main(config_args):
    """ main function
    """
    config = GSConfig(config_args)
    embs = load_gsgnn_embeddings(config.save_embed_path)

    index_dimension = embs[config.target_ntype].size(1)
    # Number of clusters (higher values lead to better recall but slower search)
    #nlist = 750
    #quantizer = faiss.IndexFlatL2(index_dimension)  # Use Flat index for quantization
    #index = faiss.IndexIVFFlat(quantizer, index_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
    #index.train(embs[config.target_ntype]) 
    index = faiss.IndexFlatIP(index_dimension)
    index.add(embs[config.target_ntype])

    #print(scores.abs().mean())
    
    gs.initialize(ip_config=config.ip_config, backend=config.backend)
    device = setup_device(config.local_rank)
    #index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(embedding_size))
    # Define the training dataset
    train_data = GSgnnNodeTrainData(
        config.graph_name,
        config.part_config,
        train_ntypes=config.target_ntype,
        eval_ntypes=config.eval_target_ntype,
        label_field=None,
        node_feat_field=None,
    )
    #for i in range(embs[config.target_ntype].shape[0]):
    #    print(embs[config.target_ntype][i,:].sum(), train_data.g.ndata['bert_h'][i].sum())
    #    breakpoint()
    #    embs[config.target_ntype][i,:] = train_data.g.ndata['bert_h'][i]
    
    #print( train_data.g.ndata['bert_h'][0,:], embs[config.target_ntype][0,:])
    #print(train_data.g.ndata['bert_h'])
    
    # TODO: devise a dataloader that can exclude targets and add train_mask like LP Loader
    test_dataloader = GSgnnNodeDataLoader(
        train_data,
        train_data.train_idxs,
        fanout=[-1],
        batch_size=config.eval_batch_size,
        device=device,
        train_task=False,
    )
    dataloader_iter = iter(test_dataloader)
    len_dataloader = max_num_batch = len(test_dataloader)
    tensor = th.tensor([len_dataloader], device=device)
    if is_distributed():
        th.distributed.all_reduce(tensor, op=th.distributed.ReduceOp.MAX)
        max_num_batch = tensor[0]
    recall = []
    max_ = []
    for iter_l in range(max_num_batch):
        ground_truth = defaultdict(set)
        input_nodes, seeds, blocks = next(dataloader_iter)
        #block_graph = dgl.block_to_graph(blocks[0])
        src_id = blocks[0].srcdata[dgl.NID].tolist()
        dst_id = blocks[0].dstdata[dgl.NID].tolist()
        #print(blocks[0].edges(form='uv', etype='also_buy'))
        #breakpoint()
        # print(dgl.NID)
        if 'also_buy' in blocks[0].etypes:
            #src, dst = block_graph.edges(form='uv', etype='also_buy')
            src, dst = blocks[0].edges(form='uv', etype='also_buy')
            for s,d in zip(src.tolist(),dst.tolist()):
                ground_truth[dst_id[d]].add(src_id[s])
                #ground_truth[src_id[s]].add(dst_id[d])
        if 'also_buy-rev' in blocks[0].etypes:
            #src, dst = block_graph.edges(form='uv', etype='also_buy-rev')
            src, dst = blocks[0].edges(form='uv', etype='also_buy-rev')
            for s,d in zip(src.tolist(),dst.tolist()):
                ground_truth[dst_id[d]].add(src_id[s])
                #ground_truth[src_id[s]].add(dst_id[d])
        query_idx = list(ground_truth.keys())
        #print(ground_truth)
        #breakpoint()
        ddd,lll = index.search(embs[config.target_ntype][query_idx],100 + 1)
        #knn_result = lll.tolist()
        
        for idx,query in enumerate(query_idx):
            recall.append(calculate_recall(lll[idx, 1:], ground_truth[query]))
            max_.append(query)
        #print(recall)
    if gs.get_rank() == 0:
        #print(query_idx, lll)
        #print(max_num_batch, len(recall), np.mean(recall))
        print(f'recall@100: {np.mean(recall)}')

def generate_parser():
    """Generate an argument parser"""
    parser = get_argument_parser()
    return parser

if __name__ == "__main__":
    arg_parser = generate_parser()

    args = arg_parser.parse_args()
    print(args)
    main(args)