import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import gzip
import dgl
from dgl.data.utils import save_graphs
from IPython import embed
import torch
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import math

# text processing function
def text_process(text):
    p_text = ' '.join(text.split('\r\n'))
    p_text = ' '.join(p_text.split('\n\r'))
    p_text = ' '.join(p_text.split('\n'))
    p_text = ' '.join(p_text.split('\t'))
    p_text = ' '.join(p_text.split('\rm'))
    p_text = ' '.join(p_text.split('\r'))
    p_text = ''.join(p_text.split('$'))
    p_text = ''.join(p_text.split('*'))

    return p_text

def tag_clean(cate):
    cate = cate.replace('</span>','')
    return cate

def encode_parquet(sub_g, edge_dict, idx2asin, asin_data, field_name):
    item, item_text, pt_lvl3 = [], [], []
    for idx, label in zip(sub_g.ndata['_ID'].tolist(), sub_g.ndata['label'].tolist()):
        item.append(idx)
        item_text.append(asin_data[idx2asin[idx]])
        if label > 0:
            pt_lvl3.append(label)
        else:
            pt_lvl3.append(math.nan)
    df = pd.DataFrame({'item': item, 'text': item_text, 'pt_lvl3': np.array(pt_lvl3)})
    table = pa.Table.from_pandas(df)
    os.makedirs(f'data/amazon_review/{field_name}/', exist_ok=True)
    pq.write_table(table, f'data/amazon_review/{field_name}/item.parquet')
    for etype in edge_dict:
        u,v = edge_dict[etype]
        edge_mask = u < v
        df = pd.DataFrame({'src_item': u[edge_mask].tolist(), 'dst_item': v[edge_mask].tolist()})
        table = pa.Table.from_pandas(df)
        pq.write_table(table, f'data/amazon_review/{field_name}/{etype}.parquet')
    return

def construct_one_domain(full_file_path, field_name, asin_dict, asin_data, edge_dict, dup_asin):
    # read raw data
    print(f"Processing {field_name}")
    with gzip.open(full_file_path) as f:
        raw_data = {}
        readin = f.readlines()
        for line in tqdm(readin):
            tmp = eval(line.strip())
            raw_data[tmp['asin']] = tmp
    data = {}
    category_dict_list = [{'':0}, {'':0}, {'':0}]
    label_freq = defaultdict(int)
    label_id = defaultdict(int)
    for idd in raw_data:
        if 'title' in raw_data[idd]:
            data[idd] = raw_data[idd]
            if idd not in asin_dict:
                asin_dict[idd] = len(asin_dict)
            else:
                dup_asin.add(idd)
            if 'category' in data[idd] and len(data[idd]['category']) > 0:
                # breakpoint()
                cate = data[idd]['category']
                cate = cate + [''] * (3-len(cate))
                if f'{cate[0]}/{cate[1]}/{cate[2]}' not in category_dict_list[2]:
                    category_dict_list[2][f'{cate[0]}/{cate[1]}/{cate[2]}'] = len(category_dict_list[2])
                label_freq[f'{cate[0]}/{cate[1]}/{cate[2]}'] += 1
    for k,v in label_freq.items():
        if v > len(raw_data) * 0.01:
            label_id[k] = len(label_id)
    item_label = []
    for idd in data:
        if 'description' in data[idd]:
            q_text = text_process(data[idd]['title'] + ' ' + ' '.join(data[idd]['description']))
        else:
            q_text = text_process(data[idd]['title'])
        asin_data[idd] = q_text

        if 'also_buy' in data[idd]:
            for dst_id in data[idd]['also_buy']:
                if dst_id in asin_dict:
                    edge_dict[('item', 'also_buy', 'item')].append([asin_dict[idd], asin_dict[dst_id]])

        if 'also_view' in data[idd]:
            for dst_id in data[idd]['also_view']:
                if dst_id in asin_dict:
                    edge_dict[('item', 'also_view', 'item')].append([asin_dict[idd], asin_dict[dst_id]])

        if 'category' in data[idd] and len(data[idd]['category']) > 0:
            cate = data[idd]['category']
            cate = cate + [''] * (3-len(cate))
            if f'{cate[0]}/{cate[1]}/{cate[2]}' in label_id:
                item_label.append(label_id[f'{cate[0]}/{cate[1]}/{cate[2]}'])
            else:
                item_label.append(-1)
        else:
            item_label.append(-1)
    return item_label, label_id



def construct_graph(directory_path, ood_fields = ['Video_Games, Automotive']):
    random.seed(2023)

    cnt = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            full_file_path = os.path.join(dirpath, filename)
            if filename.endswith('.json.gz'):
                cnt += 1
                field_name = filename.replace('.json.gz', '').replace('meta_', '')
                if field_name in ood_fields:
                    asin_dict = dict()
                    dup_asin = set()
                    asin_data = dict()
                    edge_dict = {
                        ('item', 'also_buy', 'item'): [],
                        ('item', 'also_view', 'item'): [],
                    }
                    labels, label_id = construct_one_domain(full_file_path, field_name, asin_dict, asin_data, edge_dict, dup_asin)
                    labels = torch.LongTensor(labels)
                    print(f"number of asins:{len(asin_dict)}, number of edges:{[(k,len(v)) for k,v in edge_dict.items()]}")
                    g = dgl.heterograph(edge_dict, num_nodes_dict={'item': len(asin_dict)} )
                    g.nodes['item'].data['label'] = labels

                    sub_g = dgl.edge_subgraph(g, {_etype: g.edges(form='eid', etype=_etype)  for _etype in g.etypes}, relabel_nodes=True, store_ids=True)

                    idx2asin = {v:k for k,v in asin_dict.items()}
                    encode_parquet(sub_g, {_etype: g.edges(form='uv', etype=_etype)  for _etype in g.etypes}, idx2asin, asin_data, field_name)

if __name__ == '__main__':
    directory_path = 'raw_data/'
    construct_graph(directory_path, ['Video_Games'])