import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict
import gzip
import sys
import dgl
from dgl.data.utils import save_graphs
import pandas as pd
import torch
import math
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

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

def construct_one_domain(full_file_path, asin_dict, asin_data, edge_dict, dup_asin, brand_dict, category_dict_list, item_brand, item_category):
    """ 
    Process item-related edges and node labels. Filter out low-frequency (i.e., less than 1000 times)
      product types and brands. Long-tail 
    The results are saved into parquet files.
    Input example: {"category": ["Clothing, Shoes & Jewelry", "Women", "Clothing"],  
                    "title": "Women Blouse, Ninasill Hooded Sweatshirt Coat Winter 
                    Warm Wool Zipper Pockets Cotton Coat Outwear", "also_buy": [],
                    "also_view: [], "asin": "6305121869"}
    Output example:
        item.parquet
            format: 
                item_asin item_text item_brand item_product_type_level1  
                item_product_type_level2 item_product_type_level3
        also_view.parquet
            format:
                item1_asin item2_asin
        also_buy.parquet
            format:
                item1_asin item2_asin
    """
    # read raw data
    with open(full_file_path) as f:
        raw_data = {}
        for line in tqdm(f):
            tmp = eval(line.strip())
            raw_data[tmp['asin']] = tmp
    data = {}
    label_freqs = [defaultdict(int), defaultdict(int), defaultdict(int)]
    brand_freq = defaultdict(int)
    for idd in raw_data:
        if 'title' in raw_data[idd]:
            data[idd] = raw_data[idd]
            if idd not in asin_dict:
                asin_dict[idd] = len(asin_dict)
            else:
                # cross-domain same asins
                dup_asin.add(idd)
            if 'brand' in data[idd] and data[idd]['brand'] != '':
                brand_freq[data[idd]['brand']] += 1
            if 'category' in data[idd] and len(data[idd]['category']) > 0:
                if type(data[idd]['category'][0]) is list:
                    breakpoint()
                else:
                    cate = data[idd]['category']
                    cate = cate + [''] * (3-len(cate))
                    cate[1] = f'{cate[0]}/{cate[1]}'
                    cate[2] = f'{cate[1]}/{cate[2]}'
                    label_freqs[0][cate[0]] += 1
                    label_freqs[1][cate[1]] += 1
                    label_freqs[2][cate[2]] += 1

    for idx,layer_label in enumerate(label_freqs):
        for k,freq in layer_label.items():
            if freq > 1000:
                category_dict_list[idx][k] = len(category_dict_list[idx]) + 1
    for k,freq in brand_freq.items():
        if freq > 1000:
            brand_dict[k] = len(brand_dict) + 1
        
    
    
    item, item_text, brand, pt_lvl1, pt_lvl2, pt_lvl3 = [], [], [], [], [], []
    also_buy, also_view = ([], []), ([], [])
    for idd in data:
        item.append(idd)
        if 'description' in data[idd]:
            q_text = text_process(data[idd]['title'] + ' ' + ' '.join(data[idd]['description']))
        else:
            q_text = text_process(data[idd]['title'])
        item_text.append(q_text)
        if 'brand' in data[idd] and data[idd]['brand'] != '':
            if data[idd]['brand'] in brand_dict:
                brand.append(brand_dict[data[idd]['brand']])
            else:
                brand.append(math.nan)
        else:
            brand.append(math.nan)
        if 'category' in data[idd] and len(data[idd]['category']) > 0:
            cate = data[idd]['category']
            cate = cate + [''] * (3-len(cate))
            cate[1] = f'{cate[0]}/{cate[1]}'
            cate[2] = f'{cate[1]}/{cate[2]}'
            if cate[0] in category_dict_list[0]:
                pt_lvl1.append(category_dict_list[0][cate[0]])
            else:
                pt_lvl1.append(math.nan)
            if cate[1] in category_dict_list[1]:
                pt_lvl2.append(category_dict_list[1][cate[1]])
            else:
                pt_lvl2.append(math.nan)
            if cate[2] in category_dict_list[2]:
                pt_lvl3.append(category_dict_list[2][cate[2]])
            else:
                pt_lvl3.append(math.nan)
        else:
            pt_lvl1.append(math.nan)
            pt_lvl2.append(math.nan)
            pt_lvl3.append(math.nan)
        if 'also_buy' in data[idd]:
            for dst_id in data[idd]['also_buy']:
                # define also_buy as src<dst, also_buy-rev as src>dst
                if idd < dst_id:
                    also_buy[0].append(idd)
                    also_buy[1].append(dst_id)
                else:
                    also_buy[0].append(dst_id)
                    also_buy[1].append(idd)
        if 'also_view' in data[idd]:
            for dst_id in data[idd]['also_view']:
                # define also_buy as src<dst, also_buy-rev as src>dst
                if idd < dst_id:
                    also_view[0].append(idd)
                    also_view[1].append(dst_id)
                else:
                    also_view[0].append(dst_id)
                    also_view[1].append(idd)

    df = pd.DataFrame({'item': item, 'text': item_text, 'brand': np.array(brand), 
                       'pt_lvl1': np.array(pt_lvl1), 'pt_lvl2': np.array(pt_lvl2), 'pt_lvl3': np.array(pt_lvl3)})
    table = pa.Table.from_pandas(df)
    rows_per_file = 500000

    # Calculate the total number of files needed
    num_files = len(df) // rows_per_file + (len(df) % rows_per_file > 0)

    # Iterate through chunks and write to separate Parquet files
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = (i + 1) * rows_per_file if (i + 1) * rows_per_file < len(df) else len(df)
        
        # Extract a chunk of the DataFrame
        chunk_df = df.iloc[start_idx:end_idx]

        # Convert the chunk to a pyarrow.Table
        chunk_table = pa.Table.from_pandas(chunk_df)
        
        # Write the table to a Parquet file
        file_path = f'processed_data/amazon_review/item/item_{i + 1}.parquet'
        pq.write_table(chunk_table, file_path)

        print(f"Parquet file {file_path} created for rows {start_idx + 1} to {end_idx}")
    df = pd.DataFrame({'src_item': also_buy[0], 'dst_item': also_buy[1]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'processed_data/amazon_review/also_buy.parquet')
    df = pd.DataFrame({'src_item': also_view[0], 'dst_item': also_view[1]})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f'processed_data/amazon_review/also_view.parquet')
    return

def construct_graph(full_file_path):
    random.seed(2023)
    
    asin_dict = dict()
    dup_asin = set()
    asin_data = dict()
    brand_dict = defaultdict(int)
    category_dict = [defaultdict(int),defaultdict(int), defaultdict(int)]
    item_brand = dict()
    item_category = [dict(), dict(), dict()]
    edge_dict = {
        ('item', 'also_buy', 'item'): [],
        ('item', 'also_view', 'item'): [],
    }
    cnt = 0

    construct_one_domain(full_file_path, asin_dict, asin_data, edge_dict, dup_asin, brand_dict, category_dict, item_brand, item_category)
    print(f"number of asins:{len(asin_dict)}, number of brands:{len(brand_dict)}")

if __name__ == '__main__':
    file_path = 'raw_data/All_Amazon_Meta.json'
    construct_graph(file_path)