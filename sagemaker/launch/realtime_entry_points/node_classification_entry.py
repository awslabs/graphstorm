"""
    Copyright Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GraphStorm Built-in Node Classification Real-time Inference Entry Point.
"""

import os
import json
import dgl
from datetime import datetime as dt

import torch as th
import numpy as np
import pandas as pd

import graphstorm as gs
from graphstorm.config import GSConfig
from argparse import Namespace


# Set seed to ensure prediction results constantly
th.manual_seed(6553865)
np.random.seed(6553865)


# ================== SageMaker real-time entry point functions ================== #

def input_fn(request_body, request_content_type='application/json'):
    """ Preprocessing request_body that is in JSON format.

    According to GraphStorm real-time inference API specification, 
    Parameters
    ----------
    request_body: JSON object
        The JSON object in the request. The JSON object contains the subgraph for inference.
    :param request_content_type:
    :return:
    """
    print('--START processing input data... ')

    # --------------------- receive request ------------------------------------------------ #
    input_data = json.loads(request_body)

    s_t = dt.now()

    version = input_data.get('version', None)
    gml_task = input_data.get('gml_task', None)
    targets = input_data.get('targets', None)
    graph = input_data.get('graph', None)

    print(version)
    print(gml_task)
    task_type = 'node'
    if 'edge' in gml_task or 'link' in gml_task:
        task_type = 'edge'
        target_type = targets[0]['edge_type']
    else:
        target_type = targets[0]['node_type']

    print(f'The type of {task_type} is {target_type}')

    # ==== procssing input graph json to DGL graph with node/edge features ====

    # processing node data
    nodes = graph.get('nodes', None)
    assert nodes is not None, 'Some error code and message here'

    type_node_dfs = {}
    for type_nodes in nodes:
        # processing one type of nodes, generate new interger node id and save
        # the mapping file
        node_df = pd.DataFrame()

        # generating int node ids
        str_org_node_ids = type_nodes['node_ids']
        int_node_ids = np.arange(len(str_org_node_ids))

        node_df['node_id'] = str_org_node_ids
        node_df['nid'] = int_node_ids

        for key, vals in type_nodes['features'].items():
            if key in ['node_type', 'node_ids']:
                continue
            else:
                node_df[key] = vals

        type_node_dfs[type_nodes['node_type']] = node_df

    # processing edge data
    edges = graph.get('edges', None)
    assert edges is not None,  'Some error code and message here'

    type_edge_dfs = {}
    edge_feat_dfs = {}
    for type_edges in edges:
        src_ntype, etype, dst_ntype = type_edges['edge_type']

        # process source and destination node ids to build DGL input edge lists
        edge_df = pd.DataFrame()
        str_org_src_ids = type_edges['src_node_ids']
        str_org_dst_ids = type_edges['dest_node_ids']
        edge_df['src_node_id'] = str_org_src_ids
        edge_df['dst_node_id'] = str_org_dst_ids

        # mapping orginal node ids to int ones
        edge_df = pd.merge(edge_df, type_node_dfs[src_ntype][['node_id','nid']],
                           left_on='src_node_id', right_on='node_id')
        edge_df.rename(columns={'nid': 'src_nid'}, inplace=True)
        edge_df = pd.merge(edge_df, type_node_dfs[dst_ntype][['node_id','nid']],
                           left_on='dst_node_id', right_on='node_id')
        edge_df.rename(columns={'nid': 'dst_nid'}, inplace=True)

        # edge_df = edge_df[['src_nid', 'dst_nid']]
        type_edge_dfs[(src_ntype, etype, dst_ntype)] = (th.from_numpy(edge_df['src_nid'].to_numpy()),
                                                        th.from_numpy(edge_df['dst_nid'].to_numpy()))

        # process edge features if have
        feat_df = pd.DataFrame()
        for key, vals in type_edges['features'].items():
            if key in ['edge_type', 'src_node_ids', 'dest_node_ids']:
                continue
            else:
                feat_df[key] = vals
            if feat_df.shape[0] > 0:
                edge_feat_dfs[(src_ntype, etype, dst_ntype)] = feat_df

    # Build DGL graph and assign features to nodes/edges if have
    dgl_graph = dgl.heterograph(type_edge_dfs)

    for ntype, node_df in type_node_dfs.items():
        nfeat_cols = [col for col in node_df.columns if col not in ['node_id', 'nid']]
        for nfeat_col in nfeat_cols:
            np_vals = np.stack(node_df[nfeat_col].values)
            dgl_graph.nodes[ntype].data[nfeat_col] = th.from_numpy(np_vals)

    for etype, feat_df in edge_feat_dfs.items():
        for efeat_col in feat_df.columns:
            np_vals = np.stack(feat_df[efeat_col].values)
            dgl_graph.edges[etype].data[efeat_col] = th.from_numpy(np_vals)

    # ==== get the target node ids and convert to new node ids ====
    # so far only handle the first target id set.
    target_dict = {}
    if task_type == 'node':
        target_ntype = targets[0]['node_type']
        target_nids = targets[0]['node_ids']
        node_df = type_node_dfs[target_ntype]
        target_df = node_df[node_df['node_id'].isin(target_nids)]
        target_df = target_df[['node_id', 'nid']]
        target_dict['target_ntype'] = target_ntype
        target_dict['target_df'] = target_df
    else:
        pass

    # print(dgl_graph.ndata)
    # print(dgl_graph.edata)

    e_t = dt.now()
    diff_t = int((e_t - s_t).microseconds / 1000)
    print(f'--input_fn: used {diff_t} ms ...')

    return dgl_graph, target_dict


def model_fn(model_dir):
    """
    """
    print('--START model loading... ')

    s_t = dt.now()

    gs.initialize()
    args = Namespace(yaml_config_file=os.path.join(model_dir, 'acm_nc.yaml'), local_rank=0)
    config = GSConfig(args)
    # load the dummy distributed graph
    # TODO: should get the graph name from either user's input argument or by autmatically
    #       extracted from JSON file.
    dummy_g = dgl.distributed.DistGraph('acm', \
                                        part_config=os.path.join(model_dir, 'acm_gs_1p/acm.json'))
    # rebuild the model
    # TODO: should like gsf.py to check what kind of models we need to create
    model = gs.create_builtin_node_gnn_model(dummy_g, config, train_task=False)
    model.restore_model(config.restore_model_path)

    e_t = dt.now()
    diff_t = int((e_t - s_t).microseconds / 1000)
    print(f'--model_fn: used {diff_t} ms ...')

    print(model)

    return model


def predict_fn(input_data, model):
    """ Make prediction
    """
    print('--START model prediction... ')

    s_t = dt.now()

    dgl_graph, target_dict = input_data

    # sample input graph to build blocks
    ntype = target_dict['target_ntype']
    target_df = target_dict['target_df']
    nids = th.from_numpy(target_df['nid'].to_numpy())
    print(nids)
    target_nid = {ntype: nids}

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    size = nids.shape[0]
    dataloader = dgl.dataloading.DataLoader(dgl_graph, target_nid, sampler,
                                            batch_size=size, shuffle=False,
                                            drop_last=False)
    all_nodes = []
    all_blocks = []
    for input_nodes, _, blocks in dataloader:
        all_nodes = input_nodes
        all_blocks = blocks

    nfeat_fields = {'author': ['feat'],
                    'paper': ['feat'],
                    'subject': ['feat']}
    n_h = prepare_batch_input(dgl_graph, all_nodes, feat_field=nfeat_fields)
    e_hs = []
    model.eval()
    logits, _ = model.predict(all_blocks, n_h, e_hs, all_nodes,
                              return_proba=True)
    predictions = logits[ntype][nids].cpu().detach().numpy()

    res = {'target_ntype': ntype,
           'target_nid_raw': target_df['node_id'].to_numpy().tolist(),
           'target_predictions': predictions.tolist()}

    e_t = dt.now()
    diff_t = int((e_t - s_t).microseconds / 1000)
    print(f'--predict_fn: used {diff_t} ms ...')

    return res