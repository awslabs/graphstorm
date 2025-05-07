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
"""
import json
import logging


from ..utils import (sys_tracker, get_log_level, check_graph_name)
from .utils import verify_confs


def process_json_payload_nodes(gconstruct_node_conf_list, payload_node_conf_list):
    """ Process json payload node data

    We need to process all node data before we can process edge data.

    The node conf in the payload is defined as follows:
    {
        "node_type":    "<node type>",
        "node_id":      "<node id>",
        "features":     {
            "<feat_name>": [feat_num]
        }
    }
    Return:
        node_data: {nfeat_name: node_feat_np_array}
    """
    for node_conf in payload_node_conf_list:
        assert "node_type" in node_conf, "node type must be defined in the config"
        assert "node_id" in node_conf, "node id must be defined in the config"
    return np.array([0, 0])


def process_json_payload_edges(gconstruct_edge_conf_list, payload_edge_conf_list):
    """ Process json payload edge data

    The edge conf in the edge payload json file could be:

    {
        "edge_type": "<edge type>",
        "src_node_id": <src edge id>,
        "dest_node_id": <dst edge id>
        "features: {
            "<feat_name>": [feat_num]
        }
    }

    Return:
        edges: {etype: (src_np_array, dst_np_array)}
        edge_data: {efeat_name: edge_feat_np_array}
    """
    for edge_conf in payload_edge_conf_list:
        assert "edge_type" in edge_conf, "edge type must be defined in the config"
        assert "edge_id" in edge_conf, "edge id must be defined in the config"
    return np.array([0, 0])


def process_json_payload_graph(args):
    """ Construct DGLGraph from json payload.
    """
    check_graph_name(args.graph_name)
    logging.basicConfig(level=get_log_level(args.logging_level))

    with open(args.conf_file, 'r', encoding="utf8") as json_file:
        gconstruct_confs = json.load(json_file)

    with open(args.json_payload_file, 'r', encoding="utf8") as json_file:
        json_payload_confs = json.load(json_file)

    sys_tracker.set_rank(0)
    num_processes_for_nodes = args.num_processes_for_nodes \
            if args.num_processes_for_nodes is not None else args.num_processes
    num_processes_for_edges = args.num_processes_for_edges \
            if args.num_processes_for_edges is not None else args.num_processes
    print(num_processes_for_nodes, num_processes_for_edges)
    verify_confs(gconstruct_confs)

    output_format = args.output_format
    if len(output_format) != 1 and output_format[0] != "DGL":
        logging.warning("We only support building DGLGraph for json payload")
    node_data = process_json_payload_nodes(gconstruct_confs["nodes"],
                                           json_payload_confs["graph"]["nodes"])
    sys_tracker.check('Process the node data')

    edges, edge_data = process_json_payload_edges(gconstruct_confs["edges"],
                                                  json_payload_confs["graph"]["edges"])
    sys_tracker.check('Process the edge data')

    os.makedirs(args.output_dir, exist_ok=True)

    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    sys_tracker.check('Construct DGL graph')

    for ntype in node_data:
        for name, ndata in node_data[ntype].items():
            if isinstance(ndata, ExtMemArrayWrapper):
                g.nodes[ntype].data[name] = ndata.to_tensor()
            else:
                g.nodes[ntype].data[name] = th.tensor(ndata)
    for etype in edge_data:
        for name, edata in edge_data[etype].items():
            if isinstance(edata, ExtMemArrayWrapper):
                g.edges[etype].data[name] = edata.to_tensor()
            else:
                g.edges[etype].data[name] = th.tensor(edata)
    dgl.save_graphs(os.path.join(args.output_dir, args.graph_name + ".dgl"), [g])