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
from .transform import parse_feat_ops, process_features, preprocess_features
from .utils import verify_confs

def prepare_node_data(in_file, feat_ops, read_file):
    """ Prepare node data information for data transformation.

    The function parses a node file that contains node features
    The node file is parsed according to users' configuration
    and transformation related information is extracted.

    Parameters
    ----------
    in_file : str
        The path of the input node file.
    feat_ops : dict of FeatTransform
        The operations run on the node features of the node file.
    read_file : callable
        The function to read the node file

    Returns
    -------
    dict : A dict of node feature info.
    """
    data = read_file(in_file)
    assert feat_ops is not None, "feat_ops must exist when prepare_node_data is called."
    feat_info = preprocess_features(data, feat_ops)

    return feat_info

def _process_data(user_parser,
                  two_phase_feat_ops,
                  task_info,
                  ext_mem_workspace):
    """ Process node and edge data.

    Parameter
    ---------
    user_pre_parser: func
        A function that prepares data for processing.
    user_parser: func
        A function that processes node data or edge data.
    two_phase_feat_ops: list of TwoPhaseFeatTransform
        List of TwoPhaseFeatTransform transformation ops.
    task_info: str
        Task meta info for debugging.
    """
    if len(two_phase_feat_ops) > 0:
        pre_parse_start = time.time()
        phase_one_ret = {}
        for i, in_file in enumerate(in_files):
            phase_one_ret[i] = prepare_node_data(in_file)
        update_two_phase_feat_ops(phase_one_ret, two_phase_feat_ops)

        dur = time.time() - pre_parse_start
        logging.debug("Preprocessing data files for %s takes %.3f seconds.",
                      task_info, dur)

    start = time.time()
    return_dict = multiprocessing_data_read(in_files, num_proc, user_parser,
                                            ext_mem_workspace)
    dur = time.time() - start
    logging.debug("Processing data files for %s takes %.3f seconds.",
                    task_info, dur)
    return return_dict


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

        node_type = node_conf["node_type"]
        gconstruct_node_conf = gconstruct_node_conf_list[node_type]
        (feat_ops, two_phase_feat_ops, after_merge_feat_ops, _) = \
            parse_feat_ops(gconstruct_node_conf["features"], gconstruct_node_conf["format"]["name"]) \
                if 'features' in gconstruct_node_conf else (None, [], {}, [])

        # Always do single process feature transformation as there is only payload input
    return {}, np.array([0, 0])


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
    raw_node_id_maps, node_data = process_json_payload_nodes(gconstruct_confs["nodes"],
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
            g.nodes[ntype].data[name] = th.tensor(ndata)
    for etype in edge_data:
        for name, edata in edge_data[etype].items():
            g.edges[etype].data[name] = th.tensor(edata)
    dgl.save_graphs(os.path.join(args.output_dir, args.graph_name + ".dgl"), [g])