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
import numpy as np
import dgl

from .transform import parse_feat_ops, process_features, preprocess_features

STATUS = "status_code"
MSG = "message"
GRAPH = "graph"
NODE_MAPPING = "node_mapping"

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


def get_conf(gconstruct_conf_list, type_name, structure_type):
    """ Retrieve node/edge type gconstruct config

    Paramters:
        gconstruct_conf_list: dict
            GConstruct Config Dict for either node or edge
        type_name: str
            Node/Edge Name
        structure_type: str
            One of "Node" and "Edges"
    """
    conf_list = []
    if structure_type == "Node":
        col_name = "node_type"
    elif structure_type == "Edge":
        col_name = "relation"
    else:
        raise ValueError("Expect structure_type be one of Node/Edge")
    for conf in gconstruct_conf_list:
        if conf[col_name] == type_name:
            conf_list.append(conf)

    return conf_list


def merge_payload_input(payload_input_list):
    """Merge the payload input within the same node/edge type

    Parameters:
        payload_input_list: list[dict]
            input payload list
    """
    merged_data_temp = {}

    for item in payload_input_list:
        element_category = None
        type_name = None
        current_ids = {}
        output_type_field = None

        if "node_type" in item and "node_id" in item:
            element_category = "node"
            type_name = item["node_type"]
            output_type_field = "node_type"
            current_ids["node_id"] = item["node_id"]
        elif "edge_type" in item and "src_node_id" in item and "dest_node_id" in item:
            element_category = "edge"
            type_name = item["edge_type"]
            output_type_field = "edge_type"
            current_ids["src_node_id"] = item["src_node_id"]
            current_ids["dest_node_id"] = item["dest_node_id"]

        grouping_key = (element_category, type_name)

        if grouping_key not in merged_data_temp:
            merged_entry = {output_type_field: type_name}
            for id_key, id_val in current_ids.items():
                merged_entry[id_key] = [id_val]

            if "features" in item and isinstance(item["features"], dict) and item["features"]:
                merged_entry["features"] = {}
                for feature_key, feature_value_list in item["features"].items():
                    merged_entry["features"][feature_key] = [feature_value_list]
            merged_data_temp[grouping_key] = merged_entry
        else:
            for id_key, id_val in current_ids.items():
                if id_key in merged_data_temp[grouping_key]:
                     merged_data_temp[grouping_key][id_key].append(id_val)

            if "features" in merged_data_temp[grouping_key] and \
               "features" in item and isinstance(item["features"], dict):
                for feature_key, feature_value_list in item["features"].items():
                    if feature_key in merged_data_temp[grouping_key]["features"]:
                        merged_data_temp[grouping_key]["features"][feature_key].append(feature_value_list)

    final_merged_list = list(merged_data_temp.values())
    return final_merged_list

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
    node_data_dict = {}
    merged_payload_node_conf_list = merge_payload_input(payload_node_conf_list)
    print(merged_payload_node_conf_list)
    exit(-1)
    for node_conf in merged_payload_node_conf_list:
        node_type = node_conf["node_type"]
        gconstruct_node_conf = get_conf(gconstruct_node_conf_list, node_type, "Node")
        # (feat_ops, two_phase_feat_ops, after_merge_feat_ops, _) = \
        #     parse_feat_ops(gconstruct_node_conf["features"], gconstruct_node_conf["format"]["name"]) \
        #         if 'features' in gconstruct_node_conf else (None, [], {}, [])

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
    # for edge_conf in payload_edge_conf_list:

    return np.array([0, 0])


def verify_payload_conf(request_json_payload):
    """ Verify input json payload

    Parameters:
    request_json_payload: dict
        JSON request payload
    """
    assert "graph" in request_json_payload, \
        "The JSON request must include a 'graph' definition."
    assert "nodes" in request_json_payload["graph"], \
        "The 'graph' definition in the JSON request must include a 'nodes' field."
    assert "edges" in request_json_payload["graph"], \
        "The 'graph' definition in the JSON request must include a 'edges' field."

    for node_conf in request_json_payload["graph"]["nodes"]:
        assert "node_type" in node_conf, \
            "The 'node' definition in the JSON request must include a 'node_type'"
        assert "node_id" in node_conf, \
            "The 'node' definition in the JSON request must include a 'node_id'"

    for edge_conf in request_json_payload["graph"]["edges"]:
        assert "edge_type" in edge_conf, \
            "The 'edge_type' definition in the JSON request must include a 'edge_type'"
        assert "src_node_id" in edge_conf, \
            "The 'edge' definition in the JSON request must include a 'src_node_id'"
        assert "dest_node_id" in edge_conf, \
            "The 'edge' definition in the JSON request must include a 'dest_node_id'"

    # TODO: Check if all the nodes/edges in one type have features or not have features
    # TODO: Check if all the node/edge types are in the gconstruct definition
    return True


def process_json_payload_graph(request_json_payload, gconstruct_config):
    """ Construct DGLGraph from json payload.


    Parameters:
    request_json_payload: dict
        Input json payload request

    gconstruct_config: dict
        Input Gconstruct config file
    """
    with open(gconstruct_config, 'r', encoding="utf8") as json_file:
        gconstruct_confs = json.load(json_file)

    with open(request_json_payload, 'r', encoding="utf8") as json_file:
        json_payload_confs = json.load(json_file)

    # Verify JSON payload request
    try:
        verify_payload_conf(json_payload_confs)
    except AssertionError as ae:
        error_message = str(ae)
        return {STATUS: 400, MSG: error_message}

    # Process Node Data
    try:
        raw_node_id_maps, node_data = process_json_payload_nodes(gconstruct_confs["nodes"],
                                               json_payload_confs["graph"]["nodes"])
        num_nodes = {ntype: len(raw_node_id_maps[ntype]) for ntype in raw_node_id_maps}
    except AssertionError as ae:
        error_message = str(ae)
        return {STATUS: 400, MSG: error_message}

    # Process Edge Data
    try:
        edges, edge_data = process_json_payload_edges(gconstruct_confs["edges"],
                                                  json_payload_confs["graph"]["edges"])
    except AssertionError as ae:
        error_message = str(ae)
        return {STATUS: 400, MSG: error_message}

    g = None
    # g = dgl.heterograph(edges, num_nodes_dict=num_nodes)
    #
    # # Assign node/edge features
    # for ntype in node_data:
    #     for name, ndata in node_data[ntype].items():
    #         g.nodes[ntype].data[name] = th.tensor(ndata)
    # for etype in edge_data:
    #     for name, edata in edge_data[etype].items():
    #         g.edges[etype].data[name] = th.tensor(edata)

    return {STATUS: 200, MSG: "successful build payload graph",
            GRAPH: g, NODE_MAPPING: raw_node_id_maps}
