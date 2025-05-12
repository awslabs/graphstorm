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
import torch as th

from .transform import parse_feat_ops, process_features, preprocess_features
from .transform import LABEL_STATS_FIELD, collect_label_stats

STATUS = "status_code"
MSG = "message"
GRAPH = "graph"
NODE_MAPPING = "node_mapping"

def prepare_data(input_data, feat_ops):
    """ Prepare node data information for data transformation.

    The function parses a node file that contains node features
    The node file is parsed according to users' configuration
    and transformation related information is extracted.

    Parameters
    ----------
    input_data : dict
        The json payload features input.
    feat_ops : dict of FeatTransform
        The operations run on the node features of the node file.

    Returns
    -------
    dict : A dict of node feature info.
    """
    assert feat_ops is not None, "feat_ops must exist when prepare_data is called."
    feat_info = preprocess_features(input_data, feat_ops)

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
            phase_one_ret[i] = prepare_data(in_file)
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

    if len(conf_list) >= 2:
        collected_features = [
            feature
            for conf in conf_list
            if "features" in conf
            for feature in conf.get("features", [])
        ]
        conf_list[0]["features"] = collected_features
        conf_list = [conf_list[0]]
    return conf_list[0]


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
            type_name = item["edge_type"][0] + "-" + item["edge_type"][1] + "-" + item["edge_type"][2]
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
    for item in final_merged_list:
        if "edge_type" in item:
            item["edge_type"] = item["edge_type"].split("-")
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
    node_id_map = {}
    node_data = {}
    merged_payload_node_conf_list = merge_payload_input(payload_node_conf_list)
    for node_conf in merged_payload_node_conf_list:
        node_type = node_conf["node_type"]
        node_ids = np.array(node_conf["node_id"])
        gconstruct_node_conf = get_conf(gconstruct_node_conf_list, node_type, "Node")
        (feat_ops, two_phase_feat_ops, after_merge_feat_ops, _) = \
            parse_feat_ops(gconstruct_node_conf["features"], gconstruct_node_conf["format"]["name"]) \
                if 'features' in gconstruct_node_conf else (None, [], {}, [])
        input_feat = node_conf["features"]
        for key, val in input_feat.items():
            input_feat[key] = np.array(val)
        prepare_data(input_feat, two_phase_feat_ops)
        # Always do single process feature transformation as there is only payload input
        if feat_ops is not None:
            feat_data, _ = process_features(input_feat, feat_ops, None)
        else:
            feat_data, _ = {}, {}

        for feat_name in list(feat_data):
            merged_feat = np.concatenate(feat_data[feat_name])
            if feat_name in after_merge_feat_ops:
                # do data transformation with the entire feat array.
                merged_feat = \
                    after_merge_feat_ops[feat_name].after_merge_transform(merged_feat)
            feat_data[feat_name] = merged_feat

        node_data[node_type] = feat_data
        raw_node_id_map = {}
        for index, value in enumerate(node_ids):
            if value not in raw_node_id_map:
                raw_node_id_map[value] = index
        node_id_map[node_type] = raw_node_id_map

    return node_id_map, node_data


def map_node_id(str_node_list, node_id_map, node_type):
    """ Mapping node string id into int id
    str_node_list: list
        original node id list
    node_id_map: dict of dict
        {node_type: {str_node_id: int_node_id}}
    node_type: str
        node type
    """
    type_node_id_map = node_id_map[node_type]
    int_node_list = [type_node_id_map.get(val) for val in str_node_list]
    return int_node_list

def process_json_payload_edges(gconstruct_edge_conf_list, payload_edge_conf_list, node_id_map):
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
    edges = {}
    edge_data = {}
    merged_payload_edge_conf_list = merge_payload_input(payload_edge_conf_list)
    for edge_conf in merged_payload_edge_conf_list:
        edge_type = edge_conf["edge_type"]
        src_node_ids = np.array(edge_conf["src_node_id"])
        dest_node_ids = np.array(edge_conf["dest_node_id"])
        gconstruct_edge_conf = get_conf(gconstruct_edge_conf_list, edge_type, "Edge")

        edge_type = tuple(edge_type)
        (feat_ops, two_phase_feat_ops, after_merge_feat_ops, hard_edge_neg_ops) = \
            parse_feat_ops(gconstruct_edge_conf["features"], gconstruct_edge_conf["format"]["name"]) \
                if 'features' in gconstruct_edge_conf else (None, [], {}, [])

        # We don't need to copy all node ID maps to the worker processes.
        # Only the node ID maps of the source node type and destination node type
        # are sufficient.
        id_map = {edge_type[0]: node_id_map[edge_type[0]],
                  edge_type[2]: node_id_map[edge_type[2]]}

        # For edge hard negative transformation ops, more information is needed
        for op in hard_edge_neg_ops:
            op.set_target_etype(edge_type)
            op.set_id_maps(id_map)

        input_feat = edge_conf.get("features", {})
        for key, val in input_feat.items():
            input_feat[key] = np.array(val)
        prepare_data(input_feat, two_phase_feat_ops)

        # Always do single process feature transformation as there is only payload input
        if feat_ops is not None:
            feat_data, _ = process_features(input_feat, feat_ops, None)
        else:
            feat_data, _ = {}, {}

        for feat_name in list(feat_data):
            merged_feat = np.concatenate(feat_data[feat_name])
            if feat_name in after_merge_feat_ops:
                # do data transformation with the entire feat array.
                merged_feat = \
                    after_merge_feat_ops[feat_name].after_merge_transform(merged_feat)
            feat_data[feat_name] = merged_feat
        mapped_src_node_ids = map_node_id(src_node_ids, node_id_map, edge_type[0])
        mapped_dst_node_ids = map_node_id(dest_node_ids, node_id_map, edge_type[2])
        edges[edge_type] = (mapped_src_node_ids, mapped_dst_node_ids)
        edge_data[edge_type] = feat_data

    return edges, edge_data


def verify_payload_conf(request_json_payload, gconstruct_confs):
    """ Verify input json payload

    Parameters:
    request_json_payload: dict
        JSON request payload
    gconstruct_confs: dict
        GConstruct Config
    """
    assert "graph" in request_json_payload, \
        "The JSON request must include a 'graph' definition."
    assert "nodes" in request_json_payload["graph"], \
        "The 'graph' definition in the JSON request must include a 'nodes' field."
    assert "edges" in request_json_payload["graph"], \
        "The 'graph' definition in the JSON request must include a 'edges' field."

    unique_node_types_set = {node['node_type'] for node in gconstruct_confs["nodes"]}
    unique_edge_types_set = {tuple(edge['relation']) for edge in gconstruct_confs["edges"]}

    node_feat_type = set()
    for node_conf in request_json_payload["graph"]["nodes"]:
        assert "node_type" in node_conf, \
            "The 'node' definition in the JSON request must include a 'node_type'"
        node_type = node_conf["node_type"]
        assert node_type in unique_node_types_set, \
            f"The node type {node_type} is not defined in the gconstruct config"
        assert "node_id" in node_conf, \
            "The 'node' definition in the JSON request must include a 'node_id'"
        if "features" in node_conf:
            if node_conf["node_type"] not in node_feat_type:
                node_feat_type.add(node_conf["node_type"])
    assert all(
        node_config.get("node_type") not in node_feat_type or "features" in node_config
        for node_config in request_json_payload["graph"]["nodes"]
    ), ("Validation Failed: Some nodes have the 'features' key "
        "while others of the same type do not.")

    edge_feat_type = set()
    for edge_conf in request_json_payload["graph"]["edges"]:
        assert "edge_type" in edge_conf, \
            "The 'edge_type' definition in the JSON request must include a 'edge_type'"
        edge_type = edge_conf["edge_type"]
        assert tuple(edge_type) in unique_edge_types_set, \
            f"The edge type {edge_type} is not defined in the gconstruct config"
        assert "src_node_id" in edge_conf, \
            "The 'edge' definition in the JSON request must include a 'src_node_id'"
        assert "dest_node_id" in edge_conf, \
            "The 'edge' definition in the JSON request must include a 'dest_node_id'"
        if "features" in edge_conf:
            if tuple(edge_conf["edge_type"]) not in edge_feat_type:
                edge_feat_type.add(tuple(edge_conf["edge_type"]))

    assert all(
        edge_config.get("node_type") not in edge_feat_type or "features" in edge_config
        for edge_config in request_json_payload["graph"]["edges"]
    ), ("Validation Failed: Some edges have the 'features' key "
        "while others of the same type do not.")

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
        verify_payload_conf(json_payload_confs, gconstruct_confs)
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
                                                  json_payload_confs["graph"]["edges"], raw_node_id_maps)
    except AssertionError as ae:
        error_message = str(ae)
        return {STATUS: 400, MSG: error_message}

    g = dgl.heterograph(edges, num_nodes_dict=num_nodes)

    # Assign node/edge features
    for ntype in node_data:
        for name, ndata in node_data[ntype].items():
            g.nodes[ntype].data[name] = th.tensor(ndata)
    for etype in edge_data:
        for name, edata in edge_data[etype].items():
            g.edges[etype].data[name] = th.tensor(edata)

    return {STATUS: 200, MSG: "successful build payload graph",
            GRAPH: g, NODE_MAPPING: raw_node_id_maps}
