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
import numpy as np
import dgl
import torch as th

from .transform import parse_feat_ops, process_features, preprocess_features
from .utils import update_two_phase_feat_ops

STATUS = "status_code"
MSG = "message"
GRAPH = "graph"
NODE_MAPPING = "node_mapping"
PROCESS_INDEX = 0

def prepare_data(input_data, feat_ops):
    """ Prepare phase one data for two-phase feature transformation.

    The function parses a payload input that contains node/edge features.
    The payload is parsed according to original gconstruct configuration
    and transformation.

    Parameters
    ----------
    input_data : dict
        The json payload features input.
    feat_ops : dict of FeatTransform
        The operations run on the node features of the node file.

    Returns
    -------
    dict : A dict of feature info.
    """
    assert feat_ops is not None, "feat_ops must exist when prepare_data is called."
    feat_info = preprocess_features(input_data, feat_ops)

    # update_two_phase_feat_ops function expects an input dictionary indexed by integers
    # e.g {chunk_index: chunk_data}
    # Here we are only do single process feature processing,
    # so we need to add a dummy index to the return dictionary
    return {PROCESS_INDEX: feat_info}


def get_conf(gconstruct_conf_list, type_name, structure_type):
    """ Retrieve node/edge type gconstruct config. Will combine all feature configuration
    for one node/edge type

    For either GConstruct Node/Edge, feature transformation can be defined in multiple blocks,
    e.g
    [{
        "node_id_col": "id",
        "node_type": "user",
        "features":     [
            {
                "feature_col":  "feat_one"
            }
        ]
    },
    {
        "node_type": "user",
        "features":     [
            {
                "feature_col":  "feat_two"
            }
        ]
    }]

    The function is expected to gather all the feature transformation into the first
    node/edge config definition and return. For the above example, it will be:
    [{
        "node_id_col": "id",
        "node_type": "user",
        "features":     [
            {
                "feature_col":  "feat_one"
            },
            {
                "feature_col":  "feat_two"
            }
        ]
    }]

    Parameters:
        gconstruct_conf_list: dict
            GConstruct Config Dict for either node or edge
        type_name: str
            Node/Edge Type Name
        structure_type: str
            One of "Node" or "Edge"
    Return:
        dict: merged gconstruct config
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

    # Features may be defined in multiple block for one node/edge type
    if len(conf_list) >= 2:
        # Collect all feature transformation block into the first block
        collected_features = []
        for conf in conf_list:
            if "features" in conf:
                for feature in conf["features"]:
                    collected_features.append(feature)
        conf_list[0]["features"] = collected_features
        conf_list = [conf_list[0]]
    # if there is only one node/edge type definition block,
    # no need to merge feature transformation config.
    elif len(conf_list) == 1:
        pass
    else:
        raise ValueError(f"Expect one node/edge type definition for {type_name}")
    return conf_list[0]


def merge_payload_input(payload_input_list):
    """Merge the payload input within the same node/edge type

    There may be multiple node/edge definitions within one node/edge type. For example:

    [{
        "node_type":    "user",
        "node_id":      "u1",
        "features":     {
            "feat": [feat_val1]
        }
    },
    {
        "node_type":    "user",
        "node_id":      "u2",
        "features":     {
            "feat": [feat_val2]
        }
    },
    ]

    This function is expected to group all the blocks with the same node_type/edge_type
    to fit in the gconstruct feature transformation input.
    The above return should be like:

    [{
        "node_type":    "user",
        "node_id":      ["u1", "u2"]
        "features":     {
            "feat": [[feat_val1], [feat_val2]]
        }
    }]
    Parameters:
        payload_input_list: list of dict
            input payload
    Return:
        dict: merged payload input
    """
    merged_data_temp = {}

    for item in payload_input_list:
        structure_type = None
        type_name = None
        current_ids = {}
        col_name = None

        # Merge Node/Edge IDs
        if "node_type" in item and "node_id" in item:
            structure_type = "node"
            type_name = item["node_type"]
            col_name = "node_type"
            current_ids["node_id"] = item["node_id"]
        elif "edge_type" in item and "src_node_id" in item and "dest_node_id" in item:
            structure_type = "edge"
            # List can not be hashed
            type_name = (item["edge_type"][0] + "-" + item["edge_type"][1] +
                         "-" + item["edge_type"][2])
            col_name = "edge_type"
            current_ids["src_node_id"] = item["src_node_id"]
            current_ids["dest_node_id"] = item["dest_node_id"]

        grouping_key = (structure_type, type_name)

        if grouping_key not in merged_data_temp:
            merged_entry = {col_name: type_name}
            for id_key, id_val in current_ids.items():
                merged_entry[id_key] = [id_val]

            # Merge features if necessary
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
                        merged_data_temp[grouping_key]["features"][feature_key].append(
                            feature_value_list
                        )

    final_merged_list = list(merged_data_temp.values())
    # Convert Edge Type back to a list
    for item in final_merged_list:
        if "edge_type" in item:
            item["edge_type"] = item["edge_type"].split("-")
    return final_merged_list


def process_json_payload_nodes(gconstruct_node_conf_list, payload_node_conf_list):
    """ Process json payload node input

    We need to process all node data before we can process edge data. Return node id mapping
    and node feature data.

    The node conf in the payload is defined as follows:
    {
        "node_type":    "<node type>",
        "node_id":      "<node id>",
        "features":     {
            "<feat_name>": [feat_val]
        }
    }
    Return:
        node_id_map: {str_node_id: int_node_id}
        node_data: {nfeat_name: edge_feat_np_array}
    """
    node_id_map = {}
    node_data = {}
    merged_payload_node_conf_list = merge_payload_input(payload_node_conf_list)
    for node_conf in merged_payload_node_conf_list:
        node_type = node_conf["node_type"]
        node_ids = np.array(node_conf["node_id"])
        gconstruct_node_conf = get_conf(gconstruct_node_conf_list, node_type, "Node")
        (feat_ops, two_phase_feat_ops, after_merge_feat_ops, _) = \
            parse_feat_ops(gconstruct_node_conf["features"],
                           gconstruct_node_conf["format"]["name"]) \
                if 'features' in gconstruct_node_conf else (None, [], {}, [])
        # Always do single process feature transformation as there is only one payload input
        if feat_ops is not None:
            assert "features" in node_conf, \
                "features need to be defined in the payload"
            input_feat = node_conf["features"]
            # Input features raw data should be numpy array type
            for key, val in input_feat.items():
                input_feat[key] = np.array(val)
            if len(two_phase_feat_ops) > 0:
                phase_one_ret = prepare_data(input_feat, two_phase_feat_ops)
                update_two_phase_feat_ops(phase_one_ret, two_phase_feat_ops)
            feat_data, _ = process_features(input_feat, feat_ops, None)
        else:
            feat_data = {}
        for feat_name in list(feat_data):
            if feat_name in after_merge_feat_ops:
                # do data transformation with the entire feat array.
                merged_feat = \
                    after_merge_feat_ops[feat_name].after_merge_transform(feat_data[feat_name])
                feat_data[feat_name] = merged_feat
        node_data[node_type] = feat_data
        # Avoid using id_map class as we do not save to the disk
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

    The edge conf in the edge payload json file could be like following. Return
    edges info definition and edge feature data.

    {
        "edge_type": "<edge type>",
        "src_node_id": <src edge id>,
        "dest_node_id": <dest edge id>
        "features: {
            "<feat_name>": [feat_val]
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

        # List is not hashable
        edge_type = tuple(edge_type)
        (feat_ops, two_phase_feat_ops, after_merge_feat_ops, hard_edge_neg_ops) = \
            parse_feat_ops(gconstruct_edge_conf["features"],
                           gconstruct_edge_conf["format"]["name"]) \
                if 'features' in gconstruct_edge_conf else (None, [], {}, [])

        id_map = {edge_type[0]: node_id_map[edge_type[0]],
                  edge_type[2]: node_id_map[edge_type[2]]}

        # For edge hard negative transformation ops, more information is needed
        for op in hard_edge_neg_ops:
            op.set_target_etype(edge_type)
            op.set_id_maps(id_map)

        # Always do single process feature transformation as there is only payload input
        if feat_ops is not None:
            assert "features" in edge_conf, \
                "Features need to be defined in the payload"
            input_feat = edge_conf["features"]
            # Input features raw data should be numpy array type
            for key, val in input_feat.items():
                input_feat[key] = np.array(val)
            if len(two_phase_feat_ops) > 0:
                phase_one_ret = prepare_data(input_feat, two_phase_feat_ops)
                update_two_phase_feat_ops(phase_one_ret, two_phase_feat_ops)
            feat_data, _ = process_features(input_feat, feat_ops, None)
        else:
            feat_data = {}
        for feat_name in list(feat_data):
            if feat_name in after_merge_feat_ops:
                # do data transformation with the entire feat array.
                merged_feat = \
                    after_merge_feat_ops[feat_name].after_merge_transform(feat_data[feat_name])
                feat_data[feat_name] = merged_feat
        mapped_src_node_ids = map_node_id(src_node_ids, node_id_map, edge_type[0])
        mapped_dst_node_ids = map_node_id(dest_node_ids, node_id_map, edge_type[2])
        edges[edge_type] = (mapped_src_node_ids, mapped_dst_node_ids)
        edge_data[edge_type] = feat_data

    return edges, edge_data


def verify_payload_conf(request_json_payload, gconstruct_confs):
    """ Verify input json payload.

    The json payload is expected to have input format like:
    {
        "version": "gs-realtime-v0.1",
        "gml_task": "node_classification",
        "graph": {
            "nodes": [{node_payload_definition}]，
            "edges": [{edge_payload_definition}]
        }
    }

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
        tuple(edge_config.get("edge_type")) not in edge_feat_type or "features" in edge_config
        for edge_config in request_json_payload["graph"]["edges"]
    ), ("Validation Failed: Some edges have the 'features' key "
        "while others of the same type do not.")

    return True


def process_json_payload_graph(request_json_payload, gconstruct_config):
    """ Construct DGLGraph from json payload.

    The json payload is expected to have input format like:
    {
        "version": "gs-realtime-v0.1",
        "gml_task": "node_classification",
        "graph": {
            "nodes": [{node_payload_definition}]，
            "edges": [{edge_payload_definition}]
        }
    }

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
    except AssertionError as assert_error:
        error_message = str(assert_error)
        return {STATUS: 400, MSG: error_message}

    # Process Node Data
    try:
        raw_node_id_maps, node_data = process_json_payload_nodes(gconstruct_confs["nodes"],
                                               json_payload_confs["graph"]["nodes"])
        num_nodes = {ntype: len(raw_node_id_maps[ntype]) for ntype in raw_node_id_maps}
    except AssertionError as assert_error:
        error_message = str(assert_error)
        return {STATUS: 400, MSG: error_message}

    # Process Edge Data
    try:
        edges, edge_data = process_json_payload_edges(gconstruct_confs["edges"],
                                json_payload_confs["graph"]["edges"], raw_node_id_maps)
    except AssertionError as assert_error:
        error_message = str(assert_error)
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
