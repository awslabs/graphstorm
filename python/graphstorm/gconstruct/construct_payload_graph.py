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
import numpy as np
import dgl
import torch as th

from .transform import parse_feat_ops, process_features, preprocess_features
from .utils import update_two_phase_feat_ops
from .payload_utils import (BaseApplicationError, MissingValError,
                            InvalidFeatTypeError,
                            DGLCreateError, MisMatchedTypeError,
                            MissingKeyError, MisMatchedFeatureError)

PAYLOAD_PROCESSING_STATUS = "status_code"
PAYLOAD_PROCESSING_ERROR_CODE = "error_code"
PAYLOAD_PROCESSING_RETURN_MSG = "message"
PAYLOAD_GRAPH = "graph"
PAYLOAD_GRAPH_NODE_MAPPING = "node_mapping"
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
    if feat_ops is None:
        raise MissingValError(value_name="feat_ops",
                              input_name="prepare_data function call")
    feat_info = preprocess_features(input_data, feat_ops)

    # update_two_phase_feat_ops function expects an input dictionary indexed by integers
    # e.g {chunk_index: chunk_data}
    # Here we are only do single process feature processing,
    # so we need to add a dummy index to the return dictionary
    return {PROCESS_INDEX: feat_info}


def get_gconstruct_conf(gconstruct_conf_list, type_name, structure_type):
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
    ------------
        gconstruct_conf_list: dict
            GConstruct Config Dict for either node or edge
        type_name: str
            Node/Edge Type Name
        structure_type: str
            One of "Node" or "Edge"
    Return:
    -------
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

    According to GraphStorm endpoint payload specification, nodes and edges are stored in a list,
    in which each node or edge is individually a dictionary. For one node or edge type there could
    be multiple nodes or edges. This function is expected to group all the nodes or edges with the
    same node or edge type to fit in the gconstruct feature transformation input format.
    
    For example:

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

    Return value of the above example would be:

    [{
        "node_type":    "user",
        "node_id":      ["u1", "u2"]
        "features":     {
            "feat": [[feat_val1], [feat_val2]]
        }
    }]

    Parameters:
    ------------
        payload_input_list: list of dict
            input payload
    Return:
    -------
        dict: merged payload input.
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
            type_name = (item["edge_type"][0] + "<>" + item["edge_type"][1] +
                         "<>" + item["edge_type"][2])
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
            item["edge_type"] = item["edge_type"].split("<>")

    return final_merged_list


def process_json_payload_nodes(gconstruct_node_conf_list, payload_node_conf_list):
    """ Process json payload node input

    Nodes must be processed before edges, as edges need access to the node mappings.
    This function initializes all unique node mappings and performs feature transformations.

    The node conf in the payload is defined as follows:
    {
        "node_type":    "<node type>",
        "node_id":      "<node id>",
        "features":     {
            "<feat_name>": [feat_val]
        }
    }

    Parameters
    ----------
        gconstruct_node_conf_list: dict
            GConstruct node configuration.
        payload_node_conf_list: dict
            Payload node configuration.

    Returns
    -------
        node_id_map: {str_node_id: int_node_id}
            A mapping from original node string ID to unique integer IDs.
        node_data: {nfeat_name: edge_feat_np_array}
            A structured collection of transformed features for each feature name.
    """
    node_id_map = {}
    node_data = {}
    merged_payload_node_conf_list = merge_payload_input(payload_node_conf_list)
    for node_conf in merged_payload_node_conf_list:
        node_type = node_conf["node_type"]
        node_ids = np.array(node_conf["node_id"])
        gconstruct_node_conf = get_gconstruct_conf(gconstruct_node_conf_list, node_type, "Node")
        (feat_ops, two_phase_feat_ops, after_merge_feat_ops, _) = \
            parse_feat_ops(gconstruct_node_conf["features"],
                           gconstruct_node_conf["format"]["name"]) \
                if 'features' in gconstruct_node_conf else (None, [], {}, [])
        # Always do single process feature transformation as there is only one payload input
        if feat_ops is not None:
            if "features" not in node_conf:
                raise MissingValError("features", "node payload")
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
    Parameter:
        str_node_list: list
            The original string type node id list.
        node_id_map: dict of dict
            The node id mapping for each node type in the format of
            {node_type: {str_node_id: int_node_id}}.
        node_type: str
            The node type string.
    Return:
    -------
        int_node_list: list
            Mapping result in integer node id list.
    """
    type_node_id_map = node_id_map[node_type]
    int_node_list = [type_node_id_map.get(val) for val in str_node_list]
    if None in int_node_list:
        raise MissingValError("src_node_ids/dest_node_ids", "Node Mapping")
    return int_node_list


def process_json_payload_edges(gconstruct_edge_conf_list, payload_edge_conf_list, node_id_map):
    """ Process json payload edge data
     1. Maps 'src_node_id' and 'dest_node_id' to integer IDs and stores edges information.
     2. Transforms the input edge features based on the configuration.
     Returns the edge information definition and the transformed edge feature data.

    {
        "edge_type": "<edge type>",
        "src_node_id": <src edge id>,
        "dest_node_id": <dest edge id>
        "features: {
            "<feat_name>": [feat_val]
        }
    }

    Parameters:
    ------------
        gconstruct_edge_conf_list: dict
            GConstruct node configuration.
        payload_edge_conf_list: dict
            Payload Edge configuration.
        node_id_map: dict
            A mapping from original node string ID to unique integer IDs.
    Returns:
    -------
        edges: {etype: (src_np_array, dst_np_array)}
            Edges information with mapped integer IDs.
        edge_data: {efeat_name: edge_feat_np_array}
            Edge features with transformed data.
    """
    edges = {}
    edge_data = {}
    merged_payload_edge_conf_list = merge_payload_input(payload_edge_conf_list)
    for edge_conf in merged_payload_edge_conf_list:
        edge_type = edge_conf["edge_type"]
        src_node_ids = np.array(edge_conf["src_node_id"])
        dest_node_ids = np.array(edge_conf["dest_node_id"])
        gconstruct_edge_conf = get_gconstruct_conf(gconstruct_edge_conf_list, edge_type, "Edge")

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
            if "features" not in edge_conf:
                raise MissingValError("features", "edge payload")
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
                merged_feat = (
                    after_merge_feat_ops[feat_name]
                        .after_merge_transform(feat_data[feat_name])
                )
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
    ------------
    request_json_payload: dict
        JSON request payload.
    gconstruct_confs: dict
        GConstruct configuration JSON object.
    """
    if "graph" not in request_json_payload:
        raise MissingKeyError("graph", "JSON request payload")
    if "nodes" not in request_json_payload["graph"]:
        raise MissingKeyError("nodes", "JSON request payload graph field")
    if "edges" not in request_json_payload["graph"]:
        raise MissingKeyError("edges", "JSON request payload graph field")

    unique_node_types_set = {node['node_type'] for node in gconstruct_confs["nodes"]}
    unique_edge_types_set = {tuple(edge['relation']) for edge in gconstruct_confs["edges"]}

    node_feature_keys_by_type = {}
    for node_conf in request_json_payload["graph"]["nodes"]:
        if "node_type" not in node_conf:
            raise MissingKeyError("node_type", "JSON request payload graph nodes field")
        node_type = node_conf["node_type"]
        if node_type not in unique_node_types_set:
            raise MisMatchedTypeError(structure_type="node type", type_name=node_type)
        if "node_id" not in node_conf:
            raise MissingKeyError("node_id", "JSON request payload graph nodes field")
        if "features" in node_conf:
            if not isinstance(node_conf["features"], dict):
                raise InvalidFeatTypeError()
            # Expect all node features block have the same feature schema
            current_node_feature_keys = set(node_conf["features"].keys())
            if node_type not in node_feature_keys_by_type:
                node_feature_keys_by_type[node_type] = current_node_feature_keys
            else:
                expected_keys = node_feature_keys_by_type[node_type]
                if current_node_feature_keys != expected_keys:
                    raise MisMatchedFeatureError(structural_type="node type",
                                                id_name=node_conf.get('node_id', 'N/A'),
                                                expected_keys=sorted(list(expected_keys)),
                                                actual_keys=sorted(list(current_node_feature_keys)))
    for node_config in request_json_payload["graph"]["nodes"]:
        node_type = node_config.get("node_type")
        if node_type in node_feature_keys_by_type and "features" not in node_config:
            raise MissingKeyError("features",
                        f"certain JSON request payload graph nodes field for node type {node_type}")

    edge_feature_keys_by_type = {}
    for edge_conf in request_json_payload["graph"]["edges"]:
        if "edge_type" not in edge_conf:
            raise MissingKeyError("edge_type", "JSON request payload graph edges field")
        edge_type = edge_conf["edge_type"]
        if tuple(edge_type) not in unique_edge_types_set:
            raise MisMatchedTypeError(structure_type="edge type", type_name=tuple(edge_type))
        if "src_node_id" not in edge_conf:
            raise MissingKeyError("src_node_id", "JSON request payload graph edges field")
        if "dest_node_id" not in edge_conf:
            raise MissingKeyError("dest_node_id", "JSON request payload graph edges field")
        src_node_id, dest_node_id = edge_conf["src_node_id"], edge_conf["dest_node_id"]
        if "features" in edge_conf:
            if not isinstance(edge_conf["features"], dict):
                raise InvalidFeatTypeError()
            # Expect all node features block have the same feature schema
            current_edge_feature_keys = set(edge_conf["features"].keys())
            if tuple(edge_type) not in edge_feature_keys_by_type:
                edge_feature_keys_by_type[tuple(edge_type)] = current_edge_feature_keys
            else:
                expected_keys = edge_feature_keys_by_type[tuple(edge_type)]
                if current_edge_feature_keys != expected_keys:
                    raise MisMatchedFeatureError(structural_type="edge type",
                                             id_name=(src_node_id, dest_node_id),
                                             expected_keys=sorted(list(expected_keys)),
                                             actual_keys=sorted(list(current_edge_feature_keys)))

    for edge_config in request_json_payload["graph"]["edges"]:
        edge_type = tuple(edge_config.get("edge_type"))
        if edge_type in edge_feature_keys_by_type and "features" not in edge_config:
            raise MissingKeyError("features",
                    "certain JSON request payload graph edges field "
                    f"for edge type {tuple(edge_type)}")


def process_json_payload_graph(request_json_payload, graph_construct_config):
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
    ------------
    request_json_payload: dict
        Json payload in request.
    graph_construct_config: dict
        Input Gconstruct config file
        
    Returns:
    --------
    dict
        {
            STATUS: HTTP Response Code
            MSG: Response Message
            GRAPH: Built DGLGraph
            NODE_MAPPING: str-to-int node id mapping
        }
    """
    # Verify JSON payload request
    try:
        verify_payload_conf(request_json_payload, graph_construct_config)
    except BaseApplicationError as payload_error:
        error_message = str(payload_error)
        error_code = payload_error.get_error_code()
        return {PAYLOAD_PROCESSING_STATUS: 400,
                PAYLOAD_PROCESSING_ERROR_CODE: error_code,
                PAYLOAD_PROCESSING_RETURN_MSG: error_message}

    # Process Node Data
    try:
        raw_node_id_maps, node_data = process_json_payload_nodes(graph_construct_config["nodes"],
                                               request_json_payload["graph"]["nodes"])
        num_nodes_dict = {ntype: len(raw_node_id_maps[ntype]) for ntype in raw_node_id_maps}
    except BaseApplicationError as payload_error:
        error_message = str(payload_error)
        error_code = payload_error.get_error_code()
        return {PAYLOAD_PROCESSING_STATUS: 400,
                PAYLOAD_PROCESSING_ERROR_CODE: error_code,
                PAYLOAD_PROCESSING_RETURN_MSG: error_message}

    # Process Edge Data
    try:
        edges, edge_data = process_json_payload_edges(graph_construct_config["edges"],
                                request_json_payload["graph"]["edges"], raw_node_id_maps)
    except BaseApplicationError as payload_error:
        error_message = str(payload_error)
        error_code = payload_error.get_error_code()
        return {PAYLOAD_PROCESSING_STATUS: 400,
                PAYLOAD_PROCESSING_ERROR_CODE: error_code,
                PAYLOAD_PROCESSING_RETURN_MSG: error_message}

    try:
        g = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

        # Assign node/edge features
        for ntype in node_data:
            for name, ndata in node_data[ntype].items():
                g.nodes[ntype].data[name] = th.tensor(ndata)
        for etype in edge_data:
            for name, edata in edge_data[etype].items():
                g.edges[etype].data[name] = th.tensor(edata)

        return {PAYLOAD_PROCESSING_STATUS: 200,
                PAYLOAD_PROCESSING_RETURN_MSG: "successful build payload graph",
                PAYLOAD_GRAPH: g,
                PAYLOAD_GRAPH_NODE_MAPPING: raw_node_id_maps}
    except DGLCreateError as e:
        return {PAYLOAD_PROCESSING_STATUS: 400,
                PAYLOAD_PROCESSING_ERROR_CODE: e.get_error_code(),
                PAYLOAD_PROCESSING_RETURN_MSG: str(e)}
