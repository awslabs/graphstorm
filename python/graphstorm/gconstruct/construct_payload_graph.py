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
import logging

from ..utils import sys_tracker, get_log_level, check_graph_name
from .construct_graph import verify_conf


def process_json_payload_nodes(node_conf):
    """ Process json payload node data

    We need to process all node data before we can process edge data.

    The node data of a node type is defined as follows:
    {
        "node_type":    "<node type>",
        "node_id":      "<node id>",
        "features":     {
            "<feat_name>": [feat_num]
        }
    }
    """
    print(node_conf)


def process_json_payload_edges(edge_conf):
    """ Process json payload edge data

    {
        "edge_type": "<edge type>",
        "src_node_id": <src edge id>,
        "dest_node_id": <dst edge id>
        "features: {
            "<feat_name>": [feat_num]
        }
    }
    """
    print(edge_conf)


def process_json_payload_graph(args):
    """ Construct DGLGraph from json payload.
    """
    check_graph_name(args.graph_name)
    logging.basicConfig(level=get_log_level(args.logging_level))

    with open(args.conf_file, 'r', encoding="utf8") as json_file:
        process_confs = json.load(json_file)

    sys_tracker.set_rank(0)
    num_processes_for_nodes = args.num_processes_for_nodes \
            if args.num_processes_for_nodes is not None else args.num_processes
    num_processes_for_edges = args.num_processes_for_edges \
            if args.num_processes_for_edges is not None else args.num_processes
    print(num_processes_for_nodes, num_processes_for_edges)
    verify_confs(process_confs)

    output_format = args.output_format
    if len(output_format) != 1 and output_format[0] != "DGL":
        logging.warning("We only support building DGLGraph for json payload")

    for node_conf in process_confs["graph"]["nodes"]:
        assert "node_type" in node_conf, "node type must be defined in the config"
        assert "node_id" in node_conf, "node id must be defined in the config"
        process_json_payload_nodes(node_conf)

    for edge_conf in process_confs["graph"]["edges"]:
        assert "edge_type" in edge_conf, "edge type must be defined in the config"
        assert "edge_id" in edge_conf, "edge id must be defined in the config"
        process_json_payload_edges(edge_conf)
