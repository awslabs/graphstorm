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

    SageMaker entry point file for GraphStorm node prediction realtime inference
"""

import json
import logging
import os
import traceback
import hashlib

import numpy as np
import torch as th

import graphstorm as gs
from graphstorm.utils import setup_device, get_device
from graphstorm.config.config import (
    GS_RUNTIME_GCONSTRUCT_FILENAME,
    GS_RUNTIME_TRAINING_CONFIG_FILENAME,
    )
from graphstorm.gconstruct import (PAYLOAD_PROCESSING_STATUS,
                                   PAYLOAD_PROCESSING_RETURN_MSG,
                                   PAYLOAD_PROCESSING_ERROR_CODE,
                                   PAYLOAD_GRAPH,
                                   PAYLOAD_GRAPH_NODE_MAPPING,
                                   process_json_payload_graph)
from graphstorm.dataloading import GSgnnRealtimeInferNodeDataLoader
from graphstorm.inference import GSGnnNodePredictionRealtimeInferrer
from graphstorm.sagemaker import GSRealTimeInferenceResponseMessage as RTResponseMsg

# Set seed to ensure prediction results to be constant
th.manual_seed(12345678)
np.random.seed(12345678)

DEFAULT_GS_MODEL_FILE_NAME = 'model.bin'

# set some constants as response keys
RESPONSE_NTYPE_STR = 'node_type'
RESPONSE_NID_STR = 'node_id'
RESPONSE_PREDICTION_STR = 'prediction'
PREDICTION_RESULTS_STR = 'results'


# ================== SageMaker real-time entry point functions ================== #
def model_fn(model_dir):
    """ Load GraphStorm trained model artifacts.

    GraphStorm model artifacts include three major files:
    1. The trained GraphStorm model weights. By default, it will be a `model.bin` file.
    2. The model configuration YAML file, which should be the one generated by each training job.
    3. The graph processing configuration JSON file, which should be the one generated by a gconstruct or
       GSProcessing job.
    These components should be packed in a tar file that SageMaker will download and unzip to the
    given `model_dir`, which has the following structure,

    - model_dir
        |- model.bin                                        # Binary model artifact
        |- GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml  # Training YAML config, updated with
                                                            # runtime parameters
        |- data_transform_new.json                          # GConstruct or GSProcessing
                                                            # configuration JSON used during graph
                                                            # construction process
        |- code
            |- node_prediction_entry.py # Entry point for task

    Parameters
    ----------
    model_dir: str
        The SageMaker model directory where SageMaker unzips the tar file provided in endpoint
        deployment.

    Returns
    -------
    A tuple of three elements, including:
    model: GraphStorm model
        A GraphStorm model rebuilt from model artifacts.
    gconstruct_config_dict:
        A dictionary object loaded from the given graph configuration JSON file.
    gs_train_config: GSConfig
        An instance of GSConfig object built from the given model configuration YAML file.
    """
    logging.info('-- START model loading... ')

    # find the name of artifact file names, assuming there is only one type file packed
    gs_trained_model_file = None
    gs_train_yaml_file = None
    gs_construct_json_file = None
    files = os.listdir(model_dir)

    # keep the file name or extension check logic here for easy customization as users may use
    # different artifact names or extensions from the default settings
    if DEFAULT_GS_MODEL_FILE_NAME in files:
        gs_trained_model_file = DEFAULT_GS_MODEL_FILE_NAME
    if GS_RUNTIME_TRAINING_CONFIG_FILENAME in files:
        gs_train_yaml_file = GS_RUNTIME_TRAINING_CONFIG_FILENAME
    if GS_RUNTIME_GCONSTRUCT_FILENAME in files:
        gs_construct_json_file = GS_RUNTIME_GCONSTRUCT_FILENAME

    # in case there is no built-in JSON or YAML files, use file extensions for custom names
    for file in files:
        if gs_train_yaml_file is None and file.endswith(".yaml"):
            gs_train_yaml_file = file
            logging.warning("Could not find the default training config YAML file: "
                            "%s. "
                            "Will try to use %s as the train config file.",
                            GS_RUNTIME_TRAINING_CONFIG_FILENAME,
                            file)
        if gs_construct_json_file is None and file.endswith(".json"):
            gs_construct_json_file = file
            logging.warning("Could not find the default GConstruct JSON file: "
                            "'%s'. "
                            "Will try to use '%s' as the GConstruct config file.",
                            GS_RUNTIME_GCONSTRUCT_FILENAME,
                            file)

    # check if required artifacts exist
    assert gs_trained_model_file is not None, ('Missing model file, e.g., \"model.bin\", in the ' \
                                               'tar file.')
    assert gs_train_yaml_file is not None, ('Missing model configuration YAML file in the tar ' \
                                            'file.')
    assert gs_construct_json_file is not None, ('Missing graph configuration JSON file in the tar ' \
                                                'file.')

    # load and recreate the trained model using the gsf built-in function
    try:
        model, gconstruct_config_dict, gs_train_config = \
            gs.restore_builtin_model_from_artifacts(model_dir,
                                                    gs_construct_json_file,
                                                    gs_train_yaml_file)
    except Exception as e:
        model = None
        logging.error('Fail to restore trained GraphStorm model. Details:\n %s', e)
        # This will be endpoint backend error, so not use the response class
        raise Exception('Fail to restore trained GraphStorm model. Details: %s', e)

    logging.debug(model)
    logging.debug(gconstruct_config_dict)
    logging.debug(gs_train_config)

    return (model, gconstruct_config_dict, gs_train_config)

def transform_fn(model,
                 request_body,
                 request_content_type,
                 response_content_type='application/json'):
    """ An end-to-end function to handle one request from input to output.

    #TODO: add the API specification url here:

    According to GraphStorm real-time inference API specification, the payload is like:

    {
        "version": "gs-realtime-v0.1",
        "gml_task": "node_classification",
        "graph": {
            "nodes": [
                {
                    "node_type": "author",
                    "features": {
                        "feat": [
                            0.011269339360296726,
                            ......
                        ]
                    },
                    "node_id": "a4444"
                },
                {
                    "node_type": "author",
                    "features": {
                        "feat": [
                            -0.0032965524587780237,
                            .....
                        ]
                    },
                    "node_id": "s39"
                }
            ],
            "edges": [
                {
                    "edge_type": [
                        "author",
                        "writing",
                        "paper"
                    ],
                    "features": {},
                    "src_node_id": "p4463",
                    "dest_node_id": "p4463"
                },
                ......
            ]
        },
        "targets": [
            {
                "node_type": "paper",
                "node_id": "p4463"
            },
            or
            {
                "edge_type": [
                        "paper",
                        "citing",
                        "paper"
                    ]
                "src_node_id": "p3551",
                "dest_node_id": "p3551"
            }
        ]
    }

    Parameters
    ----------
    model: a tuple of three elements
        The output of model_fn, including a model object, a graph construct config dict, and a
        GSConfig object.
    request_body: str
        The request payload as a JSON string. The JSON string contains the subgraph for inference.
    request_content_type: str
        A string to indicate what is the format of the payload. For GraphStorm built-in real-time
        input function, the format should be 'application/json'.
    response_content_type: str
        A string to indicate what is the format of the response. Default is 'application/json'.

    Returns
    -------
    res: JSON format of GSRealTimeInferenceResponseMessage
        The JSON representations of an instance of GSRealTimeInferenceResponseMessage.
    response_content_type: str
        A string to indicate the format of the response. Currently support 'application/json' only.
    """
    model_obj, gconstruct_config_dict, gs_train_config = model

    logging.debug(request_body)

    if request_content_type != 'application/json':
        res = RTResponseMsg.json_format_error(request_uid='',
            error=(f'Unsupported content type: {request_content_type}. Supported content ' \
                   'type: "application/json"'))
        return res.to_json(), response_content_type
    try:
        payload_data = json.loads(request_body)
    except Exception as e:
        res = RTResponseMsg.json_format_error(request_uid='',error=e)
        return res.to_json(), response_content_type

    # create a unique request id for easy error tracking between client and server side
    # here use the last 16 chars of the hash code of the request body.
    # The request id will be added to each response class, and return to invokers. It will
    # be logged on the server side too.
    hash_oject = hashlib.sha256(request_body.encode('utf-8'))
    request_uid = hash_oject.hexdigest()[-16: ]

    # TODO(Jian), build a unified payload content sanity checking  method under gconstruct package
    # to be shared by all entry point files

    # the version object will be used later to keep backward compatibilty for early versions
    version = payload_data.get('version', None)

    gml_task = payload_data.get('gml_task', None)
    targets = payload_data.get('targets', None)

    # 1. check if the payload is for a node prediction task
    if gml_task is None or (gml_task not in ['node_classification', 'node_regression']):
        track = (f'This endpoint is for node prediction task, but got {gml_task} task from ' \
                 'the payload. Supported task types include [\"node_classification\", ' \
                 '\"node_regression\"]')
        res = RTResponseMsg.task_mismatch_error(request_uid=request_uid, track=track)
        logging.error(res.to_json())
        return res.to_json(), response_content_type

    # 2. check if the targets field is provided
    if targets is None or len(targets)==0:
        res = RTResponseMsg.missing_required_field(request_uid=request_uid, field='targets')
        logging.error(res.to_json())
        return res.to_json(), response_content_type

    # 3. check if target has node_type and node_id. Will check values in id mapping
    for target in targets:
        if isinstance(target, dict):
            if 'node_type' not in target or 'node_id' not in target:
                res = RTResponseMsg.json_format_error(request_uid=request_uid, error=(
                    'The Element of \"targets\" field should be a dictionary that has both ' \
                    f'\"node_type\" and \"node_id\" keys, but got {target}.'))
                logging.error(res.to_json())
                return res.to_json(), response_content_type
        else:
            res = RTResponseMsg.json_format_error(request_uid=request_uid, error=(
                f'The Element of \"targets\" field should be a dictionary, but got {target}'))
            logging.error(res.to_json())
            return res.to_json(), response_content_type

    # processing payload to generate a DGL graph, and catch any errors to prevent server crash
    try:
        g_resp = process_json_payload_graph(payload_data, gconstruct_config_dict)
    except Exception as e:
        logging.error(traceback.format_exc())
        res = RTResponseMsg.graph_construction_failure(request_uid=request_uid, track=e)
        logging.error(res.to_json())
        return res.to_json(), response_content_type

    # generation failed
    if g_resp[PAYLOAD_PROCESSING_STATUS] != 200:
        track = (f'Error code: {g_resp[PAYLOAD_PROCESSING_ERROR_CODE]}, ' \
                 f'Message: {g_resp[PAYLOAD_PROCESSING_RETURN_MSG]}.')
        res = RTResponseMsg.graph_construction_failure(request_uid=request_uid, track=track)
        logging.error(res.to_json())
        return res.to_json(), response_content_type

    # generation succeeded
    if g_resp[PAYLOAD_PROCESSING_STATUS] == 200:
        dgl_graph = g_resp[PAYLOAD_GRAPH]
        raw_node_id_maps = g_resp[PAYLOAD_GRAPH_NODE_MAPPING]

    # 4. check if node or edge feature names match with model configurations
    if gs_train_config.node_feat_name is not None:
        for ntype, feat_list in gs_train_config.node_feat_name.items():
            # it is possible that some node types are not sampled
            if ntype not in dgl_graph.ntypes:
                continue

            for feat in feat_list:
                if feat not in dgl_graph.nodes[ntype].data:
                    res = RTResponseMsg.missing_feature(request_uid=request_uid,
                                                        entity_type='node',
                                                        entity_name=ntype,
                                                        feat_name=feat)
                    logging.error(res.to_json())
                    return res.to_json(), response_content_type

    if gs_train_config.edge_feat_name is not None:
        for etype, feat_list in gs_train_config.edge_feat_name.items():
            # it is possible that some edge types are not sampled
            if etype not in dgl_graph.etypes:
                continue

            for feat in feat_list:
                if feat not in dgl_graph.edges[etype].data:
                    res = RTResponseMsg.missing_feature(request_uid=request_uid,
                                                        entity_type='edge',
                                                        entity_name=etype,
                                                        feat_name=feat)
                    logging.error(res.to_json())
                    return res.to_json(), response_content_type


    # mapping the targets, a list of node objects, to new graph node IDs after dgl graph
    # construction for less overall data processing time

    # create an empty mapping dict: keys are node types, and values are two lists, 1st for
    # original(str like) node ids, 2nd for the dgl(int) node ids.
    target_mapping_dict = {ntype: ([],[]) for ntype in raw_node_id_maps.keys()}

    for target in targets:
        target_ntype = target['node_type']
        target_nid = target['node_id']

        if target_ntype in raw_node_id_maps:
            if raw_node_id_maps[target_ntype].get(target_nid, None) is None:
                # target node id is not in the payload graph
                res = RTResponseMsg.mismatch_target_nid(request_uid=request_uid,
                                                        target_nid=target_nid)
                logging.error(res.to_json())
                return res.to_json(), response_content_type
            else:
                # target node type and id are in the payload graph
                dgl_nid = raw_node_id_maps[target_ntype].get(target_nid)
                target_mapping_dict[target_ntype][0].append(target_nid)
                target_mapping_dict[target_ntype][1].append(dgl_nid)
        else:   # target node type is not in the payload graph
            res = RTResponseMsg.mismatch_target_ntype(request_uid=request_uid,
                                                      target_ntype=target_ntype)
            logging.error(res.to_json())
            return res.to_json(), response_content_type

    # extract the DGL graph nids from target ID mapping dict
    target_nids = {}
    for ntype, (_, dgl_nids) in target_mapping_dict.items():
        if len(dgl_nids) > 0:
            target_nids[ntype] = dgl_nids

    try:
        # setup device
        setup_device(0)
        device = get_device()

        # initialize a GS real-time inferrer
        inferrer = GSGnnNodePredictionRealtimeInferrer(model_obj)
        inferrer.setup_device(device)
        # initialize a GS real-time dataLoader
        dataloader = GSgnnRealtimeInferNodeDataLoader(dgl_graph,
                                                      target_nids,
                                                      gs_train_config.num_layers)
        predictions = inferrer.infer(dgl_graph, dataloader, list(target_nids.keys()),
                                     gs_train_config.node_feat_name,
                                     gs_train_config.edge_feat_name)
        # Build prediction response
        pred_list = []
        for ntype, preds in predictions.items():
            (orig_nids, _) = target_mapping_dict[ntype]
            # the dataloader ensures that the order of predictions is same as the order of dgl nids
            for orig_nid, pred in zip(orig_nids, preds):
                pred_res = {
                    RESPONSE_NTYPE_STR: ntype,
                    RESPONSE_NID_STR: orig_nid,
                    RESPONSE_PREDICTION_STR: pred.tolist()
                }
                pred_list.append(pred_res)
    except Exception as e:
        res = RTResponseMsg.internal_server_error(request_uid=request_uid, detail=e)
        logging.error(traceback.format_exc())
        return res.to_json(), response_content_type

    res = RTResponseMsg.success(request_uid=request_uid,
                                data={PREDICTION_RESULTS_STR: pred_list})
    logging.info(res.to_json())

    return res.to_json(), response_content_type
