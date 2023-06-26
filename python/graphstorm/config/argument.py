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

    Arguments and config
"""

import os
import sys
import argparse
import math

import yaml
import torch as th

from .config import BUILTIN_GNN_ENCODER
from .config import BUILTIN_ENCODER
from .config import SUPPORTED_BACKEND
from .config import BUILTIN_LP_LOSS_FUNCTION
from .config import BUILTIN_LP_LOSS_CROSS_ENTROPY

from .config import BUILTIN_TASK_NODE_CLASSIFICATION
from .config import BUILTIN_TASK_NODE_REGRESSION
from .config import BUILTIN_TASK_EDGE_CLASSIFICATION
from .config import BUILTIN_TASK_EDGE_REGRESSION
from .config import BUILTIN_TASK_LINK_PREDICTION
from .config import EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY
from .config import EARLY_STOP_AVERAGE_INCREASE_STRATEGY
from .config import GRAPHSTORM_SAGEMAKER_TASK_TRACKER
from .config import SUPPORTED_TASK_TRACKER

from .config import SUPPORTED_TASKS

from .config import BUILTIN_LP_DISTMULT_DECODER
from .config import SUPPORTED_LP_DECODER

from .config import GRAPHSTORM_MODEL_ALL_LAYERS

from .utils import get_graph_name
from ..utils import TORCH_MAJOR_VER

from ..eval import SUPPORTED_CLASSIFICATION_METRICS
from ..eval import SUPPORTED_REGRESSION_METRICS
from ..eval import SUPPORTED_LINK_PREDICTION_METRICS

from ..dataloading import BUILTIN_LP_UNIFORM_NEG_SAMPLER
from ..dataloading import BUILTIN_LP_JOINT_NEG_SAMPLER

__all__ = [
    "get_argument_parser",
]

def get_argument_parser():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="GSGNN Arguments")
    # Required parameters
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="pointer to the yaml configuration file of the experiment",
        type=str,
        required=True,
    )

    if TORCH_MAJOR_VER >= 2:
        parser.add_argument(
                "--local-rank",
                type=int,
                default=-1,
                help="local_rank for distributed training on gpus",
                )
    else:
        parser.add_argument(
                "--local_rank",
                type=int,
                default=-1,
                help="local_rank for distributed training on gpus",
                )

    # Optional parameters to override arguments in yaml config
    parser = _add_initialization_args(parser)
    # basic args
    parser = _add_gsgnn_basic_args(parser)
    # gnn args
    parser = _add_gnn_args(parser)
    parser = _add_input_args(parser)
    parser = _add_output_args(parser)
    parser = _add_task_tracker(parser)
    parser = _add_hyperparam_args(parser)
    parser = _add_rgcn_args(parser)
    parser = _add_rgat_args(parser)
    parser = _add_link_prediction_args(parser)
    parser = _add_node_classification_args(parser)
    parser = _add_edge_classification_args(parser)
    parser = _add_task_general_args(parser)
    parser = _add_lm_model_args(parser)
    return parser

# pylint: disable=no-member
class GSConfig:
    """GSgnn Argument class which contains all arguments
       from yaml config and constructs additional arguments

    Parameters:
    cmd_args: Argument
        Commend line arguments
    """
    def __init__(self, cmd_args):
        self.yaml_paths = cmd_args.yaml_config_file
        # Load all arguments from yaml config
        configuration = self.load_yaml_config(cmd_args.yaml_config_file)

        self.set_attributes(configuration)

        # Override class attributes using command-line arguments
        self.override_arguments(cmd_args)
        self.local_rank = cmd_args.local_rank
        # We do argument check as early as possible to prevent config bugs.
        self.handle_argument_conflicts()

    def set_attributes(self, configuration):
        """Set class attributes from 2nd level arguments in yaml config"""
        print(configuration)
        if 'lm_model' in configuration:
            # has language model configuration
            # lm_model:
            #   node_lm_models:
            #     -
            #       lm_type: bert
            #       model_name: "bert-base-uncased"
            #       gradient_checkpoint: true
            #       node_types:
            #         - n_0
            #         - n_1
            #     -
            #       lm_type: bert
            #       model_name: "allenai/scibert_scivocab_uncased"
            #       gradient_checkpoint: true
            #       node_types:
            #         - n_2
            lm_model = configuration['lm_model']
            assert "node_lm_models" in lm_model, "node_lm_models must be provided"
            # if node_lm_models is not defined, ignore the lm model
            node_lm_models = lm_model['node_lm_models']
            setattr(self, "_node_lm_configs", node_lm_models)

        # handle gnn config
        gnn_family = configuration['gsf']
        for family, param_family in gnn_family.items():
            for key, val in param_family.items():
                setattr(self, f"_{key}", val)

            if family == BUILTIN_TASK_LINK_PREDICTION:
                setattr(self, "_task_type", BUILTIN_TASK_LINK_PREDICTION)
            elif family == BUILTIN_TASK_EDGE_CLASSIFICATION:
                setattr(self, "_task_type", BUILTIN_TASK_EDGE_CLASSIFICATION)
            elif family == BUILTIN_TASK_EDGE_REGRESSION:
                setattr(self, "_task_type", BUILTIN_TASK_EDGE_REGRESSION)
            elif family == BUILTIN_TASK_NODE_CLASSIFICATION:
                setattr(self, "_task_type", BUILTIN_TASK_NODE_CLASSIFICATION)
            elif family == BUILTIN_TASK_NODE_REGRESSION:
                setattr(self, "_task_type", BUILTIN_TASK_NODE_REGRESSION)

        if 'udf' in configuration:
            udf_family = configuration['udf']
            # directly add udf configs as config arguments
            for key, val in udf_family.items():
                setattr(self, key, val)

    def load_yaml_config(self, yaml_path):
        """Helper function to load a yaml config file"""
        with open(yaml_path, "r", encoding='utf-8') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ValueError(f"Yaml error - check yaml file {exc}")

    def override_arguments(self, cmd_args):
        """Override arguments in yaml config using command-line arguments"""
        # TODO: Support overriding for all arguments in yaml
        cmd_args_dict = cmd_args.__dict__
        for arg_key, arg_val in cmd_args_dict.items():
            if arg_key not in ["yaml_config_file", "local_rank"]:
                if arg_key == "save_model_path" and arg_val.lower() == "none":
                    arg_val = None
                if arg_key == "save_embed_path" and arg_val.lower() == "none":
                    arg_val = None
                if arg_key == "save_prediction_path" and arg_val.lower() == "none":
                    arg_val = None

                # for basic attributes
                setattr(self, f"_{arg_key}", arg_val)
                print(f"Overriding Argument: {arg_key}")

    def handle_argument_conflicts(self):
        """Check and resolve argument conflicts
        """
        # 1. language model conflicts
        if self.node_lm_configs is not None:
            # gradient checkpoint does not work with freeze_lm_encoder_epochs
            # When freeze_lm_encoder_epochs is set, turn off gradient checkpoint
            if self.freeze_lm_encoder_epochs > 0:
                for i, _ in enumerate(self.node_lm_configs):
                    if self.node_lm_configs[i]["gradient_checkpoint"]:
                        print("WARNING: freeze_lm_encoder_epochs can not work with " \
                              "gradient checkpoint. Turn gradient checkpoint to False")
                        self.node_lm_configs[i]["gradient_checkpoint"] = False

        # TODO(xiangsx): Add more check

    ###################### Environment Info ######################
    @property
    def save_perf_results_path(self):
        """ Save performance flag
        """
        # pylint: disable=no-member
        if hasattr(self, "_save_perf_results_path"):
            return self._save_perf_results_path
        return None

    @property
    def profile_path(self):
        """ The path of the folder where the profiling results are saved.
        """
        if hasattr(self, "_profile_path"):
            return self._profile_path
        return None

    @property
    def graph_name(self):
        """ Name of the graph
        """
        return get_graph_name(self.part_config)

    @property
    def backend(self):
        """ Distributed training backend
        """
        # pylint: disable=no-member
        if hasattr(self, "_backend"):
            assert self._backend in SUPPORTED_BACKEND, \
                f"backend must be in {SUPPORTED_BACKEND}"
            return self._backend

        return "gloo"

    @property
    def ip_config(self):
        """ IP config of instances in a cluster
        """
        # pylint: disable=no-member
        assert hasattr(self, "_ip_config"), "IP config must be provided"
        assert os.path.isfile(self._ip_config), \
            f"IP config file {self._ip_config} does not exist"
        return self._ip_config

    @property
    def part_config(self):
        """ configuration of graph partition
        """
        # pylint: disable=no-member
        assert hasattr(self, "_part_config"), "Graph partition config must be provided"
        assert os.path.isfile(self._part_config), \
            f"Partition config file {self._part_config} does not exist"
        return self._part_config

    @property
    def node_id_mapping_file(self):
        """ A path to the file storing node id mapping generated by the
            graph partition algorithm.

            Graph partition will shuffle node ids and edge ids according
            to the node partition assignment. We expect partition algorithms
            will save node id mappings to map new node ids to their original
            node ids.

            We assume node_id mappings are stored as a single object
            along with part_config. (We assume the graph is partitioned
            using DGL graph partition tool)
        """
        path = os.path.dirname(self.part_config)
        # See graphstorm.gconstruct.utils.partition_graph for more detials
        node_id_mapping_file = os.path.join(path, "node_mapping.pt")
        return node_id_mapping_file if os.path.isfile(node_id_mapping_file) \
            else None

    @property
    def edge_id_mapping_file(self):
        """ A path to the file storing edge id mapping generated by the
            graph partition algorithm.

            Graph partition will shuffle node ids and edge ids according
            to the node partition assignment. We expect partition algorithms
            will save edge id mappings to map new edge ids to their original
            edge ids.

            We assume edge_id mappings are stored as a single object
            along with part_config. (We assume the graph is partitioned
            using DGL graph partition tool)
        """
        path = os.path.dirname(self.part_config)
        # See graphstorm.gconstruct.utils.partition_graph for more detials
        edge_id_mapping_file = os.path.join(path, "edge_mapping.pt")
        return edge_id_mapping_file if os.path.isfile(edge_id_mapping_file) \
            else None

    @property
    def verbose(self):
        """ verbose for print out more information.Default is False
        """
        # pylint: disable=no-member
        if hasattr(self, "_verbose"):
            assert self._verbose in [True, False]
            return self._verbose

        return False

    ###################### language model support #########################
    # Bert related
    @property
    def lm_tune_lr(self):
        """ Learning rate for BERT model(s)
        """
        # pylint: disable=no-member
        if hasattr(self, "_lm_tune_lr"):
            lm_tune_lr = float(self._lm_tune_lr)
            assert lm_tune_lr > 0.0, "Bert tune learning rate must > 0.0"
            return lm_tune_lr

        return self.lr

    @property
    def lm_train_nodes(self):
        """ Number of tunable LM model nodes
        """
        # pylint: disable=no-member
        if hasattr(self, "_lm_train_nodes"):
            assert self._lm_train_nodes >= -1, \
                "Number of LM trainable nodes must larger or equal to -1." \
                "0 means no LM trainable nodes" \
                "-1 means all nodes are LM trainable nodes"
            return self._lm_train_nodes

        # By default, do not turn on co-training
        return 0

    @property
    def lm_infer_batch_size(self):
        """ Mini batch size used to do LM model inference
        """
        # pylint: disable=no-member
        if hasattr(self, "_lm_infer_batch_size"):
            assert self._lm_infer_batch_size > 0, \
                "Batch size for LM model inference must larger than 0"
            return self._lm_infer_batch_size

        return 32

    @property
    def freeze_lm_encoder_epochs(self):
        """ Number of epochs we will take to warmup a GNN model
            before a fine-tuning LM model with GNN.
        """
        # pylint: disable=no-member
        if hasattr(self, "_freeze_lm_encoder_epochs"):
            assert self._freeze_lm_encoder_epochs >= 0, \
                "Number of warmup epochs must be larger than or equal to 0"

            assert self._freeze_lm_encoder_epochs == 0 or \
                self.model_encoder_type not in ["lm", "mlp"], \
                "Encoder type lm (language model) and mlp (encoder layer only) " \
                "do not work with language model warmup. It will cause torch " \
                "DDP error"
            return self._freeze_lm_encoder_epochs

        return 0

    def _check_lm_config(self, lm_config):
        assert "lm_type" in lm_config, "lm_type (type of language model," \
            "e.g., bert) must be provided for node_lm_models."
        assert "model_name" in lm_config, "language model model_name must " \
            "be provided for node_lm_models."
        if "gradient_checkpoint" not in lm_config:
            lm_config["gradient_checkpoint"] = False
        assert "node_types" in lm_config, "node types must be provided for " \
            "node_lm_models"
        assert len(lm_config["node_types"]) >= 1, "number of node types " \
            "must be larger than 1"

    @property
    def node_lm_configs(self):
        """ check bert config
        """
        if hasattr(self, "_node_lm_configs"):
            if self._node_lm_configs is None:
                return None

            # lm_config is not NOne
            assert isinstance(self._node_lm_configs, list), \
                "Node language model config is not None. It must be a list"
            assert len(self._node_lm_configs) > 0, \
                "Number of node language model config must larger than 0"

            for lm_config in self._node_lm_configs:
                self._check_lm_config(lm_config)

            return self._node_lm_configs

        # By default there is no node_lm_config
        return None

    ###################### general gnn model related ######################
    @property
    def model_encoder_type(self):
        """ Which graph encoder to use, it can be GNN or language model only
        """
        # pylint: disable=no-member
        assert hasattr(self, "_model_encoder_type"), \
            "Model encoder type should be provided"
        assert self._model_encoder_type in BUILTIN_ENCODER, \
            f"Model encoder type should be in {BUILTIN_ENCODER}"
        return self._model_encoder_type

    @property
    def node_feat_name(self):
        """ User defined node feature name

            It can be in following format:
            1) [feat_name]: global feature name, if a node has node feature,
            the corresponding feature name is <feat_name>
            2)["ntype0:feat0","ntype1:feat0,feat1",...]: different node
            types have different node features.
        """
        # pylint: disable=no-member
        if hasattr(self, "_node_feat_name"):
            feat_names = self._node_feat_name
            if len(feat_names) == 1 and \
                ":" not in feat_names[0]:
                # global feat_name
                return feat_names[0]

            # per node type feature
            fname_dict = {}

            for feat_name in feat_names:
                feat_info = feat_name.split(":")
                assert len(feat_info) == 2, \
                        f"Unknown format of the feature name: {feat_name}, " + \
                        "must be NODE_TYPE:FEAT_NAME"
                ntype = feat_info[0]
                if ntype in fname_dict:
                    assert False, \
                        f"You already specify the feature names of {ntype}" \
                        f"as {fname_dict[ntype]}"
                assert isinstance(feat_info[1], str), \
                    f"Feature name of {ntype} should be a string not {feat_info[1]}"
                # multiple features separated by ','
                fname_dict[ntype] = feat_info[1].split(",")
            return fname_dict

        # By default, return None which means there is no node feature
        return None

    def _check_fanout(self, fanout, fot_name):
        try:
            if fanout[0].isnumeric():
                # Fanout in format of 20,10,5,...
                fanout = [int(val) for val in fanout]
            else:
                # Fanout in format of
                # etype2:20@etype3:20@etype1:20,etype2:10@etype3:4@etype1:2
                # Each etype should be a canonical etype in format of
                # srcntype/relation/dstntype

                fanout = [{tuple(k.split(":")[0].split('/')): int(k.split(":")[1]) \
                    for k in val.split("@")} for val in fanout]
        except Exception: # pylint: disable=broad-except
            assert False, f"{fot_name} Fanout should either in format 20,10 " \
                "when all edge type have the same fanout or " \
                "etype2:20@etype3:20@etype1:20," \
                "etype2:10@etype3:4@etype1:2 when you want to " \
                "specify a different fanout for different edge types" \
                "Each etype (e.g., etype2) should be a canonical etype in format of" \
                "srcntype/relation/dstntype"

        assert len(fanout) == self.num_layers, \
            f"You have a {self.num_layers} layer GNN, " \
            f"but you only specify a {fot_name} fanout for {len(fanout)} layers."
        return fanout

    @property
    def fanout(self):
        """ training fanout
        """
        # pylint: disable=no-member
        assert hasattr(self, "_fanout"), \
            "Training fanout must be provided"

        fanout = self._fanout.split(",")
        return self._check_fanout(fanout, "Train")

    @property
    def eval_fanout(self):
        """ evaluation fanout
        """
        # pylint: disable=no-member
        if hasattr(self, "_eval_fanout"):
            fanout = self._eval_fanout.split(",")
            return self._check_fanout(fanout, "Evaluation")
        else:
            # By default use -1 as full neighbor
            return [-1] * self.num_layers

    @property
    def hidden_size(self):
        """ Hidden embedding size
        """
        # pylint: disable=no-member
        assert hasattr(self, "_hidden_size"), \
            "hidden_size must be provided when pretrain a embedding layer, " \
            "or train a GNN model"
        assert isinstance(self._hidden_size, int), \
            "Hidden embedding size must be an integer"
        assert self._hidden_size > 0, \
            "Hidden embedding size must be larger than 0"
        return self._hidden_size

    @property
    def num_layers(self):
        """ Number of GNN layers
        """
        # pylint: disable=no-member
        if self.model_encoder_type in BUILTIN_GNN_ENCODER:
            assert hasattr(self, "_num_layers"), \
                "Number of GNN layers must be provided"
            assert isinstance(self._num_layers, int), \
                "Number of GNN layers must be an integer"
            assert self._num_layers > 0, \
                "Number of GNN layers must be larger than 0"
            return self._num_layers
        else:
            # not used by non-GNN models
            return 0

    @property
    def use_mini_batch_infer(self):
        """ Whether do mini-batch inference or full graph inference
        """
        # pylint: disable=no-member
        if hasattr(self, "_use_mini_batch_infer"):
            assert self._use_mini_batch_infer in [True, False], \
                "Use mini batch inference flag must be True or False"
            return self._use_mini_batch_infer

        # By default, use mini batch inference, which requires less memory
        return True

    ###################### I/O related ######################
    ### Restore model ###
    @property
    def restore_model_layers(self):
        """ GraphStorm model layers to load.
        """
        # pylint: disable=no-member
        if hasattr(self, "_restore_model_layers"):
            assert self.restore_model_path is not None, \
                "restore-model-path must be provided"
            model_layers = self._restore_model_layers.split(',')
            for layer in model_layers:
                assert layer in GRAPHSTORM_MODEL_ALL_LAYERS, \
                    f"{layer} is not supported, must be any of {GRAPHSTORM_MODEL_ALL_LAYERS}"
            return model_layers

        return GRAPHSTORM_MODEL_ALL_LAYERS

    @property
    def restore_model_path(self):
        """ Path to the entire model including embed layer, encoder and decoder
        """
        # pylint: disable=no-member
        if hasattr(self, "_restore_model_path"):
            return self._restore_model_path
        return None

    @property
    def restore_optimizer_path(self):
        """ Path to the saved optimizer status including embed layer,
            encoder and decoder.
        """
        # pylint: disable=no-member
        if hasattr(self, "_restore_optimizer_path"):
            return self._restore_optimizer_path
        return None

    ### Save model ###
    @property
    def save_embed_path(self):
        """ Path to save the GNN embeddings from the best model
        """
        # pylint: disable=no-member
        if hasattr(self, "_save_embed_path"):
            return self._save_embed_path
        return None

    @property
    def save_model_path(self):
        """ Path to save the model.
        """
        # pylint: disable=no-member
        if hasattr(self, "_save_model_path"):
            return self._save_model_path
        return None

    @property
    def save_model_frequency(self):
        """ Save model every N iterations
        """
        # pylint: disable=no-member
        if hasattr(self, "_save_model_frequency"):
            assert self.save_model_path is not None, \
                'To save models, please specify a valid path. But got None'
            assert self._save_model_frequency > 0, \
                f'save-model-frequency must large than 0, but got {self._save_model_frequency}'
            return self._save_model_frequency
        # By default, use -1, means do not auto save models
        return -1

    @property
    def topk_model_to_save(self):
        """ the number of top k best validation performance model to save

            If topk_model_to_save is set (save_model_frequency is not set),
            GraphStorm will try to save models after each epoch and keep at
            most K models.
            If save_model_frequency is set, GraphStorm will try to save
            models every #save_model_frequency iterations and keep at
            most K models.
            By default, GraphStorm will save the latest K models unless
            eval_frequency is set. When eval_frequency is set,
            GraphStorm will evaluate the model performance every
            #eval_frequency iterations. If at the same iteration,
            #save_model_frequency is reached, it will try to save the
            best K model instead of the latest K model.
        """
        # pylint: disable=no-member
        if hasattr(self, "_topk_model_to_save"):
            assert self._topk_model_to_save > 0, "Top K best model must > 0"
            assert self.save_model_path is not None, \
                'To save models, please specify a valid path. But got None'

            if self.eval_frequency != sys.maxsize and self.save_model_frequency > 0:
                # save model within an epoch need to collaborate with evaluation
                # within an epoch
                assert self.save_model_frequency >= self.eval_frequency and \
                    self.save_model_frequency % self.eval_frequency == 0, \
                    'FATAL: save_model_frequency' \
                          f'({self.save_model_frequency}) ' \
                          'does not equal to eval_frequency' \
                          f'({self.eval_frequency}), or ' \
                          f'save_model_frequency ({self.save_model_frequency}) ' \
                          'is not divisible by eval_frequency ' \
                          f'({self.eval_frequency}). ' \
                          'GraphStorm can not guarentees that it will ' \
                          'save the best model after evaluation cycles.'

            return self._topk_model_to_save
        else:
            # By default saving all models
            return math.inf

    #### Task tracker and print options ####
    @property
    def task_tracker(self):
        """ Get the type of task_tracker
        """
        # pylint: disable=no-member
        if hasattr(self, "_task_tracker"):
            assert self._task_tracker in SUPPORTED_TASK_TRACKER
            return self._task_tracker

        # By default, use SageMaker task tracker
        # It works as normal print
        return GRAPHSTORM_SAGEMAKER_TASK_TRACKER

    @property
    def log_report_frequency(self):
        """ Get print/log frequency in number of iterations
        """
        # pylint: disable=no-member
        if hasattr(self, "_log_report_frequency"):
            assert self._log_report_frequency > 0, \
                "log_report_frequency should be larger than 0"
            return self._log_report_frequency

        # By default, use 1000
        return 1000

    ###################### Model training related ######################
    @property
    def dropout(self):
        """ Dropout
        """
        # pylint: disable=no-member
        if hasattr(self, "_dropout"):
            assert self._dropout >= 0.0 and self._dropout < 1.0
            return self._dropout
        # By default, there is no dropout
        return 0.0

    @property
    # pylint: disable=invalid-name
    def lr(self):
        """ Learning rate
        """
        assert hasattr(self, "_lr"), "Learning rate must be specified"
        lr = float(self._lr) # pylint: disable=no-member
        assert lr > 0.0, \
            "Learning rate for Input encoder, GNN encoder " \
            "and task decoder must be larger than 0.0"

        return lr

    @property
    def num_epochs(self):
        """ Number of epochs
        """
        if hasattr(self, "_num_epochs"):
            # if 0, only inference or testing
            assert self._num_epochs >= 0, "Number of epochs must >= 0"
            return self._num_epochs
        # default, inference only
        return 0

    @property
    def batch_size(self):
        """ Batch size
        """
        # pylint: disable=no-member
        assert hasattr(self, "_batch_size"), "Batch size must be specified"
        assert self._batch_size > 0
        return self._batch_size

    @property
    def sparse_optimizer_lr(self): # pylint: disable=invalid-name
        """ Sparse optimizer learning rate
        """
        if hasattr(self, "_sparse_optimizer_lr"):
            sparse_optimizer_lr = float(self._sparse_optimizer_lr)
            assert sparse_optimizer_lr > 0.0, \
                "Sparse optimizer learning rate must be larger than 0"
            return sparse_optimizer_lr

        return self.lr

    @property
    def use_node_embeddings(self):
        """ Whether to use extra learnable node embeddings
        """
        # pylint: disable=no-member
        if hasattr(self, "_use_node_embeddings"):
            assert self._use_node_embeddings in [True, False]
            return self._use_node_embeddings
        # By default do not use extra node embedding
        # It will make the model transductive
        return False

    @property
    def wd_l2norm(self):
        """ Weight decay
        """
        # pylint: disable=no-member
        if hasattr(self, "_wd_l2norm"):
            return self._wd_l2norm
        return 0

    @property
    def alpha_l2norm(self):
        """ coef for l2 norm of unused weights
        """
        # pylint: disable=no-member
        if hasattr(self, "_alpha_l2norm"):
            return self._alpha_l2norm
        return .0

    @property
    def use_self_loop(self):
        """ Whether to include self feature as a special relation
        """
        # pylint: disable=no-member
        if hasattr(self, "_use_self_loop"):
            assert self._use_self_loop in [True, False]
            return self._use_self_loop
        # By default use self loop
        return True

    ### control evaluation ###
    @property
    def eval_batch_size(self):
        """ Evaluation batch size

            Mini-batch size for computing GNN embeddings in evaluation.
        """
        # pylint: disable=no-member
        if hasattr(self, "_eval_batch_size"):
            assert self._eval_batch_size > 0
            return self._eval_batch_size
        # (Israt): Larger batch sizes significantly improve runtime efficiency. Increasing the
        # batch size from 1K to 10K reduces end-to-end inference time from 45 mins to 19 mins
        # in link prediction on OGBN-paers100M dataset with 16-dimensional length. However,
        # using an overly large batch size can lead to GPU out-of-memory (OOM) issues. Therefore,
        # a heuristic approach has been taken, and 10K has been chosen as a balanced default
        # value. More details can be found at https://github.com/awslabs/graphstorm/pull/66.
        return 10000

    @property
    def eval_frequency(self):
        """ How many iterations between evaluations
        """
        # pylint: disable=no-member
        if hasattr(self, "_eval_frequency"):
            assert self._eval_frequency > 0, "eval_frequency should larger than 0"
            return self._eval_frequency
        # set max value (Never do evaluation with in an epoch)
        return sys.maxsize

    @property
    def no_validation(self):
        """ If no_validation is True, no validation and testing will run
        """
        if hasattr(self, "_no_validation"):
            assert self._no_validation in [True, False]
            return self._no_validation

        # We do validation by default
        return False

    ### control early stop ###
    @property
    def early_stop_burnin_rounds(self):
        """ Burn-in rounds before we start checking for the early stop condition.
        """
        # pylint: disable=no-member
        if hasattr(self, "_early_stop_burnin_rounds"):
            assert isinstance(self._early_stop_burnin_rounds, int), \
                "early_stop_burnin_rounds should be an integer"
            assert self._early_stop_burnin_rounds >= 0, \
                "early_stop_burnin_rounds should be larger than or equal to 0"
            return self._early_stop_burnin_rounds

        return 0

    @property
    def early_stop_rounds(self):
        """ The number of rounds for validation scores to average to decide on early stop
        """
        # pylint: disable=no-member
        if hasattr(self, "_early_stop_rounds"):
            assert isinstance(self._early_stop_rounds, int), \
                "early_stop_rounds should be an integer"
            assert self._early_stop_rounds > 0, \
                "early_stop_rounds should be larger than 0"
            return self._early_stop_rounds

        # at least 3 iterations
        return 3

    @property
    def early_stop_strategy(self):
        """ The early stop strategy
        """
        # pylint: disable=no-member
        if hasattr(self, "_early_stop_strategy"):
            assert self._early_stop_strategy in \
                [EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY, \
                    EARLY_STOP_AVERAGE_INCREASE_STRATEGY], \
                "The supported early stop strategies are " \
                f"[{EARLY_STOP_CONSECUTIVE_INCREASE_STRATEGY}, " \
                f"{EARLY_STOP_AVERAGE_INCREASE_STRATEGY}]"
            return self._early_stop_strategy

        return EARLY_STOP_AVERAGE_INCREASE_STRATEGY

    @property
    def use_early_stop(self):
        """ whether to use early stopping by monitoring the validation value
        """
        # pylint: disable=no-member
        if hasattr(self, "_use_early_stop"):
            assert self._use_early_stop in [True, False], \
                "use_early_stop should be in [True, False]"
            return self._use_early_stop

        # By default do not enable early stop
        return False

    ## RGCN only ##
    @property
    def num_bases(self):
        """ Number of bases used in RGCN weight
        """
        # pylint: disable=no-member
        if hasattr(self, "_num_bases"):
            assert isinstance(self._num_bases, int)
            assert self._num_bases > 0 or self._num_bases == -1, \
                "num_bases should be larger than 0 or -1"
            return self._num_bases
        # By default do not use num_bases
        return -1

    ## RGAT only ##
    @property
    def num_heads(self):
        """ Number of attention heads
        """
        # pylint: disable=no-member
        if hasattr(self, "_num_heads"):
            assert self._num_heads > 0, \
                "num_heads should be larger than 0"
            return self._num_heads
        # By default use 4 heads
        return 4

    ############ task related #############
    ###classification/regression related ####
    @property
    def label_field(self):
        """ The label field in the data

            Used by node and edge classification/regression tasks.
        """
        # pylint: disable=no-member
        assert hasattr(self, "_label_field"), \
            "Must provide the feature name of labels through label_field"
        return self._label_field

    @property
    def num_classes(self):
        """ The cardinality of labels in a classification task

            Used by node classification and edge classification
        """
        # pylint: disable=no-member
        assert hasattr(self, "_num_classes"), \
            "Must provide the number possible labels through num_classes"
        assert self._num_classes > 1
        return self._num_classes

    @property
    def multilabel(self):
        """ Whether the task is a multi-label classification task

            Used by node classification and edge classification
        """
        if hasattr(self, "_multilabel"):
            assert self._multilabel in [True, False]
            return self._multilabel

        return False

    @property
    def multilabel_weights(self):
        """Used to specify label weight of each class in a
           multi-label classification task. It is feed into th.nn.BCEWithLogitsLoss.

           The weights should be in the following format 0.1,0.2,0.3,0.1,0.0
        """
        if hasattr(self, "_multilabel_weights"):
            assert self.multilabel is True, "Must be a multi-label classification task."
            try:
                weights = self._multilabel_weights.split(",")
                weights = [float(w) for w in weights]
            except Exception: # pylint: disable=broad-except
                assert False, "The weights should in following format 0.1,0.2,0.3,0.1,0.0"
            for w in weights:
                assert w >= 0., "multilabel weights can not be negative values"
            assert len(weights) == self.num_classes, \
                "Each class must have an assigned weight"

            return th.tensor(weights)

        return None

    @property
    def return_proba(self):
        """ Whether to return all the predictions or the maximum prediction.
            Set True to return predictions and False to return maximum prediction.
        """
        if hasattr(self, "_return_proba"):
            assert self._return_proba in [True, False], \
                "Return all the predictions when True else return the maximum prediction."

            if self._return_proba is True and \
                self.task_type in [BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_EDGE_REGRESSION]:
                print("WARNING: node regression and edge regression tasks "
                      "automatically ignore --return-proba flag. Regression "
                      "prediction results will be returned.")
            return self._return_proba
        # By default, return all the predictions
        return True

    @property
    def imbalance_class_weights(self):
        """ Used to specify a manual rescaling weight given to each class
            in a single-label multi-class classification task.
            It is used in imbalanced label use cases.
            It is feed into th.nn.CrossEntropyLoss

            Customer should provide the weight in following format 0.1,0.2,0.3,0.1
        """
        if hasattr(self, "_imbalance_class_weights"):
            assert self.multilabel is False, "Only used with single label classfication."
            try:
                weights = self._imbalance_class_weights.split(",")
                weights = [float(w) for w in weights]
            except Exception: # pylint: disable=broad-except
                assert False, \
                    "The rescaling weights should in following format 0.1,0.2,0.3,0.1"
            for w in weights:
                assert w > 0., "Each weight should be larger than 0."
            assert len(weights) == self.num_classes, \
                "Each class must have an assigned weight"

            return th.tensor(weights)

        return None

    ###classification/regression inference related ####
    @property
    def save_prediction_path(self):
        """ Path to save prediction results.
        """
        # pylint: disable=no-member
        if hasattr(self, "_save_prediction_path"):
            return self._save_prediction_path

        # if save_prediction_path is not specified in inference
        # use save_embed_path
        return self.save_embed_path

    ### Node related task variables ###
    @property
    def target_ntype(self):
        """ The node type for prediction
        """
        # pylint: disable=no-member
        assert hasattr(self, "_target_ntype"), \
            "Must provide the target ntype through target_ntype"
        return self._target_ntype

    #### edge related task variables ####
    @property
    def reverse_edge_types_map(self):
        """ A list of reverse edge type info.

            Each information is in the following format:
            <head,relation,reverse relation,tail>. For example:
            ["query,adds,rev-adds,asin", "query,clicks,rev-clicks,asin"]
        """
        # link prediction or edge classification
        assert self.task_type in [BUILTIN_TASK_LINK_PREDICTION, \
            BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION], \
            f"Only {BUILTIN_TASK_LINK_PREDICTION}, " \
            f"{BUILTIN_TASK_EDGE_CLASSIFICATION} and "\
            f"{BUILTIN_TASK_EDGE_REGRESSION} use reverse_edge_types_map"

        # pylint: disable=no-member
        if hasattr(self, "_reverse_edge_types_map"):
            if self._reverse_edge_types_map is None:
                return {} # empty dict
            assert isinstance(self._reverse_edge_types_map, list), \
                "Reverse edge type map should has following format: " \
                "[\"head,relation,reverse relation,tail\", " \
                "\"head,relation,reverse relation,tail\", ...]"

            reverse_edge_types_map = {}
            try:
                for etype_info in self._reverse_edge_types_map:
                    head, rel, rev_rel, tail = etype_info.split(",")
                    reverse_edge_types_map[(head, rel, tail)] = (tail, rev_rel, head)
            except Exception: # pylint: disable=broad-except
                assert False, \
                    "Reverse edge type map should has following format: " \
                    "[\"head,relation,reverse relation,tail\", " \
                    "\"head,relation,reverse relation,tail\", ...]" \
                    f"But get {self._reverse_edge_types_map}"

            return reverse_edge_types_map

        # By default return an empty dict
        return {}

    ### Edge classification and regression tasks ###
    @property
    def target_etype(self):
        """ The list of canonical etype that will be added as
            a training target in edge classification and regression tasks.

            TODO(xiangsx): Only support single task edge
            classification/regression. Support multiple tasks when needed.
        """
        # pylint: disable=no-member
        assert hasattr(self, "_target_etype"), \
            "Edge classification task needs a target etype"
        assert isinstance(self._target_etype, list), \
            "target_etype must be a list in format: " \
            "[\"query,clicks,asin\", \"query,search,asin\"]."
        assert len(self._target_etype) > 0, \
            "There must be at least one target etype."
        if len(self._target_etype) != 1:
            print(f"WARNING: only {self._target_etype[0]} will be used."
                "Currently, GraphStorm only supports single task edge "
                "classification/regression. Please contact GraphStorm "
                "dev team to support multi-task.")

        return [tuple(target_etype.split(',')) for target_etype in self._target_etype]

    @property
    def remove_target_edge_type(self):
        """ Whether to remove the training target edge type for message passing.

            Will set the fanout of training target edge type as zero

            Only used with edge classification
        """
        # pylint: disable=no-member
        if hasattr(self, "_remove_target_edge_type"):
            assert self._remove_target_edge_type in [True, False]
            return self._remove_target_edge_type

        # By default, remove training target etype during
        # message passing to avoid information leakage
        return True

    @property
    def decoder_type(self):
        """ Type of edge clasification or regression decoder
        """
        # pylint: disable=no-member
        if hasattr(self, "_decoder_type"):
            return self._decoder_type

        # By default, use DenseBiDecoder
        return "DenseBiDecoder"

    @property
    def num_decoder_basis(self):
        """ The number of basis for the decoder in edge prediction task.
        """
        # pylint: disable=no-member
        if hasattr(self, "_num_decoder_basis"):
            assert self._num_decoder_basis > 1, \
                "Decoder basis must be larger than 1"
            assert self.decoder_type == "DenseBiDecoder", \
                "num-decoder-basis only works with DenseBiDecoder"
            return self._num_decoder_basis

        # By default, return 2
        return 2

    @property
    def decoder_edge_feat(self):
        """ A list of edge features that can be used by a decoder to
            enhance its performance.
        """
        # pylint: disable=no-member
        if hasattr(self, "_decoder_edge_feat"):
            assert self.task_type in \
                (BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION), \
                "Decoder edge feature only works with " \
                "edge classification or regression tasks"
            decoder_edge_feats = self._decoder_edge_feat
            assert len(decoder_edge_feats) == 1, \
                "We only support edge classifcation or regression on one edge type"

            if ":" not in decoder_edge_feats[0]:
                # global feat_name
                return decoder_edge_feats[0]

            # per edge type feature
            feat_name = decoder_edge_feats[0]
            feat_info = feat_name.split(":")
            assert len(feat_info) == 2, \
                    f"Unknown format of the feature name: {feat_name}, " + \
                    "must be EDGE_TYPE:FEAT_NAME"
            etype = tuple(feat_info[0].split(","))
            assert etype in self.target_etype, \
                f"{etype} must in the training edge type list {self.target_etype}"
            return {etype: feat_info[1].split(",")}

        return None


    ### Link Prediction specific ###
    @property
    def train_negative_sampler(self):
        """ The algorithm of sampling negative edges for link prediction
            training.
        """
        # pylint: disable=no-member
        if hasattr(self, "_train_negative_sampler"):
            return self._train_negative_sampler
        return BUILTIN_LP_UNIFORM_NEG_SAMPLER

    @property
    def eval_negative_sampler(self):
        """ The algorithm of sampling negative edges for link prediction
            evaluation.
        """
        # pylint: disable=no-member
        if hasattr(self, "_eval_negative_sampler"):
            return self._eval_negative_sampler

        # use Joint neg for efficiency
        return BUILTIN_LP_JOINT_NEG_SAMPLER

    @property
    def num_negative_edges(self):
        """ Number of edges consider for the negative batch of edges
        """
        # pylint: disable=no-member
        if hasattr(self, "_num_negative_edges"):
            assert self._num_negative_edges > 0, \
                "Number of negative edges must larger than 0"
            return self._num_negative_edges
        # Set default value to 16.
        return 16

    @property
    def num_negative_edges_eval(self):
        """ Number of edges consider for the negative
            batch of edges for the model evaluation
        """
        # pylint: disable=no-member
        if hasattr(self, "_num_negative_edges_eval"):
            assert self._num_negative_edges_eval > 0, \
                "Number of negative edges must larger than 0"
            return self._num_negative_edges_eval
        # Set default value to 1000.
        return 1000

    @property
    def lp_decoder_type(self):
        """ Type of link prediction decoder
        """
        # pylint: disable=no-member
        if hasattr(self, "_lp_decoder_type"):
            decoder_type = self._lp_decoder_type.lower()
            assert decoder_type in SUPPORTED_LP_DECODER, \
                f"Link prediction decoder {self._lp_decoder_type} not supported. " \
                f"GraphStorm only supports {SUPPORTED_LP_DECODER}"
            return decoder_type

        # Set default value to distmult
        return BUILTIN_LP_DISTMULT_DECODER

    @property
    def lp_edge_weight_for_loss(self):
        """ The edge data fields that stores the edge weights used
            in computing link prediction loss

            The edge_weight can be in following format:
            1) [weight_name]: global weight name, if an edge has weight,
            the corresponding weight name is <weight_name>
            2) ["src0,rel0,dst0:weight0","src0,rel0,dst0:weight1",...]:
            different edge types have different edge weights.

            By default, it is none.
        """
        # pylint: disable=no-member
        if hasattr(self, "_lp_edge_weight_for_loss"):
            assert self.task_type == BUILTIN_TASK_LINK_PREDICTION, \
                "Edge weight for loss only works with link prediction"
            edge_weights = self._lp_edge_weight_for_loss
            if len(edge_weights) == 1 and \
                ":" not in edge_weights[0]:
                # global feat_name
                return edge_weights[0]

            # per edge type feature
            weight_dict = {}
            for weight_name in edge_weights:
                weight_info = weight_name.split(":")
                etype = tuple(weight_info[0].split(","))
                assert etype not in weight_dict, \
                    f"You already specify the weight names of {etype}" \
                    f"as {weight_dict[etype]}"

                # TODO: if train_etype is None, we need to check if
                # etype exists in g.
                assert self.train_etype is None or etype in self.train_etype, \
                    f"{etype} must in the training edge type list"
                assert isinstance(weight_info[1], str), \
                    f"Feature name of {etype} should be a string instead of {weight_info[1]}"
                weight_dict[etype] = [weight_info[1]]
            return weight_dict

        return None

    @property
    def train_etype(self):
        """ The list of canonical etype that will be added as
            training target with the target e type(s)

            If not provided, all edge types will be used as training target.
        """
        # pylint: disable=no-member
        if hasattr(self, "_train_etype"):
            if self._train_etype is None:
                return None
            assert isinstance(self._train_etype, list)
            assert len(self._train_etype) > 0

            return [tuple(train_etype.split(',')) for train_etype in self._train_etype]
        # By default return None, which means use all edge types
        return None

    @property
    def eval_etype(self):
        """ The list of canonical etype that will be added as
            evaluation target with the target edge type(s)

            If not provided, all edge types will be used as evaluation target.
        """
        # pylint: disable=no-member
        if hasattr(self, "_eval_etype"):
            if self._eval_etype is None:
                return None
            assert isinstance(self._eval_etype, list)
            assert len(self._eval_etype) > 0
            return [tuple(eval_etype.split(',')) for eval_etype in self._eval_etype]
        # By default return None, which means use all edge types
        return None

    @property
    def exclude_training_targets(self):
        """ Whether to remove the training targets from
            the computation graph before the forward pass.
        """
        # pylint: disable=no-member
        if hasattr(self, "_exclude_training_targets"):
            assert self._exclude_training_targets in [True, False]

            if self._exclude_training_targets is True:
                assert len(self.reverse_edge_types_map) > 0, \
                    "When exclude training targets is used, " \
                    "Reverse edge types map must be provided."
            return self._exclude_training_targets

        # By default, exclude training targets
        assert len(self.reverse_edge_types_map) > 0, \
            "By default, exclude training targets is used." \
            "Reverse edge types map must be provided."
        return True

    @property
    def gamma(self):
        """ Gamma for DistMult
        """
        assert self.lp_decoder_type == BUILTIN_LP_DISTMULT_DECODER, \
            "Only used with DistMult"
        if hasattr(self, "_gamma"):
            return float(self._gamma)

        # We use this value in DGL-KE
        return 12.0

    @property
    def lp_loss_func(self):
        """ Link prediction loss function
        """
        # pylint: disable=no-member
        if hasattr(self, "_lp_loss_func"):
            assert self._lp_loss_func in BUILTIN_LP_LOSS_FUNCTION
            return self._lp_loss_func
        # By default, return None
        # which means using the default evaluation metrics for different tasks.
        return BUILTIN_LP_LOSS_CROSS_ENTROPY

    @property
    def task_type(self):
        """ Task type
        """
        # pylint: disable=no-member
        assert hasattr(self, "_task_type"), \
            "Task type must be specified"
        assert self._task_type in SUPPORTED_TASKS, \
            f"Supported task types include {SUPPORTED_TASKS}, " \
            f"but got {self._task_type}"

        return self._task_type

    @property
    def eval_metric(self):
        """ Evaluation metric used during evaluation

            The input can be a string specifying the evaluation metric to report
            or a list of strings specifying a list of  evaluation metrics to report.
        """
        # pylint: disable=no-member
        # Task is node classification
        if self.task_type in [BUILTIN_TASK_NODE_CLASSIFICATION, \
            BUILTIN_TASK_EDGE_CLASSIFICATION]:
            assert self.num_classes > 1, \
                "For node classification, num_classes must be provided"

            # check evaluation metrics
            if hasattr(self, "_eval_metric"):
                if isinstance(self._eval_metric, str):
                    eval_metric = self._eval_metric.lower()
                    assert eval_metric in SUPPORTED_CLASSIFICATION_METRICS, \
                        f"Classification evaluation metric should be " \
                        f"in {SUPPORTED_CLASSIFICATION_METRICS}" \
                        f"but get {self._eval_metric}"
                    eval_metric = [eval_metric]
                elif isinstance(self._eval_metric, list) and len(self._eval_metric) > 0:
                    eval_metric = []
                    for metric in self._eval_metric:
                        metric = metric.lower()
                        assert metric in SUPPORTED_CLASSIFICATION_METRICS, \
                            f"Classification evaluation metric should be " \
                            f"in {SUPPORTED_CLASSIFICATION_METRICS}" \
                            f"but get {self._eval_metric}"
                        eval_metric.append(metric)
                else:
                    assert False, "Classification evaluation metric " \
                        "should be a string or a list of string"
                    # no eval_metric
            else:
                eval_metric = ["accuracy"]
        elif self.task_type in [BUILTIN_TASK_NODE_REGRESSION, \
            BUILTIN_TASK_EDGE_REGRESSION]:
            if hasattr(self, "_eval_metric"):
                if isinstance(self._eval_metric, str):
                    eval_metric = self._eval_metric.lower()
                    assert eval_metric in SUPPORTED_REGRESSION_METRICS, \
                        f"Regression evaluation metric should be " \
                        f"in {SUPPORTED_REGRESSION_METRICS}, " \
                        f"but get {self._eval_metric}"
                    eval_metric = [eval_metric]
                elif isinstance(self._eval_metric, list) and len(self._eval_metric) > 0:
                    eval_metric = []
                    for metric in self._eval_metric:
                        metric = metric.lower()
                        assert metric in SUPPORTED_REGRESSION_METRICS, \
                            f"Regression evaluation metric should be " \
                            f"in {SUPPORTED_REGRESSION_METRICS}" \
                            f"but get {self._eval_metric}"
                        eval_metric.append(metric)
                else:
                    assert False, "Regression evaluation metric " \
                        "should be a string or a list of string"
                    # no eval_metric
            else:
                eval_metric = ["rmse"]
        elif self.task_type == BUILTIN_TASK_LINK_PREDICTION:
            if hasattr(self, "_eval_metric"):
                if isinstance(self._eval_metric, str):
                    eval_metric = self._eval_metric.lower()
                    assert eval_metric in SUPPORTED_LINK_PREDICTION_METRICS, \
                        f"Link prediction evaluation metric should be " \
                        f"in {SUPPORTED_LINK_PREDICTION_METRICS}" \
                        f"but get {self._eval_metric}"
                    eval_metric = [eval_metric]
                elif isinstance(self._eval_metric, list) and len(self._eval_metric) > 0:
                    eval_metric = []
                    for metric in self._eval_metric:
                        metric = metric.lower()
                        assert metric in SUPPORTED_LINK_PREDICTION_METRICS, \
                            f"Link prediction evaluation metric should be " \
                            f"in {SUPPORTED_LINK_PREDICTION_METRICS}" \
                            f"but get {self._eval_metric}"
                        eval_metric.append(metric)
                else:
                    assert False, "Link prediction evaluation metric " \
                        "should be a string or a list of string"
                    # no eval_metric
            else:
                eval_metric = ["mrr"]
        else:
            assert False, "Unknow task type"

        return eval_metric

def _add_initialization_args(parser):
    group = parser.add_argument_group(title="initialization")
    group.add_argument(
        "--verbose",
        type=lambda x: (str(x).lower() in ['true', '1']),
        default=argparse.SUPPRESS,
        help="Print more information.",
    )
    return parser

def _add_gsgnn_basic_args(parser):
    group = parser.add_argument_group(title="graphstorm gnn")
    group.add_argument('--backend', type=str, default=argparse.SUPPRESS,
            help='PyTorch distributed backend')
    group.add_argument('--ip-config', type=str, default=argparse.SUPPRESS,
            help='The file for IP configuration')
    group.add_argument('--part-config', type=str, default=argparse.SUPPRESS,
            help='The path to the partition config file')
    group.add_argument("--save-perf-results-path",
            type=str,
            default=argparse.SUPPRESS,
            help="Folder path to save performance results of model evaluation.")
    group.add_argument("--profile-path",
            type=str,
            help="The path of the folder that contains the profiling results.")
    return parser

def _add_gnn_args(parser):
    group = parser.add_argument_group(title="gnn")
    group.add_argument('--model-encoder-type', type=str, default=argparse.SUPPRESS,
            help='Model type can either be gnn or lm to specify the model encoder')
    group.add_argument("--node-feat-name", nargs='+', type=str, default=argparse.SUPPRESS,
            help="Node feature field name. It can be in following format: "
            "1) '--node-feat-name feat_name': global feature name, "
            "if a node has node feature,"
            "the corresponding feature name is <feat_name>"
            "2)'--node-feat-name ntype0:feat0,feat1 ntype1:feat0,feat1 ...': "
            "different node types have different node features.")
    group.add_argument("--fanout", type=str, default=argparse.SUPPRESS,
            help="Fan-out of neighbor sampling. This argument can either be --fanout 20,10 or "
                 "--fanout etype2:20@etype3:20@etype1:20,etype2:10@etype3:4@etype1:2"
                 "Each etype (e.g., etype2) should be a canonical etype in format of"
                 "srcntype/relation/dstntype")
    group.add_argument("--eval-fanout", type=str, default=argparse.SUPPRESS,
            help="Fan-out of neighbor sampling during minibatch evaluation. "
                 "This argument can either be --eval-fanout 20,10 or "
                 "--eval-fanout etype2:20@etype3:20@etype1:20,etype2:10@etype3:4@etype1:2"
                 "Each etype (e.g., etype2) should be a canonical etype in format of"
                 "srcntype/relation/dstntype")
    group.add_argument("--hidden-size", type=int, default=argparse.SUPPRESS,
            help="The number of features in the hidden state")
    group.add_argument("--num-layers", type=int, default=argparse.SUPPRESS,
            help="number of layers in the GNN")
    parser.add_argument(
            "--use-mini-batch-infer",
            help="Whether to use mini-batch or full graph inference during evalution",
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS
    )

    return parser

def _add_input_args(parser):
    group = parser.add_argument_group(title="input")
    group.add_argument('--restore-model-layers', type=str, default=argparse.SUPPRESS,
                       help='Which GraphStorm neural network layers to load.'
                            'The argument ca be --restore-model-layers embed or '
                            '--restore-model-layers embed,gnn,decoder')
    group.add_argument('--restore-model-path', type=str, default=argparse.SUPPRESS,
            help='Restore the model weights saved in the specified directory.')
    group.add_argument('--restore-optimizer-path', type=str, default=argparse.SUPPRESS,
            help='Restore the optimizer snapshot saved in the specified directory.')
    return parser

def _add_output_args(parser):
    group = parser.add_argument_group(title="output")
    group.add_argument("--save-embed-path", type=str, default=argparse.SUPPRESS,
            help="Save the embddings in the specified directory. "
                 "Use none to turn off embedding saveing")
    group.add_argument('--save-model-frequency', type=int, default=argparse.SUPPRESS,
            help='Save the model every N iterations.')
    group.add_argument('--save-model-path', type=str, default=argparse.SUPPRESS,
            help='Save the model to the specified file. Use none to turn off model saveing')
    group.add_argument("--topk-model-to-save",
            type=int, default=argparse.SUPPRESS,
            help="the number of the k top best validation performance model to save")

    # inference related output args
    parser = _add_inference_args(parser)

    return parser

def _add_task_tracker(parser):
    group = parser.add_argument_group(title="task_tracker")
    group.add_argument("--task-tracker", type=str, default=argparse.SUPPRESS,
            help=f'Task tracker name. Now we only support {GRAPHSTORM_SAGEMAKER_TASK_TRACKER}')
    group.add_argument("--log-report-frequency", type=int, default=argparse.SUPPRESS,
            help="Task running log report frequency. "
                 "In training, every log_report_frequency, the task states are reported")
    return parser

def _add_hyperparam_args(parser):
    group = parser.add_argument_group(title="hp")
    group.add_argument("--dropout", type=float, default=argparse.SUPPRESS,
            help="dropout probability")
    group.add_argument("--lr", type=float, default=argparse.SUPPRESS,
            help="learning rate")
    group.add_argument("-e", "--num-epochs", type=int, default=argparse.SUPPRESS,
            help="number of training epochs")
    group.add_argument("--batch-size", type=int, default=argparse.SUPPRESS,
            help="Mini-batch size. Must be larger than 0")
    group.add_argument("--sparse-optimizer-lr", type=float, default=argparse.SUPPRESS,
            help="sparse optimizer learning rate")
    group.add_argument(
            "--use-node-embeddings",
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS,
            help="Whether to use extra learnable node embeddings")
    group.add_argument("--wd-l2norm", type=float, default=argparse.SUPPRESS,
            help="weight decay l2 norm coef")
    group.add_argument("--alpha-l2norm", type=float, default=argparse.SUPPRESS,
            help="coef for scale unused weights l2norm")
    group.add_argument(
            "--use-self-loop",
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS,
            help="include self feature as a special relation")

    # control evaluation
    group.add_argument("--eval-batch-size", type=int, default=argparse.SUPPRESS,
            help="Mini-batch size for computing GNN embeddings in evaluation.")
    group.add_argument('--eval-frequency',
            type=int,
            default=argparse.SUPPRESS,
            help="How offen to run the evaluation. "
                 "Every #eval-frequency iterations.")
    group.add_argument(
            '--no-validation',
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS,
            help="If no-validation is set to True, "
                 "there will be no evaluation during training.")
    # early stop
    group.add_argument("--early-stop-burnin-rounds",
            type=int, default=argparse.SUPPRESS,
            help="Burn-in rounds before start checking for the early stop condition.")
    group.add_argument("--early-stop-rounds",
            type=int, default=argparse.SUPPRESS,
            help="The number of rounds for validation scores to average to decide on early stop")
    group.add_argument("--early-stop-strategy",
            type=str, default=argparse.SUPPRESS,
            help="Specify the early stop strategy. "
            "It can be either consecutive_increase or average_increase")
    group.add_argument("--use-early-stop",
            type=bool, default=argparse.SUPPRESS,
            help='whether to use early stopping by monitoring the validation loss')
    return parser

def _add_lm_model_args(parser):
    group = parser.add_argument_group(title="lm model")
    group.add_argument("--lm-tune-lr", type=float, default=argparse.SUPPRESS,
            help="learning rate for fine-tuning language model")
    group.add_argument("--lm-train-nodes", type=int, default=argparse.SUPPRESS,
            help="number of nodes used in LM model fine-tuning")
    group.add_argument("--lm-infer-batch-size", type=int, default=argparse.SUPPRESS,
            help="Batch size used in LM model inference")
    group.add_argument("--freeze-lm-encoder-epochs", type=int, default=argparse.SUPPRESS,
            help="Before fine-tuning LM model, how many epochs we will take "
                 "to warmup a GNN model")
    return parser

def _add_rgat_args(parser):
    group = parser.add_argument_group(title="rgat")
    group.add_argument("--num-heads", type=int, default=argparse.SUPPRESS,
            help="number of attention heads")
    return parser

def _add_rgcn_args(parser):
    group = parser.add_argument_group(title="rgcn")
    group.add_argument("--num-bases", type=int, default=argparse.SUPPRESS,
            help="number of filter weight matrices, default: -1 [use all]")
    return parser

def _add_node_classification_args(parser):
    group = parser.add_argument_group(title="node classification")
    group.add_argument("--target-ntype", type=str, default=argparse.SUPPRESS,
                       help="the node type for prediction")
    group.add_argument("--label-field", type=str, default=argparse.SUPPRESS,
                       help="the label field in the data")
    group.add_argument(
            "--multilabel",
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS,
            help="Whether the task is a multi-label classifiction task")
    group.add_argument(
            "--multilabel-weights",
            type=str,
            default=argparse.SUPPRESS,
            help="Used to specify label weight of each class in a "
            "multi-label classifiction task."
            "It is feed into th.nn.BCEWithLogitsLoss."
            "The weights should in following format 0.1,0.2,0.3,0.1,0.0 ")
    group.add_argument(
            "--imbalance-class-weights",
            type=str,
            default=argparse.SUPPRESS,
            help="Used to specify a manual rescaling weight given to each class "
            "in a single-label multi-class classification task."
            "It is feed into th.nn.CrossEntropyLoss."
            "The weights should be in the following format 0.1,0.2,0.3,0.1,0.0 ")
    group.add_argument("--num-classes", type=int, default=argparse.SUPPRESS,
                       help="The cardinality of labels in a classifiction task")
    group.add_argument("--return-proba", type=bool, default=argparse.SUPPRESS,
                       help="Whether to return the probabilities of all the predicted \
                       results or only the maximum one. Set True to return the \
                       probabilities. Set False to return the maximum one.")
    return parser

def _add_edge_classification_args(parser):
    group = parser.add_argument_group(title="edge prediction")
    group.add_argument('--target-etype', nargs='+', type=str, default=argparse.SUPPRESS,
            help="The list of canonical etype that will be added as"
                "a training target with the target e type "
                "in this application, for example "
                "--train-etype query,clicks,asin or"
                "--train-etype query,clicks,asin query,search,asin if not specified"
                "then no aditional training target will "
                "be considered")
    group.add_argument("--decoder-edge-feat", nargs='+', type=str, default=argparse.SUPPRESS,
                       help="A list of edge features that can be used by a decoder to "
                            "enhance its performance. It can be in following format: "
                            "--decoder-edge-feat feat or "
                            "--decoder-edge-feat query,clicks,asin:feat0,feat1 "
                            "If not specified, decoder will not use edge feats")

    group.add_argument("--num-decoder-basis", type=int, default=argparse.SUPPRESS,
                       help="The number of basis for the decoder in edge prediction task")

    group.add_argument('--decoder-type', type=str, default=argparse.SUPPRESS,
                       help="Decoder type can either be  DenseBiDecoder or "
                            "MLPDecoder to specify the model decoder")

    group.add_argument(
            "--remove-target-edge-type",
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS,
            help="Whether to remove the target edge type for message passing")

    return parser

def _add_link_prediction_args(parser):
    group = parser.add_argument_group(title="link prediction")
    group.add_argument("--lp-decoder-type", type=str, default=argparse.SUPPRESS,
            help="Link prediction decoder type.")
    group.add_argument("--num-negative-edges", type=int, default=argparse.SUPPRESS,
            help="Number of edges consider for the negative batch of edges.")
    group.add_argument("--num-negative-edges-eval", type=int, default=argparse.SUPPRESS,
            help="Number of edges consider for the negative "
                 "batch of edges for the model evaluation. "
                 "If the MRR saturates at high values or has "
                 "large variance increase this number.")
    group.add_argument("--train-negative-sampler", type=str, default=argparse.SUPPRESS,
            help="The algorithm of sampling negative edges for link prediction.training ")
    group.add_argument("--eval-negative-sampler", type=str, default=argparse.SUPPRESS,
            help="The algorithm of sampling negative edges for link prediction evaluation")
    group.add_argument('--eval-etype', nargs='+', type=str, default=argparse.SUPPRESS)
    group.add_argument('--train-etype', nargs='+', type=str, default=argparse.SUPPRESS,
            help="The list of canonical etype that will be added as"
                "a training target with the target e type "
                "in this application for example "
                "--train-etype query,clicks,asin or"
                "--train-etype query,clicks,asin query,search,asin if not specified"
                "then no aditional training target will "
                "be considered")
    group.add_argument(
            '--exclude-training-targets',
            type=lambda x: (str(x).lower() in ['true', '1']),
            default=argparse.SUPPRESS,
            help="Whether to remove the training targets from the "
                 "computation graph before the forward pass.")
    group.add_argument('--reverse-edge-types-map',
            nargs='+', type=str, default=argparse.SUPPRESS,
            help="A list of reverse egde type info. Each information is in the following format:"
                    "<head,relation,reverse relation,tail>, for example "
                    "--reverse-edge-types-map query,adds,rev-adds,asin or"
                    "--reverse-edge-types-map query,adds,rev-adds,asin "
                    "query,clicks,rev-clicks,asin")
    group.add_argument(
            "--gamma",
            type=float,
            default=argparse.SUPPRESS,
            help="Used in DistMult score func"
    )
    group.add_argument("--lp-edge-weight-for-loss", nargs='+', type=str, default=argparse.SUPPRESS,
            help="Edge feature field name for edge weights. It can be in following format: "
            "1) '--lp-edge-weight-for-loss feat_name': global feature name, "
            "if all edge types use the same edge weight field."
            "The corresponding feature name is <feat_name>"
            "2)'--lp-edge-weight-for-loss query,adds,asin:weight0 query,clicks,asin:weight1 ..."
            "Different edge types have different weight fields.")

    return parser

def _add_task_general_args(parser):
    group = parser.add_argument_group(title="train task")
    group.add_argument('--eval-metric', nargs='+', type=str, default=argparse.SUPPRESS,
            help="The list of canonical etype that will be added as"
                "the evaluation metric used. Supported metrics are accuracy,"
                "precision_recall, or roc_auc multiple metrics"
                "can be specified e.g. --eval-metric accuracy precision_recall")
    return parser

def _add_inference_args(parser):
    group = parser.add_argument_group(title="infer")
    group.add_argument("--save-prediction-path", type=str, default=argparse.SUPPRESS,
                       help="Where to save the prediction results.")
    return parser

# Users can add their own udf parser
