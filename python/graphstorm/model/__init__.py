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

    Package initialization. Import necessary classes.
"""
from .embed import GSNodeEncoderInputLayer
from .lm_embed import GSLMNodeEncoderInputLayer, GSPureLMNodeInputLayer

from .utils import sparse_emb_initializer

from .gnn import GSgnnModel, GSgnnModelBase, GSOptimizer
from .gnn import do_full_graph_inference
from .gnn import do_mini_batch_inference
from .node_gnn import GSgnnNodeModel, GSgnnNodeModelBase, GSgnnNodeModelInterface
from .node_gnn import (node_mini_batch_gnn_predict,
                       node_mini_batch_predict,
                       run_node_mini_batch_predict)
from .edge_gnn import GSgnnEdgeModel, GSgnnEdgeModelBase, GSgnnEdgeModelInterface
from .edge_gnn import (edge_mini_batch_gnn_predict,
                       edge_mini_batch_predict,
                       run_edge_mini_batch_predict)
from .lp_gnn import (GSgnnLinkPredictionModel,
                     GSgnnLinkPredictionModelBase,
                     GSgnnLinkPredictionModelInterface,
                     run_lp_mini_batch_predict)
from .multitask_gnn import (GSgnnMultiTaskModelInterface,
                            GSgnnMultiTaskSharedEncoderModel)
from .multitask_gnn import (multi_task_mini_batch_predict,
                            gen_emb_for_nfeat_reconstruct)
from .rgcn_encoder import RelationalGCNEncoder, RelGraphConvLayer
from .rgat_encoder import RelationalGATEncoder, RelationalAttLayer
from .sage_encoder import SAGEEncoder, SAGEConv
from .gat_encoder import GATEncoder, GATConv
from .gatv2_encoder import GATv2Encoder, GATv2Conv
from .hgt_encoder import HGTEncoder, HGTLayer

from .node_decoder import EntityClassifier, EntityRegression
from .edge_decoder import (DenseBiDecoder,
                           MLPEdgeDecoder,
                           MLPEFeatEdgeDecoder,
                           LinkPredictDotDecoder,
                           LinkPredictDistMultDecoder,
                           LinkPredictWeightedDotDecoder,
                           LinkPredictWeightedDistMultDecoder,
                           LinkPredictContrastiveDotDecoder,
                           LinkPredictContrastiveDistMultDecoder,
                           LinkPredictRotatEDecoder,
                           LinkPredictContrastiveRotatEDecoder,
                           LinkPredictWeightedRotatEDecoder)

from .gnn_encoder_base import GraphConvEncoder

from .loss_func import ClassifyLossFunc, RegressionLossFunc, LinkPredictBCELossFunc
