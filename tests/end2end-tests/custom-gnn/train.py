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


    Train GSgnn custom model.
"""

import argparse
import dgl
import torch as th
import graphstorm as gs
from graphstorm import model as gsmodel
from graphstorm.trainer import GSgnnNodePredictionTrainer
from graphstorm.dataloading import GSgnnData, GSgnnNodeDataLoader
from graphstorm.utils import get_device

class MyGNNModel(gsmodel.GSgnnNodeModelBase):
    def __init__(self, g, feat_size, hidden_size, num_classes):
        super(MyGNNModel, self).__init__()
        self._node_input = gsmodel.GSNodeEncoderInputLayer(g, feat_size, hidden_size)
        self._gnn = gsmodel.RelationalGCNEncoder(g, hidden_size, hidden_size, num_hidden_layers=1)
        self._decoder = gsmodel.EntityClassifier(hidden_size, num_classes, multilabel=False)
        self._loss_fn = gsmodel.ClassifyLossFunc(multilabel=False)

    def forward(self, blocks, node_feats, _, labels, input_nodes=None):
        input_nodes = {ntype: blocks[0].srcnodes[ntype].data[dgl.NID].cpu() \
                for ntype in blocks[0].srctypes}
        embs = self._node_input(node_feats, input_nodes)
        embs = self._gnn(blocks, embs)
        target_ntype = list(labels.keys())[0]
        emb = embs[target_ntype]
        labels = labels[target_ntype]
        logits = self._decoder(emb)

        pred_loss = self._loss_fn(logits, labels)
        # L2 regularization of trainable parameters, this also solves the unused weights error
        reg_loss = th.tensor(0.).to(pred_loss.device)
        for d_para in self.parameters():
            reg_loss += d_para.square().sum()
        reg_loss = 0. * reg_loss

        total_loss = pred_loss + reg_loss
        return total_loss

    def predict(self, blocks, node_feats, _):
        input_nodes = {ntype: blocks[0].srcnodes[ntype].data[dgl.NID].cpu() \
                for ntype in blocks[0].srctypes}
        device = blocks[0].device
        embs = self._node_input(node_feats, input_nodes)
        embs = {name: emb.to(device) for name, emb in embs.items()}
        embs = self._gnn(blocks, embs)
        assert len(embs) == 1
        emb = list(embs.values())[0]
        return self._decoder.predict(emb)

    def restore_model(self, restore_model_path, model_layer_to_load):
        pass

    def save_model(self, model_path):
        pass

    def create_optimizer(self):
        # Here we don't set up an optimizer for sparse embeddings.
        return th.optim.Adam(self.parameters(), lr=0.001)

def main(args):
    gs.initialize(ip_config=args.ip_config, backend="gloo",
                  local_rank=args.local_rank)
    train_data = GSgnnData(args.part_config,
                           node_feat_field=args.node_feat)
    for ntype in train_data.g.ntypes:
        print(ntype, train_data.g.nodes[ntype].data.keys())
    feat_size = gs.get_node_feat_size(train_data.g, args.node_feat)
    model = MyGNNModel(train_data.g, feat_size, 16, args.num_classes)
    trainer = GSgnnNodePredictionTrainer(model, topk_model_to_save=1)
    trainer.setup_device(device=get_device())
    train_idxs = train_data.get_node_train_set(args.target_ntype, mask="train_mask")
    dataloader = GSgnnNodeDataLoader(train_data, train_idxs, fanout=[10, 10],
                                     batch_size=1000, node_feats=args.node_feat,
                                     label_field=args.label, train_task=True)
    trainer.fit(train_loader=dataloader, num_epochs=2)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Training GNN model")
    argparser.add_argument("--ip-config", type=str, required=True,
                           help="The IP config file for the cluster.")
    argparser.add_argument("--graph-name", type=str, required=True,
                           help="The graph name.")
    argparser.add_argument("--part-config", type=str, required=True,
                           help="The partition config file.")
    argparser.add_argument("--target-ntype", type=str, required=True,
                           help="The node type for prediction.")
    argparser.add_argument("--node-feat", type=str, required=True,
                           help="The name of the node feature.")
    argparser.add_argument("--label", type=str, required=True,
                           help="The name of the label.")
    argparser.add_argument("--num-classes", type=int, required=True,
                           help="The number of classes.")
    argparser.add_argument("--local_rank", type=int,
                           help="The rank of the trainer.")
    argparser.add_argument("--verbose",
                           type=lambda x: (str(x).lower() in ['true', '1']),
                           default=argparse.SUPPRESS,
                          help="Print more information.")
    args = argparser.parse_args()
    main(args)
