.. _use-own-models:

Use Your Own Models
======================
Currently GraphStorm has two built-in GNN models, i.e., the RGCN and the RGAT model. If users want to further explore different GNN models and leverage the GraphStorm's ease-of-use and scalability, you can create your own GNN models according to the GraphStorm's customer model APIs. This tutorial will explain in detail how to do this with a runnable `example <https://github.com/awslabs/graphstorm/tree/main/examples/customized_models/HGT>`_ that customizes the HGT model implementation.

.. _use-own-models-prerequisites:

Prerequisites
---------------
Before following GraphStorm's customized model APIs, please make sure your GNN models meet the prerequisites.

.. _use-own-models-prerequisites-1:

Use DGL to implement your GNN models
.....................................
The GraphStorm Framework relies on the `DGL library <https://www.dgl.ai/>`_ to implement and run GNN models. Particularly, the GraphStorm's scalability comes from the DGL's distributed libraries. For this reason, your GNN models should be implemented with the DGL Library. You can learn how to do this via the DGL's `User Guide <https://docs.dgl.ai/guide/index.html>`_. In addition, there are many `GNN model examples <https://github.com/dmlc/dgl/tree/master/examples>`_ implemented by the DGL community. Please explore these materials to check if there is any model that may meet your requirements.

.. _use-own-models-prerequisites-2:

Modify you GNN models to use mini-batch training/inference
..........................................................
Many existing GNN models were implemented for running on popular academic graphs, which, compared to enterprise-level graphs, are relatively small and lack node/edge features. Therefore, implementors use the full-graph training/inference mode, i.e., feed the entire graph along with its node/edge features into GNN models in one epoch. When dealing with large graphs, this mode will fail due to either the limits of the GPUs' memory, or the slow speed if using CPUs.

In order to tackle large graphs, we can change GNN models to perform stochastic mini-batch training. You can learn how to modify GNN models into mini-batch training/inference mode via the `DGL User Guide Chapter 6 <https://docs.dgl.ai/en/1.0.x/guide/minibatch.html>`_. For examples of the different implementations between full-graph mode and mini-batch mode, please look for DGL model examples, in which mini-batch mode files normally have a file name ended with `_mb`` string, like the `RGCN model <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify_mb.py>`_, or file names including `dist string, like the `GraphSage distributed model <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/dist/train_dist.py#L26>`_.

.. _use-own-models-prerequisites-3:

Learn how to run GraphStorm in a Docker environment
......................................................
Currently GraphStorm runs on Docker environment. The rest of the tutorial assumes execution within the GraphStorm Docker container. Please refer to the first two sections in the :ref:`Environment Setup<setup>` to learn how to run GraphStorm in a Docker environment, and set up your environment.

Modifications required for customer models
---------------------------------------------------------------

.. _step-1:

Step 1: Convert your graph data into required format
.....................................................
Users can follow the :ref:`User Your Own Graph Data <use-own-data>` tutorial to prepare your graph data for GraphStorm.

.. _step-2:

Step 2: Modify your GNN model to use the GraphStorm APIs
.........................................................
To plug your GNN models into GraphStorm, you need to use the GraphStorm model APIs. The key model APIs are the class `GSgnnNodeModelBase <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/model/node_gnn.py#L76>`_, `GSgnnEdgeModelBase <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/model/edge_gnn.py#L80>`_, and `GSgnnLinkPredictionModelBase <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/model/lp_gnn.py#L58>`_. Your GNN models should inherit one of the three classes depending on your task.

Here we use the `DGL HGT example model <https://github.com/dmlc/dgl/blob/master/examples/pytorch/hgt/model.py>`_ to demonstrate how to modify the GNN models.

.. code-block:: python

    class HGT(nn.Module):
        def __init__(self, ......):
            super(HTG, self).__init__()
            ......

        def forward(self, G, out_key):
            h = {}
            for ntype in G.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data["inp"]))
            for i in range(self.n_layers):
                h = self.gcs[i](G, h)
            return self.out(h[out_key])

The original HGT model implement uses full-graph training and inference mode. Its ``forward()`` function takes a DGL graph, ``G``, and the to-be predicted node type, ``out_key``, as input arguments.

As the :ref:`Prerequisites <use-own-models-prerequisites-2>` required, we first revise this model to use mini-batch training and inference mode as shown below.

.. code-block:: python

    class HGT_mb(nn.Module):
        def __init__(self, ......)
            super(HGT_mb, self).__init__()
            ......

        def forward(self, blocks, n_feats_dict, out_ntype):
            h = {}
            for ntype in blocks[0].ntypes:
                if self.adapt_ws[ntype] is None:
                    n_id = self.node_dict[ntype]
                    emb_id = self.ntype_id_map[n_id]
                    n_embed = self.ntype_embed(torch.Tensor([emb_id] * blocks[0].num_nodes(ntype)).long().to(self.device))
                else:
                    n_embed = self.adapt_ws[ntype](n_feats_dict[ntype])
                h[ntype] = F.gelu(n_embed)

            for i in range(self.n_layers):
                h = self.gcs[i](blocks[i], h)

            return self.out(h[out_ntype])

The new ``HGT_mb`` model's ``forward()`` function takes mini-batch blocks, ``blocks``, and their corresponding node feature dictionary, ``n_feats_dict``, as inputs to replace the original full graph data, ``G``.

Then to further make this ``HGT_mb`` model work in GraphStorm, we need replace the PyTorch ``nn.Module`` with GraphStorm's ``GSgnnNodeModelBase`` and implement required functions.

The ``GSgnnNodeModelBase`` class, which is also a PyTorch Module extension, has three required functions that users' own GNN model need to implement, including ``forward(self, blocks, node_feats, edge_feats, labels, input_nodes)``, ``predict(self, blocks, node_feats, edge_feats, input_nodes)``, and ``create_optimizer(self)``.

The ``GSgnnNodeModelBase`` class' ``forward()`` function is similar to the PyTorch Module's ``forward()`` function except that its input arguments **MUST** include:

* **blocks**, which are DGL blocks sampled for a mini-batch.
* **labels**, which is a dictionary, whose key is the to-be predicted node type, and value is the labels of the to-be predicted nodes in a mini-batch.
* **node_feats**, which is a dictionary, whose keys are node types in the graph, and values are the node features associated to.
* **edge_feats**. Currently GraphStorm does **NOT** support edge features. So, leave as it is.
* **input_nodes**, optional only if your GNN model needs them.

Unlike common cases where forward function returns logits computed by models, the return value of ``forward()`` should be a loss value, which GraphStorm will use to perform backward operations. Because of this change, you need to include a loss function within your GNN models, instead of computing loss outside. Following these requirements, our revised model will have a few more lines added as shown below.

.. code-block:: python

    class HGT(gsmodel.GSgnnNodeModelBase):
        def __init__(self, ......)

        # use GraphStorm loss function components
        self._loss_fn = gsmodel.ClassifyLossFunc(multilabel=False)

    def forward(self, blocks, node_feats, edge_feats, labels, input_nodes):
        h = {}
        for ntype in blocks[0].ntypes:
            if self.adapt_ws[ntype] is None:
                n_id = self.node_dict[ntype]
                emb_id = self.ntype_id_map[n_id]
                embeding = self.ntype_embed(torch.Tensor([emb_id]).long().to('cuda'))
                n_embed = embeding.expand(blocks[0].num_nodes(ntype), -1)
            else:
                n_embed = self.adapt_ws[ntype](node_feats[ntype])
            h[ntype] = F.gelu(n_embed)
        for i in range(self.num_layers):
            h = self.gcs[i](blocks[i], h)
        for ntype, emb in h.items():
            h[ntype] = self.out(emb)
        pred_loss = self._loss_fn(h[self.target_ntype], labels[self.target_ntype])

        return pred_loss

You may notice that GraphStorm already provides common loss functions for classification, regression and link prediction, which can be easily imported and used in your model. But you are free to use any PyTorch loss functions or even your own loss function. In the above example, we also change the to-be predicted node type as a class variable, and use it for computing the loss value.

The ``predict()`` function is for inference and it will not be used for backward. Its input arguments are similar to the forward() function, but no need for labels. The ``predict()`` will return two values. The first is the prediction results. The second will be the probability if the argument ``return_proba`` is True, otherwise will return the raw logits, which could be used for some specific purposes. With these requirements, the ``predict()`` function of the modified HGT model is like the code below.

.. code-block:: python

    def predict(self, blocks, node_feats, _, input_nodes, return_proba):
        h = {}
        for ntype in blocks[0].ntypes:
            if self.adapt_ws[ntype] is None:
                n_id = self.node_dict[ntype]
                emb_id = self.ntype_id_map[n_id]
                embeding = self.ntype_embed(torch.Tensor([emb_id]).long().to('cuda'))
                n_embed = embeding.expand(blocks[0].num_nodes(ntype), -1)
            else:
                n_embed = self.adapt_ws[ntype](node_feats[ntype])
            h[ntype] = F.gelu(n_embed)
        for i in range(self.num_layers):
            h = self.gcs[i](blocks[i], h)
        for ntype, emb in h.items():
            h[ntype] = self.out(emb)
        if return_proba:
            return h[self.target_ntype].argmax(dim=1), torch.softmax(h[self.target_ntype], 1)
        else:
            return h[self.target_ntype].argmax(dim=1), h[self.target_ntype]

The ``create_optimizer()`` function is for users to define their own optimizer, like the code below.

.. code-block:: python

    def create_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

There are other two required functions in the `GSgnnNodeModelBase <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/model/node_gnn.py#L76>`_ class, including ``restore_model(self, restore_model_path)`` and ``save_model(self, model_path)``, which are used to restore and save models. If you want to save or restore models, implement these two functions too. If not, you can just leave it unimplemented as the below code:

.. code-block:: python

    def restore_model(self, restore_model_path):
        pass

    def save_model(self, model_path):
        pass

Step 3. Modify the training/inference flow with the GraphStorm APIs
....................................................................
With the modified GNN models ready, the next step is to modify the training/inference loop by replacing datasets and dataloaders with the GraphStorm's dataloading classes.

The original HGT_mb model uses the `DGL Stochastic Trainingon Large Graph Guide <https://docs.dgl.ai/guide/minibatch-node.html#guide-minibatch-node-classification-sampler>`_ method for the training/infernece flow. GraphStorm training/inference flow is similar with a few modifications.

Start training process with GraphStorm's iniatilization
```````````````````````````````````````````````````````````
Any GraphStorm training process **MUST** start with a proper initialization. You can use the following codes at the beginning of training flow.

.. code-block:: python

    import graphstorm as gs
    ......

    def main(args):
        gs.initialize(ip_config=args.ip_config, backend="gloo", local_rank=args.local_rank)

the ``ip_config`` argument specifies a ip configuration file, which contains the IP addresses of machines in a GraphStorm distributed cluster. You can find its description at the :ref:`Launch Training<launch-training>` section of the :ref:`Quick Start Tutorial <quick-start-standalone>`. The ``local_rank`` argument specifies the PyTorch local rank of the current process. It is used by GraphStorm to setup the GPU device.

Replace DGL DataLoader with the GraphStorm's dataset and dataloader
`````````````````````````````````````````````````````````````````````
Because the GraphStorm uses distributed graphs, we need to first load the partitioned graph, which is created in the :ref:`Step 1 <step-1>`, with the `GSgnnData <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/dataloading/dataset.py#L57>`_ class (for edge tasks, the same class is used). The ``GSgnnData`` could be created as shown in the code below.

.. code-block:: python

    train_data = GSgnnData(config.part_config)

Arguments of this class include the partition configuration JSON file path, which are the outputs of the :ref:`Step 1 <step-1>`.

Then we can put this dataset into GraphStorm's `GSgnnNodeDataLoader <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/dataloading/dataloading.py#L1237>`_, which is like:

.. code-block:: python

    # Get train idx
    train_idxs = train_data.get_node_train_set(config.target_ntype)
    # Define the GraphStorm train dataloader
    dataloader = GSgnnNodeDataLoader(train_data,
                                     train_idxs, fanout=config.fanout,
                                     batch_size=config.batch_size,
                                     label_field=config.label_field,
                                     node_feats=node_feat_fields,train_task=True)

    # Optional: Define the evaluation dataloader
    val_idxs = train_data.get_node_val_set(eval_ntype)
    eval_dataloader = GSgnnNodeDataLoader(train_data,
                                          val_idxs,
                                          fanout=config.fanout,
                                          batch_size=config.eval_batch_size,
                                          label_field=config.label_field,
                                          node_feats=node_feat_fields,
                                          train_task=False)
    # Optional: Define the evaluation dataloader
    test_idxs = train_data.get_node_test_set(eval_ntype)
    test_dataloader = GSgnnNodeDataLoader(train_data,
                                          test_idxs,
                                          fanout=config.fanout,
                                          batch_size=config.eval_batch_size,
                                          label_field=config.label_field,
                                          node_feats=node_feat_fields,
                                          train_task=False)

GraphStorm provides a set of dataloaders for different GML tasks. Here we deal with a node task, hence using the node dataloader, which takes the graph data created above as the first argument. The second argument is the label index that the GraphStorm dataset extracts from the graph as indicated in the target nodes' ``train_mask``, ``val_mask``, and ``test_mask``, which are automatically generated by GraphStorm graph construction tool with the specified ``split_pct`` field. The ``GSgnnData`` provides functions to get the indexes of train data, validation data and test data through ``get_node_train_set``, ``get_node_val_set`` and ``get_node_test_set``, respectively.
The ``label_field`` is also required by the GSgnnNodeDataLoader to get the labels for model training and evaluation.
The ``node_feats`` and ``edge_feats`` are optional to GSgnnNodeDataLoader, which define the node features and edge features, respectively, to be used for the task associated with the dataloader.
The rest of arguments are similar to the common training flow, except that we set the ``train_task`` to be ``False`` for the evaluation and test dataloader.

Use GraphStorm's model trainer to wrap your model and attach evaluator and task tracker to it
````````````````````````````````````````````````````````````````````````````````````````````````
Unlike the common flow, GraphStorm wraps GNN models with different trainers just like other frameworks, e.g. scikit-learn. GraphStorm provides node prediction, edge prediction, and link prediction trainers. Creation of them is easy.

First we create the modified HGT model like the following code.

.. code-block:: python

    # Define the HGT model
    model = HGT(node_dict, edge_dict,
                n_inp_dict=nfeat_dims,
                n_hid=config.hidden_size,
                n_out=config.num_classes,
                num_layers=num_layers,
                num_heads=args.num_heads,
                target_ntype=config.target_ntype,
                use_norm=True,
                alpha_l2norm=config.alpha_l2norm)

Then we can use the `GSgnnNodePredictionTrainer <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/trainer/np_trainer.py#L29>`_ class to wrap it like:

.. code-block:: python

    # Create a trainer for the node classification task.
    trainer = GSgnnNodePredictionTrainer(model)

The ``GSgnnNodePredictionTrainer`` takes a GraphStorm model as the first argument. The seconde argument is for using different GPUs.

The GraphStorm trainers can have evaluators and task trackers associated. The following code shows how to do this.

.. code-block:: python

    # Optional: set up a evaluator
    evaluator = GSgnnClassificationEvaluator(config.eval_frequency,
                                             config.eval_metric,
                                             config.multilabel,
                                             config.use_early_stop,
                                             config.early_stop_burnin_rounds,
                                             config.early_stop_rounds,
                                             config.early_stop_strategy)
    trainer.setup_evaluator(evaluator)
    # Optional: set up a task tracker to show the progress of training.
    tracker = GSSageMakerTaskTracker(config.eval_frequency)
    trainer.setup_task_tracker(tracker)

GraphStorm's `evaluators <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/eval/evaluator.py>`_ could help to compute the required evaluation metrics, such as ``accuracy``, ``f1``, ``mrr``, and etc. Users can select the proper evaluator and use the trainer's ``setup_evaluator()`` method to attach them. GraphStorm's `task trackers <https://github.com/awslabs/graphstorm/blob/main/python/graphstorm/tracker/graphstorm_tracker.py>`_ serve as log collectors, which are used to show the process information.

Use trainer's ``fit()`` function to run training
``````````````````````````````````````````````````
Once all trainers, evaluators, and task trackers are set, the last step is to use the trainer's ``fit()`` function to run training, validating, and testing on the three sets like the code below.

.. code-block:: python

    # Start the training process.
    trainer.fit(train_loader=dataloader,
                num_epochs=config.num_epochs,
                val_loader=eval_dataloader,
                test_loader=test_dataloader,
                save_model_path=config.save_model_path,
                use_mini_batch_infer=True)

The ``fit()`` function wraps dataloaders, number of epochs, to replace the common "**for loops**" as seen in the common training flow. The ``fit()`` function also takes additional arguments, such as ``save_model_path`` to save different model artifacts. **BUT** before set these arguments, you need to implement the ``restore_model(self, restore_model_path)`` and ``save_model(self, model_path)`` functions in the :ref:`Step 2 <step-2>`.

Step 4. Handle the unused weights error
...................................................
Uncommonly seen in the full-graph training or mini-batch training on a single GPU, the unused weights error could frequently occur when we start to train models on multiple GPUs in parallel. PyTorch distributed framework's inner mechanism causes this problem. One easy way to solve this error is to add a regularization to all trainable parameters into the loss computation like the code blow.

.. code-block:: python

        pred_loss = self._loss_fn(h[self.target_ntype], labels[self.target_ntype])

        reg_loss = torch.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.parameters():
            reg_loss += d_para.square().sum()

        reg_loss = self.alpha_l2norm * reg_loss

        return pred_loss + reg_loss

You can add a coefficient, like the ``alpha_l2norm``, to control the influence of the regularization.

Step 5. Add a few additional arguments for the Python main function
......................................................................
Because GraphStorm relys on a few arguments to launch training and inference command, including: ``part-config``, ``ip-config``, ``verbose``, and ``local_rank``. GraphStorm's built-in launch scripts have this argument configured already. But for customized models, it is required to add them as arguments of the Python main function although these arguments are not used anywhere in the customized model. A sample code is shown below.

.. code-block:: python

    if __name__ == '__main__':
        argparser = argparse.ArgumentParser("Training HGT model with the GraphStorm Framework")
        ......
        argparser.add_argument("--part-config", type=str, required=True,
                            help="The partition config file. \
                                    For customized models, MUST have this argument!!")
        argparser.add_argument("--ip-config", type=str, required=True,
                            help="The IP config file for the cluster. \
                                    For customized models, MUST have this argument!!")
        argparser.add_argument("--verbose",
                            type=lambda x: (str(x).lower() in ['true', '1']),
                            default=argparse.SUPPRESS,
                            help="Print more information. \
                                    For customized models, MUST have this argument!!")
        argparser.add_argument("--local_rank", type=int,
                            help="The rank of the trainer. \
                                    For customized models, MUST have this argument!!")

.. note:: PyTorch v2.0 change the argument ``local_rank`` to ``local-rank``. Therefore, if users use PyTorch v2.0 or later version, please change this argument accordingly.

Step 6. Setup GraphStorm configuration YAML file
.....................................................................
GraphStorm has a set of parameters that control the various perspectives of the model training and inference process. You can find the details of these parameters in the GraphStorm :ref:`Training and Inference Configurations <configurations-run>`. These parameters could be either passed as input arguments or set in a YAML format file. Below is an example of the YAML file.

.. code-block:: yaml

    ---
    version: 1.0
    gsf:
    basic:
        model_encoder_type: rgcn
        backend: gloo
        verbose: false
        alpha_l2norm: 0.
    gnn:
        fanout: "50,50"
        num_layers: 2
        hidden_size: 256
        use_mini_batch_infer: false
    input:
        restore_model_path: null
    output:
        topk_model_to_save: 7
        save_model_path: /data/outputs
        save_embeds_path: /data/outputs
        save_prediction_path: /data/outputs
    hyperparam:
        dropout: 0.
        lr: 0.0001
        num_epochs: 20
        batch_size: 1024
        wd_l2norm: 0
    rgcn:
        num_bases: -1
        use_self_loop: true
        sparse_optimizer_lr: 1e-2
        use_node_embeddings: false
    node_classification:
        target_ntype: "paper"
        label_field: "label"
        multilabel: false
        num_classes: 14

Users can use an argument to read in this YAML file, and construct a ``GSConfig`` object like the below code. And then use the GSConfig instance, e.g., ``config``, to provide arguments that the GraphStorm supports.

.. code-block:: python

    from graphstorm.config import GSConfig
    ......
    argparser.add_argument("--yaml-config-file", type=str, required=True, help="The GraphStorm YAML configuration file path.")
    args = argparser.parse_args()
    config = GSConfig(args)

For users' own configurations, you still can pass them as input argument of the training script, and extract them from the ``args`` object.

Put Everything Together and Run them
-------------------------------------
With all required modifications ready, let's put everything of the modified HGT model together in a Python file, e.g, ``hgt_nc.py``. We can put the Python file and the related artifacts, such as the YAML file, ``acm_nc.yaml``, in a folder, e.g. ``/hgt_nc/``. And then use the GraphStorm's launch script to run this modified HGT model.

.. code-block:: python

    python -m graphstorm.run.launch \
              --workspace /graphstorm/examples/customized_models/HGT \
              --part-config /data/acm_nc/acm.json \
              --num-trainers 1 \
              --num-servers 1 \
              --num-samplers 0 \
              hgt_nc.py --yaml-config-file acm_nc.yaml \
                        --node-feat paper:feat-author:feat-subject:feat \
                        --num-heads 8

The argument value of ``--part-config`` is the JSON file coming from the :ref:`outputs <output-graph-construction>` of the :ref:`Step 1 <step-1>`.

.. note:: To try this runnable example, please follow the `GraphStorm examples readme <https://github.com/awslabs/graphstorm/tree/main/examples/customized_models/HGT>`_.
