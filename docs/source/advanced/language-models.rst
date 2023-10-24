.. _language_models:

Use Language Model in GraphStorm
==================================
Many real world graphs have text contents as nodes' features, e.g., the title and description of a product, and the questions and comments from users. To leverage these text contents, GraphStorm supports language models (LMs), i.e., HuggingFace BERT models, to embed text contents and use these embeddings for various Graph Machine Learning tasks.

There are a set of modes to use LMs in GraphStorm.

#. Use Pre-trained LMs only without fine-tuning
    In this mode, users can embed text contents with pre-trained LMs, and then use them as the input node features to train GNN models, but do not fine-tune the LMs. Model training speed in this mode is fast, and memory consumption will be lower. However, in some cases, pre-trained LMs may not fit to the graph data well, hence not improving performance.

#. Fine-tune LMs on graph data

    To achieve better performance, it is better to fine-tune LMs with graph data. To achieve this goal, GraphStorm provides four training strategies.

    * Fine-tune LMs only

    In this mode, users can fine-tune LMs relying on labels at graph data, e.g. labels of node/edge or positive/negative edges for link prediction. In this mode, there is no GNN models involved, which means the final performance depends on nodes' features (including text features) only without using graph information. Using this mode may not achieve the best performance, but could pave the way of other modes, e.g. two-step co-training.

    * Two-step co-training manually

    In this mode, users can train both LMs and GNNs in two steps. The first step is similar as the Fine-tune LMs only mode, and the fine-tuned LMs are saved. In the second step, GraphStorm loads the saved LMs, computes embeddings on text features, and then use them to as input to train GNN models. In this mode, users have the options to fine-tune LMs on the GML tasks different from GNN models. For example, the fine-tuning could be done on a link prediction task, and the trained LMs then could be used by GNN models for node classification tasks.

    * Auto Two-step co-trainig (GLEM)

    GraphStorm currently provdies an experimental feature based on `GLEM <https://arxiv.org/abs/2210.14709>`_, which trains LM and GNN models iteratively for node classification. In this mode, GLEM automatically conducts the two step co-training iteratively. Under certain prerequsites, this mode can achieve good performance on node classification.

    * Co-train LMs and GNN

    In this mode, LMs and GML models are co-train in the same training epoch simultaneousely, which will better fit the LMs and GNN models to graph data. However, co-train LMs and GNN models will consume much more memory, particularly GPU memory, and take much longer time to complete training loops.

This tutorial will help users to learn how to use GraphStorm for all of the above modes. Given that using language models on text could take longer time than general GNN training, this tutorial uses a relatively small demo graph. Users can follow the `GraphStorm MAG example <https://github.com/awslabs/graphstorm/tree/main/examples/mag>`_ for GraphStorm's performance of large texture graphs.

.. Note::

    All commands below should be run in a GraphStorm Docker container. Please refer to the :ref:`GraphStorm Docker environment setup<setup>` to prepare your environment.

    If you :ref:`set up the GraphStorm environment with pip Packages<setup_pip>`, please replace all occurrences of "2222" in the argument ``--ssh-port`` with **22**, and clone GraphStorm toolkits.

Prepare Text Graph Data
------------------------

This tutorial will use the same ACM data as the :ref:`Use Your Own Data<use-own-data>` tutorial but add text data as node features.

First go the ``/graphstorm/examples/`` folder.

.. code-block:: bash

    cd /graphstorm/examples 

Then run the command to create the ACM data with the required ``raw_w_text`` format.

.. code-block:: bash
    
    python3 -m /graphstorm/examples/acm_data.py --output-path /tmp/acm_raw --output-type raw_w_text

Once successful, the command will create a set of folders and files under the ``/tmp/acm_raw/`` folder, similar to the :ref:`outputs<acm-raw-data-output>` in the :ref:`Use Your Own Data<use-own-data>` tutorial except that there is one new column, called "text", in node data files as demonstrated in the figure below.

But the contents of the ``config.json`` file have a few extra lines that list the text feature columns and specify how they should be processed during graph contruction. 

The following snippet shows the information of ``author`` nodes. It indicates that the "**text**" column contains text features, and it require the GraphStorm's graph contruction tool to use a `HuggingFace BERT model <https://huggingface.co/models>`_ named ``bert-base-uncased`` to tokenize these text features during construction.

.. code-block:: json

    "nodes": [
        {
            "node_type": "author",
            "format": {
                "name": "parquet"
            },
            "files": [
                "/tmp/acm_raw/nodes/author.parquet"
            ],
            "node_id_col": "node_id",
            "features": [
                {
                    "feature_col": "feat",
                    "feature_name": "feat"
                },
                {
                    "feature_col": "text",
                    "feature_name": "text",
                    "transform": {
                        "name": "tokenize_hf",
                        "bert_model": "bert-base-uncased",
                        "max_seq_length": 16
                    }
                }
            ]
        }

Construct Graph
------------------
Then we use the graph construction tool to process the ACM raw data with the following command for GraphStorm model training.

.. code-block:: bash

    python3 -m graphstorm.gconstruct.construct_graph \
               --conf-file /tmp/acm_raw/config.json \
               --output-dir /tmp/acm_nc \
               --num-parts 1 \
               --graph-name acm

Outcomes of this command are also same as the :ref:`Outputs of Graph Construction<output-graph-construction>`. But users may notice that the ``paper``, ``author``, and ``subject`` nodes all have three additional features, named ``input_ids``,``attention_mask``, and ``token_type_ids``, which are generated by the BERT tokenizer.

GraphStorm Language Model Configuration
-----------------------------------------
Users can set up language model in GraphStorm's configuration YAML file. Below is an example of such configuration for the ACM data. The full configuration YAML file, `acm_lm_nc.yaml <https://github.com/awslabs/graphstorm/blob/main/examples/use_your_own_data/acm_lm_nc.yaml>`_, is located under GraphStorm's ``examples/use_your_own_data`` folder.

.. code-block:: yaml

  lm_model:
  node_lm_models:
    -
      lm_type: bert
      model_name: "bert-base-uncased"
      gradient_checkpoint: true
      node_types:
        - paper
        - author
        - subject

The current version of GraphStorm supports several types of pre-trained LM models from HuggingFace reposity on nodes only. Users can choose any `HuggingFace LM models <https://huggingface.co/models>`_ under the following ``lm_type``: ``"bert", "roberta", "albert", "camembert", "ernie", "ibert", "luke", "mega", "mpnet", "nezha", "qdqbert","roc_bert"``. But the value of ``model_name`` **MUST** be the same as the one specified in the raw data JSON file's ``bert_model`` field. Here in the example, it is the ``bert-base-uncased`` model.

The ``node_type`` field lists the types of nodes that have tokenized text features. In this ACM example, all three types of nodes have tokenized text features, which are all list in the configuration YAML file.

Launch GraphStorm Trainig without Fine-tuning BERT Models
------------------------------------------------------------
With the above GraphStorm configuration YAML file, we can launch GraphStorm model training with the same commands as in the :ref:`Step 3: Launch training script on your own graphs<launch_training_oyog>`. 

First, we create the ``ip_list.txt`` file for the standalone mode.

.. code-block:: bash

    touch /tmp/ip_list.txt
    echo 127.0.0.1 > /tmp/ip_list.txt

Then, the launch command is almost the same except that in this case the configuration file is ``acm_lm_nc.yaml``, which contains the language model configurations.

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
            --workspace /tmp \
            --part-config /tmp/acm_nc/acm.json \
            --ip-config /tmp/ip_list.txt \
            --num-trainers 4 \
            --num-servers 1 \
            --num-samplers 0 \
            --ssh-port 2222 \
            --cf /tmp/acm_lm_nc.yaml \
            --save-model-path /tmp/acm_nc/models \
            --node-feat-name paper:feat author:feat subject:feat

In the training process, GraphStorm will first use the specified BERT model to compute the text embeddings in the specified node types. And then the text embeddings and other node features are concatenated together as the input node feature for GNN models training.

Launch GraphStorm Trainig for both BERT and GNN Models
---------------------------------------------------------
To co-train BERT and GNN models, we need to add one more argument, ``--lm-train-nodes``, to either the launch command or the configuration YAML file. Below command sets this argument to the launch command.

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
            --workspace /tmp \
            --part-config /tmp/acm_nc/acm.json \
            --ip-config /tmp/ip_list.txt \
            --num-trainers 4 \
            --num-servers 1 \
            --num-samplers 0 \
            --ssh-port 2222 \
            --cf /tmp/acm_lm_nc.yaml \
            --save-model-path /tmp/acm_nc/models \
            --node-feat-name paper:feat author:feat subject:feat \
            --lm-train-nodes 10

The ``--lm-train-nodes`` argument determines how many nodes will be used in each mini-batch per GPU to tune the BERT models. Because the BERT models are normally large, training of them will consume many memories. If use all nodes to co-train BERT and GNN models, it could cause GPU out of memory (OOM) errors. Use a smaller number for the ``--lm-train-nodes`` could reduce the overall GPU memory consumption.

.. note:: It will take longer time to co-train BERT and GNN models compared to no co-train.

Only Use BERT Models
------------------------
GraphStorm also allows users to only use BERT models to perform graph tasks. We can add another argument, ``--lm-encoder-only``, to control whether only use BERT models or not.

If users want to fine tune the BERT model only, just add the ``--lm-train-nodes`` argument as the command below:

.. code-block:: bash

    python3 -m graphstorm.run.gs_node_classification \
            --workspace /tmp \
            --part-config /tmp/acm_nc/acm.json \
            --ip-config /tmp/ip_list.txt \
            --num-trainers 4 \
            --num-servers 1 \
            --num-samplers 0 \
            --ssh-port 2222 \
            --cf /tmp/acm_lm_nc.yaml \
            --save-model-path /tmp/acm_nc/models \
            --node-feat-name paper:feat author:feat subject:feat \
            --lm-encoder-only \
            --lm-train-nodes 10

.. note:: The current version of GraphStorm requires **ALL** node types must have text features when users want to do the above graph-aware LM fine-tuning only.
