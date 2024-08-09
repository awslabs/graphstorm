.. _multi_task_learning:

Multi-task Learning in GraphStorm
=========================================
In real world graphs, it is common to have multiple tasks defined on the same graph. For example, people
may want to do link prediction as well as node feature reconstruction at the same time to supervise the
training of a GNN model. As another example, people may want to do fraud detection on both seller and
buyer nodes in a seller-product-buyer graph. To support such scenarios, GraphStorm supports
multi-task learning, allowing users to define multiple training targets on different nodes and edges
within a single training loop. The supported training supervisions for multi-task learning include node classification/regression, edge classification/regression, link prediction and node feature reconstruction.


Preparing the Training Data
---------------------------
You can follow the :ref:`Use Your Own Data tutorial<use-own-data>` to prepare your graph data for
multi-task learning. You can define multiple tasks on the same node type or edge type as shown in the JSON example below.

.. code-block:: json

    {
        "version": "gconstruct-v0.1",
        "nodes": [

            ......

            {
                "node_type": "paper",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/paper.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat"
                    }
                ],
                "labels": [
                    {
                        "label_col": "label_class",
                        "task_type": "classification",
                        "split_pct":	[0.8, 0.1, 0.1],
                        "mask_field_names": ["train_mask_class",
                                             "val_mask_class",
                                             "test_mask_class"]
                    },
                    {
                        "label_col": "label_reg",
                        "task_type": "regression",
                        "split_pct":	[0.8, 0.1, 0.1],
                        "mask_field_names": ["train_mask_reg",
                                             "val_mask_reg",
                                             "test_mask_reg"]
                    }
                ]
            },

            ......

        ],
        ......
    }

In the above configuration, we define two tasks for the **paper** nodes. One is a classification task
with the label name of `label_class` and the train/validation/test mask fields as `train_mask_class`,
`val_mask_class` and `test_mask_class`, respectively. Another one is a regression task with label name of `label_reg`
and the train/validation/test mask fields as `train_mask_reg`, `val_mask_reg` and `test_mask_reg`, respectively.

You can also define multiple tasks on different node and edge types as shown in the JSON example below.

.. code-block:: json

    {
        "version": "gconstruct-v0.1",
        "nodes": [

            ......

            {
                "node_type": "paper",
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/nodes/paper.parquet"
                ],
                "node_id_col": "node_id",
                "features": [
                    {
                        "feature_col": "feat",
                        "feature_name": "feat"
                    }
                ],
                "labels": [
                    {
                        "label_col": "label",
                        "task_type": "classification",
                        "split_pct":	[0.8, 0.1, 0.1],
                        "mask_field_names": ["train_mask_class",
                                             "val_mask_class",
                                             "test_mask_class"]
                    }
                ]
            },

                ......

        ],
        "edges": [

            ......

            {
                "relation": [
                    "paper",
                    "citing",
                    "paper"
                ],
                "format": {
                    "name": "parquet"
                },
                "files": [
                    "/tmp/acm_raw/edges/paper_citing_paper.parquet"
                ],
                "source_id_col": "source_id",
                "dest_id_col": "dest_id",
                "labels": [
                    {
                        "task_type": "link_prediction",
                        "split_pct":	[0.8, 0.1, 0.1],
                        "mask_field_names": ["train_mask_lp",
                                             "val_mask_lp",
                                             "test_mask_lp"]
                    }
                ]
            },

        ......

        ]
    }

In the above configuration, we define one task for the **paper** node and one task for the
**paper,citing,paper** edge. The node classification task will take the label name of `label_class` and the train/validation/test mask fields as `train_mask_class`,
`val_mask_class` and `test_mask_class`, respectively. The link prediction task will take the train/validation/test mask fields as `train_mask_lp`, `val_mask_lp` and `test_mask_lp`, respectively.


Construct Graph
~~~~~~~~~~~~~~~~
You can follow the instructions in :ref:`Run graph construction<run-graph-construction>` to use the
GraphStorm construction tool for creating partitioned graph data. Please ensure you
customize the command line arguments such as `--conf-file`, `--output-dir`, `--graph-name` to your
specific values.


Run Multi-task Learning Training
--------------------------------
Running a multi-task learning training task is similar to running other GraphStorm built-in tasks as
detailed in :ref:`Launch Training<launch-training>`. The main difference is to define multiple training
targets in the YAML configuration file.


Define Multi-task for training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can specify multiple training tasks for a training job by providing the `multi_task_learning`
configurations in the YAML file. The following configuration defines two training tasks, one for node
classification and one for edge classification.

.. code-block:: yaml

    ---
    version: 1.0
    gsf:
        basic:
            ...
        ...
        multi_task_learning:
            - node_classification:
                target_ntype: "paper"
                label_field: "label_class"
                mask_fields:
                    - "train_mask_class"
                    - "val_mask_class"
                    - "test_mask_class"
                num_classes: 10
                task_weight: 1.0
            - node_regression:
                target_ntype: "paper"
                label_field: "label_reg"
                mask_fields:
                    - "train_mask_reg"
                    - "val_mask_reg"
                    - "test_mask_reg"
                task_weight: 1.0
            - link_prediction:
                num_negative_edges: 4
                num_negative_edges_eval: 100
                train_negative_sampler: joint
                train_etype:
                    - "paper,citing,paper"
                mask_fields:
                    - "train_mask_lp"
                    - "val_mask_lp"
                    - "test_mask_lp"
                task_weight: 0.5 # weight of the task

Task specific hyperparameters in multi-task learning are same as those for single task learning as
detailed in :ref:`Training and Inference<configurations-run>`, except that two new configs are required,
i.e., `mask_fields` and `task_weight`. The `mask_fields` provides the specific training, validation and
test masks for a task. The `task_weight` defines a task's loss weight value to be multiplied with
its loss value when aggregating all task losses to compute the total loss during training.

In multi-task learning, GraphStorm provides a new unsupervised training signal, i.e., node feature
reconstruction (`BUILTIN_TASK_RECONSTRUCT_NODE_FEAT = "reconstruct_node_feat"`). You can define a
node feature reconstruction task as the following example:

.. code-block:: yaml

    ---
    version: 1.0
    gsf:
        basic:
            ...
        ...
        multi_task_learning:
            - node_classification:
                ...
            - reconstruct_node_feat:
                reconstruct_nfeat_name: "title"
                target_ntype: "movie"
                batch_size: 128
                mask_fields:
                    - "train_mask_c0" # node classification mask 0
                    - "val_mask_c0"
                    - "test_mask_c0"
                task_weight: 1.0
                eval_metric:
                    - "mse"

In the configuration, `target_ntype` defines the target node type, the reconstruct node feature
learning will be applied. `reconstruct_nfeat_name`` defines the name of the feature to be
re-construct. The other configs are same as node regression tasks.


Run Model Training
~~~~~~~~~~~~~~~~~~~
GraphStorm introduces a new command line `graphstorm.run.gs_multi_task_learning` with an additional
argument `--inference` to run multi-task learning tasks. You can use the following command to start a multi-task training job:

.. code-block:: bash

    python -m graphstorm.run.gs_multi_task_learning \
              --workspace <PATH_TO_WORKSPACE> \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config <PATH_TO_GRAPH_DATA> \
              --cf <PATH_TO_CONFIG> \

Run Model Inference
~~~~~~~~~~~~~~~~~~~~
You can use the same command line `graphstorm.run.gs_multi_task_learning` to run inference as following:

.. code-block:: bash

    python -m graphstorm.run.gs_multi_task_learning \
              --inference \
              --workspace <PATH_TO_WORKSPACE> \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config <PATH_TO_GRAPH_DATA> \
              --cf <PATH_TO_CONFIG> \
              --save-prediction-path <PATH_TO_OUTPUT>

The prediction results of each prediction tasks (node classification, node regression,
edge classification and edge regression) will be saved into different sub-directories under PATH_TO_OUTPUT. The sub-directories are prefixed with the `<task_type>_<node/edge_type>_<label_name>`.

Run Model Training on SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GraphStorm supports running multi-task training on :ref:`SageMaker<distributed-sagemaker>` as following:

.. code-block:: bash

    python3 launch/launch_train.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region <REGION> \
        --entry-point run/train_entry.py \
        --role <ROLE_ARN> \
        --graph-data-s3 s3://<PATH_TO_DATA> \
        --graph-name <GRAPH_NAME> \
        --task-type multi_task \
        --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG> \
        --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
        --instance-count <INSTANCE_COUNT> \
        --instance-type <INSTANCE_TYPE>

Run Model Inference on SageMaker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
GraphStorm supports to run multi-task inference on :ref:`SageMaker<distributed-sagemaker>` as following:

.. code-block:: bash

    python3 launch/launch_infer.py \
        --image-url <AMAZON_ECR_IMAGE_URI> \
        --region <REGION> \
        --entry-point run/infer_entry.py \
        --role <ROLE_ARN> \
        --graph-data-s3 s3://<PATH_TO_DATA> \
        --yaml-s3 s3://<PATH_TO_TRAINING_CONFIG> \
        --model-artifact-s3 s3://<PATH_TO_SAVE_TRAINED_MODEL>/ \
        --raw-node-mappings-s3 s3://<PATH_TO_DATA>/raw_id_mappings \
        --output-emb-s3 s3://<PATH_TO_SAVE_GENERATED_NODE_EMBEDDING>/ \
        --output-prediction-s3 s3://<PATH_TO_SAVE_PREDICTION_RESULTS> \
        --graph-name <GRAPH_NAME> \
        --task-type multi_task \
        --instance-count <INSTANCE_COUNT> \
        --instance-type <INSTANCE_TYPE>

Multi-task Learning Output
--------------------------

Saved Node Embeddings
~~~~~~~~~~~~~~~~~~~~~~
When ``save_embed_path`` is provided in the training config or inference condig,
GraphStorm will save the node embeddings in the corresponding path.
In multi-task learning, xxx


Saved Prediction Results
~~~~~~~~~~~~~~~~~~~~~~~~~
When ``save_prediction_path`` is provided in the inference condig,
GraphStorm will save the prediction results in the corresponding path.
In multi-task learning, xxx
