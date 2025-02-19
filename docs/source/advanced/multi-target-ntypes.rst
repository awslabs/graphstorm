.. _multi_target_ntypes:

Multiple Target Node Types Training
===================================

When training on a heterogeneous graph, we often need to train a model by minimizing the objective
function on more than one node type. GraphStorm provides supports to achieve this goal. The recommended
method is to leverage GraphStorm's multi-task learning method, i.e., using multiple node tasks, and each
trained on one target node type. 

More detailed guide of using multi-task learning can be found in
:ref:`Multi-task Learning in GraphStorm<multi_task_learning>`. This guide provides two examples of how
to conduct two target node type classification training on the `movielen 100k <https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset>`_
data, where the **movie** ("item" in the original data) and **user** node types have classification
labels associated.

Using multi-task learning for multiple target node types training (Recommended)
--------------------------------------------------------------------------------

Preparing the training data
............................

During graph construction step, you can define two classification tasks on the two node type as
shown in the JSON example below.

.. code-block:: json

    {
        "version": "gconstruct-v0.1",
        "nodes": [
            {
                "node_type": "movie",
                ......
                ],
                "labels": [
                    {
                        "label_col": "label_movie",
                        "task_type": "classification",
                        "split_pct":	[0.8, 0.1, 0.1],
                        "mask_field_names": ["train_mask_movie",
                                             "val_mask_movie",
                                             "test_mask_movie"]
                    },
                ]
            },
            {
                "node_type": "user",
                ......
                ],
                "labels": [
                    {
                        "label_col": "label_user",
                        "task_type": "classification",
                        "split_pct":	[0.2, 0.2, 0.6],
                        "mask_field_names": ["train_mask_user",
                                             "val_mask_user",
                                             "test_mask_user"]
                    },
                ]
            },
        ],
        ......
    }

The above configuration defines two classification tasks for the **movie** nodes and **user** nodes.
Each node type has its own "lable_col" and train/validation/test mask fields associated. Then you can
follow the instructions in :ref:`Run graph construction<run-graph-construction>` to use the GraphStorm
construction tool for creating partitioned graph data.

Define multi-task for training
...............................

Now, you can specify two training tasks by providing the `multi_task_learning` configurations in
the training configuration YAML file, like the example below.

.. code-block:: yaml

    ---
    version: 1.0
    gsf:
        basic:
            ...
        multi_task_learning:
            - node_classification:
                target_ntype: "movie"
                label_field: "label_movie"
                mask_fields:
                    - "train_mask_movie"
                    - "val_mask_movie"
                    - "test_mask_movie"
                num_classes: 10
                task_weight: 0.5
            - node_classification:
                target_ntype: "user"
                label_field: "label_user"
                mask_fields:
                    - "train_mask_user"
                    - "val_mask_user"
                    - "test_mask_user"
                task_weight: 1.0
            ...

The above configuration defines one classification task for the **movie** node type and another one
for the **user** node type. The two node classification tasks will take their own label name, i.e.,
`label_movie` and `label_user`, and their own train/validation/test mask fields. It also defines
different `task_weight` values, which want models to focus more on **user** nodes classification
(`task_weight = 1.0`) than classification on **movie** nodes (`task_weight = 0.5`).

Run multi-task model training
..............................

You can use the `graphstorm.run.gs_multi_task_learning` command to run multi-task learning tasks,
like the following example.

.. code-block:: bash

    python -m graphstorm.run.gs_multi_task_learning \
              --workspace <PATH_TO_WORKSPACE> \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config <PATH_TO_GRAPH_DATA> \
              --cf <PATH_TO_CONFIG> \

Run multi-task model Inference
...............................

For inference, you can use the same command line `graphstorm.run.gs_multi_task_learning`  with an
additional argument `--inference` as the following:

.. code-block:: bash

    python -m graphstorm.run.gs_multi_task_learning \
              --inference \
              --workspace <PATH_TO_WORKSPACE> \
              --num-trainers 1 \
              --num-servers 1 \
              --part-config <PATH_TO_GRAPH_DATA> \
              --cf <PATH_TO_CONFIG> \
              --save-prediction-path <PATH_TO_OUTPUT>

The prediction results of each prediction tasks will be saved into different sub-directories under
<PATH_TO_OUTPUT>. The sub-directories are prefixed with the `<task_type>_<node/edge_type>_<label_name>`.

Using multi-target node type training (Not Recommended)
-------------------------------------------------------

You can also use GraphStorm's multi-target node types configuration. But this method lacks of the
flexibility that the multi-task learning method provides.

- Train on multiple node types: The users only need to edit the ``target_ntype`` in model config
YAML file to minimize the objective function defined on mutiple target node types. For example,
by setting ``target_ntype`` as following, we can jointly optimize the objective function defined
on "movie" and "user" node types.

  .. code-block:: yaml

    target_ntype:
    -  movie
    -  user

  During evuation, the users can set a single node type for evaluation. For example, by setting
  ``eval_target_ntype:  movie``, we will only perform evaluation on "movie" node type.

- Evaluate on single node type: During evuation, the users can set a single node type for evaluation. For example, by setting ``eval_target_ntype:  movie``, we will only perform evaluation on "movie" node type. Our current implementation only support evaluating on a single node type.

- Per target node type decoder: The users may also want to use a different decoder on each node type, where the output dimension for each decoder maybe different. We can achieve this by setting ``num_classes`` in model config YAML file. For example, by setting ``num_classes`` as following, GraphStorm will create a decoder with output dimension as 3 for movie node type, and a decoder with output dimension as 7 for user node type.

  .. code-block:: yaml

    num_classes:
      movie:  3
      user:  7

- Reweighting on loss function: The users may also want to use a customized loss function reweighting on each node type, which can be achieved by setting ``multilabel``, ``multilabel_weights``, and ``imbalance_class_weights``. Examples are illustrated as following. Our current implementation does not support different node types with different ``multilabel`` setting.

  .. code-block:: yaml

    multilabel:
      movie:  true
      user:  true
    multilabel_weights:
      movie:  0.1,0.2,0.3
      user:  0.1,0.2,0.3,0.4,0.5,0.0

    multilabel:
      movie:  false
      user:  false
    imbalance_class_weights:
      movie:  0.1,0.2,0.3
      user:  0.1,0.2,0.3,0.4,0.5,0.0
