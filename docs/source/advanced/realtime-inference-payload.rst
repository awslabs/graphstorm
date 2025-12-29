.. _realtime_inference_payload:

Advanced Real-time Inference Payload Prepare
====================================================

.. _introduction:

Introduction
------------

GraphStorm real-time inference enables users to deploy trained graph machine learning models as SageMaker endpoints for immediate predictions on subgraphs. Unlike offline inference that processes large datasets in batches, real-time inference provides low-latency responses for individual prediction requests, making it ideal for applications such as recommendation systems, fraud detection, and social network analysis.

To enhance the flexibility and capabilities of real-time inference, in v0.5.1 GraphStorm introduces two new payload features:

#. **Raw Text Input Support**:
   Users can now submit raw text directly in the inference payload without pre-processing. The deployed model will handle text tokenization and embed them automatically using the same language models configured (or trained) during training.

#. **Learnable Embeddings Input Support**:
   Users can include learnable node embeddings in the payload, enabling both transductive inference (using embeddings from training nodes) and inductive inference (generating embeddings for new nodes).

These enhancements allow users to leverage the full configuration capabilities of GraphStorm during model training while maintaining the same rich feature support in real-time inference scenarios. Users can now deploy models trained with complex text processing pipelines and learnable embeddings without losing functionality when moving from offline to online inference.


.. _raw_text_support:

Raw Text Input Support
----------------------

.. _raw_text_overview:

Feature Overview
~~~~~~~~~~~~~~~~

Raw text input support allows users to submit unprocessed text strings directly in the real-time inference payload. This feature eliminates the need for client-side text preprocessing and ensures consistency between training and inference text processing pipelines. This feature is particularly useful when models are trained with text features using GraphStorm's language model integration, including BERT, RoBERTa, and other transformer-based models.

Before the release of this feature, users needed to embed text into numerical embeddings before graph construction and use these embeddings as node/edge features. With this approach, users could not train language models with GNN models as introduced in :ref:`Use Language Models in GraphStorm<language_models>`.

.. _raw_text_payload:

Payload Format for Raw Text
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When submitting raw text in the inference payload, text features should be provided as string values instead of numerical arrays. The feature names must match exactly with the text feature names defined during graph construction.

**Basic syntax for text features:**

.. code-block:: json

   {
       "features": {
           "text_feature_name": "This is the raw text content to be processed"
       }
   }

**Complete payload example with raw text:**

.. code-block:: json

   {
       "version": "gs-realtime-v0.1",
       "gml_task": "node_classification",
       "graph": {
           "nodes": [
               {
                   "node_type": "paper",
                   "node_id": "p1234",
                   "features": {
                       "title": "Graph Neural Networks for Recommendation Systems",
                       "abstract": "This paper presents a novel approach to recommendation systems using graph neural networks...",
                       "numerical_feat": [0.1, 0.2, 0.3]
                   }
               },
               {
                   "node_type": "author",
                   "node_id": "a5678",
                   "features": {
                       "bio": "Dr. Smith is a researcher in machine learning and graph theory",
                       "affiliation": "University of Technology"
                   }
               }
           ],
           "edges": [
               {
                   "edge_type": ["author", "writes", "paper"],
                   "src_node_id": "a5678",
                   "dest_node_id": "p1234",
                   "features": {}
               }
           ]
       },
       "targets": [
           {
               "node_type": "paper",
               "node_id": "p1234"
           }
       ]
   }

.. _raw_text_configuration:

Configuration Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~

To use raw text input in real-time inference, the model must be trained with text features properly configured:

**During Graph Construction:**

* Text features must be defined in the graph construction configuration (GConstruct or GSProcessing).
* Features names must match the ``feature_name`` entries defined in :ref:`GConstruct JSON specification
  <gconstruction-json>`, or the ``name`` values of ``features`` fields defined in
  :ref:`GSProcessing JSON specification <gsprocessing_input_configuration>`.

**During Model Training:**

* Language models must be configured using GraphStorm's text processing capabilities.
* The model configuration should specify the appropriate tokenizer and language model.

**Model Artifacts:**

The following files are required for text processing during inference:

* ``model.bin``: Contains the trained model weights, including language model parameters if it was trained.
* ``data_transform_new.json``: Contains feature transformation information including text feature definitions.
* ``GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml``: Contains the complete model configuration including language model settings.

.. _raw_text_best_practices:

Best Practices and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Performance Implications:**

* Text tokenization and encoding add computational overhead compared to pre-processed features.
* For high-throughput scenarios, consider the trade-off between convenience and latency.
* GPU instances may provide better performance for text-heavy workloads.

**Error Handling:**

* Invalid text encoding will result in error code 411 (payload conversion errors).
* Missing text features will result in error code 402 (missing field values).
* Text feature name mismatches will result in error code 401 (missing required fields).

.. _learnable_embeddings_support:

Learnable Embeddings Input Support
-----------------------------------

.. _learnable_embeddings_overview:

Feature Overview
~~~~~~~~~~~~~~~~

Learnable embeddings input support enables users to include node embeddings directly in the real-time inference payload. These embeddings can be either pre-computed from the training graph or dynamically generated for new nodes, e.g., averaging neighors' learnable embeddings, providing flexibility for both transductive and inductive inference scenarios. This feature is particularly valuable when models are trained with learnable node embeddings that capture complex graph structural information and node-specific patterns.

Key benefits include:

* **Transductive inference**: Leverage embeddings learned during training for known nodes.
* **Inductive inference**: Support for new nodes not seen during training with dynamical generation.
* **Enhanced model performance**: Utilize rich node representations beyond traditional features.

.. _transductive_inductive_inference:

Transductive vs Inductive GNN Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GraphStorm supports two distinct inference modes when using learnable embeddings:

**Transductive Inference**: 
In transductive inference, the model uses pre-computed embeddings for nodes that were present during training. These embeddings are learned parameters that capture node-specific information from the training process. GraphStorm will save these embeddings along with GNN models. Details could be found at :ref:`GraphStorm Saved Node Embeddings<gs-output-embs>`.

**Inductive Inference**:
In inductive inference, the model generates embeddings for previously unseen nodes using their features and local graph structure. This enables inference on nodes not present during training.

To enable inductive inference for unseen nodes without learnable embeddings, you can use several methods, including:

- use the overall average (min or max) of all saved learnable embeddings.
- use the average (min or max) of learnable embeddings of all (or partial) first-hop neighbors.
- use the average (min or max) of learnable embeddings of all (or partial) multiple-hop neighbors.

Or, you can find more advanced methods in papers like `this one <https://arxiv.org/html/2506.05039>`_.

.. _learnable_embeddings_payload:

Payload Format for Learnable Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learnable embeddings are specified as numerical arrays in the node features using the reserved key name ``gs_learnable_embedding``. The embedding dimension must match the configuration used during model training.

**Basic syntax for learnable embeddings:**

.. code-block:: json

   {
       "features": {
           "gs_learnable_embedding": [0.1, -0.2, 0.5, 0.8, -0.1, 0.3]
       }
   }

**Complete payload example with learnable embeddings:**

.. code-block:: json

   {
       "version": "gs-realtime-v0.1",
       "gml_task": "node_classification",
       "graph": {
           "nodes": [
               {
                   "node_type": "user",
                   "node_id": "u1234",
                   "features": {
                       "gs_learnable_embedding": [0.15, -0.23, 0.67, 0.89, -0.12, 0.34, 0.56, -0.78],
                       "age": [25.0],
                       "category": "premium"
                   }
               },
               {
                   "node_type": "item",
                   "node_id": "i5678",
                   "features": {
                       "gs_learnable_embedding": [-0.45, 0.67, -0.12, 0.34, 0.78, -0.23, 0.91, 0.15],
                       "price": [99.99],
                       "brand": "TechCorp"
                   }
               }
           ],
           "edges": [
               {
                   "edge_type": ["user", "purchased", "item"],
                   "src_node_id": "u1234",
                   "dest_node_id": "i5678",
                   "features": {}
               }
           ]
       },
       "targets": [
           {
               "node_type": "user",
               "node_id": "u1234"
           }
       ]
   }

.. _learnable_embeddings_configuration:

Configuration and Setup
~~~~~~~~~~~~~~~~~~~~~~~

To use learnable embeddings in real-time inference, the model must be trained with embedding layers properly configured:

**During Model Training:**

* Learnable embeddings must be enabled in the model configuration, either use featureless nodes or set ``use_node_embeddings`` to be ``true``.
* Embedding dimensions must be consistent across training and inference.

**Model Artifacts Requirements:**

* ``model.bin``: **NOT** contain learnable embedding, but the GNN model parameters only. Learnable embeddings should be stored separately, and used for payload construction.
* ``data_transform_new.json``: Includes embedding feature definitions and dimensions.
* ``GRAPHSTORM_RUNTIME_UPDATED_TRAINING_CONFIG.yaml``: Contains embedding layer configurations.

Advanced Usage Patterns
~~~~~~~~~~~~~~~~~~~~~~~~

**Error Handling:**

* Dimension mismatch will result in error code 500 (internal server errors)
* Invalid embedding values (NaN, infinity) will cause error code 500 (internal server errors)
* Missing embeddings for nodes configured to use them will result in error code 500 (internal server errors)