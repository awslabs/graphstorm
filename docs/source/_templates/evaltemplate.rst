.. role:: hidden
    :class: hidden-section
.. currentmodule:: {{ module }}


{{ name | underline}}

.. autoclass:: {{ name }}
    :show-inheritance:
    :members: prepare_data, get_node_feats, get_edge_feats, get_labels, forward, get_sparse_params, 
              get_general_dense_parameters, get_lm_dense_parameters, save_model, remove_saved_model,
              save_topk_models, get_best_model_path, restore_model, fit, eval, infer, evaluate,
              do_eval, compute_score, predict, history