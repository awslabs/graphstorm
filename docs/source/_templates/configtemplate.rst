.. role:: hidden
    :class: hidden-section
.. currentmodule:: {{ module }}

{{ name | underline}}

.. autoclass:: {{ name }}
    :show-inheritance:
    :members:
    :member-order: alphabetical
    :exclude-members: handle_argument_conflicts,
                      load_yaml_config,
                      set_attributes,
                      set_task_attributes,
                      override_arguments,
                      verify_node_feat_reconstruct_arguments,
                      verify_node_class_arguments,
                      verify_node_regression_arguments,
                      verify_edge_class_arguments,
                      verify_edge_regression_arguments,
                      verify_link_prediction_arguments,
                      verify_arguments,
                      training_method,
                      node_lm_configs,
                      distill_lm_configs,
                      cache_lm_embed,
                      save_embed_format,
                      log_report_frequency,
                      use_pseudolabel,
                      eval_target_ntype,
                      report_eval_per_type,
                      model_select_etype,
                      decoder_norm,
                      reconstruct_nfeat_name,
                      reconstruct_efeat_name,
                      multi_tasks,
                      construct_feat_ntype,
                      construct_feat_encoder,
                      construct_feat_fanout,
                      profile_path
