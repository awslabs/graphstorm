.. role:: hidden
    :class: hidden-section
.. currentmodule:: {{ module }}


{{ name | underline}}

.. autoclass:: {{ name }}
    :show-inheritance:
    :members:
    :member-order: bysource
    :inherited-members: restore_dense_model,
                        restore_sparse_model,
                        save_dense_model,
                        save_sparse_model,
                        normalize_node_embs,
                        restore_model,
                        save_model,
                        create_optimizer,
                        prepare_input_encoder,
                        freeze_input_encoder,
                        unfreeze_input_encoder,
                        device
