.. _apigraphstorm:

.. currentmodule:: graphstorm

graphstorm
============

    The ``graphstorm`` package contains a function for environment setup and a set of
    utilization functions. Users can directly use the following code to use these functions.

    >>> import graphstorm as gs
    >>> gs.initialize()
    >>> feat_size = gs.get_node_feat_size(g, node_feat_names={"author": "feat"})
    >>> relation_names = gs.get_rel_names_for_reconstruct(
    >>>                     g,
    >>>                     reconstructed_embed_ntype=["paper"],
    >>>                     feat_size=feat_size)

.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    gsf.initialize
    gsf.get_node_feat_size
    gsf.get_rel_names_for_reconstruct
