.. _apiconfig:

graphstorm.config
===================

GraphStorm model training and inference CLIs have a set of :ref:`built-in configurations <configurations-run>`.
These configurations are defined and managed in ``graphstorm.config`` module whose ``get_argument_parser()``
method can load these built-in configurations either from a yaml file or from CLI arguments. By default,
GraphStorm parses the yaml config file first, and then it parses CLI arguments to overwrite configurations
defined in the yaml file or add new configurations. In GraphStorm, these configurations are stored in a
``GSConfig`` object, which will be passed to other modules to extract related configurations. 

.. currentmodule:: graphstorm.config

Configuration
--------------

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: configtemplate.rst

    get_argument_parser
    GSConfig