.. _apiconfig:

graphstorm.config
===================

GraphStorm model training and inference CLIs have a set of :ref:`built-in configurations <configurations-run>`.
These configurations are defined and managed in ``graphstorm.config`` module whose ``get_argument_parser()``
method can help users to read in these built-in configurations either from a yaml file or from CLI arguments.
This ``get_argument_parser()`` is useful when users want to convert your customized models to use GraphStorm CLIs.

.. currentmodule:: graphstorm.config

Configuration Argument Parser
------------------------------

    Users can call the `get_argument_parser` method to obtain a GraphStorm configuration parse in
    the `main` function, and parse launch CLIs arguments. For example,

    >>> from graphstorm.config import get_argument_parser
    >>> if __name__ == '__main__':
    >>>     # Leverage GraphStorm's argument parser to accept configuratioin yaml file
    >>>     arg_parser = get_argument_parser()
    >>>
    >>>     # parse all arguments and split GraphStorm's built-in arguments from the customized ones
    >>>     gs_args, unknown_args = arg_parser.parse_known_args()
    >>>
    >>>     print(f'GS arguments: {gs_args}')
    >>>     print(f'Non GS arguments: {unknown_args}')

.. autosummary::
    :toctree: ../generated/
    :nosignatures:

    get_argument_parser