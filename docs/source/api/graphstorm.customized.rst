.. _apidataloading:

customized model APIs
==========================

    GraphStorm .

.. currentmodule:: graphstorm

Dataloaders
------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    .. dataloading.AbsNodeDataLoader
    .. dataloading.AbsEdgeDataLoader

Models
------------
.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    model.GSgnnNodeModelBase
    model.GSgnnModelBase
    model.GSgnnNodeModelInterface
    model.GSgnnEdgeModelInterface
    model.GSgnnLinkPredictionModelInterface

Evaluators
------------

    If users want to implement customized evaluators or evaluation methods, a best practice is to
    extend the ``eval.GSgnnInstanceEvaluator`` class, and implement the abstract methods.

.. autosummary::
    :toctree: ../generated/
    :nosignatures:
    :template: classtemplate.rst

    eval.GSgnnInstanceEvaluator
