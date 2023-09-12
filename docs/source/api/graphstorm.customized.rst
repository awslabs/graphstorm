.. _apicustomized:

customized model APIs
==========================

    GraphStorm provides a set of APIs for users to integrate their own customized models with
    the framework of GraphStorm, so that users' own models can leverage GraphStorm's easy-to-use
    and distributed capabilities.

    For how to modify users' own models, please refer to this :ref:`Use Your Own Model Tutorial
    <use-own-models>`.

    In general, there are three sets of APIs involved in programming customized models.

    * Dataloaders: users need to extend GraphStorm's abstract node or edge dataloader to implement
      their own graph samplers or mini_batch generators.
    * Models: depending on specific GML tasks, users need to extend the corresponding ModelBase and
      ModelInterface, and then implement the required abstract functions.
    * Evaluators: if necessary, users can also extend the two evaluator templates to implement their
      own performance evaluation method.

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

    model.GSgnnModelBase
    model.GSgnnNodeModelBase
    model.GSgnnEdgeModelBase
    model.GSgnnLinkPredictionModelBase
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
    eval.GSgnnLPEvaluator