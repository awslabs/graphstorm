"""This module provides utility functions for working with WholeGraph"""
from .utils import is_wholegraph, is_wholegraph_sparse_emb

def is_wholegraph_embedding(data):
    """ Check if the data is in WholeMemory emedding format which
        is required to use wholegraph framework.
    """
    try:
        import pylibwholegraph
        return isinstance(data, pylibwholegraph.torch.WholeMemoryEmbedding)
    except ImportError:
        return False

def is_wholegraph_embedding_module(data):
    """Check if the data is in WholeMemory emedding format which
    is required to use wholegraph framework.
    """
    try:
        import pylibwholegraph
        return isinstance(data, pylibwholegraph.torch.WholeMemoryEmbeddingModule)
    except:  # pylint: disable=bare-except
        return False


def is_wholegraph_optimizer(data):
    """Check if the data is in WholeMemoryOptimizer format which
    is required to use wholegraph framework.
    """
    try:
        import pylibwholegraph
        return isinstance(data, pylibwholegraph.torch.WholeMemoryOptimizer)
    except:  # pylint: disable=bare-except
        return False

