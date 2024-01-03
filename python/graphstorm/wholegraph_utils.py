"""This module provides utility functions for working with WholeGraph"""
from .utils import is_wholegraph

def is_wholegraph_embedding(data):
    """ Check if the data is in WholeMemory emedding format which
        is required to use wholegraph framework.
    """
    try:
        import pylibwholegraph
        assert (
            is_wholegraph()
        ), "WholeGraph needs to be enabled first."
        return isinstance(data, pylibwholegraph.torch.WholeMemoryEmbedding)
    except ImportError:
        return False
