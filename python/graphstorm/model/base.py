"""
Module for base types and utilities
"""

class GraphModelBase():
    """ Base Graph model
    """
    def __init__(self, g, config):
        print(config)

        self._g = g
        self._debug = config.debug

    @property
    def g(self):
        """ graph
        """
        return self._g

    @property
    def debug(self):
        """ Debug flag
        """
        return self._debug
