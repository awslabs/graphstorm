"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Base dataset for GSF dataset
"""
import os
import dgl
from dgl.data.dgl_dataset import DGLDataset


class GSgnnDataset(DGLDataset):
    """r Basic class of GSgnn dataset
    """
    def __init__(self, name, url, raw_dir, force_reload=False, verbose=True, reverse_edge=True):
        self._encoding = 'utf-8'
        self._raw_text_feat = None
        self.reverse_edge = reverse_edge

        super(GSgnnDataset, self).__init__(name,
                                           url=url,
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def load(self):
        pass

    def has_cache(self):
        return False

    def save(self, path=None):
        pass

    def download(self):
        if self.raw_dir is not None:
            return
        assert False, "Will add auto download in the future"

    def process(self):
        pass

    def __getitem__(self, idx):
        # Add this pylint disable because the DGLDataset has the ._g attribute.
        # pylint: disable=no-member
        r"""Gets the data object at index.
        """
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

class ConstructedGraphDataset(GSgnnDataset):
    """ A general dgl graph dataset wrapper used for loading the
        output DGL Graph of construct_graph.py.

        By default, construct_graph.py creates two output: 1) constructed graph; 2) partioned graph.
        ConstructedGraphDataset is used to load the constructed graph.

        Note: The graph filename should be path/<name>.dgl

        Examples:
        python3 construct_graph.py --name example --filepath input --output data ...

        >>> dataset = ConstructedGraphDataset(example, data)
        >>> g = dataset[0]

    """
    def __init__(self, name, path, verbose=True):
        super(ConstructedGraphDataset, self).__init__(name,
                                                      url=None,
                                                      raw_dir=path,
                                                      force_reload=False,
                                                      verbose=verbose)

    def download(self):
        pass

    def process(self):
        root_path = self._raw_dir
        gname = self._name + ".dgl"
        assert os.path.exists(os.path.join(root_path, gname)), \
            "A saved DGLGraph file named {} should exist under {}".format(gname, root_path)
        g, _ = dgl.load_graphs(os.path.join(root_path, gname))
        self._g = g[0]
