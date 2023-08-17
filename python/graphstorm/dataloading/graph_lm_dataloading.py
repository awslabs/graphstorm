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

    Dataloaders for graph aware LM training/inference
"""
import dgl

from .dataloading import GSgnnNodeDataLoader
from .utils import trim_data

from ..model.graph_transformer import BFS_TRANSVERSE

def get_graph_lm_dataloader(dataloader):
    if dataloader == "lm_hat_node_centric":
        return GSlmHatNodeDataLoader

    raise RuntimeError(f"Unknow dataloader type {dataloader}")


class GSlmHatNodeDataLoader(GSgnnNodeDataLoader):
    """ The minibatch dataloader for node centric tasks
    """
    def __init__(self, dataset, prepare_input_fn, target_idx, fanout, batch_size,
                 device, train_task, max_sequence_length, max_sentence_length,
                 pin_memory, shuffle, num_workers=0, drop_last=True,
                 transverse_format=BFS_TRANSVERSE,
                 shuffle_neighbor_order=True):
        self._max_sentence_len = max_sentence_length
        self._max_sequence_len = max_sequence_length
        self._transverse_format = transverse_format
        self._shuffle_neighbor_order = shuffle_neighbor_order
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._pin_memory = pin_memory
        self._prepare_input_fn = prepare_input_fn
        super(GSlmHatNodeDataLoader, self).__init__(dataset, target_idx, fanout, batch_size, device, train_task)

    def _prepare_dataloader(self, g, target_idx, fanout, batch_size, train_task, device):
        for ntype in target_idx:
            target_idx[ntype] = trim_data(target_idx[ntype], device)
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        loader = dgl.dataloading.DistNodeDataLoader(g, target_idx, sampler,
            batch_size=batch_size, shuffle=train_task,
            num_workers=self._num_workers, drop_last=self._drop_last)

        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        input_nodes, seeds, blocks = self.dataloader.__next__()
        batch = self._prepare_input_fn(self.data, input_nodes, seeds, blocks,
                                       max_sentence_len=self._max_sentence_len,
                                       max_sequence_len=self._max_sequence_len,
                                       transverse_format=self._transverse_format,
                                       shuffle_neighbor_order=self._shuffle_neighbor_order)

        if self._pin_memory:
            batch = tuple([data.pin_memory() for data in list(batch)])

        # Build batch from sampled graph.
        return batch


    def __len__(self):
        """ Size of dataset
        """
        return sum([len(idx) for _, idx in self._target_nidx.items()])
