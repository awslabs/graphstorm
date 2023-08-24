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

from transformers import BatchEncoding

from ..dataloading import GSgnnNodeDataLoader
from ..utils import trim_data

DFS_TRANSVERSE = "dfs"
BFS_TRANSVERSE = "bfs"

class GSlmHatNodeDataLoader(GSgnnNodeDataLoader):
    """ The minibatch dataloader for node centric tasks
    """
    def __init__(self, dataset, prepare_input_fn,
                 target_idx, fanout, batch_size,
                 device, train_task, max_sequence_length, max_sentence_length,
                 pin_memory, data_collator,
                 num_workers=0, drop_last=True,
                 transverse_format=BFS_TRANSVERSE,
                 shuffle_neighbor_order=True):
        self._max_sentence_len = max_sentence_length
        self._max_sequence_len = max_sequence_length
        self._transverse_format = transverse_format
        self._shuffle_neighbor_order = shuffle_neighbor_order
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._pin_memory = pin_memory
        self._data_collator = data_collator
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

        # TODO: we assume there is only one target node type
        # We need to support multiple node types later
        batch = [list(data.values())[0] if len(data) == 1 else None for data in list(batch)]

        if self._pin_memory:
            batch = [data.pin_memory() if data is not None else None for data in list(batch)]

        if self._shuffle_neighbor_order:
            # Build batch from sampled graph.
            return self._data_collator([{
                "input_ids": batch[2],
                "attention_mask": batch[3],
                "doc_position_ids": batch[5],
            }])
        else:
            # Build batch from sampled graph.
            return self._data_collator([{
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "doc_position_ids": batch[5],
            }])

    def __len__(self):
        """ Size of dataset
        """
        return sum([len(idx) for _, idx in self._target_nidx.items()])
