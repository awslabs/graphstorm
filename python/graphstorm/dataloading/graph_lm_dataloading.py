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

def get_graph_lm_dataloader(dataloader):
    if dataloader == "lm_hat_node_centric":
        return GSlmHatNodeDataLoader

    raise RuntimeError(f"Unknow dataloader type {dataloader}")


class GSlmHatNodeDataLoader(GSgnnNodeDataLoader):
    """ The minibatch dataloader for node centric tasks
    """
    def __init__(self, dataset, prepare_input_fn, target_idx, fanout, batch_size,
                 device, train_task, max_sequence_length, max_sentence_length,
                 num_workers, pin_memory, shuffle, drop_last):
        self._prepare_input_fn = prepare_input_fn
        super(GSlmHatNodeDataLoader, self).__init__(dataset, target_idx, fanout, batch_size, device, train_task)

    def _prepare_dataloader(self, g, target_idx, fanout, batch_size, train_task, device):
        for ntype in target_idx:
            target_idx[ntype] = trim_data(target_idx[ntype], device)
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        loader = dgl.dataloading.DistNodeDataLoader(g, target_idx, sampler,
            batch_size=batch_size, shuffle=train_task, num_workers=0, drop_last=True)

        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def __next__(self):
        input_nodes, seeds, blocks = self.dataloader.__next__()
        batch = self._prepare_input_fn(self.data, input_nodes, seeds, blocks)

        # Build batch from sampled graph.
        return batch


    def __len__(self):
        """ Size of dataset
        """
        return sum([len(idx) for _, idx in self._target_nidx.items()])
