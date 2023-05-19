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

    Utils to estimate the CPU memory requirement to run GFS on a specific graph data.
"""

import argparse
from graphstorm.utils import estimate_mem_train, estimate_mem_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BertModel')
    parser.add_argument('--root-path', type=str, help='The root path of the partition files.')
    parser.add_argument('--supervise-task', type=str, help='The supervision task to train models. It is either node or edge.')
    parser.add_argument('--is-train', type=lambda x: (str(x).lower() in ['true', '1']),
            help='Indicate whether this is a training or inference task.')
    parser.add_argument('--hidden-size', type=int, help='The number of hidden dimensions.')
    parser.add_argument('--num-layers', type=int, help='The number of GNN layers.')
    parser.add_argument('--graph-name', type=str, help='The graph name.')
    args = parser.parse_args()

    assert args.is_train is not None
    if args.is_train:
        assert args.supervise_task in ('node', 'edge'), 'The supervision task should be either train or inference.'
        peak_mem, shared_mem = estimate_mem_train(args.root_path, args.supervise_task)
        print('We need {:.3f} GB memory to train the graph data and {:.3f} GB shared memory'.format(peak_mem, shared_mem))
    else:
        assert args.hidden_size is not None
        assert args.num_layers is not None
        assert args.graph_name is not None
        peak_mem, shared_mem = estimate_mem_infer(args.root_path, args.graph_name, args.hidden_size, args.num_layers)
        print('We need {:.3f} GB memory to run inference on the graph data and {:.3f} GB shared memory'.format(peak_mem, shared_mem))
