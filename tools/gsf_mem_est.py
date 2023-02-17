import argparse
from graphstorm.utils import estimate_mem_train, estimate_mem_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BertModel')
    parser.add_argument('--root_path', type=str, help='The root path of the partition files.')
    parser.add_argument('--supervise_task', type=str, help='The supervision task to train models. It is either node or edge.')
    parser.add_argument('--is_train', type=lambda x: (str(x).lower() in ['true', '1']),
            help='Indicate whether this is a training or inference task.')
    parser.add_argument('--num_hidden', type=int, help='The number of hidden dimensions.')
    parser.add_argument('--num_layers', type=int, help='The number of GNN layers.')
    parser.add_argument('--graph_name', type=str, help='The graph name.')
    args = parser.parse_args()

    assert args.is_train is not None
    if args.is_train:
        assert args.supervise_task in ('node', 'edge'), 'The supervision task should be either train or inference.'
        peak_mem, shared_mem = estimate_mem_train(args.root_path, args.supervise_task)
        print('We need {:.3f} GB memory to train the graph data and {:.3f} GB shared memory'.format(peak_mem, shared_mem))
    else:
        assert args.num_hidden is not None
        assert args.num_layers is not None
        assert args.graph_name is not None
        peak_mem, shared_mem = estimate_mem_infer(args.root_path, args.graph_name, args.num_hidden, args.num_layers)
        print('We need {:.3f} GB memory to run inference on the graph data and {:.3f} GB shared memory'.format(peak_mem, shared_mem))
