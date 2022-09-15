import os
from graphstorm.data import StandardM5gnnDataset
from m5_dataloaders.datasets.constants import REGRESSION_TASK, CLASSIFICATION_TASK
from graphstorm.data.constants import TOKEN_IDX


def test_gsgnn_dataset(input_dir):
    dataset = StandardM5gnnDataset(input_dir, "test",
        hf_bert_model='bert-base-uncased',
        max_node_seq_length={'node':16},
        ntext_fields={'node':['text']},
        nlabel_fields={'node':'label'},
        ntask_types={'node': CLASSIFICATION_TASK},
        feat_format='npy')
    g = dataset[0]
    assert g.number_of_nodes('node') == 4
    assert g.number_of_nodes('item') == 4
    assert g.number_of_edges('r0') == 6
    assert g.number_of_edges('r1') == 4
    assert TOKEN_IDX in g.nodes['node'].data
    assert g.nodes['node'].data[TOKEN_IDX].shape[0] == 4
    assert g.nodes['node'].data[TOKEN_IDX].shape[1] == 16
    assert g.nodes['node'].data['nfeat'].shape[0] == 4
    assert g.nodes['node'].data['nfeat'].shape[1] == 3
    assert g.nodes['node'].data['nfeat1'].shape[0] == 4
    assert g.nodes['node'].data['nfeat1'].shape[1] == 3

def test_gsgnn_dataset_nfeats(input_dir):
    dataset = StandardM5gnnDataset(input_dir, "test",
        hf_bert_model='bert-base-uncased',
        max_node_seq_length={'node':16},
        ntext_fields={'node':['text']},
        nlabel_fields={'node':'label'},
        ntask_types={'node': CLASSIFICATION_TASK},
        feat_format='npy')
    g = dataset[0]
    print(g)
    assert g.number_of_nodes('node') == 4
    assert g.number_of_nodes('item') == 4
    assert g.number_of_edges('r0') == 6
    assert g.number_of_edges('r1') == 4
    assert TOKEN_IDX in g.nodes['node'].data
    assert g.nodes['node'].data[TOKEN_IDX].shape[0] == 4
    assert g.nodes['node'].data[TOKEN_IDX].shape[1] == 16
    assert g.nodes['node'].data['nfeat'].shape[0] == 4
    assert g.nodes['node'].data['nfeat'].shape[1] == 3
    assert g.nodes['node'].data['nfeat1'].shape[0] == 4
    assert g.nodes['node'].data['nfeat1'].shape[1] == 3

def test_gsgnn_dataset_eclass(input_dir):
    dataset = StandardM5gnnDataset(input_dir, "test",
        hf_bert_model='bert-base-uncased',
        max_node_seq_length={'node':16},
        ntext_fields={'node':['text']},
        nlabel_fields={'node':'label'},
        ntask_types={'node': CLASSIFICATION_TASK},
        elabel_fields={("node","r0","item"): 'label'},
        etask_types={("node","r0","item"): CLASSIFICATION_TASK},
        split_ntypes=["node"],
        split_etypes=[("node","r0","item")])
    g = dataset[0]
    print(g)
    print(g.edges['r0'].data)
    print(g.nodes['node'].data)
    assert g.number_of_nodes('node') == 4
    assert g.number_of_nodes('item') == 4
    assert g.number_of_edges('r0') == 6
    assert g.number_of_edges('r1') == 4
    assert TOKEN_IDX in g.nodes['node'].data
    assert g.nodes['node'].data[TOKEN_IDX].shape[0] == 4
    assert g.nodes['node'].data[TOKEN_IDX].shape[1] == 16
    assert g.edges['r0'].data['label'].shape[0] == 6

if __name__ == '__main__':
    #test_gsgnn_dataset(os.path.join('./data/', 'node_class'))
    test_gsgnn_dataset_eclass(os.path.join('./data/', 'edge_class'))