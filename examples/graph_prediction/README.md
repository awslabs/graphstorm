# Example Code for Graph-level Prediction Solutions

## Super-node Graph Data Processing

**Step 1**: Generate super-node format graph data
``` bash
python gen_ogbn_supernode.py --ogbg-data-name molhiv
```

**Step 2**: Run GraphStorm graph construction CLI
``` bash
python -m graphstorm.gconstruct.construct_graph \
        --conf-file ./dataset/ogbg_molhiv/gs_raw/config.json \
        --output-dir ./dataset/ogbg_molhiv/gs_1p/ \
        --num-parts 1 \
        --graph-name supernode_molhiv
```

To dive deep the super-node format graph structure, we provide a dummy super-node graph generation script, i.e., the `dummy_supernode_data.py`. You can run the following commands to build a dummy super-node format graph dataset. This dummy data can also be used for debugging the super-node customized GraphStorm models.

``` bash
python dummy_supernode_data.py --num-subgraphs 200 --save-path ./dummy_raw/

python -m graphstorm.gconstruct.construct_graph \
        --conf-file ./dummy_raw/config.json \
        --output-dir ./dummy_gs_1p/ \
        --num-parts 1 \
        --graph-name dummy_supernode
```

## Customized RGCN Encoder for Super-node Formated Graph
``` bash
python supernode_ogbg_gc.py
```

