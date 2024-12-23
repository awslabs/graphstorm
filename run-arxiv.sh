python3 /graphstorm/tools/partition_graph.py --dataset ogbn-papers100M \
                                            --filepath /data \
                                            --num-parts 3 \
                                            --train-pct 0.1 \
                                            --balance-train \
                                            --balance-edges \
                                            --output /data/ogbn_papers100M_3p 


python -m graphstorm.run.gs_node_classification \
          --workspace /tmp/ogbn-arxiv-nc \
          --num-trainers 1 \
          --num-servers 1 \
          --part-config /tmp/ogbn_arxiv_nc_1p/ogbn-arxiv.json \
          --cf /graphstorm/training_scripts/gsgnn_np/arxiv_nc.yaml \
          --save-model-path /tmp/ogbn-arxiv-nc/models