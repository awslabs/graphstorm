# python3 gen_ogb_dataset.py --savepath /data \
#                            --dataset ogbn-papers100M \
#                            --edge-pct 0.8

python /graphstorm/tools/gen_ogb_dataset.py --savepath /data --dataset ogbn-papers100M --retain-original-features true


# python3 /graphstorm/tools/partition_graph.py --dataset ogbn-papers100M \
#                                             --filepath /data \
#                                             --num-parts 3 \
#                                             --train-pct 0.1 \
#                                             --balance-train \
#                                             --balance-edges \
#                                             --output /data/ogbn_papers100M_3p 


# python3 -m graphstorm.run.gs_node_classification \
#            --workspace /data/ \
#            --num-trainers 4 \
#            --num-servers 1 \
#            --num-samplers 0 \
#            --part-config /data/ogbn_papers100M_3p/ogbn-papers100M.json \
#            --ip-config /data/ip_list.txt \
#            --ssh-port 2222 \
#            --graph-format csc,coo \
#            --cf /data/ogbn_papers100M_nc_p3.yaml \
#            --node-feat-name feat