WORKSPACE=/shared_data
dataset=amazon_review
domain=$1
rm -rf ./data/"$dataset"/predict_pt/"$domain"
python "$WORKSPACE"/graphstorm/tools/partition_graph.py --dataset graph_200_200_nc.opt_gs.bin \
    --filepath "$WORKSPACE"/GraphPEFT/dataset/"$dataset"/"$domain" --target-ntype item \
    --nlabel-field item:label --num-parts 1\
    --output ./data/"$dataset"/predict_pt/"$domain"

