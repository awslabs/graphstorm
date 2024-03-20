WORKSPACE=/shared_data/graphstorm/examples/knn_retriever/
DATASPACE=/shared_data/graphstorm/examples/peft_llm_gnn/
dataset=amazon_review
domain=$1

python -m graphstorm.run.launch \
    --workspace "$WORKSPACE" \
    --part-config "$DATASPACE"/datasets/amazon_review_"$domain"/amazon_review.json \
    --ip-config "$DATASPACE"/ip_list.txt \
    --num-trainers 1 \
    --num-servers 1 \
    --num-samplers 0 \
    --ssh-port 22 \
    --do-nid-remap False \
    build_index.py \
    --cf "$WORKSPACE"/embedding_config.yaml \
    --save-model-path "$DATASPACE"/model/lp/"$domain"/ \
    --save-embed-path "$DATASPACE"/results/lp/"$domain"/