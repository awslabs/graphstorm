WORKSPACE=/shared_data/graphstorm/examples/peft_llm_gnn/
dataset=amazon_review
domain=$1

python -m graphstorm.run.launch \
    --workspace "$WORKSPACE" \
    --part-config "$WORKSPACE"/dataset/amazon_review_"$domain"/amazon_review.json \
    --ip-config ./ip_list.txt \
    --num-trainers 8 \
    --num-servers 1 \
    --num-samplers 0 \
    --ssh-port 22 \
    main_lp.py \
    --cf ./lp_config_"$domain".yaml \
    --save-model-path "$WORKSPACE"/model/lp/"$domain"/ \
    --save-prediction-path "$WORKSPACE"/results/lp/"$domain"/
#  
#--part-config /shared_data/GPEFT_processing/datasets/amazon_review_nc_"$domain"_opt/amazon_review.json \
#     --logging-level debug \
