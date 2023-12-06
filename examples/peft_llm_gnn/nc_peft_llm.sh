WORKSPACE=/shared_data/graphstorm/examples/peft_llm_gnn/
dataset=amazon_review
domain=$1

python3 -m graphstorm.run.launch \
    --workspace "$WORKSPACE" \
    --part-config "$WORKSPACE"/data/amazon_review/predict_pt/"$domain"/graph_200_200_nc.opt_gs.bin.json \
    --ip-config ./ip_list.txt \
    --num-trainers 8 \
    --num-servers 1 \
    --num-samplers 0 \
    --ssh-port 22 \
    main_nc.py \
    --cf ./nc_config_"$domain".yaml \
    --save-model-path "$WORKSPACE"/model/nc/"$domain"/ \
    --save-prediction-path "$WORKSPACE"/results/nc/"$domain"/