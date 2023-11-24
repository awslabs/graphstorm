DGL_HOME=/root/dgl
GS_HOME=$(pwd)
NUM_TRAINERS=4
NUM_INFO_TRAINERS=2
export PYTHONPATH=$GS_HOME/python/
cd $GS_HOME/training_scripts/gsgnn_lp
echo "127.0.0.1" > ip_list.txt
cd $GS_HOME/inference_scripts/lp_infer
echo "127.0.0.1" > ip_list.txt

# train a model, save model and embeddings
python3 -m graphstorm.run.gs_link_prediction --workspace $GS_HOME/training_scripts/gsgnn_lp --num-trainers $NUM_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --eval-batch-size 1024 --exclude-training-targets True --reverse-edge-types-map user,rating,rating-rev,movie  --save-model-path /data/gsgnn_lp_ml_dot/ --topk-model-to-save 1 --save-model-frequency 1000 --save-embed-path /data/gsgnn_lp_ml_dot/emb/ --logging-file /tmp/train_log.txt --logging-level debug --preserve-input True


best_epoch_dot=$(grep "successfully save the model to" /tmp/train_log.txt | tail -1 | tr -d '\n' | tail -c 1)
echo "The best model is saved in epoch $best_epoch_dot"

echo "**************dataset: Movielens, do inference on saved model, decoder: dot"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --eval-batch-size 1024 --save-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/ --logging-file /tmp/log.txt --preserve-input True

# inference for retrieval setting
echo "**************dataset: Movielens, do inference on saved model, decoder: dot, retrieval setting:"
python3 -m graphstorm.run.gs_link_prediction --inference --workspace $GS_HOME/inference_scripts/lp_infer --num-trainers $NUM_INFO_TRAINERS --num-servers 1 --num-samplers 0 --part-config /data/movielen_100k_lp_train_val_1p_4t/movie-lens-100k.json --ip-config ip_list.txt --ssh-port 2222 --cf ml_lp_infer.yaml --fanout '10,15' --num-layers 2 --use-mini-batch-infer false --eval-batch-size 1024 --restore-embed-path /data/gsgnn_lp_ml_dot/infer-emb/ --restore-model-path /data/gsgnn_lp_ml_dot/epoch-$best_epoch_dot/ --preserve-input True --eval-negative-sampler full --save-embed-path none