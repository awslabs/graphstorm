```
python3 -m graphstorm.run.gs_link_prediction \
			--num_trainers 8 --num_servers 4 \
			--part_config ~/workspace/data/oagv2.1/mag_bert_constructed/mag.json \
			--ip_config ip_list.txt \
			--cf mag_lp.yaml \
			--num-epochs 1 \
			--save-model-path ~/workspace/data/oagv2.1/mag_model \
			--node-feat-name fos:feat paper:feat 
```
