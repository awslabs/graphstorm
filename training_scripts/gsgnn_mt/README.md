# Multi-task Learning Example Yaml Files
This folder presents example yaml files for multi-task learning with Movielens datasets.

## Build a graph for multi-task learning on Movielens dataset
```
python3 $GS_HOME/tests/end2end-tests/data_gen/process_movielens.py

python3 -m graphstorm.gconstruct.construct_graph \
	--conf-file $GS_HOME/tests/end2end-tests/data_gen/movielens_multi_task.json \
	--num-processes 1 \
	--output-dir movielen_100k_multi_task_train_val_1p_4t \
	--graph-name movie-lens-100k \
	--add-reverse-edges
```

## Run the example
```
python3 -m graphstorm.run.gs_multi_task_learning \
	--workspace $GS_HOME/training_scripts/gsgnn_mt  \
	--num-trainers 1 \
	--num-servers 1 \
	--part-config movielen_100k_multi_task_train_val_1p_4t/movie-lens-100k.json \
	--cf ml_nc_ec_er_lp.yaml \
	--save-model-path /data/gsgnn_mt/ \
	--save-model-frequency 1000
```