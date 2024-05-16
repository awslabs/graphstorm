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
```