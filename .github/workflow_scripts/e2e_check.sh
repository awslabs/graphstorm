# Move to parent directory
cd ../../

sh ./tests/end2end-tests/setup.sh

sh ./tests/end2end-tests/create_data.sh

sh ./tests/end2end-tests/tools/test_mem_est.sh

sh ./tests/end2end-tests/custom-gnn/run_test.sh

sh ./tests/end2end-tests/graphstorm-nc/test.sh

sh ./tests/end2end-tests/graphstorm-lp/test.sh

sh ./tests/end2end-tests/graphstorm-ec/test.sh

sh ./tests/end2end-tests/graphstorm-er/test.sh