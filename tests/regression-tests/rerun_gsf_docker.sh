docker_id=`docker ps | grep graphstorm | awk '{print $1}'`
if [ ! -z $docker_id ]; then
        docker kill $docker_id
        docker rm $docker_id
fi
nvidia-docker run -v /home/ubuntu/workspace/gsf-data/:/data -v /home/ubuntu/workspace/graph-storm/:/graph-storm -v /dev/shm:/dev/shm --network=host --name regression_test -d 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm_alpha:v3
