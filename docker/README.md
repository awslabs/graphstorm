# How to build Docker images to run in local instances

GraphStorm can be installed as a pip package. However, running GraphStorm in a distributed environment is non-trivial.
Users need to install dependencies and configure distributed Pytorch running environments. For this reason, we
recommend that our users use Docker as the base running environment to use GraphStorm in a distributed environment.

For users who want to create their own GraphStorm Docker images because they want to add additional functions,
e.g. graph data building, you can use the provided scripts to build your own GraphStorm Docker images.

For instructions refer to the
[GraphStorm documentation](https://graphstorm.readthedocs.io/en/latest/install/env-setup.html#setup-graphstorm-docker-environment)
