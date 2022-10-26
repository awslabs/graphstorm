FROM 911734752298.dkr.ecr.us-east-1.amazonaws.com/graphstorm_alpha:v3

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN apt-get update \
 && apt-get upgrade -y \
 && apt-get autoremove -y \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG PYTHON
ARG PYTHON_VERSION
ARG PYTHON_SHORT_VERSION
ARG MAMBA_VERSION

ARG CUBLAS_VERSION=11.5.1.109
ARG EFA_PATH=/opt/amazon/efa
ARG CUDA_HOME=/usr/local/cuda

ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV OPEN_MPI_PATH=/opt/amazon/openmpi

ENV DGLBACKEND=pytorch
ENV MANUAL_BUILD=0
ENV RDMAV_FORK_SAFE=1
ENV DLC_CONTAINER_TYPE=training

# Python won’t try to write .pyc or .pyo files on the import of source modules
# Force stdin, stdout and stderr to be totally unbuffered. Good for logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST="3.7 5.0 7.0+PTX 8.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

ENV CUDNN_VERSION=8.2.0.53
ENV NCCL_VERSION=2.10.3
ENV HOROVOD_VERSION=0.24.3
ENV EFA_VERSION=1.15.2
ENV OMPI_VERSION=4.1.1
ENV BRANCH_OFI=1.4.0-aws

RUN apt-get update \
 # TODO: Remove systemd upgrade once it is updated in base image
 && apt-get -y upgrade --only-upgrade systemd \
 && apt-get install -y --allow-change-held-packages --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    cuda-command-line-tools-11-3 \
    cuda-cudart-11-3 \
    cuda-libraries-11-3 \
    curl \
    emacs \
    git \
    hwloc \
    jq \
    libcublas-11-3=${CUBLAS_VERSION}-1 \
    libcublas-dev-11-3=${CUBLAS_VERSION}-1 \
    libcudnn8=$CUDNN_VERSION-1+cuda11.3 \
    libcufft-dev-11-3 \
    libcurand-dev-11-3 \
    libcurl4-openssl-dev \
    libcusolver-dev-11-3 \
    libcusparse-dev-11-3 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libibverbs-dev \
    libhwloc-dev \
    libnuma1 \
    libnuma-dev \
    libssl1.1 \
    libssl-dev \
    libtool \
    openssl \
    python3-dev \
    unzip \
    vim \
    wget \
    zlib1g-dev \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

 # Install NCCL
RUN cd /tmp \
 && git clone https://github.com/NVIDIA/nccl.git -b v${NCCL_VERSION}-1 \
 && cd nccl \
 && make -j64 src.build BUILDDIR=/usr/local \
 && rm -rf /tmp/nccl

# Install EFA alone without AWS OPEN_MPI
RUN mkdir /tmp/efa \
 && cd /tmp/efa \
 && curl -O https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz \
 && tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz \
 && cd aws-efa-installer \
 && apt-get update \
 && ./efa_installer.sh -y --skip-kmod -g \
 && rm -rf $OPEN_MPI_PATH \
 && rm -rf /tmp/efa \
 && rm -rf /tmp/aws-efa-installer-${EFA_VERSION}.tar.gz \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Install OpenMPI without libfabric support
RUN mkdir /tmp/openmpi \
 && cd /tmp/openmpi \
 && wget --quiet https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-${OMPI_VERSION}.tar.gz \
 && tar zxf openmpi-${OMPI_VERSION}.tar.gz \
 && cd openmpi-${OMPI_VERSION} \
 && ./configure --enable-orterun-prefix-by-default --prefix=$OPEN_MPI_PATH --with-cuda \
 && make -j $(nproc) all \
 && make install \
 && ldconfig \
 && cd / \
 && rm -rf /tmp/openmpi

ENV PATH="$OPEN_MPI_PATH/bin:$PATH"
ENV LD_LIBRARY_PATH=$OPEN_MPI_PATH/lib/:$EFA_PATH/lib/:$LD_LIBRARY_PATH

WORKDIR /root

RUN pip install --no-cache-dir -U \
    "awscli<2" \
    boto3 \
    click \
    cmake==3.18.2.post1 \
    "cryptography>3.2" \
    ipython==8.1.0 \
    mpi4py==3.1.3 \
    numpy==1.22.2 \
    opencv-python \
    packaging \
    "Pillow>=9.0.0" \
    psutil \
    pybind11 \
    "pyyaml>=5.4,<5.5" \
    scipy

RUN curl -o /usr/local/bin/deep_learning_container.py https://raw.githubusercontent.com/aws/deep-learning-containers/master/src/deep_learning_container.py

RUN chmod +x /usr/local/bin/deep_learning_container.py

RUN curl -o /license.txt  https://aws-dlc-licenses.s3.amazonaws.com/pytorch-1.11/license.txt

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

ARG PYTHON
ARG PYTHON_VERSION
ARG PYTHON_SHORT_VERSION

ARG RMM_VERSION=0.15.0

# The smdebug pipeline relies for following format to perform string replace and trigger DLC pipeline for validating
# the nightly builds. Therefore, while updating the smdebug version, please ensure that the format is not disturbed.
ARG SMDEBUG_VERSION=1.0.22

ENV SAGEMAKER_TRAINING_MODULE=sagemaker_pytorch_container.training:main

# SMD model parallel and data parallel binaries
ARG SMD_DATA_PARALLEL_URL
ARG SMD_MODEL_PARALLEL_URL

# PyTorch Binaries
ARG PT_SM_TRAINING_URL
ARG PT_TORCHVISION_URL
ARG PT_TORCHAUDIO_URL
ARG PT_TORCHDATA_URL
ARG PT_S3_WHL_GPU

# Install AWS-PyTorch and other torch packages
RUN pip uninstall -y torch torchvision torchaudio torchdata \
 && pip install --no-cache-dir -U ${PT_SM_TRAINING_URL} ${PT_TORCHVISION_URL} ${PT_TORCHAUDIO_URL} ${PT_TORCHDATA_URL} torchnet

# Configure Open MPI and configure NCCL parameters
RUN mv $OPEN_MPI_PATH/bin/mpirun $OPEN_MPI_PATH/bin/mpirun.real \
 && echo '#!/bin/bash' > $OPEN_MPI_PATH/bin/mpirun \
 && echo "${OPEN_MPI_PATH}/bin/mpirun.real --allow-run-as-root \"\$@\"" >> $OPEN_MPI_PATH/bin/mpirun \
 && chmod a+x $OPEN_MPI_PATH/bin/mpirun \
 && echo "hwloc_base_binding_policy = none" >> $OPEN_MPI_PATH/etc/openmpi-mca-params.conf \
 && echo "rmaps_base_mapping_policy = slot" >> $OPEN_MPI_PATH/etc/openmpi-mca-params.conf \
 && echo NCCL_DEBUG=INFO >> /etc/nccl.conf \
 && echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# Install AWS OFI NCCL plug-in
RUN apt-get update && apt-get install -y autoconf
RUN mkdir /tmp/efa-ofi-nccl \
 && cd /tmp/efa-ofi-nccl \
 && git clone https://github.com/aws/aws-ofi-nccl.git -b v${BRANCH_OFI} \
 && cd aws-ofi-nccl \
 && ./autogen.sh \
 && ./configure --with-libfabric=/opt/amazon/efa \
  --with-mpi=/opt/amazon/openmpi \
  --with-cuda=/usr/local/cuda \
  --with-nccl=/usr/local --prefix=/usr/local \
 && make \
 && make install \
 && rm -rf /tmp/efa-ofi-nccl \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Install OpenSSH for MPI to communicate between containers, allow OpenSSH to talk to containers without asking for confirmation
RUN apt-get update \
 && apt-get install -y  --allow-downgrades --allow-change-held-packages --no-install-recommends \
 && apt-get install -y --no-install-recommends openssh-client openssh-server \
 && mkdir -p /var/run/sshd \
 && cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
 && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
 && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Configure OpenSSH so that nodes can communicate with each other
RUN mkdir -p /var/run/sshd && \
 sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
 mkdir -p /root/.ssh/ && \
 ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
 cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
 && printf "Host *\n StrictHostKeyChecking no\n" >> /root/.ssh/config

# Install Horovod
RUN pip uninstall -y horovod \
 && ldconfig /usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs \
 && HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_CUDA_HOME=/usr/local/cuda-11.3 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod==${HOROVOD_VERSION} \
 && ldconfig


# Install PT S3 plugin
RUN mkdir -p /etc/pki/tls/certs && cp /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

WORKDIR /


# Install scikit-learn and pandas
RUN pip3 install scikit-learn \
    scikit-learn \
    pandas

WORKDIR /

# Install libboost from source. This package is needed for smdataparallel functionality [for networking asynchronous IO].
RUN wget https://sourceforge.net/projects/boost/files/boost/1.73.0/boost_1_73_0.tar.gz/download -O boost_1_73_0.tar.gz \
 && tar -xzf boost_1_73_0.tar.gz \
 && cd boost_1_73_0 \
 && ./bootstrap.sh \
 && ./b2 threading=multi --prefix=/opt/conda -j 64 cxxflags=-fPIC cflags=-fPIC install || true \
 && cd .. \
 && rm -rf boost_1_73_0.tar.gz \
 && rm -rf boost_1_73_0 \
 && cd /opt/conda/include/boost

WORKDIR /opt/pytorch

# Copy workaround script for incorrect hostname
COPY changehostname.c /
COPY start_with_right_hostname.sh /usr/local/bin/start_with_right_hostname.sh

RUN chmod +x /usr/local/bin/start_with_right_hostname.sh

WORKDIR /root

RUN pip install --no-cache-dir -U \
    # disable smdebug pip install until available stable smdebug releases
    # smdebug==${SMDEBUG_VERSION} \
    smclarify \
    #Pin sagemaker version to 2.102.0 as the latest version has an issue, should revert it when the issue has gone
    "sagemaker==2.102.0" \
    sagemaker-experiments==0.* \
    sagemaker-pytorch-training

# Install smdebug from souce
RUN cd /tmp \
  && git clone -b ${SMDEBUG_VERSION} https://github.com/awslabs/sagemaker-debugger \
  && cd sagemaker-debugger \
  && python3 setup.py install \
  && rm -rf /tmp/*

# Install extra packages
# numba 0.54 only works with numpy>=1.20. See https://github.com/numba/numba/issues/7339
RUN pip install --no-cache-dir -U \
    "bokeh>=2.3,<3" \
    "imageio>=2.9,<3" \
    "opencv-python>=4.3,<5" \
    "plotly>=5.1,<6" \
    "seaborn>=0.11,<1" \
    "numba<0.54" \
    "shap>=0.39,<1"

# Install RAPIDSMemoryManager.
# Requires cmake>=3.14.
RUN  wget -nv https://github.com/rapidsai/rmm/archive/v${RMM_VERSION}.tar.gz \
 && tar -xvf v${RMM_VERSION}.tar.gz \
 && cd rmm-${RMM_VERSION} \
 && INSTALL_PREFIX=/usr/local ./build.sh librmm \
 && cd .. \
 && rm -rf v${RMM_VERSION}.tar* \
 && rm -rf rmm-${RMM_VERSION}

ENV LD_LIBRARY_PATH="/opt/conda/lib/python${PYTHON_SHORT_VERSION}/site-packages/smdistributed/dataparallel/lib:$LD_LIBRARY_PATH"

ENV PATH="/opt/ml/code:${PATH}"

RUN pip3 install --no-cache --upgrade sagemaker-training
RUN pip3 install awscli

RUN pip3 uninstall -y transformers
RUN pip3 install transformers==4.15.0

# This TEMP solution for dist training
RUN sed -n 215p /usr/local/lib/python3.8/dist-packages/transformers/models/bert/modeling_bert.py

RUN sed -i '215s/.*/        token_type_embeddings = self.token_type_embeddings(token_type_ids.clone())/' /usr/local/lib/python3.8/dist-packages/transformers/models/bert/modeling_bert.py

RUN sed -n 215p /usr/local/lib/python3.8/dist-packages/transformers/models/bert/modeling_bert.py

RUN sed -n 219p /usr/local/lib/python3.8/dist-packages/transformers/models/bert/modeling_bert.py

RUN sed -i '219s/.*/            position_embeddings = self.position_embeddings(position_ids.clone())/' /usr/local/lib/python3.8/dist-packages/transformers/models/bert/modeling_bert.py

ENV OLD_SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > /tmp/sshd_config && \
    sed "0,/Port ${OLD_SSH_PORT}/s//Port 22/" /tmp/sshd_config > /etc/ssh/sshd_config

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY graphstorm /opt/ml/code/graphstorm/
COPY graphstorm/sagemaker/scripts/sagemaker_train.py /opt/ml/code/

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV PYTHONPATH="/opt/ml/code/graphstorm/python/:${PYTHONPATH}"

WORKDIR /

# Starts framework
ENTRYPOINT ["bash", "-m", "start_with_right_hostname.sh"]
CMD ["/bin/bash"]
