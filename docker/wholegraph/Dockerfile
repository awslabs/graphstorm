# Docker file for building a docker image for running GraphStorm code on Amazon SageMaker
# Note: Distributed graph partition will use another docker image which will come soon.

ARG DEVICE=gpu

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2
ENV dev_type=GPU
# Install DGL GPU version
RUN pip3 install dgl==1.0.4+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html && rm -rf /root/.cache

LABEL maintainer="Amazon AI Graph ML team"

# Install related Python packages
RUN pip3 install ogb==1.3.6 scipy pyarrow boto3 scikit-learn transformers

# Install other dependencies
RUN apt-get update
RUN apt-get install -y apt-transport-https
RUN apt-get install -y cython3 libicu-dev
RUN pip3 install h5py psutil

RUN apt-get install -y unzip

# Install GraphStorm from source code
RUN mkdir -p /graphstorm
COPY code/python/graphstorm /graphstorm/python/graphstorm
ENV PYTHONPATH="/graphstorm/python/:${PYTHONPATH}"

# Copy GraphStorm scripts and tools
COPY code/inference_scripts /graphstorm/inference_scripts
COPY code/tools /graphstorm/tools
COPY code/training_scripts /graphstorm/training_scripts

RUN ls /usr/local/cuda*

#################################################
## Install NCCL
RUN apt-get clean
RUN apt-get reinstall ca-certificates
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnccl2_2.15.1-1+cuda11.8_amd64.deb
RUN dpkg -i libnccl2_2.15.1-1+cuda11.8_amd64.deb
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libnccl-dev_2.15.1-1+cuda11.8_amd64.deb
RUN dpkg -i libnccl-dev_2.15.1-1+cuda11.8_amd64.deb

#################################################
## Install EFA installer
ARG EFA_INSTALLER_VERSION=latest
RUN cd $HOME \
    && curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && tar -xf $HOME/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz \
    && cd aws-efa-installer \
    && apt-get update \
    && apt-get install -y libhwloc-dev \
    && ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /var/lib/apt/lists/*



RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key add 3bf863cc.pub
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt --fix-broken -y install
RUN apt update
RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt update
RUN apt install -y cuda-11-8
RUN apt install -y cuda-libraries-dev-11-8

###################################################
## Install AWS-OFI-NCCL plugin
ARG AWS_OFI_NCCL_VERSION=v1.7.1-aws
RUN git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/ \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=/usr/local/cuda-11.8/  \
    && make && make install

ENV PATH "/opt/amazon/efa/bin:$PATH"

# Install WholeGraph
RUN apt install -y ninja-build
RUN apt-get remove -y cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.27.6/cmake-3.27.6-linux-x86_64.sh
RUN mv cmake-3.27.6-linux-x86_64.sh /opt/cmake-3.27.6-linux-x86_64.sh \
    && cd /opt \
    && chmod +x cmake-3.27.6-linux-x86_64.sh \
    && yes | bash cmake-3.27.6-linux-x86_64.sh
RUN pip3 install cmake==3.27.5

COPY wholegraph/install-wholegraph.sh install-wg.sh
RUN bash install-wg.sh

# Increase nofile limit
RUN echo "root                soft    nofile          1048576" >> /etc/security/limits.conf \
    && echo "root                hard    nofile          1048576" >> /etc/security/limits.conf

# Set up SSH
RUN apt-get install -y openssh-client openssh-server
ENV SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > /tmp/sshd_config && \
    sed "0,/^#Port 22/s//Port ${SSH_PORT}/" /tmp/sshd_config > /etc/ssh/sshd_config
ENV SSHDIR $HOME/.ssh
RUN mkdir -p ${SSHDIR}
RUN ssh-keygen -t rsa -f ${SSHDIR}/id_rsa -N ''
RUN cp ${SSHDIR}/id_rsa.pub ${SSHDIR}/authorized_keys

EXPOSE 2222
CMD ["/usr/sbin/sshd", "-D"]
