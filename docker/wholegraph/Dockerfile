FROM nvcr.io/nvidia/dgl:23.07-py3

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

###################################################
## Install AWS-OFI-NCCL plugin
ARG AWS_OFI_NCCL_VERSION=v1.7.1-aws
RUN git clone https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl \
    && cd /opt/aws-ofi-nccl \
    && git checkout ${AWS_OFI_NCCL_VERSION} \
    && ./autogen.sh \
    && ./configure --prefix=/opt/aws-ofi-nccl/ \
       --with-libfabric=/opt/amazon/efa/ \
       --with-cuda=/usr/local/cuda \
    && make && make install

ENV PATH "/opt/amazon/efa/bin:$PATH"

# Install WholeGraph
COPY wholegraph/install_wholegraph.sh install_wholegraph.sh
RUN bash install_wholegraph.sh

# Install GraphStorm
RUN pip install --no-cache-dir boto3 'h5py>=2.10.0' scipy tqdm 'pyarrow>=3' 'transformers==4.28.1' pandas pylint scikit-learn ogb psutil
RUN git clone https://github.com/awslabs/graphstorm

# Increase nofile limit
RUN echo "root                soft    nofile          1048576" >> /etc/security/limits.conf \
    && echo "root                hard    nofile          1048576" >> /etc/security/limits.conf

# Make EFA NCCL plugin the default plugin
RUN sed -i '/nccl_rdma_sharp_plugin/d' /etc/ld.so.conf.d/hpcx.conf \
    && echo "/opt/aws-ofi-nccl/lib" >> /etc/ld.so.conf.d/hpcx.conf \
    && ldconfig

# Set up SSH
RUN apt-get update && apt-get install -y openssh-client openssh-server && rm -rf /var/lib/apt/lists/*
ENV SSH_PORT=2222
RUN cat /etc/ssh/sshd_config > /tmp/sshd_config && \
    sed "0,/^#Port 22/s//Port ${SSH_PORT}/" /tmp/sshd_config > /etc/ssh/sshd_config
ENV HOME=/root
ENV SSHDIR $HOME/.ssh
RUN mkdir -p ${SSHDIR}
RUN ssh-keygen -t rsa -f ${SSHDIR}/id_rsa -N ''
RUN cp ${SSHDIR}/id_rsa.pub ${SSHDIR}/authorized_keys
RUN touch /root/.ssh/config;echo -e "Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile=/dev/null\n  Port ${SSH_PORT}" > /root/.ssh/config
EXPOSE 2222
RUN mkdir /run/sshd

CMD ["/usr/sbin/sshd", "-D"]
