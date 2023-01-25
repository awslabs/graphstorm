ARG CUDA_VER=11.3
ARG PYTHON_VERSION=3.8.13

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.12.1-gpu-py38-cu113-ubuntu20.04-sagemaker

LABEL maintainer="Amazon AI"
LABEL dlc_major_version="1"

RUN pip install --no-cache-dir transformers==4.15.0

# This TEMP solution for dist training
RUN sed -n 215p /opt/conda/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py

RUN sed -i '215s/.*/        token_type_embeddings = self.token_type_embeddings(token_type_ids.clone())/' /opt/conda/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py

RUN sed -n 215p /opt/conda/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py

RUN sed -n 219p /opt/conda/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py

RUN sed -i '219s/.*/            position_embeddings = self.position_embeddings(position_ids.clone())/' /opt/conda/lib/python3.8/site-packages/transformers/models/bert/modeling_bert.py

# Install Parmetis
RUN apt-get update; \
            apt-get install -y --no-install-recommends build-essential python3-dev make g++ vim net-tools

# Install mpi4.0 for 64-bit parmetis
RUN apt-get install -y --no-install-recommends libopenmpi-dev

# Download GKlib, METIS, ParMETIS
RUN cd /root; git clone https://github.com/KarypisLab/GKlib.git
RUN cd /root/GKlib; make config prefix=/root/GKlib/; make install
RUN cd /root; git clone https://github.com/KarypisLab/METIS.git
RUN cd /root/METIS; make config gklib_path=/root/GKlib/ prefix=/root/METIS/ i64=1; make -j8; make -j8 install
RUN cd /root; git clone https://github.com/classicsong/ParMETIS.git
RUN cd /root/ParMETIS; git checkout dgl_part3_dist
RUN cd /root/ParMETIS; make config cc=mpicc prefix=/root/ParMETIS/ gklib_path=/root/GKlib/ metis_path=/root/METIS/; make -j8; make -j8 install

# Update the Path Variable.
ENV LD_LIBRARY_PATH=/root/ParMETIS/lib:/root/METIS/lib:/root/GKlib/lib:$LD_LIBRARY_PATH

ENV PATH=/root/ParMETIS/bin:/root/GKlib/bin:/root/METIS/bin:$PATH

# Install DGL
RUN pip install --pre --no-cache-dir dgl-cu113 -f https://data.dgl.ai/wheels-test/repo.html

# Install related Python packages
RUN pip install --no-cache-dir ogb
RUN pip install pydantic

# Download DGL source code
RUN cd /root; git clone --recursive https://github.com/dmlc/dgl.git


# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY graphstorm /opt/ml/code/graphstorm/
COPY graphstorm/sagemaker/scripts/utils.py /opt/ml/code/
COPY graphstorm/sagemaker/scripts/sagemaker_infer.py /opt/ml/code/
COPY graphstorm/sagemaker/scripts/sagemaker_parmetis.py /opt/ml/code/
COPY graphstorm/sagemaker/scripts/sagemaker_train.py /opt/ml/code/

ENV PYTHONPATH="/opt/ml/code/graphstorm/python/:${PYTHONPATH}"

WORKDIR /

CMD ["/bin/bash"]
