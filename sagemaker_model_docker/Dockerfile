# SageMaker MxNet image
ARG REGION=us-east-1

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38-ubuntu20.04-sagemaker

RUN pip3 install sagemaker-training
RUN pip install smdebug

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
COPY code/ /opt/ml/code

# this environment variable is used by the SageMaker container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

ENV PATH="/opt/ml/code:${PATH}"

# Defines train.py as script entrypoint 
ENV SAGEMAKER_PROGRAM preprocess.py
