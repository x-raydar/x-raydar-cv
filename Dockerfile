# set base image (host OS)
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:19.10-py3
FROM ${FROM_IMAGE_NAME}

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN conda install -y -c conda-forge --file requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# Port:
EXPOSE 5050

# command to run on container start
ENTRYPOINT [ "python", "./wt_ai_main.py" ]