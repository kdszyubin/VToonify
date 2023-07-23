FROM continuumio/miniconda3:23.5.2-0
ENV HTTP_PROXY "http://192.168.31.57:1087"
ENV HTTPS_PROXY "http://192.168.31.57:1087"
ENV NO_PROXY "localhost,127.0.0.1,docker-registry.example.com,.corp"

RUN apt-get update
RUN apt-get install -y build-essential cmake libgl1-mesa-glx

COPY ./environment/vtoonify_env.yaml /tmp/vtoonify_env.yml
RUN conda env create -f  /tmp/vtoonify_env.yml
COPY . /data/workspace/williamyang1991/VToonify
RUN pip cache purge
COPY resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth

# Set the working directory to /app
WORKDIR /data/workspace/williamyang1991/VToonify

# Run any command you want when the container launches
CMD [ "/bin/bash", "-c", "source activate vtoonify_env && python style_transfer_server.py --cpu" ]

