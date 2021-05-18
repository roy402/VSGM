ARG CUDA_VERSION

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
ARG NVIDIA_VERSION

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install python3-pip libxrender1 libsm6 xserver-xorg-core xorg python3-venv vim pciutils wget git module-init-tools

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY requirements.txt scripts/install_nvidia.sh /app/
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && python3 -c "import ai2thor.controller; ai2thor.controller.Controller(download_only=True)"
RUN NVIDIA_VERSION=$NVIDIA_VERSION /app/install_nvidia.sh

COPY ai2thor_docker /app/ai2thor_docker
COPY example_agent.py ./

# edit your userpsd
ENV ssspsd='root:cilab'

# install ssh
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd

RUN echo ${ssspsd} | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# install vim wget
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y vim

# link host folder to container folder
RUN mkdir -p /home/web
RUN mkdir -p /home/host

# Setting GPU
ENV NVIDIA_VISIBLE_DEVICES all

# redirect ssh login 
RUN echo 'cd /home' >> /root/.bashrc

# install conda
# conda can not install by this file
# please install conda in the container with command lineyes

# After install conda add command to tell user login to change password
# -> passwd --expire root

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]