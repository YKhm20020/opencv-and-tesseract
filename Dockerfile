FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL maintainer="mshinoda"
RUN apt-get update

# Python 3.7.1 から 3.10 に移行予定
ENV PYTHON_VERSION 3.10
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

# timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# apt
RUN apt update
RUN apt-get install -y curl
RUN apt-get update
RUN apt install -y libopencv-dev
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:git-core/ppa

RUN apt-get install -y git
RUN apt-get install wget

# install python and pip
RUN apt install -y python3.10 python3-pip
RUN pip install --upgrade pip
RUN python3 -m pip install --upgrade pip setuptools

# set working directory and copy files
WORKDIR /usr/src/app
COPY ./ /usr/src/app

# install the NVIDIA Container Toolkit
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && \
    apt-get update
RUN apt-get install -y nvidia-container-toolkit

# install opencv
RUN pip install opencv-python
RUN pip install numpy
RUN pip install pandas
RUN pip install pyocr
RUN pip install Pillow
RUN pip install pdf2image
RUN apt-get install poppler-utils

# install PaddleOCR
RUN apt remove -y python3-blinker
RUN pip install paddleocr
# Python のバージョンを変更した場合は paddlepaddle の対応バージョンも確認して合わせる
# 以下リンクから Python のバージョンと合わせる https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/install/pip/windows-pip_en.html
# RUN python3 -m pip install paddlepaddle==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pyinstaller
RUN pip install loguru

# install Fugashi, unidic
RUN pip install fugashi[unidic]
RUN python3 -m unidic download

# install for using Llama API
RUN pip install replicate

RUN pip install llama-cpp-python
RUN pip install --force-reinstall --ignore-installed --no-cache-dir llama-cpp-python==0.1.65

# for DeblurGANv2
RUN pip install torch>=1.0.1
RUN pip install torchvision
RUN pip install torchsummary
RUN pip install pretrainedmodels
RUN pip install joblib
RUN pip install albumentations>=1.0.0
RUN pip install scikit-image==0.19
RUN pip install tqdm
RUN pip install glog
RUN pip install tensorboardx
RUN pip install fire