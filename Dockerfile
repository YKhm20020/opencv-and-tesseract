FROM ubuntu:latest

ENV PYTHON_VERSION 3.7.1
ENV HOME /root
ENV PYTHON_ROOT $HOME/local/python-$PYTHON_VERSION
ENV PATH $PYTHON_ROOT/bin:$PATH
ENV PYENV_ROOT $HOME/.pyenv

# timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

# apt
RUN apt update
RUN apt install -y libopencv-dev
RUN apt install -y tesseract-ocr
RUN apt install -y libtesseract-dev

RUN apt-get update -y \
    && apt-get upgrade -y \
    && apt-get autoremove -y \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:git-core/ppa

RUN apt-get install -y git

# install python and pip
RUN apt install -y python3 python3-pip
RUN pip install --upgrade pip
RUN python3 -m pip install --upgrade pip setuptools

# set working directory and copy files
WORKDIR /usr/src/app
COPY ./ /usr/src/app

# install opencv
RUN pip install opencv-python
RUN pip install numpy
RUN pip install pyocr
RUN pip install Pillow

# install PaddleOCR
RUN apt remove -y python3-blinker
RUN pip install paddleocr
RUN python3 -m pip install paddlepaddle==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install pyinstaller
RUN pip install loguru

# install MeCab and dic
RUN pip install mecab-python3
ENV MECABRC ./MeCab/etc/mecabrc
RUN pip install unidic-lite