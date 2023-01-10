FROM tensorflow/tensorflow:latest-gpu

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install tensorflow-probability
