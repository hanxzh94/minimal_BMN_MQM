FROM tensorflow/tensorflow:2.6.0-gpu

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install tensorflow-probability==0.14.0
