FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime


RUN apt-get update
RUN apt install tesseract-ocr -y
RUN pip install transformers
RUN pip install pytesseract
RUN mkdir /document_classification

ENV TEST_IMG_NAME=""

WORKDIR /document_classification