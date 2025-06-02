FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip install pandas==2.2.3 numpy==2.0.1 \
    beautifulsoup4==4.13.3 selenium==4.28.1 webdriver_manager==4.0.2 \
    transformers==4.49.0 

RUN pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/bin/bash"]
