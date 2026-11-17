FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TORCH_NUM_THREADS=2
COPY requirements.txt .
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
COPY . .
CMD ["sh", "start.sh"]
