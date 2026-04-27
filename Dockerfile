# 使用官方 Python 3.10 基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装 OpenCV 所需的系统依赖（适用于最新 Debian）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖（包括 gunicorn 和 opencv-python-headless）
RUN pip install --no-cache-dir -r requirements.txt

# 如果你担心 gunicorn 没装上，可以再加一行显式安装
RUN pip install --no-cache-dir gunicorn

# 复制所有项目文件
COPY . .

# 暴露端口（Railway 会通过环境变量注入）
EXPOSE $PORT

# 启动命令
CMD gunicorn app:app --bind 0.0.0.0:$PORT