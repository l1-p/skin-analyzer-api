# 使用官方 Python 3.10 轻量镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖（OpenCV 所需底层库）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制所有项目文件（包括模型文件 best_model.pth）
COPY . .

# 暴露端口（Railway 会自动注入 PORT 环境变量）
EXPOSE $PORT

# 启动命令
CMD gunicorn app:app --bind 0.0.0.0:$PORT