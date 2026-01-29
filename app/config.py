"""
配置管理
"""
import os

# VLLM Server 配置
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://192.168.6.146:8780/v1")

# 线程池配置
THREAD_WORKERS = int(os.getenv("THREAD_WORKERS", "10"))

# Gunicorn 配置（在 Dockerfile CMD 中使用）
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# API 版本
API_VERSION = "1.0.0"
BACKEND_NAME = "paddleocr"
