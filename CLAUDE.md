# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发原则

Linus 开发原则：不要拉屎，过多的解释和写文档，就如同拉屎。只需要专注于核心功能的实现，保持数据结构简洁、科学、规范，但不要过度设计。

## 项目架构

这是一个基于 PaddleOCR 的高并发 OCR 服务，采用微服务架构：

### 核心组件

1. **VLLM Server** (`vllm-server` 服务)
   - 运行 PaddleOCR-VL-0.9B 模型
   - 提供底层 OCR 推理能力
   - GPU 加速，暴露在 8000 端口

2. **OCR API Server** (`api_server/app.py`)
   - FastAPI 应用，主端点：`/file_parse`
   - 兼容 MinerU API 格式
   - 多进程 + 多线程架构（Gunicorn + ThreadPoolExecutor）
   - 使用线程锁保证 PaddleOCR 单例线程安全
   - 支持图片和 PDF（带页码范围过滤）
   - 输出格式：Markdown、JSON、图片 base64、ZIP 打包

3. **Streamlit 客户端** (`test/streamlit_client.py`)
   - Web UI 测试工具
   - 支持多文件上传、参数配置、结果预览

### 关键设计

- **单例模式**: PaddleOCRSingleton 确保模型只加载一次，节省内存
- **线程安全**: 使用 `threading.Lock()` 保护 OCR 推理调用
- **异步处理**: FastAPI 异步端点 + ThreadPoolExecutor 执行同步 OCR
- **PDF 处理**: 支持页码范围提取（pypdf），多页结果自动合并
- **MinerU 兼容**: 响应格式遵循 MinerU API 规范

## 常用命令

### 启动服务
```bash
# 使用 Docker Compose 启动完整服务栈
docker-compose up -d

# 仅启动 OCR API（需要手动配置 VLLM_SERVER_URL）
docker run -p 8080:8080 \
  -e VLLM_SERVER_URL=http://vllm-server:8000/v1 \
  quantatrisk/paddleocr-api:latest
```

### 构建镜像
```bash
# 从根目录构建
docker build -t paddleocr-api .
```

### 本地开发
```bash
# 安装依赖
pip install -r api_server/requirements.txt

# 启动 API 服务器（开发模式）
cd api_server
uvicorn app:app --reload --host 0.0.0.0 --port 8080

# 启动 Streamlit 客户端
streamlit run test/streamlit_client.py
```

### 测试
```bash
# 基础测试
python test/test_paddleocr.py

# 使用 curl 测试 API
curl -X POST http://localhost:8080/file_parse \
  -F "files=@test.pdf" \
  -F "return_md=true" \
  -F "start_page_id=0" \
  -F "end_page_id=5"
```

## 环境变量

- `VLLM_SERVER_URL`: VLLM 服务地址（默认: `http://192.168.6.146:8780/v1`）
- `THREAD_WORKERS`: 线程池大小（默认: 10）
- `MAX_WORKERS`: Gunicorn 进程数（默认: 4）

## API 端点

### POST `/file_parse`
主要 OCR 端点，参数：
- `files`: 上传文件（支持多文件）
- `return_md`: 返回 Markdown（默认: true）
- `return_middle_json`: 返回完整 JSON（默认: false）
- `return_images`: 返回图片 base64（默认: false）
- `response_format_zip`: ZIP 打包下载（默认: false）
- `start_page_id`: PDF 起始页（默认: 0）
- `end_page_id`: PDF 结束页（默认: 99999）

### GET `/health`
健康检查端点

## 文件结构

```
api_server/
  ├── app.py              # FastAPI 主应用
  └── requirements.txt    # Python 依赖

test/
  ├── streamlit_client.py # Streamlit 测试客户端
  ├── test_paddleocr.py  # 基础测试
  └── files/             # 测试文件

Dockerfile              # OCR API 镜像定义
docker-compose.yml      # 完整服务编排
```

## 关键代码路径

- PDF 页面提取: `api_server/app.py:extract_pdf_pages()` (341-380 行)
- 多页合并逻辑: `api_server/app.py:merge_json_files()`, `merge_markdown_files()` (94-227 行)
- OCR 处理入口: `api_server/app.py:process_ocr_sync()` (383-541 行)
- 响应构建: `api_server/app.py:build_file_result()` (230-338 行)
