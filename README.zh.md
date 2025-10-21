# PaddleOCR-VL API

基于 PaddleOCR-VL-0.9B 模型的高性能 OCR 微服务，提供兼容 MinerU API 格式的文档识别能力。

## 特性

- **高性能**: 多进程 + 多线程架构，充分利用 GPU 加速
- **MinerU 兼容**: 完全兼容 MinerU API 响应格式
- **多格式支持**: 支持图片、PDF 文档识别
- **灵活输出**: Markdown、JSON、图片 base64、ZIP 打包等多种输出格式
- **PDF 增强**: 支持页码范围提取，自动合并多页结果
- **容器化部署**: Docker Compose 一键启动完整服务栈

## 快速开始

### 使用 Docker Compose（推荐）

```bash
# 克隆仓库
git clone <your-repo-url>
cd paddleocr-vl-api

# 启动完整服务栈
docker-compose up -d

# 查看日志
docker-compose logs -f
```

服务将在以下端口启动：
- OCR API: http://localhost:8080
- VLLM Server: http://localhost:8000

### 使用预构建镜像

```bash
# 仅启动 OCR API（需要配置 VLLM 服务地址）
docker run -p 8080:8080 \
  -e VLLM_SERVER_URL=http://your-vllm-server:8000/v1 \
  quantatrisk/paddleocr-api:latest
```

### 本地开发

```bash
# 安装依赖
pip install -r api_server/requirements.txt

# 启动 API 服务器
cd api_server
uvicorn app:app --reload --host 0.0.0.0 --port 8080

# 启动 Web UI 测试工具
streamlit run test/streamlit_client.py
```

## API 使用

### 基础请求

```bash
curl -X POST http://localhost:8080/file_parse \
  -F "files=@document.pdf" \
  -F "return_md=true"
```

### 高级参数

```bash
curl -X POST http://localhost:8080/file_parse \
  -F "files=@document.pdf" \
  -F "return_md=true" \
  -F "return_middle_json=true" \
  -F "return_images=true" \
  -F "start_page_id=0" \
  -F "end_page_id=5" \
  -F "response_format_zip=true"
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `files` | File[] | 必填 | 上传文件（支持多文件） |
| `return_md` | Boolean | true | 返回 Markdown 格式 |
| `return_middle_json` | Boolean | false | 返回完整 JSON 结果 |
| `return_images` | Boolean | false | 返回图片 base64 编码 |
| `response_format_zip` | Boolean | false | ZIP 打包下载 |
| `start_page_id` | Integer | 0 | PDF 起始页码 |
| `end_page_id` | Integer | 99999 | PDF 结束页码 |

### 响应格式

```json
{
  "status": "success",
  "message": "OCR processing completed",
  "results": [
    {
      "filename": "document.pdf",
      "md_content": "# 识别结果...",
      "middle_json": {...},
      "images": ["data:image/png;base64,..."]
    }
  ]
}
```

## 架构设计

### 服务组件

```
┌─────────────────┐
│  Streamlit UI   │  (测试客户端)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OCR API       │  FastAPI + Gunicorn
│   (port 8080)   │  多进程 + 线程池
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VLLM Server    │  PaddleOCR-VL-0.9B
│   (port 8000)   │  GPU 加速推理
└─────────────────┘
```

### 关键技术

- **线程安全**: PaddleOCR 单例 + 线程锁保护
- **异步处理**: FastAPI 异步端点 + ThreadPoolExecutor
- **PDF 处理**: pypdf 页面提取 + 自动合并
- **容器化**: Docker 多阶段构建 + Compose 编排

## 配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VLLM_SERVER_URL` | VLLM 服务地址 | http://192.168.6.146:8780/v1 |
| `THREAD_WORKERS` | 线程池大小 | 10 |
| `MAX_WORKERS` | Gunicorn 进程数 | 4 |

### Docker Compose 配置

修改 `docker-compose.yml` 调整资源分配：

```yaml
vllm-server:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1  # 调整 GPU 数量
```

## 性能优化

- **多进程**: Gunicorn 启动多个 worker 进程
- **线程池**: ThreadPoolExecutor 并发处理 OCR 任务
- **GPU 加速**: VLLM 后端充分利用 GPU 算力
- **单例模式**: 模型只加载一次，节省内存

## 测试

```bash
# 基础功能测试
python test/test_paddleocr.py

# 启动 Web UI 测试工具
streamlit run test/streamlit_client.py

# API 健康检查
curl http://localhost:8080/health
```

## 项目结构

```
paddleocr-vl-api/
├── api_server/
│   ├── app.py              # FastAPI 主应用
│   └── requirements.txt    # Python 依赖
├── test/
│   ├── streamlit_client.py # Streamlit 测试客户端
│   ├── test_paddleocr.py  # 基础测试
│   └── files/             # 测试文件
├── Dockerfile              # OCR API 镜像
├── docker-compose.yml      # 服务编排
└── CLAUDE.md              # 开发指南
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

## 致谢

基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 和 [vLLM](https://github.com/vllm-project/vllm) 构建。
