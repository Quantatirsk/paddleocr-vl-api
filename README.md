# PaddleOCR-VL API

[中文文档](docs/README.zh.md) | English

A high-performance OCR microservice based on PaddleOCR-VL-1.5-0.9B model, providing document recognition capabilities compatible with MinerU API format.

## Model Information

- **Model**: [PaddleOCR-VL-1.5-0.9B](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)
- **Backend**: VLLM for high-performance GPU inference
- **Features**: Vision-Language model optimized for document understanding

## Features

- **High Performance**: Multi-process + multi-threading architecture with GPU acceleration
- **MinerU Compatible**: Fully compatible with MinerU API response format
- **Multi-format Support**: Supports image and PDF document recognition
- **Flexible Output**: Multiple output formats including Markdown, JSON, image base64, and ZIP packaging
- **PDF Enhancement**: Supports page range extraction with automatic multi-page result merging
- **Containerized Deployment**: One-click startup with Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd paddleocr-vl-api

# Copy environment configuration
cp .env.example .env

# Start the complete service stack
docker-compose up -d

# View logs
docker-compose logs -f
```

Services will start on the following ports:
- OCR API: http://localhost:8781
- VLLM Server: http://localhost:8780

### Using Pre-built Image

```bash
# Start OCR API only (requires VLLM server configuration)
docker run -p 8080:8080 \
  -e VLLM_SERVER_URL=http://your-vllm-server:8000/v1 \
  quantatrisk/paddleocr-api:latest
```

### Local Development

```bash
# Install dependencies
pip install -r app/requirements.txt

# Start API server
cd app
uvicorn main:app --reload --host 0.0.0.0 --port 8080

# Start Web UI testing tool
streamlit run test/streamlit_client.py
```

## API Usage

### Basic Request

```bash
curl -X POST http://localhost:8781/file_parse \
  -F "files=@document.pdf" \
  -F "return_md=true"
```

### Advanced Parameters

```bash
curl -X POST http://localhost:8781/file_parse \
  -F "files=@document.pdf" \
  -F "return_md=true" \
  -F "return_middle_json=true" \
  -F "return_images=true" \
  -F "start_page_id=0" \
  -F "end_page_id=5" \
  -F "response_format_zip=true"
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `files` | File[] | Required | Upload files (supports multiple files) |
| `return_md` | Boolean | true | Return Markdown format |
| `return_middle_json` | Boolean | false | Return complete JSON result |
| `return_images` | Boolean | false | Return base64 encoded images |
| `response_format_zip` | Boolean | false | Package as ZIP for download |
| `start_page_id` | Integer | 0 | PDF starting page number |
| `end_page_id` | Integer | 99999 | PDF ending page number |

### Response Format

```json
{
  "status": "success",
  "message": "OCR processing completed",
  "results": [
    {
      "filename": "document.pdf",
      "md_content": "# Recognition result...",
      "middle_json": {...},
      "images": ["data:image/png;base64,..."]
    }
  ]
}
```

## Architecture

### Service Components

```
┌─────────────────┐
│  Streamlit UI   │  (Testing Client)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   OCR API       │  FastAPI + Gunicorn
│  (port 8781)    │  Multi-process + Thread Pool
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  VLLM Server    │  PaddleOCR-VL-1.5-0.9B
│  (port 8780)    │  GPU Accelerated Inference
└─────────────────┘
```

### Key Technologies

- **Thread Safety**: PaddleOCR singleton + thread lock protection
- **Async Processing**: FastAPI async endpoints + ThreadPoolExecutor
- **PDF Processing**: pypdf page extraction + automatic merging
- **Containerization**: Docker multi-stage build + Compose orchestration

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_SERVER_URL` | VLLM service address | http://vllm-server:8000/v1 |
| `THREAD_WORKERS` | Thread pool size | 10 |
| `MAX_WORKERS` | Gunicorn worker processes | 4 |
| `NVIDIA_VISIBLE_DEVICES` | GPU device selection | 0 |

Edit `.env` file to customize configuration.

### Docker Compose Configuration

Modify `docker-compose.yml` to adjust resource allocation:

```yaml
vllm-server:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1  # Adjust GPU count
```

## Performance Optimization

- **Multi-processing**: Gunicorn starts multiple worker processes
- **Thread Pool**: ThreadPoolExecutor for concurrent OCR tasks
- **GPU Acceleration**: VLLM backend fully utilizes GPU computing power
- **Singleton Pattern**: Model loaded only once to save memory

## Testing

```bash
# Basic functionality test
python test/test_paddleocr.py

# Start Web UI testing tool
streamlit run test/streamlit_client.py

# API health check
curl http://localhost:8781/health
```

## Project Structure

```
paddleocr-vl-api/
├── app/
│   ├── main.py              # FastAPI main application
│   ├── config.py            # Configuration management
│   ├── models.py            # Pydantic data models
│   ├── ocr/                 # OCR core modules
│   │   ├── singleton.py     # PaddleOCR singleton
│   │   ├── processor.py     # OCR processing logic
│   │   └── pdf.py           # PDF utilities
│   ├── utils/               # Utility functions
│   │   ├── file_merge.py    # Multi-page file merging
│   │   └── response.py      # Response building
│   └── requirements.txt     # Python dependencies
├── test/
│   ├── streamlit_client.py  # Streamlit testing client
│   ├── test_paddleocr.py    # Basic tests
│   └── files/               # Test files
├── Dockerfile               # OCR API image
├── docker-compose.yml       # Single-node deployment
├── docker-compose-cluster.yml # Multi-GPU cluster deployment
├── nginx.conf               # Load balancer configuration
├── .env.example             # Environment template
└── CLAUDE.md                # Development guide
```

## License

MIT License

## Contributing

Issues and Pull Requests are welcome!

## Acknowledgments

Built on [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [vLLM](https://github.com/vllm-project/vllm).
