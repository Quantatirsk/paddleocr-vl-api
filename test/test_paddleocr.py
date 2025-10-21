"""
PaddleOCR 核心功能测试
"""
from paddleocr import PaddleOCRVL
from pathlib import Path

# 初始化 PaddleOCR
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server",
    vl_rec_server_url="http://localhost:8780/v1"
)

# 运行 OCR
test_image = "files/arxiv_paper1.pdf"
output = pipeline.predict(test_image)

# 创建输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 输出并保存结果
for res in output:
    res.print()
    # res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
