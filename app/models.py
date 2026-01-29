"""
Pydantic 数据模型
"""
from typing import Optional, Dict
from pydantic import BaseModel


class FileResult(BaseModel):
    """单个文件的处理结果 - 兼容 MinerU 格式"""
    md_content: Optional[str] = None  # Markdown 内容字符串
    middle_json: Optional[str] = None  # 完整 JSON 字符串
    content_list: Optional[str] = None  # 内容列表 JSON 字符串
    images: Optional[Dict[str, str]] = None  # 图片名 -> base64 映射


class OCRResponse(BaseModel):
    """API 响应模型 - 兼容 MinerU 格式"""
    backend: str = "paddleocr"
    version: str = "1.0.0"
    results: Dict[str, FileResult]  # 文件名 -> 处理结果映射
