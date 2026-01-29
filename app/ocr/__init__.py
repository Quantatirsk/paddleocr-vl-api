"""
OCR 处理模块
"""
from .singleton import PaddleOCRSingleton
from .processor import process_ocr_sync, process_ocr_async
from .pdf import extract_pdf_pages

__all__ = [
    "PaddleOCRSingleton",
    "process_ocr_sync",
    "process_ocr_async",
    "extract_pdf_pages"
]
