"""
PDF 文件处理工具
"""
import tempfile
import logging
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def extract_pdf_pages(pdf_path: str, start_page: int, end_page: int) -> str:
    """
    从 PDF 中提取指定页码范围，生成新的临时 PDF 文件

    参数:
        pdf_path: 原始 PDF 文件路径
        start_page: 起始页码（从 0 开始，包含）
        end_page: 结束页码（包含）

    返回:
        临时 PDF 文件路径
    """
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)

        # 调整页码范围
        start_page = max(0, start_page)
        end_page = min(total_pages - 1, end_page)

        logger.info(f"Extracting PDF pages {start_page}-{end_page} from {total_pages} total pages")

        # 创建新的 PDF writer
        writer = PdfWriter()

        # 添加指定范围的页面
        for page_num in range(start_page, end_page + 1):
            writer.add_page(reader.pages[page_num])

        # 写入临时文件
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        writer.write(temp_pdf.name)
        temp_pdf.close()

        logger.info(f"Created temporary PDF with {end_page - start_page + 1} pages: {temp_pdf.name}")
        return temp_pdf.name

    except Exception as e:
        logger.error(f"Error extracting PDF pages: {e}")
        raise
