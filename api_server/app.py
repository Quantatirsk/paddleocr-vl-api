"""
优化的 PaddleOCR API - 支持图片和 PDF 文件的高并发 OCR 识别
参考 MinerU API 设计，提供灵活的配置选项和多种返回格式
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from paddleocr import PaddleOCRVL
import tempfile
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import time
from contextlib import asynccontextmanager
import threading
import zipfile
import io
import shutil
from pypdf import PdfReader, PdfWriter
import json
import base64
import uuid

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局线程池
thread_pool: Optional[ThreadPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global thread_pool
    # 启动时
    thread_pool = ThreadPoolExecutor(max_workers=THREAD_WORKERS)
    logger.info("Thread pool started")

    # 预加载 PaddleOCR 模型
    logger.info("Preloading PaddleOCR pipeline...")
    PaddleOCRSingleton.get_pipeline()
    logger.info("PaddleOCR pipeline preloaded successfully")

    yield
    # 关闭时
    if thread_pool:
        thread_pool.shutdown(wait=True)
        logger.info("Thread pool shut down")


app = FastAPI(title="PaddleOCR Optimized API", lifespan=lifespan)

# 配置参数
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://192.168.6.146:8780/v1")
THREAD_WORKERS = int(os.getenv("THREAD_WORKERS", "10"))  # 线程池大小


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


# 使用单例模式初始化 PaddleOCR（避免重复加载模型）
class PaddleOCRSingleton:
    _instance = None
    _pipeline = None
    _lock = threading.Lock()  # 添加线程锁，确保线程安全

    @classmethod
    def get_pipeline(cls):
        if cls._pipeline is None:
            logger.info("Initializing PaddleOCR pipeline...")
            cls._pipeline = PaddleOCRVL(
                vl_rec_backend="vllm-server",
                vl_rec_server_url=VLLM_SERVER_URL
            )
            logger.info("PaddleOCR pipeline initialized successfully")
        return cls._pipeline

    @classmethod
    def get_lock(cls):
        """获取锁对象"""
        return cls._lock


def merge_json_files(output_dir: Path, filename_stem: str) -> Optional[str]:
    """
    合并多页 PDF 的 JSON 文件
    将 filename_0_res.json, filename_1_res.json 等合并成一个 filename_res.json

    返回: 合并后的文件路径，如果没有文件则返回 None
    """
    try:
        # 查找所有分页 JSON 文件
        json_files = list(output_dir.glob(f"{filename_stem}_*_res.json"))

        if not json_files:
            # 没有分页文件，可能是单页或图片
            single_file = output_dir / f"{filename_stem}_res.json"
            if single_file.exists():
                return str(single_file)
            return None

        # 按照页码数字排序（而不是字符串排序）
        # 文件名格式: filename_0_res.json, filename_1_res.json, ...
        def extract_page_num(filepath):
            # 提取 filename_N_res.json 中的 N
            stem = filepath.stem  # 去掉 .json 得到 filename_N_res
            parts = stem.split('_')
            # 倒数第二个部分是页码 (最后一个是 'res')
            return int(parts[-2])

        json_files = sorted(json_files, key=extract_page_num)

        logger.info(f"Found {len(json_files)} JSON files to merge: {[f.name for f in json_files]}")

        # 读取所有 JSON 文件
        pages_data = []
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                pages_data.append(data)

        # 合并后的文件名
        merged_file = output_dir / f"{filename_stem}_res.json"

        # 如果只有一个文件，直接重命名
        if len(json_files) == 1:
            shutil.move(str(json_files[0]), str(merged_file))
            logger.info(f"Single page, renamed to: {merged_file.name}")
            return str(merged_file)

        # 多页合并：创建一个包含所有页面的数组
        merged_data = {
            "input_path": pages_data[0]["input_path"],
            "total_pages": len(pages_data),
            "model_settings": pages_data[0]["model_settings"],
            "pages": pages_data  # 所有页面数据的数组
        }

        # 写入合并后的 JSON
        with open(merged_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)

        # 删除原始分页文件
        for json_file in json_files:
            json_file.unlink()
            logger.debug(f"Deleted: {json_file.name}")

        logger.info(f"Merged {len(json_files)} JSON files into: {merged_file.name}")
        return str(merged_file)

    except Exception as e:
        logger.error(f"Error merging JSON files: {e}")
        raise


def merge_markdown_files(output_dir: Path, filename_stem: str) -> Optional[str]:
    """
    合并多页 PDF 的 Markdown 文件
    将 filename_0.md, filename_1.md 等合并成一个 filename.md

    返回: 合并后的文件路径，如果没有文件则返回 None
    """
    try:
        # 查找所有分页 Markdown 文件
        md_files = list(output_dir.glob(f"{filename_stem}_*.md"))

        if not md_files:
            # 没有分页文件，可能是单页或图片
            single_file = output_dir / f"{filename_stem}.md"
            if single_file.exists():
                return str(single_file)
            return None

        # 按照页码数字排序（而不是字符串排序）
        # 文件名格式: filename_0.md, filename_1.md, ...
        def extract_page_num(filepath):
            # 提取 filename_N.md 中的 N
            stem = filepath.stem  # 去掉 .md 得到 filename_N
            parts = stem.split('_')
            # 最后一个部分是页码
            return int(parts[-1])

        md_files = sorted(md_files, key=extract_page_num)

        logger.info(f"Found {len(md_files)} Markdown files to merge: {[f.name for f in md_files]}")

        # 合并后的文件名
        merged_file = output_dir / f"{filename_stem}.md"

        # 如果只有一个文件，直接重命名
        if len(md_files) == 1:
            shutil.move(str(md_files[0]), str(merged_file))
            logger.info(f"Single page, renamed to: {merged_file.name}")
            return str(merged_file)

        # 多页合并：直接追加原始内容，不添加任何额外字符
        merged_content = []
        for md_file in md_files:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            merged_content.append(content)

        # 写入合并后的 Markdown
        with open(merged_file, 'w', encoding='utf-8') as f:
            f.write(''.join(merged_content))

        # 删除原始分页文件
        for md_file in md_files:
            md_file.unlink()
            logger.debug(f"Deleted: {md_file.name}")

        logger.info(f"Merged {len(md_files)} Markdown files into: {merged_file.name}")
        return str(merged_file)

    except Exception as e:
        logger.error(f"Error merging Markdown files: {e}")
        raise


def build_file_result(
    output_dir: str,
    filename_stem: str,
    return_markdown: bool = True,
    return_json: bool = True,
    return_content_list: bool = False,
    return_images: bool = True
) -> FileResult:
    """
    构建 MinerU 格式的文件结果

    参数:
        output_dir: 输出目录
        filename_stem: 文件名（不含扩展名）
        return_markdown: 是否返回 Markdown 内容
        return_json: 是否返回 JSON 内容
        return_content_list: 是否返回 content_list
        return_images: 是否返回图片 base64

    返回:
        FileResult 对象
    """
    output_path = Path(output_dir)
    result = FileResult()

    # 读取 Markdown 内容
    if return_markdown:
        md_file = output_path / f"{filename_stem}.md"
        if md_file.exists():
            with open(md_file, 'r', encoding='utf-8') as f:
                result.md_content = f.read()

    # 读取 JSON 内容（作为字符串）
    if return_json:
        json_file = output_path / f"{filename_stem}_res.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                result.middle_json = f.read()  # 直接读取字符串，不解析

    # 读取 content_list（作为字符串）
    if return_content_list:
        # 从 JSON 文件构建简化的 content_list
        json_file = output_path / f"{filename_stem}_res.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 构建简化的 content_list
                content_list = []

                # 如果是多页 PDF 的合并结果
                if 'pages' in data:
                    for page_data in data['pages']:
                        page_idx = page_data.get('page_index', 0)
                        for block in page_data.get('parsing_res_list', []):
                            content_item = {
                                'type': block.get('block_label', 'text'),
                                'bbox': block.get('block_bbox', []),
                                'page_idx': page_idx
                            }
                            if block.get('block_label') == 'text':
                                content_item['text'] = block.get('block_content', '')
                            elif block.get('block_label') == 'image':
                                # 图片路径相对于 output_dir
                                content_item['img_path'] = 'imgs/' + Path(block.get('block_content', '')).name if block.get('block_content') else ''
                            content_list.append(content_item)
                else:
                    # 单页情况
                    page_idx = data.get('page_index', 0)
                    for block in data.get('parsing_res_list', []):
                        content_item = {
                            'type': block.get('block_label', 'text'),
                            'bbox': block.get('block_bbox', []),
                            'page_idx': page_idx
                        }
                        if block.get('block_label') == 'text':
                            content_item['text'] = block.get('block_content', '')
                        elif block.get('block_label') == 'image':
                            content_item['img_path'] = 'imgs/' + Path(block.get('block_content', '')).name if block.get('block_content') else ''
                        content_list.append(content_item)

                # 转换为 JSON 字符串
                result.content_list = json.dumps(content_list, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.warning(f"Failed to build content_list: {e}")

    # 编码图片为 base64
    if return_images:
        imgs_dir = output_path / "imgs"
        if imgs_dir.exists() and imgs_dir.is_dir():
            images_dict = {}
            for img_file in imgs_dir.rglob("*"):
                if img_file.is_file():
                    try:
                        with open(img_file, 'rb') as f:
                            img_data = f.read()
                            # 转换为 base64
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            # 添加 data URI 前缀
                            mime_type = "image/jpeg" if img_file.suffix.lower() in ['.jpg', '.jpeg'] else "image/png"
                            images_dict[img_file.name] = f"data:{mime_type};base64,{img_base64}"
                    except Exception as e:
                        logger.warning(f"Failed to encode image {img_file}: {e}")

            if images_dict:
                result.images = images_dict

    return result


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


def process_ocr_sync(
    image_path: str,
    original_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    return_json: bool = True,
    return_markdown: bool = False,
    start_page_id: int = 0,
    end_page_id: int = 99999
) -> Dict[str, Any]:
    """
    同步 OCR 处理（在线程池中执行）
    支持图片和 PDF 文件格式
    使用锁确保线程安全
    返回完整的 OCR 数据结构

    参数:
        image_path: 图片或 PDF 文件路径
        original_filename: 原始文件名（用于保存输出文件）
        start_page_id: 起始页码（从 0 开始，包含）
        end_page_id: 结束页码（包含）

    对于 PDF 文件，会先提取指定页码范围的页面，然后再进行 OCR
    """
    temp_extracted_pdf = None
    try:
        pipeline = PaddleOCRSingleton.get_pipeline()
        lock = PaddleOCRSingleton.get_lock()

        # 检查是否是 PDF 文件，且需要页码过滤
        is_pdf = image_path.lower().endswith('.pdf')
        needs_extraction = is_pdf and (start_page_id > 0 or end_page_id < 99999)

        # 确定原始文件名（用于输出）
        if original_filename is None:
            original_filename = Path(image_path).name

        # 如果是 PDF 且需要页码过滤，先提取页面
        processing_path = image_path
        if needs_extraction:
            logger.info(f"PDF page extraction required: pages {start_page_id}-{end_page_id}")
            temp_extracted_pdf = extract_pdf_pages(image_path, start_page_id, end_page_id)
            processing_path = temp_extracted_pdf
        elif is_pdf:
            logger.info(f"Processing full PDF without page extraction")

        # 使用锁确保同一时间只有一个线程在使用 OCR 模型
        with lock:
            logger.debug(f"Processing {processing_path} with lock acquired")
            output = pipeline.predict(processing_path)

        output_files = {}

        # 保存文件到指定目录（使用 PaddleOCR 原生方法）
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # 使用原始文件名（不含扩展名）
            filename_stem = Path(original_filename).stem

            # 使用 PaddleOCR 原生方法保存所有页面
            for res in output:
                if return_json:
                    res.save_to_json(save_path=str(output_path))
                if return_markdown:
                    res.save_to_markdown(save_path=str(output_path))

            # 对于多页 PDF，PaddleOCR 会生成多个文件：
            # - {processing_stem}_0_res.json, {processing_stem}_1_res.json, ...
            # - {processing_stem}_0.md, {processing_stem}_1.md, ...
            # 我们需要：
            # 1. 重命名为使用原始文件名
            # 2. 合并多个文件为一个

            processing_stem = Path(processing_path).stem

            # 处理 JSON 文件
            if return_json:
                # 查找所有生成的 JSON 文件
                json_pattern = f"{processing_stem}_*_res.json"
                json_files = list(output_path.glob(json_pattern))

                if len(json_files) == 0:
                    # 单页情况，文件名是 {processing_stem}_res.json
                    original_json = output_path / f"{processing_stem}_res.json"
                    target_json = output_path / f"{filename_stem}_res.json"
                    if original_json.exists() and original_json != target_json:
                        shutil.move(str(original_json), str(target_json))
                        logger.debug(f"Renamed single-page JSON: {original_json.name} -> {target_json.name}")
                    output_files['json'] = str(target_json)
                else:
                    # 多页情况，需要重命名然后合并
                    for json_file in sorted(json_files):
                        # 提取页码：{processing_stem}_0_res.json -> 0
                        page_num = json_file.stem.split('_')[-2]
                        target_name = f"{filename_stem}_{page_num}_res.json"
                        target_path = output_path / target_name
                        if json_file != target_path:
                            shutil.move(str(json_file), str(target_path))
                            logger.debug(f"Renamed page {page_num} JSON: {json_file.name} -> {target_name}")

                    # 合并所有 JSON 文件
                    merged_json = merge_json_files(output_path, filename_stem)
                    if merged_json:
                        output_files['json'] = merged_json
                        logger.info(f"Merged JSON files into: {merged_json}")

            # 处理 Markdown 文件
            if return_markdown:
                # 查找所有生成的 Markdown 文件
                md_pattern = f"{processing_stem}_*.md"
                md_files = list(output_path.glob(md_pattern))

                if len(md_files) == 0:
                    # 单页情况，文件名是 {processing_stem}.md
                    original_md = output_path / f"{processing_stem}.md"
                    target_md = output_path / f"{filename_stem}.md"
                    if original_md.exists() and original_md != target_md:
                        shutil.move(str(original_md), str(target_md))
                        logger.debug(f"Renamed single-page Markdown: {original_md.name} -> {target_md.name}")
                    output_files['markdown'] = str(target_md)
                else:
                    # 多页情况，需要重命名然后合并
                    for md_file in sorted(md_files):
                        # 提取页码：{processing_stem}_0.md -> 0
                        page_num = md_file.stem.split('_')[-1]
                        target_name = f"{filename_stem}_{page_num}.md"
                        target_path = output_path / target_name
                        if md_file != target_path:
                            shutil.move(str(md_file), str(target_path))
                            logger.debug(f"Renamed page {page_num} Markdown: {md_file.name} -> {target_name}")

                    # 合并所有 Markdown 文件
                    merged_md = merge_markdown_files(output_path, filename_stem)
                    if merged_md:
                        output_files['markdown'] = merged_md
                        logger.info(f"Merged Markdown files into: {merged_md}")

        # 记录处理统计
        page_count = len(output)
        if needs_extraction:
            logger.info(f"Processed {page_count} pages from extracted PDF (range: {start_page_id}-{end_page_id})")
        else:
            logger.info(f"Processed {page_count} pages")

        return {
            'output_files': output_files
        }
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise
    finally:
        # 清理临时提取的 PDF 文件
        if temp_extracted_pdf and os.path.exists(temp_extracted_pdf):
            try:
                os.unlink(temp_extracted_pdf)
                logger.debug(f"Cleaned up temporary PDF: {temp_extracted_pdf}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary PDF: {e}")


async def process_ocr_async(
    image_path: str,
    original_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
    return_json: bool = True,
    return_markdown: bool = False,
    start_page_id: int = 0,
    end_page_id: int = 99999
) -> Dict[str, Any]:
    """
    异步 OCR 处理（将同步任务放到线程池）
    支持图片和 PDF 文件格式
    支持 PDF 页码范围过滤
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        thread_pool,
        lambda: process_ocr_sync(
            image_path,
            original_filename,
            output_dir,
            return_json,
            return_markdown,
            start_page_id,
            end_page_id
        )
    )


async def create_zip_response(output_dir: str, output_files: Dict[str, Dict[str, str]]) -> StreamingResponse:
    """
    创建 ZIP 格式的响应
    包含所有输出文件和 imgs 目录（可视化图片）
    目录结构：每个文件有自己的目录（使用文件名不带后缀）
    """
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # 添加输出文件（JSON, Markdown 等）- 按文件名创建目录
        for original_filename, files in output_files.items():
            # 使用文件名（不带后缀）作为目录名
            dir_name = Path(original_filename).stem

            for _, file_path in files.items():
                if os.path.exists(file_path):
                    # 添加目录层级：dir_name/filename
                    arcname = f"{dir_name}/{Path(file_path).name}"
                    zip_file.write(file_path, arcname)
                    logger.debug(f"Added to ZIP: {arcname}")

        # 添加 imgs 目录（保持目录结构，但放在对应文件的目录下）
        imgs_dir = Path(output_dir) / "imgs"
        if imgs_dir.exists() and imgs_dir.is_dir():
            logger.info(f"Adding imgs directory to ZIP: {imgs_dir}")
            # 对于多文件情况，imgs 可能包含多个文件的图片
            # 这里我们将 imgs 目录添加到第一个文件的目录下
            # 如果需要更精细的控制，需要追踪每个图片属于哪个文件
            if output_files:
                first_filename = next(iter(output_files.keys()))
                dir_name = Path(first_filename).stem

                for img_file in imgs_dir.rglob("*"):
                    if img_file.is_file():
                        # 添加到文件目录下：dir_name/imgs/image.png
                        rel_path = img_file.relative_to(Path(output_dir))
                        arcname = f"{dir_name}/{rel_path}"
                        zip_file.write(img_file, arcname)
                        logger.debug(f"Added image to ZIP: {arcname}")
        else:
            logger.debug(f"No imgs directory found at: {imgs_dir}")

    zip_buffer.seek(0)
    logger.info(f"Created ZIP file with {len(output_files)} file groups")
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=results.zip"}
    )


@app.post("/file_parse")
async def parse_files(
    files: List[UploadFile] = File(...),
    output_dir: str = Form(default="./output"),
    lang_list: str = Form(default='["ch"]'),
    # MinerU 兼容参数
    backend: str = Form(default="paddleocr"),  # MinerU uses "pipeline"
    parse_method: str = Form(default="auto"),
    formula_enable: bool = Form(default=True),
    table_enable: bool = Form(default=True),
    server_url: Optional[str] = Form(default=None),
    return_md: bool = Form(default=True),  # MinerU 参数名
    return_middle_json: bool = Form(default=False),  # MinerU 参数名
    return_model_output: bool = Form(default=False),
    return_content_list: bool = Form(default=False),
    return_images: bool = Form(default=False),
    response_format_zip: bool = Form(default=False),  # MinerU 默认 false
    start_page_id: int = Form(default=0),
    end_page_id: int = Form(default=99999)
):
    """
    文件解析端点 - 类似 MinerU 的 /file_parse
    支持批量上传图片或 PDF 文件进行 OCR 识别
    参数设计参考 MinerU API

    支持 PDF 页码范围过滤（start_page_id, end_page_id）
    未使用的参数保留以保持 API 兼容性，未来版本可能实现
    """
    start_time = time.time()
    tmp_paths = []

    # 为每个请求创建唯一的输出目录
    request_id = str(uuid.uuid4())
    actual_output_dir = os.path.join(output_dir, request_id)
    os.makedirs(actual_output_dir, exist_ok=True)
    logger.info(f"Created unique output directory: {actual_output_dir}")

    # 以下参数预留供未来版本使用
    _ = (lang_list, backend, parse_method, formula_enable,
         table_enable, server_url, return_model_output)

    # 如果请求 ZIP 格式，强制返回 JSON 和 Markdown（忽略相关参数）
    if response_format_zip:
        actual_return_json = True
        actual_return_markdown = True
        logger.info("ZIP format requested: forcing return_md=True and return_middle_json=True")
    else:
        # 使用 MinerU 参数名
        actual_return_json = return_middle_json
        actual_return_markdown = return_md

    try:
        # 处理所有上传的文件
        for file in files:
            filename = file.filename or "image.png"
            suffix = Path(filename).suffix

            # 保存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
                tmp_paths.append(tmp_path)

            logger.info(f"Processing file: {filename}, size: {len(content)} bytes")

            # 执行 OCR（传递原始文件名和页码范围参数）
            # 注意：我们只需要生成文件，不需要使用返回的 result 对象
            await process_ocr_async(
                tmp_path,
                original_filename=filename,  # 传递原始文件名
                output_dir=actual_output_dir,  # 使用唯一的输出目录
                return_json=actual_return_json,
                return_markdown=actual_return_markdown,
                start_page_id=start_page_id,
                end_page_id=end_page_id
            )

        processing_time = time.time() - start_time
        logger.info(f"Processed {len(files)} files in {processing_time:.2f}s")

        # 如果需要返回 ZIP 格式
        if response_format_zip:
            # 为 ZIP 格式重新组织输出文件
            zip_output_files = {}
            for file in files:
                fname = file.filename or "image.png"
                fname_stem = Path(fname).stem
                file_outputs = {}

                json_file = Path(actual_output_dir) / f"{fname_stem}_res.json"
                md_file = Path(actual_output_dir) / f"{fname_stem}.md"

                if json_file.exists():
                    file_outputs['json'] = str(json_file)
                if md_file.exists():
                    file_outputs['markdown'] = str(md_file)

                if file_outputs:
                    zip_output_files[fname] = file_outputs

            return await create_zip_response(actual_output_dir, zip_output_files)

        # 返回 MinerU 格式的 JSON
        results_dict = {}
        for file in files:
            filename = file.filename or "image.png"
            filename_stem = Path(filename).stem

            # 构建 MinerU 格式的文件结果
            file_result = build_file_result(
                output_dir=actual_output_dir,
                filename_stem=filename_stem,
                return_markdown=actual_return_markdown,
                return_json=actual_return_json,
                return_content_list=return_content_list,
                return_images=return_images
            )

            results_dict[filename_stem] = file_result

        return OCRResponse(
            backend="paddleocr",
            version="1.0.0",
            results=results_dict
        )

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        for tmp_path in tmp_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # 清理唯一的输出目录（包括所有生成的文件）
        if os.path.exists(actual_output_dir):
            try:
                shutil.rmtree(actual_output_dir)
                logger.info(f"Cleaned up output directory: {actual_output_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up output directory {actual_output_dir}: {e}")


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    output_dir: str = "./output"
):
    """
    简单的 OCR 端点（向后兼容，返回 MinerU 格式）
    上传图片或 PDF 文件进行 OCR 识别
    """
    start_time = time.time()
    tmp_path = None

    # 为每个请求创建唯一的输出目录
    request_id = str(uuid.uuid4())
    actual_output_dir = os.path.join(output_dir, request_id)
    os.makedirs(actual_output_dir, exist_ok=True)
    logger.info(f"Created unique output directory: {actual_output_dir}")

    try:
        # 异步保存文件
        filename = file.filename or "image.png"
        filename_stem = Path(filename).stem
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Processing image: {filename}, size: {len(content)} bytes")

        # 异步执行 OCR，生成文件
        await process_ocr_async(
            tmp_path,
            original_filename=filename,
            output_dir=actual_output_dir,
            return_json=True,
            return_markdown=True
        )

        processing_time = time.time() - start_time
        logger.info(f"OCR completed in {processing_time:.2f}s")

        # 构建 MinerU 格式的响应
        file_result = build_file_result(
            output_dir=actual_output_dir,
            filename_stem=filename_stem,
            return_markdown=True,
            return_json=True,
            return_content_list=False,
            return_images=True
        )

        return OCRResponse(
            backend="paddleocr",
            version="1.0.0",
            results={filename_stem: file_result}
        )

    except Exception as e:
        logger.error(f"Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 清理临时文件
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        # 清理唯一的输出目录
        if os.path.exists(actual_output_dir):
            try:
                shutil.rmtree(actual_output_dir)
                logger.info(f"Cleaned up output directory: {actual_output_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up output directory {actual_output_dir}: {e}")


@app.post("/ocr/batch")
async def ocr_batch(files: List[UploadFile] = File(...)):
    """
    批量 OCR 处理
    支持批量上传图片和 PDF 文件进行识别
    """
    start_time = time.time()

    try:
        # 并发处理多个文件
        tasks = []
        for file in files:
            # 重置文件指针
            await file.seek(0)
            tasks.append(ocr_image(file))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        success_count = sum(1 for r in results if isinstance(r, OCRResponse))

        # 序列化结果
        processed_results = []
        for r in results:
            if isinstance(r, OCRResponse):
                processed_results.append(r.model_dump())
            elif isinstance(r, Exception):
                processed_results.append({"error": str(r), "success": False})
            else:
                processed_results.append({"error": "Unknown error", "success": False})

        return JSONResponse(content={
            "success": True,
            "total": len(files),
            "success_count": success_count,
            "processing_time": time.time() - start_time,
            "results": processed_results
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        _ = PaddleOCRSingleton.get_pipeline()
        return {
            "status": "healthy",
            "vllm_server": VLLM_SERVER_URL,
            "thread_workers": THREAD_WORKERS
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
