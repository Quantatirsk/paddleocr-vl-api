"""
OCR 核心处理逻辑
"""
import os
import logging
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .singleton import PaddleOCRSingleton
from .pdf import extract_pdf_pages
from ..utils.file_merge import merge_json_files, merge_markdown_files

logger = logging.getLogger(__name__)

# 全局线程池（由 app.py 的 lifespan 管理）
thread_pool: Optional[ThreadPoolExecutor] = None


def set_thread_pool(pool: ThreadPoolExecutor):
    """设置全局线程池"""
    global thread_pool
    thread_pool = pool


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
