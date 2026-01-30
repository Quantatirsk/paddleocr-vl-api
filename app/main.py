"""
优化的 PaddleOCR API - 支持图片和 PDF 文件的高并发 OCR 识别
参考 MinerU API 设计，提供灵活的配置选项和多种返回格式
"""
from fastapi_offline import FastAPIOffline  # 用于创建实例（支持离线文档）
from fastapi import FastAPI, File, UploadFile, HTTPException, Form  # FastAPI 用于类型注解
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Optional
import time
from contextlib import asynccontextmanager
import shutil
import uuid

from .config import VLLM_SERVER_URL, THREAD_WORKERS, API_VERSION, BACKEND_NAME
from .models import OCRResponse
from .ocr import PaddleOCRSingleton, process_ocr_async
from .ocr.processor import set_thread_pool
from .utils import build_file_result, create_zip_response

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
    set_thread_pool(thread_pool)
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


app = FastAPIOffline(title="PaddleOCR Optimized API", lifespan=lifespan)


@app.post("/file_parse")
async def parse_files(
    files: List[UploadFile] = File(...),
    output_dir: str = Form(default="./output"),
    lang_list: str = Form(default='["ch"]'),
    # MinerU 兼容参数
    backend: str = Form(default="paddleocr"),
    parse_method: str = Form(default="auto"),
    formula_enable: bool = Form(default=True),
    table_enable: bool = Form(default=True),
    server_url: Optional[str] = Form(default=None),
    return_md: bool = Form(default=True),
    return_middle_json: bool = Form(default=False),
    return_model_output: bool = Form(default=False),
    return_content_list: bool = Form(default=False),
    return_images: bool = Form(default=False),
    response_format_zip: bool = Form(default=False),
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
            await process_ocr_async(
                tmp_path,
                original_filename=filename,
                output_dir=actual_output_dir,
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
            backend=BACKEND_NAME,
            version=API_VERSION,
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
            backend=BACKEND_NAME,
            version=API_VERSION,
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
