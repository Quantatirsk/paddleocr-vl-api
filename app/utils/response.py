"""
响应构建工具 - 生成 MinerU 格式的响应和 ZIP 打包
"""
import os
import io
import json
import base64
import zipfile
import logging
from pathlib import Path
from typing import Dict
from fastapi.responses import StreamingResponse

from ..models import FileResult

logger = logging.getLogger(__name__)


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
