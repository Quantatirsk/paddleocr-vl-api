"""
文件合并工具 - 处理多页 PDF 的 JSON 和 Markdown 合并
"""
import json
import shutil
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
