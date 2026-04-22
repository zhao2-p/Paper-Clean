#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文解析模块

此模块定义了 PaperParser 类，用于将原始页面文本转换为带有章节上下文的论文感知块。
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from rag_pipeline.schemas.models import PaperBlock, PaperMetadata


# 正则表达式常量
SECTION_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVX]+)\s+[A-Z].+")  # 章节正则
SIMPLE_SECTION_LINE_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVX]+)\s+[A-Z][A-Za-z0-9\s\-(),/:]{1,80}$")  # 简单章节行正则
MIXED_SECTION_RE = re.compile(r"^\d+(?:\.\d+)\s+[A-Z][^.]{1,80}$")  # 混合章节正则
CHINESE_SECTION_RE = re.compile(r"^(第[一二三四五六七八九十百]+[章节])[\s .．、:：-]*.+")  # 中文章节正则
FIGURE_RE = re.compile(r"^(figure|fig\.)\s*(\d+)", re.IGNORECASE)  # 图标题正则
TABLE_RE = re.compile(r"^table\s*(\d+)", re.IGNORECASE)  # 表标题正则
CHINESE_FIGURE_RE = re.compile(r"^(图|图表)\s*(\d+)[.．、:：\s-]*")  # 中文图标题正则
CHINESE_TABLE_RE = re.compile(r"^表\s*(\d+)[.．、:：\s-]*")  # 中文表标题正则
AUTHOR_SPLIT_RE = re.compile(r",| and ")  # 作者分隔正则
KEYWORDS_RE = re.compile(r"^(关键词|Key words?)[:：]", re.IGNORECASE)  # 关键词正则
ABSTRACT_RE = re.compile(r"^(摘要|abstract)[:：]?\s*", re.IGNORECASE)  # 摘要正则
MARKDOWN_HEADER_RE = re.compile(r"^#{1,6}\s+")  # Markdown标题正则
MARKDOWN_STRONG_RE = re.compile(r"\*\*(.*?)\*\*")  # Markdown加粗正则
MARKDOWN_EM_RE = re.compile(r"_(.*?)_")  # Markdown斜体正则
TOC_HINT_RE = re.compile(r"(目录|contents)", re.IGNORECASE)  # 目录提示正则
DOT_LEADER_RE = re.compile(r"\.{5,}")  # 点领导者正则
AUTHOR_HINT_RE = re.compile(r"(作者|专业班级|指导教师|学院|学校)", re.IGNORECASE)  # 作者提示正则
AFFILIATION_HINT_RE = re.compile(
    r"(school of|college of|university|laboratory|institute|department of|@|china|australia)",
    re.IGNORECASE,
)  # 机构提示正则
PAGE_ARTIFACT_RE = re.compile(r"^\d{3,4}$")  # 页码artifact正则
ALGO_STEP_RE = re.compile(
    r"^\d+\s+(?:for|foreach|while|if|return|end|compute|create|initialize|select|update|deliver)\b",
    re.IGNORECASE,
)  # 算法步骤正则


class PaperParser:
    """
    论文解析器类
    
    将原始页面文本转换为带有章节上下文的论文感知块。
    """

    def extract_metadata(self, pages: list[dict]) -> PaperMetadata:
        """
        提取论文元数据
        
        Args:
            pages: 页面数据列表
        
        Returns:
            PaperMetadata: 论文元数据
        """
        first_page = pages[0] if pages else {}
        first_page_blocks = first_page.get("blocks", [])
        # 提取标题
        title = self._extract_title(first_page_blocks, first_page.get("font_size_median", 0.0))
        authors: list[str] = []
        if title:
            # 找到标题所在的索引
            title_index = next(
                (idx for idx, block in enumerate(first_page_blocks) if self._normalize_block_text(block.get("text", "")) == title),
                -1,
            )
            # 尝试从标题后提取作者
            for block in first_page_blocks[title_index + 1 : title_index + 5]:
                line = self._normalize_block_text(block.get("text", ""))
                if "@" in line or len(line.split()) > 20:
                    continue
                if AUTHOR_HINT_RE.search(line) or AFFILIATION_HINT_RE.search(line):
                    continue
                # 分割作者
                parts = [part.strip() for part in AUTHOR_SPLIT_RE.split(line) if 1 < len(part.strip()) < 30]
                if 1 < len(parts) <= 8:
                    authors = parts
                    break
        return {
            "paper_title": title,  # 论文标题
            "authors": authors,  # 作者列表
        }

    def parse_blocks(self, pages: list[dict], doc_id: str) -> list[PaperBlock]:
        """
        解析页面块
        
        Args:
            pages: 页面数据列表
            doc_id: 文档ID
        
        Returns:
            list[PaperBlock]: 论文块列表
        """
        blocks: list[PaperBlock] = []  # 存储解析后的块
        order = 0  # 块顺序
        current_section: list[str] = []  # 当前章节路径
        in_abstract = False  # 是否在摘要中

        for page in pages:
            # 跳过目录页
            if self._is_toc_page(page):
                continue

            # 遍历页面块
            for raw_block in self._iter_page_blocks(page):
                # 分割块
                for segment in self._split_block(raw_block):
                    # 标准化文本
                    raw_text = self._normalize_block_text(segment["text"])
                    # 跳过封面噪声
                    if self._looks_like_cover_noise(raw_text, page_number=page["page"]):
                        continue

                    # 推断块类型
                    block_type = self._infer_block_type(raw_text, in_abstract=in_abstract)
                    # 验证章节标题
                    if block_type == "section_header" and not self._is_valid_section_header(raw_text):
                        block_type = "paragraph"

                    # 更新章节路径和摘要状态
                    if block_type == "section_header":
                        in_abstract = False
                        current_section = self._update_section_path(current_section, raw_text)
                    elif block_type == "abstract":
                        in_abstract = True
                        current_section = ["Abstract"]
                    elif in_abstract and self._looks_like_new_section_start(raw_text):
                        in_abstract = False
                        block_type = self._infer_block_type(raw_text, in_abstract=False)
                        if block_type == "section_header" and self._is_valid_section_header(raw_text):
                            current_section = self._update_section_path(current_section, raw_text)

                    # 创建块
                    block: PaperBlock = {
                        "block_id": f"{doc_id}-b{order:05d}",  # 块ID
                        "block_type": block_type,  # 块类型
                        "text": raw_text,  # 块文本
                        "page": page["page"],  # 页码
                        "section_path": list(current_section),  # 章节路径
                        "order": order,  # 顺序
                        "metadata": {
                            "line_count": len(segment.get("lines", [])),  # 行数
                            "char_count": len(raw_text),  # 字符数
                            "bbox": segment.get("bbox"),  # 边界框
                            "font_size_avg": segment.get("font_size_avg", 0.0),  # 平均字体大小
                            "font_size_max": segment.get("font_size_max", 0.0),  # 最大字体大小
                            "column_hint": segment.get("column_hint", 0),  # 列提示
                        },
                    }
                    # 添加图/表ID
                    if block_type == "figure_caption":
                        block["figure_id"] = self._extract_number(raw_text)
                    if block_type == "table":
                        block["table_id"] = self._extract_number(raw_text)
                    blocks.append(block)
                    order += 1
        return blocks

    def _iter_lines(self, pages: Iterable[dict]) -> Iterable[str]:
        """
        遍历页面行
        
        Args:
            pages: 页面数据迭代器
        
        Yields:
            str: 非空行文本
        """
        for page in pages:
            for line in page.get("text", "").splitlines():
                text = line.strip()
                if text:
                    yield text

    def _iter_page_blocks(self, page: dict) -> Iterable[dict[str, Any]]:
        """
        遍历页面块
        
        Args:
            page: 页面数据
        
        Yields:
            dict[str, Any]: 非空块
        """
        for block in page.get("blocks", []):
            text = block.get("text", "").strip()
            if not text:
                continue
            yield {**block, "text": text}

    def _split_block(self, block: dict[str, Any]) -> list[dict[str, Any]]:
        """
        分割块
        
        Args:
            block: 块数据
        
        Returns:
            list[dict[str, Any]]: 分割后的块列表
        """
        lines = [self._normalize_block_text(line) for line in block.get("lines", []) if self._normalize_block_text(line)]
        if not lines:
            return [block]

        results: list[dict[str, Any]] = []
        buffer: list[str] = []

        def flush_buffer() -> None:
            """刷新缓冲区"""
            if not buffer:
                return
            text = " ".join(buffer).strip()
            if text:
                results.append({**block, "text": text, "lines": list(buffer)})
            buffer.clear()

        for line in lines:
            if self._looks_like_new_section_start(line) and self._is_valid_section_header(line):
                flush_buffer()
                results.append({**block, "text": line, "lines": [line]})
                continue
            buffer.append(line)

        flush_buffer()
        return results or [block]

    def _infer_block_type(self, text: str, in_abstract: bool = False) -> str:
        """
        推断块类型
        
        Args:
            text: 块文本
            in_abstract: 是否在摘要中
        
        Returns:
            str: 块类型
        """
        lowered = text.lower()
        if ABSTRACT_RE.match(text):
            return "abstract"  # 摘要
        if self._looks_like_new_section_start(text):
            return "section_header"  # 章节标题
        if FIGURE_RE.match(text) or CHINESE_FIGURE_RE.match(text):
            return "figure_caption"  # 图标题
        if TABLE_RE.match(text) or CHINESE_TABLE_RE.match(text):
            return "table"  # 表标题
        if lowered.startswith("references"):
            return "references"  # 参考文献
        if lowered.startswith("appendix"):
            return "appendix"  # 附录
        if KEYWORDS_RE.match(text):
            return "paragraph"  # 段落
        if in_abstract:
            return "abstract"  # 摘要
        return "paragraph"  # 段落

    def _looks_like_new_section_start(self, text: str) -> bool:
        """
        判断是否像新章节的开始
        
        Args:
            text: 文本
        
        Returns:
            bool: 是否像新章节的开始
        """
        return bool(
            SIMPLE_SECTION_LINE_RE.match(text)
            or MIXED_SECTION_RE.match(text)
            or CHINESE_SECTION_RE.match(text)
        )

    def _extract_number(self, text: str) -> str:
        """
        提取数字
        
        Args:
            text: 文本
        
        Returns:
            str: 提取的数字
        """
        match = re.search(r"(\d+)", text)
        return match.group(1) if match else ""

    def _update_section_path(self, current_section: list[str], header: str) -> list[str]:
        """
        更新章节路径
        
        Args:
            current_section: 当前章节路径
            header: 章节标题
        
        Returns:
            list[str]: 更新后的章节路径
        """
        numeric_match = re.match(r"^(\d+(?:\.\d+)*)", header)
        chinese_match = re.match(r"^(第[一二三四五六七八九十百]+[章节])", header)
        if chinese_match:
            return [header]  # 中文章节直接返回
        if not numeric_match:
            return [header]  # 非数字章节直接返回
        # 计算章节级别
        level = numeric_match.group(1).count(".") + 1
        # 截断并添加新章节
        next_path = current_section[: max(level - 1, 0)]
        next_path.append(header)
        return next_path

    def _extract_title(self, blocks: list[dict[str, Any]], median_font_size: float) -> str:
        """
        提取标题
        
        Args:
            blocks: 块列表
            median_font_size: 字体大小中位数
        
        Returns:
            str: 标题
        """
        candidates: list[tuple[float, str]] = []
        for block in blocks[:8]:
            text = self._normalize_block_text(block.get("text", ""))
            if len(text) < 6 or AUTHOR_HINT_RE.search(text) or ABSTRACT_RE.match(text):
                continue
            if AFFILIATION_HINT_RE.search(text) and len(text) > 80:
                continue
            size = block.get("font_size_max", 0.0)
            # 选择字体较大的作为标题候选
            if size >= max(median_font_size * 1.25, median_font_size + 1.5):
                candidates.append((size, text))
        if candidates:
            # 按字体大小和长度排序
            candidates.sort(key=lambda item: (-item[0], len(item[1])))
            return candidates[0][1]
        # 如果没有找到候选，尝试返回第一个较长的文本
        for block in blocks[:8]:
            text = self._normalize_block_text(block.get("text", ""))
            if len(text) > 12 and not AUTHOR_HINT_RE.search(text) and not ABSTRACT_RE.match(text):
                return text
        return ""

    def _is_toc_page(self, page: dict[str, Any]) -> bool:
        """
        判断是否为目录页
        
        Args:
            page: 页面数据
        
        Returns:
            bool: 是否为目录页
        """
        text = page.get("text", "")
        if not TOC_HINT_RE.search(text):
            return False
        # 计算包含点领导者的行数
        dot_lines = sum(1 for line in text.splitlines() if DOT_LEADER_RE.search(line))
        return dot_lines >= 3

    def _looks_like_cover_noise(self, text: str, page_number: int) -> bool:
        """
        判断是否为封面噪声
        
        Args:
            text: 文本
            page_number: 页码
        
        Returns:
            bool: 是否为封面噪声
        """
        if len(text) < 3:
            return True
        if page_number == 1 and ("@" in text or AFFILIATION_HINT_RE.search(text)):
            return True
        return AUTHOR_HINT_RE.search(text) is not None and len(text) < 100

    def _is_valid_section_header(self, text: str) -> bool:
        """
        判断是否为有效的章节标题
        
        Args:
            text: 文本
        
        Returns:
            bool: 是否为有效的章节标题
        """
        if PAGE_ARTIFACT_RE.match(text):
            return False
        if ALGO_STEP_RE.match(text):
            return False
        if re.match(r"^\d+\s", text) and not self._looks_like_new_section_start(text):
            return False
        if AFFILIATION_HINT_RE.search(text):
            return False
        if len(text) > 120:
            return False
        if len(text.split()) > 16:
            return False
        return True

    def _normalize_block_text(self, text: str) -> str:
        """
        标准化块文本
        
        Args:
            text: 原始文本
        
        Returns:
            str: 标准化后的文本
        """
        normalized = text.replace("\n", " ").strip()
        normalized = MARKDOWN_HEADER_RE.sub("", normalized).strip()
        normalized = MARKDOWN_STRONG_RE.sub(r"\1", normalized)
        normalized = MARKDOWN_EM_RE.sub(r"\1", normalized)
        normalized = normalized.replace("`", "")
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()
