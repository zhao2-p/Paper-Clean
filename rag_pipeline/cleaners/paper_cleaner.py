#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文清理模块

此模块定义了 PaperCleaner 类，用于在分块前对论文内容进行轻量级的清理，
移除噪声内容，如页码、目录、 affiliations等。
"""

from __future__ import annotations

import re

from rag_pipeline.schemas.models import PaperBlock


# 正则表达式常量
PAGE_NUMBER_RE = re.compile(r"^\d+$")  # 匹配页码（纯数字）
PAGE_ARTIFACT_RE = re.compile(r"^\d{3,4}$")  # 匹配页码 artifacts（3-4位数字）
TOC_RE = re.compile(r"(目录|contents)", re.IGNORECASE)  # 匹配目录标记
DOT_LEADER_RE = re.compile(r"\.{5,}")  # 匹配目录中的点领导者（连续5个以上的点）
AFFILIATION_HINT_RE = re.compile(
    r"(school of|college of|university|laboratory|institute|department of|@)",
    re.IGNORECASE,
)  # 匹配机构/ affiliation 提示词
ALGO_STEP_RE = re.compile(
    r"^\d+\s+(?:for|foreach|while|if|return|end|compute|create|initialize|select|update|deliver)\b",
    re.IGNORECASE,
)  # 匹配算法步骤标记
MARKDOWN_ARTIFACT_RE = re.compile(r"^[*_#`\-\s]+$")  # 匹配 markdown  artifacts


class PaperCleaner:
    """
    论文清理类
    
    在分块前对论文内容应用轻量级的论文特定清理。
    """

    def clean_blocks(self, blocks: list[PaperBlock]) -> list[PaperBlock]:
        """
        清理论文块列表
        
        Args:
            blocks: 论文块列表
        
        Returns:
            list[PaperBlock]: 清理后的论文块列表
        """
        cleaned: list[PaperBlock] = []  # 存储清理后的块
        seen_text: set[tuple[int, str, str]] = set()  # 用于去重的集合
        
        for block in blocks:
            # 标准化文本
            text = self._normalize_text(block.get("text", ""))
            
            # 过滤空文本或页码
            if not text or PAGE_NUMBER_RE.match(text) or PAGE_ARTIFACT_RE.match(text):
                continue
            
            # 过滤 markdown artifacts
            if MARKDOWN_ARTIFACT_RE.match(text):
                continue
            
            # 过滤目录噪声
            if self._is_toc_noise(text):
                continue
            
            # 过滤机构/affiliation 噪声
            if self._is_affiliation_noise(text, block.get("page", 0)):
                continue
            
            # 过滤算法噪声
            if self._is_algorithm_noise(text, block.get("block_type", "")):
                continue
            
            # 过滤可能的运行标题
            if self._is_probable_running_header(text, block.get("block_type", "")):
                continue

            # 去重
            page = block.get("page", 0)
            dedupe_key = (page, block.get("block_type", ""), text)
            if dedupe_key in seen_text:
                continue
            seen_text.add(dedupe_key)

            # 更新文本并添加到清理列表
            updated = dict(block)
            updated["text"] = text
            cleaned.append(updated)
        
        return cleaned

    def _normalize_text(self, text: str) -> str:
        """
        标准化文本
        
        Args:
            text: 原始文本
        
        Returns:
            str: 标准化后的文本
        """
        # 合并连字符连接的单词
        merged = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
        # 合并多个空格为单个空格
        merged = re.sub(r"\s+", " ", merged)
        # 去除首尾空格
        return merged.strip()

    def _is_probable_running_header(self, text: str, block_type: str) -> bool:
        """
        判断是否为可能的运行标题
        
        Args:
            text: 文本内容
            block_type: 块类型
        
        Returns:
            bool: 是否为运行标题
        """
        if block_type not in {"paragraph", "unknown"}:
            return False
        if len(text) > 140:
            return False
        lower = text.lower()
        # 检查是否以 "proceedings of" 开头，以 "arxiv:" 开头，或以 "conference" 结尾
        return lower.startswith("proceedings of") or lower.startswith("arxiv:") or lower.endswith("conference")

    def _is_toc_noise(self, text: str) -> bool:
        """
        判断是否为目录噪声
        
        Args:
            text: 文本内容
        
        Returns:
            bool: 是否为目录噪声
        """
        if TOC_RE.search(text):
            return True
        # 检查是否包含点领导者且长度小于2000
        return DOT_LEADER_RE.search(text) is not None and len(text) < 2000

    def _is_affiliation_noise(self, text: str, page: int) -> bool:
        """
        判断是否为机构/affiliation 噪声
        
        Args:
            text: 文本内容
            page: 页码
        
        Returns:
            bool: 是否为机构/affiliation 噪声
        """
        if page > 1:
            return False
        if "@" in text:
            return True
        # 检查是否包含机构提示词且长度小于600
        return AFFILIATION_HINT_RE.search(text) is not None and len(text) < 600

    def _is_algorithm_noise(self, text: str, block_type: str) -> bool:
        """
        判断是否为算法噪声
        
        Args:
            text: 文本内容
            block_type: 块类型
        
        Returns:
            bool: 是否为算法噪声
        """
        if block_type != "section_header":
            return False
        # 检查是否匹配算法步骤模式
        return ALGO_STEP_RE.match(text) is not None
