#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分块构建模块

此模块提供了将论文块分块为适合检索的片段的功能。

功能说明：
- 使用递归字符分割器将长文本分割为较小的分块
- 保持论文元数据与分块的关联
- 支持自定义分块大小和重叠

主要组件：
- ChunkBuilder: 分块构建器，负责将论文块转换为检索分块
"""

from __future__ import annotations

from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.schemas.models import PaperBlock, PaperChunk, PaperMetadata


class ChunkBuilder:
    """
    分块构建器
    
    负责将论文块分块为适合检索的片段，同时保持论文元数据与分块的关联
    
    功能：
    - 使用递归字符分割器将长文本分割为较小的分块
    - 为每个分块生成唯一的ID
    - 保留原始块的元数据信息
    - 支持自定义分块大小和重叠
    """

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
        """
        初始化分块构建器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        # 创建递归字符分割器
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def build_chunks(
        self,
        blocks: Iterable[PaperBlock],
        paper_metadata: PaperMetadata,
    ) -> list[PaperChunk]:
        """
        构建检索分块
        
        将论文块列表转换为检索分块列表，保留论文元数据
        
        Args:
            blocks: 论文块可迭代对象
            paper_metadata: 论文元数据
        
        Returns:
            list[PaperChunk]: 检索分块列表
        """
        chunks: list[PaperChunk] = []
        # 遍历每个论文块
        for block in blocks:
            # 获取块文本
            text = block.get("text", "")
            # 跳过空文本
            if not text:
                continue
            # 初始化分块部分
            parts = [text]
            # 如果文本超过分块大小，进行分割
            if len(text) > self.chunk_size:
                parts = self.splitter.split_text(text)

            # 为每个分块部分创建分块对象
            for part_index, part in enumerate(parts):
                # 生成分块ID
                chunk_id = f"{block['block_id']}-c{part_index:03d}"
                # 构建分块元数据
                metadata = {
                    **paper_metadata,
                    "page": block.get("page"),
                    "section_path": block.get("section_path", []),
                    "block_type": block.get("block_type"),
                    "table_id": block.get("table_id", ""),
                    "figure_id": block.get("figure_id", ""),
                }
                # 添加分块到列表
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "text": part,
                        "block_id": block["block_id"],
                        "block_type": block.get("block_type", "unknown"),
                        "page": block.get("page", 0),
                        "section_path": block.get("section_path", []),
                        "metadata": metadata,
                    }
                )
        return chunks
