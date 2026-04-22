#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型模块

此模块定义了项目中使用的各种数据类型和结构。
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


BlockType = Literal[
    "title",  # 标题
    "abstract",  # 摘要
    "section_header",  # 章节标题
    "paragraph",  # 段落
    "table",  # 表格
    "figure_caption",  # 图表标题
    "references",  # 参考文献
    "appendix",  # 附录
    "unknown",  # 未知类型
]


class PaperMetadata(TypedDict, total=False):
    """
    论文元数据类型
    
    包含论文的基本信息
    """
    doc_id: str  # 文档ID
    source_file: str  # 源文件路径
    paper_title: str  # 论文标题
    authors: list[str]  # 作者列表
    year: int  # 年份


class PaperBlock(TypedDict, total=False):
    """
    论文块类型
    
    表示论文中的一个内容块
    """
    block_id: str  # 块ID
    block_type: BlockType  # 块类型
    text: str  # 块文本
    page: int  # 页码
    section_path: list[str]  # 章节路径
    table_id: str  # 表格ID
    figure_id: str  # 图表ID
    order: int  # 顺序
    metadata: dict[str, Any]  # 元数据


class PaperChunk(TypedDict, total=False):
    """
    论文分块类型
    
    表示论文分块后的一个片段
    """
    chunk_id: str  # 分块ID
    text: str  # 分块文本
    block_id: str  # 原始块ID
    block_type: BlockType  # 块类型
    page: int  # 页码
    section_path: list[str]  # 章节路径
    metadata: dict[str, Any]  # 元数据


class IndexingState(TypedDict, total=False):
    """
    索引状态类型
    
    表示论文索引过程中的状态信息
    """
    input_dir: str  # 输入目录
    current_pdf: str  # 当前处理的PDF文件
    pdf_paths: list[str]  # PDF文件路径列表
    parser_backend: str  # 解析器后端
    paper_metadata: PaperMetadata  # 论文元数据
    raw_pages: list[dict[str, Any]]  # 原始页面数据
    blocks: list[PaperBlock]  # 解析后的块
    cleaned_blocks: list[PaperBlock]  # 清理后的块
    chunks: list[PaperChunk]  # 分块后的片段
    parsed_output_path: str  # 解析输出路径
    cleaned_output_path: str  # 清理输出路径
    chunk_output_path: str  # 分块输出路径
    vectorstore_path: str  # 向量存储路径
    collection_name: str  # 集合名称
    embedding_model: str  # 嵌入模型
    skip_vectorstore: bool  # 是否跳过向量存储
    embedded_count: int  # 嵌入数量
    report_path: str  # 报告路径
    errors: list[str]  # 错误列表
