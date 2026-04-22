#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
索引图构建模块

此模块定义了论文索引流程的工作流图，使用 LangGraph 构建状态图来管理整个索引过程。

流程说明：
1. parse_pdf: 解析PDF文件，提取原始页面数据
2. extract_blocks: 从原始页面中提取论文块
3. clean_blocks: 清理论文块，去除噪声
4. build_chunks: 将论文块分块为适合检索的片段
5. write_chroma: 将分块写入向量数据库
6. save_reports: 保存处理报告和中间结果

主要组件：
- build_index_graph: 构建索引状态图
- 各个节点函数: 执行索引流程的各个步骤
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from rag_pipeline.cleaners.paper_cleaner import PaperCleaner
from rag_pipeline.config import PipelinePaths
from rag_pipeline.embeddings import create_embeddings
from rag_pipeline.loaders.pdf_loader import PDFLoader
from rag_pipeline.parsers.paper_parser import PaperParser
from rag_pipeline.schemas.models import IndexingState
from rag_pipeline.transformers.chunk_builder import ChunkBuilder
from rag_pipeline.utils.io import write_json, write_jsonl
from rag_pipeline.vectorstores.chroma_store import ChromaWriter


def build_index_graph() -> StateGraph:
    """
    构建索引状态图
    
    创建并配置论文索引流程的状态图，定义节点和边
    
    Returns:
        StateGraph: 配置好的索引状态图
    """
    # 创建状态图
    graph = StateGraph(IndexingState)
    # 添加各个处理节点
    graph.add_node("parse_pdf", parse_pdf_node)
    graph.add_node("extract_blocks", extract_blocks_node)
    graph.add_node("clean_blocks", clean_blocks_node)
    graph.add_node("build_chunks", build_chunks_node)
    graph.add_node("write_chroma", write_chroma_node)
    graph.add_node("save_reports", save_reports_node)

    # 定义节点之间的边（执行顺序）
    graph.add_edge(START, "parse_pdf")
    graph.add_edge("parse_pdf", "extract_blocks")
    graph.add_edge("extract_blocks", "clean_blocks")
    graph.add_edge("clean_blocks", "build_chunks")
    graph.add_edge("build_chunks", "write_chroma")
    graph.add_edge("write_chroma", "save_reports")
    graph.add_edge("save_reports", END)
    return graph


def parse_pdf_node(state: IndexingState) -> IndexingState:
    """
    解析PDF节点
    
    使用 PDFLoader 加载PDF文件并提取原始页面数据
    
    Args:
        state: 索引状态
    
    Returns:
        IndexingState: 更新后的索引状态，包含原始页面数据和解析器后端信息
    """
    # 创建 PDF 加载器
    loader = PDFLoader()
    # 获取当前处理的PDF文件路径
    current_pdf = state["current_pdf"]
    # 加载PDF文件
    raw_pages = loader.load(current_pdf)
    # 获取解析器后端信息
    parser_backend = raw_pages[0].get("parser_backend", "") if raw_pages else ""
    return {
        **state,
        "raw_pages": raw_pages,
        "parser_backend": parser_backend,
    }


def extract_blocks_node(state: IndexingState) -> IndexingState:
    """
    提取块节点，使用 PaperParser 从原始页面中提取论文块和元数据
    
    Args:
        state: 索引状态
    
    Returns:
        IndexingState: 更新后的索引状态，包含论文元数据和块列表
    """
    # 创建论文解析器
    parser = PaperParser()
    # 解析文档ID
    doc_id = _resolve_doc_id(state["current_pdf"])
    # 提取论文元数据
    metadata = parser.extract_metadata(state.get("raw_pages", []))
    metadata["doc_id"] = doc_id
    metadata["source_file"] = state["current_pdf"]
    # 解析论文块
    blocks = parser.parse_blocks(state.get("raw_pages", []), doc_id=doc_id)
    return {
        **state,
        "paper_metadata": metadata,
        "blocks": blocks,
    }


def clean_blocks_node(state: IndexingState) -> IndexingState:
    """
    清理块节点
    
    使用 PaperCleaner 清理论文块，去除噪声和无关内容
    
    Args:
        state: 索引状态
    
    Returns:
        IndexingState: 更新后的索引状态，包含清理后的块列表
    """
    # 创建论文清理器
    cleaner = PaperCleaner()
    # 清理论文块
    cleaned_blocks = cleaner.clean_blocks(state.get("blocks", []))
    return {
        **state,
        "cleaned_blocks": cleaned_blocks,
    }


def build_chunks_node(state: IndexingState) -> IndexingState:
    """
    构建分块节点
    
    使用 ChunkBuilder 将清理后的论文块分块为适合检索的片段
    
    Args:
        state: 索引状态
    
    Returns:
        IndexingState: 更新后的索引状态，包含分块列表
    """
    # 创建分块构建器
    builder = ChunkBuilder()
    # 构建分块
    chunks = builder.build_chunks(
        blocks=state.get("cleaned_blocks", []),
        paper_metadata=state.get("paper_metadata", {}),
    )
    return {
        **state,
        "chunks": chunks,
    }


def write_chroma_node(state: IndexingState) -> IndexingState:
    """
    写入Chroma节点
    
    将分块写入 Chroma 向量数据库，如果跳过向量存储则直接返回
    
    Args:
        state: 索引状态
    
    Returns:
        IndexingState: 更新后的索引状态，包含嵌入数量和向量存储路径
    """
    # 如果跳过向量存储，直接返回
    if state.get("skip_vectorstore", False):
        return {
            **state,
            "embedded_count": 0,
        }

    # 解析工作空间路径
    workspace = Path(state["input_dir"]).resolve().parent
    # 创建管道路径对象
    paths = PipelinePaths(workspace)
    # 确保目录存在
    paths.ensure()
    # 获取集合名称
    collection_name = state.get("collection_name", "paper_chunks")
    # 创建嵌入模型
    embeddings = create_embeddings(state.get("embedding_model"))
    # 创建 Chroma 写入器
    writer = ChromaWriter(
        persist_directory=paths.chroma,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    # 添加分块到向量存储
    embedded_count = writer.add_chunks(state.get("chunks", []))
    return {
        **state,
        "embedded_count": embedded_count,
        "vectorstore_path": str(paths.chroma),
        "collection_name": collection_name,
    }


def save_reports_node(state: IndexingState) -> IndexingState:
    """
    保存报告节点
    
    保存处理报告和中间结果到文件系统
    
    Args:
        state: 索引状态
    
    Returns:
        IndexingState: 更新后的索引状态，包含输出文件路径
    """
    # 获取当前PDF文件路径
    current_pdf = Path(state["current_pdf"])
    # 解析工作空间路径
    workspace = Path(state["input_dir"]).resolve().parent
    # 创建管道路径对象
    paths = PipelinePaths(workspace)
    # 确保目录存在
    paths.ensure()

    # 生成输出文件名
    stem = current_pdf.stem
    parsed_output_path = paths.parsed_json / f"{stem}.json"
    cleaned_output_path = paths.cleaned_json / f"{stem}.json"
    chunk_output_path = paths.chunk_jsonl / f"{stem}.jsonl"
    report_path = paths.reports / f"{stem}.json"

    # 保存解析结果
    write_json(
        parsed_output_path,
        {
            "parser_backend": state.get("parser_backend", ""),
            "paper_metadata": state.get("paper_metadata", {}),
            "blocks": state.get("blocks", []),
        },
    )
    # 保存清理后的块
    write_json(cleaned_output_path, state.get("cleaned_blocks", []))
    # 保存分块结果
    write_jsonl(chunk_output_path, state.get("chunks", []))
    # 保存处理报告
    write_json(
        report_path,
        {
            "pdf": state["current_pdf"],
            "block_count": len(state.get("blocks", [])),
            "cleaned_block_count": len(state.get("cleaned_blocks", [])),
            "chunk_count": len(state.get("chunks", [])),
            "embedded_count": state.get("embedded_count", 0),
            "vectorstore_path": state.get("vectorstore_path", ""),
            "collection_name": state.get("collection_name", ""),
            "parser_backend": state.get("parser_backend", ""),
            "status": "ok",
        },
    )

    return {
        **state,
        "parsed_output_path": str(parsed_output_path),
        "cleaned_output_path": str(cleaned_output_path),
        "chunk_output_path": str(chunk_output_path),
        "report_path": str(report_path),
    }


def _resolve_doc_id(current_pdf: str) -> str:
    """

    根据PDF文件路径生成唯一的文档ID
    
    Args:
        current_pdf: PDF文件路径
    
    Returns:
        str: 文档ID，格式为 {文件名}-{UUID前8位}
    """
    return f"{Path(current_pdf).stem}-{uuid4().hex[:8]}"
