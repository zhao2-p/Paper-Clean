#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chroma查询脚本

此脚本用于在本地 Chroma 向量存储中执行相似性搜索。

流程说明：
1. 解析工作空间路径，确保 Chroma 目录存在
2. 创建嵌入模型
3. 初始化 ChromaWriter
4. 执行相似性搜索
5. 打印搜索结果
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_pipeline.config import PipelinePaths
from rag_pipeline.embeddings import create_embeddings
from rag_pipeline.vectorstores.chroma_store import ChromaWriter


# 默认的向量数据库集合名称
DEFAULT_COLLECTION_NAME = "paper_chunks"
# 默认返回的检索结果数量
DEFAULT_TOP_K = 5


def run_query(
    query: str,
    *,
    workspace: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    执行相似性搜索
    
    对本地 Chroma 集合执行相似性搜索
    
    Args:
        query: 搜索查询文本
        workspace: 工作空间路径
        collection_name: Chroma集合名称
        embedding_model: 嵌入模型名称
        top_k: 返回的检索结果数量
    
    Returns:
        list[dict]: 搜索结果列表
    """
    # 解析工作空间路径
    workspace = Path(workspace).resolve()
    # 创建管道路径对象
    paths = PipelinePaths(workspace)
    # 检查 Chroma 目录是否存在
    if not paths.chroma.exists():
        raise FileNotFoundError(f"未找到 Chroma 目录: {paths.chroma}")

    # 创建嵌入模型
    embeddings = create_embeddings(embedding_model)
    # 创建 ChromaWriter 对象
    writer = ChromaWriter(
        persist_directory=paths.chroma,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    # 执行相似性搜索并返回结果
    return writer.similarity_search(query, k=top_k)


def main(
    query: str,
    *,
    workspace: str | Path,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> None:
    """
    主函数
    
    为在PyCharm中直接执行而打印查询结果
    
    Args:
        query: 搜索查询文本
        workspace: 工作空间路径
        collection_name: Chroma集合名称
        embedding_model: 嵌入模型名称
        top_k: 返回的检索结果数量
    """
    # 执行查询
    matches = run_query(
        query,
        workspace=workspace,
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=top_k,
    )
    # 打印搜索结果
    print(json.dumps(matches, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 用于检索质量检查的本地验证入口
    # 当你想检查不同的召回情况时，修改 QUERY_TEXT
    project_root = Path(__file__).resolve().parents[2]
    QUERY_TEXT = "federated graph neural network"

    main(
        QUERY_TEXT,
        workspace=project_root,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=None,
        top_k=DEFAULT_TOP_K,
    )
