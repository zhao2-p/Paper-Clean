#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chroma向量存储模块

此模块提供了将论文分块持久化到本地 Chroma 向量数据库的功能。

功能说明：
- 初始化 Chroma 向量存储客户端
- 添加论文分块到向量数据库
- 执行相似性搜索
- 清理和规范化元数据

主要组件：
- ChromaWriter: 向量存储写入器，负责管理 Chroma 集合的操作
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from rag_pipeline.schemas.models import PaperChunk


class ChromaWriter:
    """
    Chroma向量存储写入器
    
    负责将论文分块持久化到本地 Chroma 集合中，并提供相似性搜索功能。
    
    功能：
    - 初始化和管理 Chroma 向量存储
    - 批量添加文档分块
    - 执行相似性搜索查询
    - 清理和规范化元数据
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embeddings: Embeddings,
    ) -> None:
        """
        初始化 ChromaWriter
        
        Args:
            persist_directory: 向量存储持久化目录路径
            collection_name: Chroma 集合名称
            embeddings: 嵌入模型实例
        """
        # 初始化 Chroma 向量存储客户端
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(Path(persist_directory)),
            embedding_function=embeddings,
        )

    def add_chunks(self, chunks: list[PaperChunk]) -> int:
        """
        添加论文分块到向量存储
        
        将论文分块列表转换为 LangChain Document 对象并添加到 Chroma 集合中
        
        Args:
            chunks: 论文分块列表
        
        Returns:
            int: 成功添加的分块数量
        """
        # 将分块转换为 Document 对象
        documents = [
            Document(
                page_content=chunk["text"],
                metadata=self._sanitize_metadata(
                    {
                        **chunk.get("metadata", {}),
                        "chunk_id": chunk["chunk_id"],
                        "block_id": chunk["block_id"],
                    }
                ),
                id=chunk["chunk_id"],
            )
            for chunk in chunks
        ]
        # 如果没有文档，直接返回0
        if not documents:
            return 0
        # 添加文档到向量存储
        self.vectorstore.add_documents(documents=documents)
        return len(documents)

    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        """
        执行相似性搜索
        
        在向量存储中搜索与查询文本最相似的文档
        
        Args:
            query: 查询文本
            k: 返回的最相似文档数量
        
        Returns:
            list[dict]: 搜索结果列表，每个结果包含文本和元数据
        """
        # 执行相似性搜索
        results = self.vectorstore.similarity_search(query, k=k)
        # 转换结果格式
        return [
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in results
        ]

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """
        清理元数据
        
        将元数据中的值转换为 Chroma 支持的类型（字符串、整数、浮点数、布尔值）
        
        Args:
            metadata: 原始元数据字典
        
        Returns:
            dict[str, str | int | float | bool]: 清理后的元数据字典
        """
        sanitized: dict[str, str | int | float | bool] = {}
        # 遍历元数据中的每个键值对
        for key, value in metadata.items():
            # 规范化元数据值
            normalized = self._normalize_metadata_value(value)
            if normalized is None:
                continue
            sanitized[key] = normalized
        return sanitized

    def _normalize_metadata_value(self, value: Any) -> str | int | float | bool | None:
        """
        规范化元数据值
        
        将任意类型的值转换为 Chroma 支持的类型
        
        Args:
            value: 任意类型的值
        
        Returns:
            str | int | float | bool | None: 规范化后的值
        """
        # 处理 None 值
        if value is None:
            return None
        # 布尔值直接返回
        if isinstance(value, bool):
            return value
        # 字符串、整数、浮点数直接返回
        if isinstance(value, (str, int, float)):
            # 空字符串返回 None
            if isinstance(value, str) and not value.strip():
                return None
            return value
        # 列表转换为 JSON 字符串
        if isinstance(value, list):
            if not value:
                return None
            return json.dumps(value, ensure_ascii=False)
        # 字典转换为 JSON 字符串
        if isinstance(value, dict):
            if not value:
                return None
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        # 其他类型转换为字符串
        return str(value)
