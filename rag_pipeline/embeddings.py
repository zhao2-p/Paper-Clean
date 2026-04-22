#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型模块

此模块用于创建和配置嵌入模型，为向量索引提供支持。
"""

from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

def create_embeddings(model: str | None = None) -> Embeddings:
    """
    创建嵌入模型实例
    
    构建用于索引的嵌入后端。虽然文档处理过程保持离线，但向量化仍然
    依赖于您配置的嵌入后端。
    
    Args:
        model: 嵌入模型名称，如果为 None，则使用环境变量 DASHSCOPE_EMBEDDING_MODEL
              或默认值 "text-embedding-v4"
    
    Returns:
        Embeddings: 嵌入模型实例
    """
    # 选择嵌入模型，如果未提供则使用环境变量或默认值
    selected_model = model or os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")
    # 创建并返回 DashScopeEmbeddings 实例
    return DashScopeEmbeddings(model=selected_model)
