#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chroma查询运行脚本

此脚本用于在本地 Chroma 向量存储中执行相似性搜索。

流程说明：
1. 配置查询参数（查询文本、集合名称、嵌入模型等）
2. 调用查询脚本执行相似性搜索
3. 输出搜索结果

使用方法：
- 在 PyCharm 中直接运行此脚本
- 修改 QUERY_TEXT 变量来查询不同的概念
- 调整 TOP_K 参数来控制返回的结果数量
"""

from __future__ import annotations

from pathlib import Path

from rag_pipeline.scripts.query_chroma import main as run_query_main


# 项目根目录路径
PROJECT_ROOT = Path(__file__).resolve().parent
# 向量数据库中的集合名称
COLLECTION_NAME = "paper_chunks"
# 嵌入模型名称，None表示使用默认模型
EMBEDDING_MODEL: str | None = None
# 返回的检索结果数量
TOP_K = 5
# 查询文本，可以修改为任何你想搜索的概念
QUERY_TEXT = "TWAFL算法的准确率如何"


def main() -> None:
    """
    主函数，执行查询流程
    
    PyCharm直接运行设置：
    1. 将 QUERY_TEXT 设置为你想要验证的概念
    2. 当需要检查更多返回的分块时，增加 TOP_K 的值
    3. 仅当你想要使用非默认的嵌入模型时，覆盖 EMBEDDING_MODEL
    """
    # 调用查询脚本执行相似性搜索
    run_query_main(
        QUERY_TEXT,
        workspace=PROJECT_ROOT,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        top_k=TOP_K,
    )


if __name__ == "__main__":
    main()
