#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文索引脚本

此脚本用于将本地论文PDF文件索引到调试artifacts和向量数据库中。

流程说明：
1. 解析输入目录，确保目录存在
2. 查找并排序目录中的PDF文件
3. 构建并编译索引图
4. 对每个PDF文件执行索引流程
"""

from __future__ import annotations

from pathlib import Path

from rag_pipeline.workflows.index_graph import build_index_graph


# 默认的向量数据库集合名称
DEFAULT_COLLECTION_NAME = "paper_chunks"


def run_ingestion(
    input_dir: str | Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
    skip_vectorstore: bool = False,
) -> None:
    """
    执行论文索引流程
    
    对输入目录下的每个PDF文件运行索引工作流
    
    Args:
        input_dir: 包含PDF文件的目录路径
        collection_name: Chroma集合名称
        embedding_model: 嵌入模型名称
        skip_vectorstore: 是否跳过向量存储步骤
    """
    # 解析并创建输入目录
    input_dir = Path(input_dir).resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找并排序PDF文件路径
    pdf_paths = sorted(str(path) for path in input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"在目录中未找到PDF文件: {input_dir}")

    # 构建并编译索引图
    app = build_index_graph().compile()

    # 处理每个PDF文件
    for pdf_path in pdf_paths:
        app.invoke(
            {
                "input_dir": str(input_dir),
                "current_pdf": pdf_path,
                "pdf_paths": pdf_paths,
                "collection_name": collection_name,
                "embedding_model": embedding_model,
                "skip_vectorstore": skip_vectorstore,
                "errors": [],
            }
        )


def main(
    input_dir: str | Path,
    *,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
    skip_vectorstore: bool = False,
) -> None:
    """
    主函数
    
    为从PyCharm直接执行而保留的薄包装器
    
    Args:
        input_dir: 包含PDF文件的目录路径
        collection_name: Chroma集合名称
        embedding_model: 嵌入模型名称
        skip_vectorstore: 是否跳过向量存储步骤
    """
    run_ingestion(
        input_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        skip_vectorstore=skip_vectorstore,
    )


if __name__ == "__main__":
    # 此模块的本地验证入口
    # 在PyCharm中运行此文件会执行与项目级run_indexing.py相同的工作流
    # 同时将演示代码与核心逻辑分开
    project_root = Path(__file__).resolve().parents[2]
    papers_dir = project_root / "papers"
    main(
        papers_dir,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=None,
        skip_vectorstore=False,
    )
