"""
论文索引运行脚本

此脚本用于启动论文索引流程，将论文目录中的PDF文件导入到向量数据库中。
"""

from __future__ import annotations

from pathlib import Path

from rag_pipeline.scripts.ingest_papers import run_ingestion


# 项目根目录路径
PROJECT_ROOT = Path(__file__).resolve().parent
# 论文存储目录路径
PAPERS_DIR = PROJECT_ROOT / "papers"

# PyCharm直接运行设置:
# 1. 将源PDF文件放入项目级别的 `papers/` 目录中。
# 2. 保持 `SKIP_VECTORSTORE=False` 以正常索引到Chroma向量数据库。
# 3. 仅当需要调试解析器/分块器时，才将其设置为 `True`。

# 向量数据库中的集合名称
COLLECTION_NAME = "paper_chunks"
# 嵌入模型名称，None表示使用默认模型
EMBEDDING_MODEL: str | None = None
# 是否跳过向量存储步骤（用于调试）
SKIP_VECTORSTORE = False


def main() -> None:
    """
    主函数，执行论文索引流程
    
    调用 run_ingestion 函数，传入论文目录路径和相关配置参数
    """
    run_ingestion(
        PAPERS_DIR,
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        skip_vectorstore=SKIP_VECTORSTORE,
    )


if __name__ == "__main__":
    main()
