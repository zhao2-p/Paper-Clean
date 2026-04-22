# QCYZ Paper Clean

基于 LangChain 1.x + LangGraph 1.x 的论文 PDF 清洗+入库流水线。

当前项目目标不是聊天问答，而是先把论文索引链路做扎实：

`PDF -> 结构化解析 -> 论文清洗 -> 分块 -> 向量化 -> 写入 Chroma`

## 当前做到哪里了

当前已经完成一版可运行的 indexing MVP：

- 用 LangGraph 串起完整 workflow
- 支持从 `papers/` 目录批量读取 PDF
- 支持输出中间调试结果
- 支持正文、图注、表格标题的抽取
- 支持论文专用清洗
- 支持保留 `page`、`section_path`、`figure_id`、`table_id` 等 metadata
- 支持切块后写入本地 Chroma
- 支持在输出中记录当前实际使用的解析后端 `parser_backend`

当前主流程：

`parse_pdf -> extract_blocks -> clean_blocks -> build_chunks -> write_chroma -> save_reports`

## 当前架构

核心模块：

- `rag_pipeline/loaders/pdf_loader.py`
  - PDF 解析入口
  - 当前统一使用 `pymupdf4llm`
- `rag_pipeline/parsers/paper_parser.py`
  - 标题、摘要、章节、图注、表格标题等结构识别
  - 适配 `pymupdf4llm` 的 markdown 风格输出
- `rag_pipeline/cleaners/paper_cleaner.py`
  - 页眉页脚、页码、目录、机构信息、部分噪声清洗
- `rag_pipeline/transformers/chunk_builder.py`
  - 结构块转 chunk
  - 保留 metadata
- `rag_pipeline/vectorstores/chroma_store.py`
  - 本地 Chroma 写入
- `rag_pipeline/workflows/index_graph.py`
  - LangGraph workflow 编排
- `run_indexing.py`
  - PyCharm 直接运行入口

## 目录约定

- `papers/`
  - 放原始论文 PDF
- `outputs/parsed_json/`
  - 结构化 block 调试输出
- `outputs/cleaned_json/`
  - 清洗后的 block 输出
- `outputs/chunk_jsonl/`
  - 最终 chunk 输出
- `outputs/reports/`
  - 每篇论文的统计报告
- `data/chroma/`
  - 本地 Chroma 持久化目录

## 如何运行

推荐直接在 PyCharm 中运行根目录的 `run_indexing.py`。

使用方式：

1. 把原始论文 PDF 放进 `papers/`
2. 打开 `run_indexing.py`
3. 默认保持 `SKIP_VECTORSTORE = False`
4. 先验证解析、清洗、切块和调试输出
5. 如果只想调试解析 / 清洗 / 切块，再把 `SKIP_VECTORSTORE = True`

不要直接运行 `rag_pipeline/workflows/index_graph.py`，它只是工作流定义文件。

如果仍想走命令行或脚本入口：

- 可以直接运行 `rag_pipeline/scripts/ingest_papers.py`
- 该文件底部保留了 `if __name__ == "__main__":` 的本地验证入口
- 它会按默认参数读取项目根目录下的 `papers/` 并执行完整入库流程

完成入库后，可以用下面的方式做简单检索验证：

- 推荐直接运行根目录的 `run_query_chroma.py`
- 如果想调整查询词、返回数量或 collection，可以修改其中的常量后运行
- 也可以直接打开 `rag_pipeline/scripts/query_chroma.py`，修改 `if __name__ == "__main__":` 中的查询词和 `top_k`

## 依赖说明

当前至少需要这些关键依赖：

- `langchain`
- `langgraph`
- `langchain-chroma`
- `langchain-openai`
- `langchain-text-splitters`
- `langchain-community`
- `pymupdf4llm`

建议直接在项目解释器里安装：

```bash
pip install -e .
```

如果要走 `pymupdf4llm` 主路径，请确认当前 PyCharm 解释器里确实装好了：

```python
import pymupdf4llm
print("ok")
```

## 如何判断当前用了哪个解析器

每次运行后，可以看：

- `outputs/reports/*.json` 里的 `parser_backend`
- `outputs/parsed_json/*.json` 顶层的 `parser_backend`

当前固定记录为：

- `pymupdf4llm`

## 当前效果总结

以目前测试论文为例，`pymupdf4llm` 路径下已经能做到：

- 正确抽取论文标题
- 正确抽取作者列表
- 识别 `Abstract`
- 识别主章节与子章节
  - 如 `1 Introduction`
  - `2 Related Work`
  - `3 Our Method`
  - `3.1 Preliminaries`
  - `4 Evaluation`
  - `4.1 Experimental Settings`
  - `5 Conclusion`
- 抽取图注
- 抽取表格标题
- 清洗掉大部分会议页眉和页码噪声

## 当前还存在的问题

这版已经可用，但还不是最终版。主要问题有：

- 标题、作者目前仍然被标成 `paragraph`
  - 后续可以细分为 `title`、`author_list`
- 表格当前主要抓到的是表格标题
  - 还没有做高保真表格结构重建
- 图相关块当前主要是图注
  - 还没有把邻近正文和图片内容进一步绑定
- 对不同来源论文 PDF 的鲁棒性还需要继续验证
  - 当前主要基于一篇测试论文做了多轮调试
- Chroma 写入链路虽然已经接好
  - 但还没有进入系统性的检索效果验证阶段

## 下一步建议

当前最合理的下一阶段是从“解析调优”转向“入库验证”：

1. 打开 `SKIP_VECTORSTORE = False`
2. 把当前 chunk 正式写入本地 Chroma
3. 做一轮简单检索验证
4. 检查 metadata 过滤与召回效果
5. 再决定是否继续增强图表和表格结构

## 当前结论

当前项目已经从“工程骨架”推进到“可运行的论文 indexing MVP”。

如果后续目标是做论文问答 / RAG，那么现在最值得推进的不是继续大改 parser，而是：

- 先完成 Chroma 正式入库
- 再补检索验证
- 最后按效果决定是否继续增强表格、图注和多模态部分
