from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any


class PDFLoader:
    """
    PDF 加载器。

    当前项目将 PDF 加载层收敛为单一路径：只使用 ``pymupdf4llm``。

    这样做的目的有两个：
    1. 降低维护成本，避免同时维护两套输入结构。
    2. 保证下游 parser 面对的始终是同一种 markdown 风格输出。

    这里的职责只负责“把 PDF 转成按页组织的原始文本块”，
    不负责论文结构识别、清洗或 chunk 切分。
    """

    def load(self, pdf_path: str | Path) -> list[dict[str, Any]]:
        """
        加载单个 PDF，并返回按页组织的原始页面数据。

        返回结果中的每一页都保持统一结构，供后续 parser 使用：
        - ``page``: 页码
        - ``text``: 该页完整文本
        - ``blocks``: 由 markdown 切分出的文本块
        - ``parser_backend``: 当前固定为 ``pymupdf4llm``

        如果 ``pymupdf4llm`` 不可用，或解析结果为空，会直接抛出异常。
        这是有意为之：当前项目不再提供隐式 fallback，
        这样失败原因会更清晰，也更方便在 PyCharm 中直接定位问题。
        """

        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        pages = self._load_with_pymupdf4llm(path)
        if not pages:
            raise ValueError(f"pymupdf4llm returned no usable pages for: {path}")
        return pages

    def _load_with_pymupdf4llm(self, pdf_path: Path) -> list[dict[str, Any]]:
        """
        使用 pymupdf4llm 将 PDF 转成按页 markdown。

        优先尝试 ``page_chunks=True``，因为这是当前最理想的返回形式：
        能直接拿到按页切分后的内容，避免整篇文档被误判为单页。

        如果当前版本的 pymupdf4llm 不支持该返回形式，
        或返回的数据结构不符合预期，则退回到完整 markdown 文本，
        再基于换页符做保守切分。
        """

        try:
            import pymupdf4llm  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "未检测到 pymupdf4llm。当前项目只支持 pymupdf4llm 解析路径，"
                "请先在 PyCharm 对应解释器中安装并确认环境可用。"
            ) from exc

        try:
            page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        except Exception:
            page_chunks = None

        parsed_pages = self._normalize_markdown_pages(page_chunks)
        if parsed_pages:
            return parsed_pages

        try:
            markdown_text = pymupdf4llm.to_markdown(str(pdf_path))
        except Exception as exc:
            raise RuntimeError(f"pymupdf4llm failed to parse PDF: {pdf_path}") from exc

        if not markdown_text.strip():
            return []

        fallback_page_texts = self._split_markdown_pages(markdown_text)
        pages: list[dict[str, Any]] = []
        for index, page_text in enumerate(fallback_page_texts):
            blocks = self._markdown_to_blocks(page_text)
            pages.append(
                {
                    "page": index + 1,
                    "width": 0.0,
                    "height": 0.0,
                    "text": page_text,
                    "blocks": blocks,
                    "font_size_median": self._median_font_size(blocks),
                    "parser_backend": "pymupdf4llm",
                }
            )
        return pages

    def _split_markdown_pages(self, markdown: str) -> list[str]:
        """
        将整篇 markdown 文本按页拆分。

        这里主要处理旧版本或特殊情况下的兜底结果：
        如果返回文本中存在换页符 ``\\f``，就按换页符拆分；
        否则只能把整篇文本视为单页。

        注意：
        这个方法只是保守兜底，不代表分页质量最佳。
        真正推荐的路径仍然是 ``page_chunks=True``。
        """

        normalized = markdown.replace("\r\n", "\n")
        if "\f" in normalized:
            return [chunk.strip() for chunk in normalized.split("\f") if chunk.strip()]
        return [normalized]

    def _normalize_markdown_pages(self, payload: Any) -> list[dict[str, Any]]:
        """
        规范化 pymupdf4llm 的按页输出。

        不同版本的 pymupdf4llm 可能返回：
        - ``list[str]``
        - ``list[dict]``

        因此这里统一做一层兼容，将其转成项目内部稳定的 page schema。
        """

        if not isinstance(payload, list) or not payload:
            return []

        pages: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if isinstance(item, str):
                page_number = index + 1
                page_text = item.strip()
            elif isinstance(item, dict):
                page_number = int(item.get("page") or item.get("page_number") or index + 1)
                page_text = str(
                    item.get("text")
                    or item.get("markdown")
                    or item.get("md")
                    or ""
                ).strip()
            else:
                continue

            if not page_text:
                continue

            blocks = self._markdown_to_blocks(page_text)
            pages.append(
                {
                    "page": page_number,
                    "width": 0.0,
                    "height": 0.0,
                    "text": page_text,
                    "blocks": blocks,
                    "font_size_median": self._median_font_size(blocks),
                    "parser_backend": "pymupdf4llm",
                }
            )
        return pages

    def _markdown_to_blocks(self, page_text: str) -> list[dict[str, Any]]:
        """
        将单页 markdown 文本切分成粗粒度 block。

        这里的 block 不是最终论文结构块，而是加载层的原始块：
        仅用于给 parser 提供更稳定的输入单位。

        当前切分策略比较简单：
        - 以空行分段
        - 保留段内逐行文本
        - 基于 markdown 标题符号粗略估计字体大小

        之所以保留这种轻量策略，是因为真正的结构识别应该在 parser 中完成，
        避免加载层承担过多论文理解逻辑。
        """

        raw_blocks = [chunk.strip() for chunk in re_split_blocks(page_text) if chunk.strip()]
        blocks: list[dict[str, Any]] = []

        for index, text in enumerate(raw_blocks):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                continue

            header_like = lines[0].startswith("#")
            font_size = 16.0 if header_like else 11.0
            blocks.append(
                {
                    "text": "\n".join(lines),
                    "lines": lines,
                    # 这里没有真实版面坐标，因此只保留一个稳定的伪 bbox，
                    # 让下游仍然可以使用统一字段，而不误以为这是精确视觉位置。
                    "bbox": (0.0, float(index), 0.0, float(index + 1)),
                    "column_hint": 0,
                    "font_size_max": font_size,
                    "font_size_min": font_size,
                    "font_size_avg": font_size,
                }
            )
        return blocks

    def _median_font_size(self, blocks: list[dict[str, Any]]) -> float:
        """
        计算当前页 block 的字体中位数。

        下游标题识别会参考该值，因此这里仍然保留这个统计量。
        """

        values = [block.get("font_size_avg", 0.0) for block in blocks if block.get("font_size_avg", 0.0) > 0]
        return statistics.median(values) if values else 0.0


def re_split_blocks(page_text: str) -> list[str]:
    """
    按空行切分 markdown block。

    这是一个刻意保持简单的切分函数：
    加载层只需要给 parser 提供较稳定的输入块，
    不应该在这里提前做复杂结构判断。
    """

    return page_text.split("\n\n")
