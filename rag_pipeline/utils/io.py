#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输入输出工具模块

此模块提供了文件输入输出相关的工具函数。

功能说明：
- 写入 JSON 文件
- 写入 JSONL 文件（每行一个 JSON 对象）

主要组件：
- write_json: 写入 JSON 文件
- write_jsonl: 写入 JSONL 文件
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_json(path: str | Path, payload: Any) -> None:
    """
    写入 JSON 文件
    
    将数据写入 JSON 文件，自动创建父目录
    
    Args:
        path: 文件路径
        payload: 要写入的数据，可以是任何可序列化的 Python 对象
    """
    # 确保父目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 写入 JSON 文件
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    """
    写入 JSONL 文件
    
    将数据写入 JSONL 文件（每行一个 JSON 对象），自动创建父目录
    
    Args:
        path: 文件路径
        rows: 可迭代的数据对象，每个对象将被序列化为 JSON 并写入一行
    """
    # 确保父目录存在
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # 将每行数据序列化为 JSON 并用换行符连接
    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    # 写入 JSONL 文件
    Path(path).write_text(content, encoding="utf-8")
