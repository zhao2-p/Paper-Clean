#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块

此模块定义了项目中使用的路径管理类，用于集中管理项目的磁盘布局。
"""

from __future__ import annotations

from pathlib import Path


class PipelinePaths:
    """
    管道路径管理类
    
    集中管理项目的磁盘布局，包括调试 artifacts 和向量存储数据的路径。
    """

    def __init__(self, workspace: str | Path) -> None:
        """
        初始化管道路径对象
        
        Args:
            workspace: 工作空间路径，可以是字符串或 Path 对象
        """
        root = Path(workspace).resolve()
        self.root = root  # 项目根目录
        self.outputs = root / "outputs"  # 输出目录
        self.parsed_json = self.outputs / "parsed_json"  # 解析后的 JSON 文件目录
        self.cleaned_json = self.outputs / "cleaned_json"  # 清理后的 JSON 文件目录
        self.chunk_jsonl = self.outputs / "chunk_jsonl"  # 分块后的 JSONL 文件目录
        self.reports = self.outputs / "reports"  # 报告文件目录
        self.chroma = root / "data" / "chroma"  # Chroma 向量存储目录

    def ensure(self) -> None:
        """
        确保所有必要的目录存在
        
        遍历所有路径，创建不存在的目录（包括父目录）
        """
        for path in (
            self.parsed_json,
            self.cleaned_json,
            self.chunk_jsonl,
            self.reports,
            self.chroma,
        ):
            path.mkdir(parents=True, exist_ok=True)
