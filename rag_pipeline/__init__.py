#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 论文索引管道

此模块是 RAG 论文索引管道的包初始化文件。

该管道用于离线处理和索引学术论文，包括：
- PDF 文件加载和解析
- 论文内容清理和分块
- 向量嵌入和存储
- 相似性搜索

模块结构：
- cleaners: 论文内容清理模块
- loaders: PDF 加载模块
- parsers: 论文解析模块
- schemas: 数据模型定义
- scripts: 执行脚本
- vectorstores: 向量存储模块
- workflows: 工作流定义
"""
