"""MCP Server：通过 Model Context Protocol 对外暴露工具。

这个包是一个**完全独立**的进程，不 import FastAPI、SQLAlchemy、Milvus 客户端，
也不读应用配置。它只依赖 Python 标准库和 MCP SDK。

为什么要把这层独立性守住：MCP 的定位就是「把工具的实现和调用方解耦」。
一旦 Server 开始依赖应用内部的东西（数据库会话、配置对象），
它就不再是「一个可以被任何 MCP Client 使用的工具服务」，
而只是应用的一块内部代码 —— 那用普通函数调用就够了，没必要上 MCP。
"""
