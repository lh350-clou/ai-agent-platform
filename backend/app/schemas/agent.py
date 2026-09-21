"""Agent 接口的请求 / 响应模型。

这里的 response 结构和 service 层的 AgentResult 是刻意分开的：
service 定义的是「程序内部怎么表达结果」，schema 定义的是「对外承诺什么」。
两者现在字段一样，但改动理由完全不同 —— 将来接口想少暴露一个字段，
改的是 schema，不该牵动 service。
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

# 问题文本的长度上限，与其它几个接口保持一致。
MAX_QUESTION_LENGTH: int = 2000


class AgentRequest(BaseModel):
    """POST /api/knowledge-bases/{id}/agent 的请求体。"""

    question: str = Field(
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="用户问题",
        examples=["Milvus 是什么？"],
    )

    @field_validator("question", mode="before")
    @classmethod
    def _strip_question(cls, value: object) -> object:
        """先去首尾空白再做长度校验，理由同其它接口：
        mode="before" 才能让 "   " 在清洗后正确地判为空、返回 422。"""
        return value.strip() if isinstance(value, str) else value


class AgentToolCall(BaseModel):
    """一次实际执行过的工具调用。

    只回「真执行过的」，不回「模型想执行但被拒绝的」——
    这份列表的用途是让调用方能解释「这个回答是怎么来的」，
    混进失败项只会让人误以为检索成功了。
    """

    tool: str = Field(
        description="工具名。内置工具是 search_knowledge_base，MCP 工具带 mcp_ 前缀",
        examples=["search_knowledge_base", "mcp_get_current_time"],
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="本次调用实际使用的参数",
        examples=[{"query": "Milvus 是什么", "top_k": 3}],
    )
    # 下面两个字段是给内置检索工具的便捷入口：它是本项目的核心工具，
    # 调用方多半只想直接拿到「查了什么词、取了几条」，不想每次去 arguments 里翻。
    # MCP 工具调用时它们是 null —— 各自的参数在 arguments 里。
    query: str | None = Field(
        default=None, description="检索工具实际使用的查询文本；非检索工具为 null"
    )
    top_k: int | None = Field(
        default=None,
        description="检索工具实际使用的 top_k（可能已被收敛到上限）；非检索工具为 null",
    )


class AgentResponse(BaseModel):
    """POST /api/knowledge-bases/{id}/agent 的响应体。"""

    answer: str = Field(description="Agent 的最终回答")
    tool_calls: list[AgentToolCall] = Field(
        default_factory=list,
        description="本次回答过程中实际执行过的工具调用；模型直接作答时为空列表",
    )
