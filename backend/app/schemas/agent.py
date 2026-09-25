"""Agent 接口的请求 / 响应模型。

这里的 response 结构和 service 层的 AgentResult 是刻意分开的：
service 定义的是「程序内部怎么表达结果」，schema 定义的是「对外承诺什么」。
两者现在字段一样，但改动理由完全不同 —— 将来接口想少暴露一个字段，
改的是 schema，不该牵动 service。
"""

from typing import Any

from uuid import UUID

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

    # 会话 ID。不传表示「开一个新会话」，服务端创建后随响应返回；
    # 传上一次的返回值则延续同一个会话，模型能看到之前的问答。
    conversation_id: UUID | None = Field(
        default=None,
        description="会话 ID；不传或传 null 表示新建会话",
        examples=[None],
    )


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


class AgentLLMCall(BaseModel):
    """一次 LLM 调用。

    它和 AgentToolCall 一起构成「这次回答是怎么来的」的完整链条：
    哪几轮问了模型、每轮等了多久，中间又查了什么。
    """

    model: str = Field(
        description="本次调用使用的模型名", examples=["deepseek-chat"]
    )
    duration_ms: float = Field(
        description="本次调用耗时（毫秒），含网络等待", examples=[1180.25]
    )
    success: bool = Field(description="本次调用是否成功")
    # 失败原因只用一句话说明（异常类型 + 消息），不含堆栈 ——
    # 堆栈里的本机路径这类内部信息不该顺着接口出去。
    error: str | None = Field(
        default=None, description="失败原因；成功时为 null", examples=[None]
    )


class AgentResponse(BaseModel):
    """POST /api/knowledge-bases/{id}/agent 的响应体。"""

    # 本次问答所属的会话 ID。新建会话时这里是新生成的 ID，
    # 调用方把它存下来，下一轮原样传回来就能接上上下文。
    conversation_id: UUID = Field(description="会话 ID，下一轮请求带上它即可延续对话")
    answer: str = Field(description="Agent 的最终回答")
    tool_calls: list[AgentToolCall] = Field(
        default_factory=list,
        description="本次回答过程中实际执行过的工具调用；模型直接作答时为空列表",
    )

    # ---- 运行记录（Trace）----
    # 这几个字段回答的是「这一轮跑得怎么样」：慢在哪、卡在哪、是不是在打转。
    # 刻意只挑这四项暴露：更细的（每次都调了哪个工具的第几个参数、
    # 被拒绝的调用等）留在服务端，接口不需要，也不该把内部结构整个摊出去。
    trace_id: str = Field(
        description="本次运行的 Trace ID，可在服务端日志里按它捞到同一轮的完整记录",
        examples=["3f2a1c8e-9b4d-4f0a-8f1e-2b6c7d5a9e10"],
    )
    iterations: int = Field(
        description="「调模型 → 执行工具」循环了几轮；等于 5 说明模型在打转、已被强制收敛",
        examples=[2],
    )
    llm_calls: list[AgentLLMCall] = Field(
        default_factory=list,
        description="每一次 LLM 调用的耗时与成败，按发生顺序排列",
    )
    total_duration_ms: float = Field(
        description="整个 Agent Run 的总耗时（毫秒），从收到问题到拿到回答",
        examples=[1820.4],
    )
