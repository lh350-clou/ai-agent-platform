"""会话历史的读取与转换。

抽成独立模块，是因为「取最近若干条消息喂给模型」这件事现在有两个调用方：
RAG 问答（api/qa.py）和 Agent（api/agent.py）。两份各写一遍的话，
「最近 N 条」「哪些角色能进提示词」这些规则迟早会改歪一处 ——
而改歪的后果是模型上下文里混进不该有的东西，从回答上看不出来。
"""

import logging
from uuid import UUID

from openai.types.chat import ChatCompletionMessageParam
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import Message

logger = logging.getLogger(__name__)

# 每次请求最多带多少条历史消息给模型。
#
# 为什么必须设上限：历史会随对话无限增长，而每次请求都要把它们完整发出去 ——
# token 消耗和费用随之线性上涨，迟早撞上模型的上下文窗口，
# 而且很久之前的问答对当前问题基本没有帮助。
#
# 第一版用最简单的策略：只取最近 N 条，不做摘要、不做相关性筛选。
MAX_HISTORY_MESSAGES: int = 10

# 允许进入提示词的历史消息角色。
#
# 只放行 user 和 assistant。数据库里的 messages.role 是自由字符串
# （当初为了「加新角色不用改表结构」才没用 enum），万一有代码写入了
# role="system" 的记录，把它原样拼进 messages 就等于让【库里的数据】
# 坐上了系统指令的位置 —— 那正是提示词注入最想要的入口。
# 在这里白名单过滤，比指望所有写入方都规矩更可靠。
ALLOWED_HISTORY_ROLES: frozenset[str] = frozenset({"user", "assistant"})


async def load_recent_messages(db: AsyncSession, conversation_id: UUID) -> list[Message]:
    """读取会话最近的消息，按时间正序返回。

    为什么要「先倒序取 N 条、再翻转」而不是直接 `ORDER BY created_at ASC LIMIT N`：
    后者取到的是【最早】的 N 条，恰恰是对话开头那几句。
    而我们要的是【最近】的 N 条。这个错误很隐蔽 ——
    短对话里两者结果一样，只有消息超过上限之后才会显形，
    表现为「模型突然忘了刚才说过什么」。
    """
    rows = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
    )
    # 倒序取出来的是「从新到旧」，翻转成「从旧到新」才是给模型的时间顺序
    return list(reversed(rows.scalars().all()))


def history_to_messages(rows: list[Message]) -> list[ChatCompletionMessageParam]:
    """把历史消息转成模型能读的 messages，并过滤掉不允许的角色。

    返回的是可直接拼接进 messages 列表的结构，调用方负责决定
    它放在 system 之后还是别的位置。
    """
    messages: list[ChatCompletionMessageParam] = []
    for row in rows:
        # 白名单过滤：任何非 user / assistant 的角色都不进提示词。
        if row.role not in ALLOWED_HISTORY_ROLES:
            logger.warning(
                "历史消息角色不在白名单，已跳过：message_id=%s role=%r", row.id, row.role
            )
            continue
        messages.append({"role": row.role, "content": row.content})
    return messages
