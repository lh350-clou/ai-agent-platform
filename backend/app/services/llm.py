"""LLM 服务：封装 DeepSeek 的对话调用。

DeepSeek 官方提供的是「OpenAI 兼容」接口，所以这里直接用 openai 官方 SDK，
只把 base_url 指向 https://api.deepseek.com。

约定（见 CLAUDE.md）：业务代码只使用本模块导出的 chat()，
不自己 new AsyncOpenAI，也不直接读 settings.DEEPSEEK_API_KEY。

本模块目前只做「一次对话请求」，不含 Agent 循环、工具调用、检索等逻辑。
"""

import logging

from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---- 客户端 ----
# 这里用一个模块级变量 + 惰性创建，而不是像 database.py 那样在导入时就建好实例，
# 原因有两个：
# 1. openai 3.x 的 AsyncOpenAI 在没有 api_key 时会直接抛 OpenAIError。
#    如果在模块导入时创建，那么「还没配 Key」的环境下 import 本模块就会失败，
#    连 FastAPI 都起不来 —— 而我们希望没配 Key 只是「调不通模型」，不是「服务挂了」。
# 2. 整个进程共用一个 client 就够了：它内部维护 HTTP 连接池，
#    每个请求都新建一个会反复重建连接，比复用更慢。
_llm_client: AsyncOpenAI | None = None


def get_llm_client() -> AsyncOpenAI:
    """返回全局唯一的 AsyncOpenAI 客户端，第一次调用时才真正创建。

    注意：创建客户端本身不会发起任何网络请求，也不会产生 API 费用，
    所以在这里创建是安全的；真正的调用只发生在 chat() 里。
    """
    global _llm_client

    if _llm_client is None:
        # 只判断「有没有配」，绝不把 Key 本身写进日志或异常信息。
        if not settings.DEEPSEEK_API_KEY.get_secret_value():
            raise RuntimeError(
                "未配置 DEEPSEEK_API_KEY：请在仓库根目录的 .env 中填写后重试"
                "（可参考 .env.example）"
            )

        # 显式调用 get_secret_value() 取出明文交给 SDK —— 这是唯一需要明文的地方。
        _llm_client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY.get_secret_value(),
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    return _llm_client


async def chat(
    messages: list[ChatCompletionMessageParam],
    temperature: float = 0.7,
) -> str:
    """调用 DeepSeek 对话接口，返回模型回答的纯文本。

    参数：
        messages:    对话消息列表，结构沿用 OpenAI 规范，
                     例如 [{"role": "user", "content": "你好"}]。
                     角色可以是 system / user / assistant，
                     多轮对话就是把历史消息按顺序一起传进来（模型本身不记上下文）。
        temperature: 采样温度，0 最稳定、越大越发散。默认 0.7 适合日常问答。

    返回：
        模型回复的文本内容。只返回文本，不返回 usage、finish_reason 等原始结构，
        是为了让调用方（后面的 RAG / Agent）拿到的是能直接用的一等公民。

    异常：
        未配置 Key、网络不通、接口返回 4xx/5xx 时抛 RuntimeError，
        消息里只包含「哪一步出错了」，不含密钥。
    """
    completion = await _call_completions(messages, temperature)

    # 正常情况下 choices 至少有一项；返回空列表属于异常响应，
    # 与其让后面 completion.choices[0] 抛出难懂的 IndexError，不如在这里给出明确提示。
    if not completion.choices:
        raise RuntimeError("DeepSeek 返回结果中没有 choices，无法取出文本内容")

    # content 理论上可能为 None（例如模型只返回了工具调用），
    # 用 or "" 兜底成空字符串，避免调用方拿到 None 还要再判一次。
    return completion.choices[0].message.content or ""


async def _call_completions(
    messages: list[ChatCompletionMessageParam],
    temperature: float,
    tools: list[ChatCompletionToolParam] | None = None,
) -> ChatCompletion:
    """真正发起一次 chat.completions 请求，并统一处理异常。

    把这段抽出来，是因为调用方从「一种」变成了「两种」：
    不带工具的普通对话（chat）和带工具的工具调用（chat_with_tools）。
    两者的差别只在 tools 这一个参数，错误处理完全一样 ——
    复制一份的话，将来改错误文案或补一种异常处理，就得记得改两处，
    而漏掉的那一处恰恰最不容易被发现（另一条路径平时不跑）。
    """
    client = get_llm_client()

    # tools 为空时【不能】把它当成 None 传进去：
    # 传 tools=None 相当于明确声明「本次没有可用工具」，
    # 而「不传这个参数」与「传 None」在部分 OpenAI 兼容实现里行为并不一致。
    # 用展开的方式，只在真的有工具时才带上这个键。
    extra: dict = {"tools": tools} if tools else {}

    try:
        return await client.chat.completions.create(
            # 模型名同样从配置读，不写死在调用处，
            # 将来想换 deepseek-reasoner 只需要改 .env，不用动代码。
            model=settings.DEEPSEEK_MODEL,
            messages=messages,
            temperature=temperature,
            **extra,
        )
    except APIConnectionError as exc:
        # 网络层失败：DNS 解析不了、超时、连不上 api.deepseek.com。
        # 这类问题要看到完整堆栈才好排查，所以用 logger.exception 记全。
        logger.exception("调用 DeepSeek 失败：无法建立连接")
        raise RuntimeError(
            "无法连接 DeepSeek 服务，请检查网络或 .env 中的 DEEPSEEK_BASE_URL"
        ) from exc
    except APIStatusError as exc:
        # 服务端有响应但状态码非 2xx：401 密钥无效、402 余额不足、429 限流、5xx 服务异常。
        # 这里只把状态码和接口给的错误说明写进日志 —— 它们能定位问题，又不含密钥。
        logger.error("调用 DeepSeek 失败：HTTP %s - %s", exc.status_code, exc.message)
        raise RuntimeError(
            f"DeepSeek 接口返回错误（HTTP {exc.status_code}），详情见服务端日志"
        ) from exc


async def chat_with_tools(
    messages: list[ChatCompletionMessageParam],
    tools: list[ChatCompletionToolParam],
    temperature: float = 0.0,
) -> ChatCompletionMessage:
    """带工具的对话：返回模型这一轮的完整消息。

    和 chat() 的区别在于返回什么：
    chat() 只回文本，够用是因为普通对话里模型只会说话；
    而工具调用这一轮模型可能【什么都没说、只要求调用工具】，
    此时 content 是 None，真正有用的信息在 message.tool_calls 里。
    所以这里把整条消息返回出去，由调用方（agent.py）自己判断该看哪个字段。

    参数：
        messages:    对话消息列表，含 system 规则、历史、以及上一轮的工具结果。
        tools:       可用工具的定义列表（OpenAI 规范格式）。
        temperature: 默认 0.0 —— 比普通对话低得多，这是刻意的：
                     选哪个工具、传什么参数必须稳定可复现，
                     0.7 那种「发散」在这里只会让同一个问题时而调工具时而不调。

    异常：
        未配置 Key、网络不通、接口返回 4xx/5xx 时抛 RuntimeError，消息不含密钥。
    """
    completion = await _call_completions(messages, temperature, tools=tools)

    if not completion.choices:
        raise RuntimeError("DeepSeek 返回结果中没有 choices，无法取出消息")

    return completion.choices[0].message
