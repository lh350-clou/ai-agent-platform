"""文本切分：把长文本切成适合向量化的片段。

为什么必须切分：embedding 模型对输入长度有上限，而且一整篇文档压成一个向量后，
语义会被「平均」掉 —— 检索时既定位不到具体段落，也会把无关内容一起召回。
切成 300~500 字的片段，是检索精度和上下文完整度之间的常见折中。

本模块只处理「字符串 → 字符串列表」，不碰 embedding、不碰 Milvus。
"""

import logging

logger = logging.getLogger(__name__)

# 切分时优先在哪些字符之后断开。
# 分成两档而不是一档，是因为「段落/句号」和「逗号/空格」的语义权重完全不同：
# 在句号后断开，得到的片段是完整的句子；在逗号后断开，片段会断在半句话上。
# 所以先在窗口里找强断点，找不到才退而求其次找弱断点。
_STRONG_BREAKS: tuple[str, ...] = ("\n", "。", "！", "？", "；", ".", "!", "?", ";")
_WEAK_BREAKS: tuple[str, ...] = ("，", ",", " ")

# 往回找断点时最多回退多远（占 chunk_size 的比例）。
# 这个值不能太大：如果整段文字没有任何标点，回退太远会让每个 chunk 都远小于 chunk_size，
# 白白浪费额度；也不能太小，否则等于没找。
_MAX_LOOKBACK_RATIO: float = 0.3


def _pick_break_point(text: str, start: int, end: int, lookback: int) -> int:
    """在 (start, end] 区间内从右往左找最靠右的自然断点。

    返回的是「下一个片段应该从哪里开始」的位置，也就是断点字符的下一个下标。
    找不到任何断点时返回 end，表示这一刀只能硬切。

    从右往左找（而不是从左往右）是为了让片段尽量接近 chunk_size：
    最靠右的断点浪费的额度最少。
    """
    # 回退下限。至少要比 start 大 1，否则可能返回一个不大于 start 的位置，
    # 导致调用方的 start 不前进、陷入死循环。
    floor = max(start + 1, end - lookback)

    for breaks in (_STRONG_BREAKS, _WEAK_BREAKS):
        for i in range(end - 1, floor - 1, -1):
            if text[i] in breaks:
                return i + 1  # 断在断点字符「之后」，让这个标点留在前一个片段里
    return end


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """把文本按字符长度切成片段。

    参数：
        text:          待切分的文本。
        chunk_size:    单个片段的**最大**字符数（不是固定长度，见下方说明）。
        chunk_overlap: 相邻片段的重叠字符数，必须小于 chunk_size。

    返回：
        片段列表，顺序与原文一致。输入为空或只有空白字符时返回空列表。

    异常：
        ValueError：chunk_size 不是正数、chunk_overlap 为负数，或
                    chunk_overlap >= chunk_size。

    ── 为什么 chunk_size 是「上限」而不是「固定值」──
    函数会尽量在断点处收尾：如果第 chunk_size 个字符落在句子中间，
    就往前退到最近的句号或换行之后。所以实际长度通常略小于 chunk_size。
    这样做的代价是片段长度不齐，收益是片段不会断在半句话上 ——
    而检索质量对「片段是否语义完整」非常敏感。

    ── 为什么需要 chunk_overlap ──
    只要切分，就一定会有一个完整的语义单元正好横跨切分点的情况。
    让相邻片段重叠一段，可以保证这类内容至少在某一个片段里是完整的。
    重叠越多越不容易漏，但存储和向量化的成本也越高，一般取 chunk_size 的 10% 左右。
    """
    # ---- 参数校验：非法参数必须在干活之前就拒绝 ----
    if chunk_size <= 0:
        raise ValueError(f"chunk_size 必须为正整数，收到 {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap 不能为负数，收到 {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        # 这条不只是「参数不合理」：如果 overlap 大于等于 size，
        # 下一段的起点就不会超过上一段的起点，循环永远走不到结尾。
        raise ValueError(
            f"chunk_overlap({chunk_overlap}) 必须小于 chunk_size({chunk_size})，"
            "否则切分无法向前推进"
        )

    if not text or not text.strip():
        return []

    lookback = max(1, int(chunk_size * _MAX_LOOKBACK_RATIO))
    total = len(text)
    chunks: list[str] = []

    start = 0
    while start < total:
        end = min(start + chunk_size, total)

        # 只有当后面还有内容时才需要找断点；最后一段直接切到结尾
        if end < total:
            end = _pick_break_point(text, start, end, lookback)

        # strip 掉片段首尾的空白，避免产出「整段都是空白」的片段。
        # 这不会丢正文：被 strip 掉的是空白字符，而且下一段的起点在上一个
        # 片段的末尾之前（因为有 overlap），这些位置会被重新覆盖到。
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= total:
            break

        # 下一个片段的起点往回退 overlap 个字符，形成重叠。
        # max(..., start + 1) 是兜底：保证起点无论如何都比上一轮更靠后，
        # 否则一旦 end 因为找断点而回退得过多，这里就可能原地打转。
        start = max(end - chunk_overlap, start + 1)

    logger.info(
        "切分完成：原文 %d 字符 -> %d 个片段（chunk_size=%d, overlap=%d）",
        total, len(chunks), chunk_size, chunk_overlap,
    )
    return chunks
