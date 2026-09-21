"""文档解析：把不同格式的文件转成纯文本。

目前只支持 TXT。PDF / Word / Markdown 后续按同样方式在这里扩展 ——
一种格式一个 parse_xxx() 函数，再由上层按扩展名分派。
不在这里做「一个函数处理所有格式」的大分派，是因为每种格式的失败模式完全不同
（PDF 可能是扫描件没有文字层、Word 可能是老版本 .doc），
混在一起会让错误信息变得含糊。

本模块只做「文件 → 字符串」，不做切分（那是 splitter.py 的事）。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_txt(file_path: str | Path) -> str:
    """读取 UTF-8 编码的 TXT 文件，返回纯文本。

    参数：
        file_path: 文件路径，str 或 Path 都接受。

    返回：
        文件内容原文。不做 strip —— 首尾空白由下游的切分器处理，
        这里保持「读什么就是什么」，避免解析环节悄悄改动原文。

    异常：
        FileNotFoundError：文件不存在。
        ValueError：        路径不是文件、文件不是 UTF-8 编码、或文件内容为空。
        PermissionError 等：由 Python 原样抛出，本函数不吞异常。
    """
    path = Path(file_path)

    # 显式检查而不是直接 read_text，是为了给出更明确的信息。
    # read_text 在文件不存在时也会抛 FileNotFoundError，但带上的信息更少；
    # 这里的判断还能顺带挡住「传进来一个目录」这种情况。
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    if not path.is_file():
        raise ValueError(f"路径不是文件：{path}")

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        # 这里必须给出明确提示，因为「中文 txt 其实是 GBK 编码」在 Windows 上极其常见。
        # 如果只抛原始的 UnicodeDecodeError，读代码的人看到的是一串字节位置，
        # 很难立刻反应过来是编码问题。
        # 注意这是「补充说明后重新抛出」，不是吞掉异常 —— 用 from exc 保留了原始调用栈。
        raise ValueError(
            f"文件不是合法的 UTF-8 编码：{path}。"
            "如果这是 Windows 上用记事本保存的中文文档，很可能是 GBK/ANSI 编码，"
            "请先转成 UTF-8 再上传。"
        ) from exc

    # 空文件（或只有空白字符）直接拒绝，而不是返回空串。
    # 让它流到下游的话，切分会产出空列表、向量化会拿到空文本，
    # 最后表现为「文档上传成功了但什么都没入库」—— 这种静默失败最难查。
    if not text.strip():
        raise ValueError(f"文件内容为空：{path}")

    logger.info("已解析 TXT：%s（%d 字符）", path, len(text))
    return text
