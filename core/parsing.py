"""文档解析层（F2 流水线第一步）：原始文件字节 -> 结构化 ParsedDoc。

设计书 §4.1：Markdown/HTML 直接解析；PDF 用 MinerU；Office 用 Apache Tika。
本模块输出统一的 block 流（段落 / 表格 / 列表 / 代码块），供 core.chunking
做标题感知切分；除 Tika/MinerU 两个重型解析器外均为纯函数，方便单测。

约定：
- ParsedBlock.headings 为标题层级路径（顶层在前），如 ["指南", "安装"]；
- 表格块 is_table=True，text 保留 Markdown 表格表示（切分层整体不切）；
- md/html/txt 无页码概念，page_no 为 None（PDF 接入后回填）。
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

_MD_EXTS = {".md", ".markdown"}
_HTML_EXTS = {".html", ".htm"}
_TXT_EXTS = {".txt"}
_PDF_EXTS = {".pdf"}
_OFFICE_EXTS = {".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt"}

# 中文优先：先按 utf-8 解码，失败退 gb18030（GBK/GB2312 超集），再不行替换非法字符
_TEXT_ENCODINGS = ("utf-8", "gb18030")

# 句边界（中英标点），用于切分层的句子级原子化；lookbehind 保留分隔符在句尾
_SENTENCE_DELIM_RE = re.compile(r"(?<=[。！？!?；;…])")


@dataclass
class ParsedBlock:
    """一个内容块：段落 / 表格 / 列表 / 代码块等。"""

    text: str
    headings: list[str]  # 所在标题路径（顶层在前）；无标题时为空列表
    page_no: int | None  # 原文页码（溯源用）；md/html/txt 为 None
    is_table: bool  # 表格块：切分时整体保留、不再切分


@dataclass
class ParsedDoc:
    """一篇文档的解析结果。"""

    title: str
    blocks: list[ParsedBlock]


# --------------------------------------------------------------------------
# 分发入口
# --------------------------------------------------------------------------
def parse_document_file(filename: str, data: bytes) -> ParsedDoc:
    """按文件扩展名分发到具体解析器。

    filename 仅用于取扩展名与兜底标题，不要求真实路径。
    不支持的扩展名抛 ValueError；解析器缺失 / 外部服务不可用抛 RuntimeError。
    """
    ext = os.path.splitext(filename)[1].strip().lower()
    if ext in _MD_EXTS:
        return _parse_markdown(filename, data)
    if ext in _HTML_EXTS:
        return _parse_html(filename, data)
    if ext in _TXT_EXTS:
        return _parse_plain_text(filename, data)
    if ext in _PDF_EXTS:
        return _parse_pdf(filename, data)
    if ext in _OFFICE_EXTS:
        return _parse_office(filename, data)
    raise ValueError(f"不支持的文件类型：{filename or '<空文件名>'}（扩展名 {ext or '<无>'}）")


# --------------------------------------------------------------------------
# Markdown（mistune 3.x token 级解析，保留标题层级）
# --------------------------------------------------------------------------
def _parse_markdown(filename: str, data: bytes) -> ParsedDoc:
    import mistune  # 顶层依赖，但延迟导入以加快无需解析的导入路径

    text = _decode_text(data)
    stem = _file_stem(filename)
    # renderer=None 返回 token 树而非 HTML；table 插件让 | 表格 | 识别为 table token
    md = mistune.create_markdown(renderer=None, plugins=["table"])
    result = md.parse(text)
    tokens: list = result[0] if isinstance(result, tuple) else result

    title: str | None = None
    path: list[str] = []
    blocks: list[ParsedBlock] = []

    def emit(block_text: str, is_table: bool = False) -> None:
        """追加一个内容块（丢弃空白块）。"""
        if block_text and block_text.strip():
            blocks.append(ParsedBlock(text=block_text.strip("\n"), headings=list(path), page_no=None, is_table=is_table))

    for tok in tokens:
        if not isinstance(tok, dict):
            continue
        ttype = tok.get("type")
        if ttype == "heading":
            level = int((tok.get("attrs") or {}).get("level", 1) or 1)
            heading_text = _inline_text(tok.get("children", [])).strip()
            if not heading_text:
                continue
            if level == 1 and title is None:
                title = heading_text
            # 维护标题路径：h2 覆盖 path[1:]，跳级标题（如无 h2 直接 h3）顺序拼接
            path[level - 1 :] = [heading_text]
        elif ttype == "paragraph":
            emit(_inline_text(tok.get("children", [])))
        elif ttype == "table":
            emit(_table_tokens_to_markdown(tok), is_table=True)
        elif ttype == "list":
            emit(_list_tokens_to_text(tok))
        elif ttype == "block_code":
            raw = tok.get("raw") or ""
            info = (tok.get("attrs") or {}).get("info") or ""
            emit(f"```{info}\n{raw}```")
        elif ttype == "block_quote":
            # 引用内是嵌套的块级 token，通用递归抽文本即可
            emit(_inline_text(tok.get("children", [])))
        # blank_line / 其他 token 忽略
    return ParsedDoc(title=title or stem, blocks=blocks)


def _inline_text(children: Iterable) -> str:
    """从行内 token 树抽取纯文本（softbreak 还原为换行）。"""
    parts: list[str] = []

    def _walk(nodes: Iterable) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("type") in ("softbreak", "linebreak"):
                parts.append("\n")
            raw = node.get("raw")
            if isinstance(raw, str):
                parts.append(raw)
            kids = node.get("children")
            if isinstance(kids, list):
                _walk(kids)

    _walk(children)
    return "".join(parts)


def _table_tokens_to_markdown(tok: dict) -> str:
    """把 mistune table token 还原为 Markdown 表格文本（切分层整体不切）。"""
    head: list[str] = []
    body: list[list[str]] = []
    for child in tok.get("children", []):
        ctype = child.get("type")
        if ctype == "table_head":
            head = [
                _inline_text(c.get("children", [])).strip()
                for c in child.get("children", [])
                if isinstance(c, dict) and c.get("type") == "table_cell"
            ]
        elif ctype == "table_body":
            for row in child.get("children", []):
                if not (isinstance(row, dict) and row.get("type") == "table_row"):
                    continue
                body.append(
                    [
                        _inline_text(c.get("children", [])).strip()
                        for c in row.get("children", [])
                        if isinstance(c, dict) and c.get("type") == "table_cell"
                    ]
                )
    ncols = max([len(head)] + [len(r) for r in body]) if (head or body) else 0
    if ncols == 0:
        return ""
    head = (head + [""] * ncols)[:ncols]
    body = [(r + [""] * ncols)[:ncols] for r in body]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _list_tokens_to_text(tok: dict) -> str:
    """列表 token -> 每项一行的纯文本（保留条目结构信息）。"""
    items: list[str] = []
    for item in tok.get("children", []):
        if isinstance(item, dict) and item.get("type") == "list_item":
            items.append("- " + _inline_text(item.get("children", [])).strip())
    return "\n".join(items)


# --------------------------------------------------------------------------
# HTML（BeautifulSoup + lxml）
# --------------------------------------------------------------------------
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# 内容容器：嵌套在这些元素里的同类元素随外层整体输出，避免内容重复
_CONTAINER_TAGS = ["p", "table", "pre", "blockquote", "ul", "ol"]
_COLLECT_TAGS = list(_HEADING_LEVELS) + _CONTAINER_TAGS


def _parse_html(filename: str, data: bytes) -> ParsedDoc:
    from bs4 import BeautifulSoup

    stem = _file_stem(filename)
    soup = BeautifulSoup(data, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    title = title or None

    path: list[str] = []
    blocks: list[ParsedBlock] = []
    root = soup.body or soup
    for el in root.find_all(_COLLECT_TAGS):
        name = el.name or ""
        if name in _HEADING_LEVELS:
            level = _HEADING_LEVELS[name]
            heading_text = el.get_text(" ", strip=True)
            if not heading_text:
                continue
            path[level - 1 :] = [heading_text]
            if level == 1 and title is None:
                title = heading_text
            continue
        # 嵌套在容器内的段落/表格等已被外层捕获，跳过
        if el.find_parent(_CONTAINER_TAGS) is not None:
            continue
        if name == "table":
            text = _html_table_to_markdown(el)
            if text:
                blocks.append(ParsedBlock(text=text, headings=list(path), page_no=None, is_table=True))
        else:
            text = el.get_text("\n", strip=True)
            if text:
                blocks.append(ParsedBlock(text=text, headings=list(path), page_no=None, is_table=False))
    return ParsedDoc(title=title or stem, blocks=blocks)


def _html_table_to_markdown(table) -> str:
    """HTML <table> -> Markdown 表格文本（第一行视为表头，无论 th/td）。"""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        row = [c.get_text(" ", strip=True) for c in cells]
        if row:
            rows.append(row)
    if not rows:
        return ""
    ncols = max(len(r) for r in rows)
    rows = [(r + [""] * ncols)[:ncols] for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 纯文本
# --------------------------------------------------------------------------
def _parse_plain_text(filename: str, data: bytes) -> ParsedDoc:
    """整篇一个 block（无结构信息，切分层按纯文本递归切）。"""
    text = _decode_text(data).strip()
    blocks = [ParsedBlock(text=text, headings=[], page_no=None, is_table=False)] if text else []
    return ParsedDoc(title=_file_stem(filename), blocks=blocks)


# --------------------------------------------------------------------------
# PDF（MinerU，重型可选依赖）
# --------------------------------------------------------------------------
def _parse_pdf(filename: str, data: bytes) -> ParsedDoc:
    try:
        import mineru  # noqa: F401  # 仅探测可用性
    except ImportError as exc:
        raise RuntimeError("mineru 未安装，请 pip install -e .[parsing-heavy]") from exc
    # TODO(集成阶段): MinerU 2.x 真实流水线（薄接入）：
    #   1) 落盘临时文件，调用 mineru 的 do_parse / CLI 得到 markdown 中间产物 + middle.json；
    #   2) middle.json 含页码、标题层级、表格 bbox —— 据此映射为 ParsedBlock：
    #      headings 取标题层级路径、page_no 取所在页、表格 is_table=True；
    #   3) markdown 正文部分可直接复用 _parse_markdown 的 token 解析。
    raise RuntimeError("MinerU 已安装，但 PDF 解析流水线尚未接入（TODO）")


# --------------------------------------------------------------------------
# Office（Apache Tika server，HTTP PUT /tika 抽纯文本）
# --------------------------------------------------------------------------
def _parse_office(filename: str, data: bytes) -> ParsedDoc:
    # Tika 地址走环境变量（框架 config 无此字段，不得修改 infra/**）
    url = os.getenv("TIKA_URL", "http://localhost:9998/tika")
    import httpx

    try:
        resp = httpx.put(url, content=data, headers={"Accept": "text/plain"}, timeout=60.0)
        resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(f"Tika 解析失败，请确认 Tika server 可用（PUT {url}）：{exc}") from exc
    text = resp.text.strip()
    # Tika 只出纯文本：整篇一个 block（无标题层级/页码）
    blocks = [ParsedBlock(text=text, headings=[], page_no=None, is_table=False)] if text else []
    return ParsedDoc(title=_file_stem(filename), blocks=blocks)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _decode_text(data: bytes) -> str:
    """中文优先解码：utf-8 -> gb18030 -> 替换非法字符。"""
    for enc in _TEXT_ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _file_stem(filename: str) -> str:
    """文件名去扩展名（兜底标题）。"""
    return os.path.splitext(os.path.basename(filename))[0] or filename
