"""core.chunking 单测：标题感知递归切分 + small-to-big 父子块 + 表格整体不切。"""
from itertools import pairwise

from core.chunking import (
    CHUNK_TARGET_TOKENS,
    approx_token_len,
    chunk_document,
)
from core.parsing import ParsedBlock, ParsedDoc


def _block(text, headings=(), is_table=False, page_no=None):
    return ParsedBlock(text=text, headings=list(headings), page_no=page_no, is_table=is_table)


def test_approx_token_len():
    assert approx_token_len("") == 0
    assert approx_token_len("中文") == 2  # CJK 每字 1
    assert approx_token_len("abcd") == 1  # ASCII 连续段每 4 字符 1
    assert approx_token_len("abc") == 1  # 不足 4 向上取整
    assert approx_token_len("abcdefgh") == 2
    assert approx_token_len("中文abcd") == 3  # 2 + 1
    assert approx_token_len("Hello 世界") == 4  # "Hello "（6 ascii -> 2）+ 世 + 界
    assert approx_token_len("中\n\n文") == 2  # 换行空白不计


def test_multilevel_headings_parent_child_structure():
    doc = ParsedDoc(
        title="t",
        blocks=[
            _block("开篇说明，无标题章节。", []),
            _block("顶层说明。", ["指南"]),
            _block("安装第一步。", ["指南", "安装"]),
            _block("安装第二步。", ["指南", "安装"]),
            _block("配置说明。", ["指南", "配置"]),
        ],
    )
    drafts = chunk_document(doc)
    parents = [d for d in drafts if d.parent_seq is None]
    children = [d for d in drafts if d.parent_seq is not None]
    by_seq = {d.seq: d for d in drafts}

    # 4 个叶子章节 -> 4 个父块（无标题块自成一组）
    assert len(parents) == 4
    assert {p.headings for p in parents} == {"", "指南", "指南 > 安装", "指南 > 配置"}

    # 子块的 parent_seq 都指向父块，且父块 seq 在子块之前
    for c in children:
        parent = by_seq[c.parent_seq]
        assert parent.parent_seq is None
        assert parent.seq < c.seq
        assert c.headings == parent.headings

    # 父块内容 = 章节全文聚合（含全部子块内容）
    install = next(p for p in parents if p.headings == "指南 > 安装")
    assert install.content == "安装第一步。\n\n安装第二步。"
    for p in parents:
        kids = [c for c in children if c.parent_seq == p.seq]
        assert kids and all(k.content in p.content for k in kids)


def test_seq_continuous_across_document():
    doc = ParsedDoc(
        title="t",
        blocks=[
            _block("第一章内容。", ["一"]),
            _block("表格", ["一"], is_table=True),
            _block("第二章内容。", ["二"]),
        ],
    )
    drafts = chunk_document(doc)
    assert sorted(d.seq for d in drafts) == list(range(len(drafts)))
    assert drafts[0].parent_seq is None  # seq=0 一定是第一个父块


def test_long_text_split_with_overlap():
    sentence = "这是用于验证递归切分逻辑的中文句子，包含标点符号与常见词汇。"
    text = "".join(f"第{i}句：{sentence}" for i in range(120))  # 约 4000+ token 的单段落
    doc = ParsedDoc(title="t", blocks=[_block(text, ["长章"])])
    drafts = chunk_document(doc)
    children = [d for d in drafts if d.parent_seq is not None]

    # 切出多块，每块（含重叠在内）都不超过 512 token
    assert len(children) >= 3
    for c in children:
        assert c.token_count == approx_token_len(c.content)
        assert c.token_count <= CHUNK_TARGET_TOKENS

    # 相邻块重叠非空，且重叠取自上一块尾部、落在句子/行边界
    for prev, nxt in pairwise(children):
        first_line = nxt.content.split("\n")[0]
        assert first_line.strip()
        assert first_line in prev.content.split("\n")
    # 父块不切分：等于整段原文
    parent = next(d for d in drafts if d.parent_seq is None)
    assert parent.content == text
    assert parent.token_count > CHUNK_TARGET_TOKENS


def test_table_never_split():
    table_text = "| A | B |\n|---|---|\n" + "\n".join(
        f"| 行{i} | 内容说明{i} |" for i in range(120)  # 远超 512 token
    )
    assert approx_token_len(table_text) > CHUNK_TARGET_TOKENS
    doc = ParsedDoc(title="t", blocks=[_block("本章说明。", ["表"]), _block(table_text, ["表"], is_table=True)])
    drafts = chunk_document(doc)
    children = [d for d in drafts if d.parent_seq is not None]
    tables = [c for c in children if c.content == table_text]
    assert len(tables) == 1  # 表格整体一块，原样输出
    assert tables[0].token_count > CHUNK_TARGET_TOKENS
    assert tables[0].headings == "表"
    # 其余子块均来自普通文本且不超限
    assert all(c.token_count <= CHUNK_TARGET_TOKENS for c in children if c is not tables[0])


def test_short_section_single_child():
    doc = ParsedDoc(title="t", blocks=[_block("很短的一节。", ["短"])])
    drafts = chunk_document(doc)
    assert len(drafts) == 2  # 1 父 + 1 子
    parent, child = drafts
    assert child.content == parent.content == "很短的一节。"


def test_page_no_carried_from_block():
    doc = ParsedDoc(
        title="t",
        blocks=[_block("第一页内容。", ["章"], page_no=3), _block("第二页内容。", ["章"], page_no=4)],
    )
    drafts = chunk_document(doc)
    parent = next(d for d in drafts if d.parent_seq is None)
    assert parent.page_no == 3  # 父块取章节首页
    kids = [d for d in drafts if d.parent_seq is not None]
    assert [k.page_no for k in kids] == [3, 4]


def test_parse_then_chunk_integration():
    from core.parsing import parse_document_file

    md = "# 部署指南\n\n概要说明。\n\n## 安装\n\n安装说明第一段。\n\n| 步骤 | 命令 |\n|---|---|\n| 1 | ls |\n"
    doc = parse_document_file("guide.md", md.encode("utf-8"))
    drafts = chunk_document(doc)
    parents = [d for d in drafts if d.parent_seq is None]
    children = [d for d in drafts if d.parent_seq is not None]
    assert len(parents) == 2  # 概要 / 安装 两个叶子章节
    assert len(children) == 3  # 概要 1 + 安装段落 1 + 表格 1
    assert any(c.content.startswith("| 步骤") for c in children)
