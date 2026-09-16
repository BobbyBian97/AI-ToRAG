"""core.parsing 单测：md/html/txt 真实解析 + pdf/office 分发与报错路径（mock，零外部服务）。"""
import httpx
import pytest

from core.parsing import parse_document_file

MD = """# 部署指南

本指南介绍系统的部署流程与注意事项。

## 安装

执行安装脚本即可完成基础安装。

| 步骤 | 命令 |
|---|---|
| 1 | pip install -e . |
| 2 | python -m services.worker |

### 注意事项

安装完成后需要重启服务才能生效。
"""

HTML = """<html>
  <head><title>运维手册</title></head>
  <body>
    <h1>运维手册</h1>
    <p>系统概述文本。</p>
    <h2>巡检</h2>
    <p>每日巡检说明。</p>
    <table>
      <tr><th>项目</th><th>频率</th></tr>
      <tr><td>备份</td><td>每日</td></tr>
    </table>
    <h3>告警</h3>
    <p>告警处理说明。</p>
  </body>
</html>
"""


def test_markdown_structure():
    doc = parse_document_file("guide.md", MD.encode("utf-8"))
    # 标题取第一个 h1
    assert doc.title == "部署指南"
    paths = [b.headings for b in doc.blocks]
    # 标题路径逐级深入，三级标题完整保留
    assert ["部署指南"] in paths
    assert ["部署指南", "安装"] in paths
    assert ["部署指南", "安装", "注意事项"] in paths
    # 普通段落
    para = next(b for b in doc.blocks if b.text.startswith("本指南介绍"))
    assert not para.is_table and para.page_no is None


def test_markdown_table_block():
    doc = parse_document_file("guide.md", MD.encode("utf-8"))
    tables = [b for b in doc.blocks if b.is_table]
    assert len(tables) == 1
    table = tables[0]
    # text 保留 Markdown 表格表示
    assert "| 步骤 | 命令 |" in table.text
    assert "| 1 | pip install -e . |" in table.text
    assert table.text.count("\n") == 3  # 表头 + 分隔行 + 2 数据行 -> 4 行 3 个换行
    assert table.headings == ["部署指南", "安装"]


def test_markdown_list_and_code_block():
    src = "- 一号事项\n- 二号事项\n\n```python\nprint(1)\n```\n"
    doc = parse_document_file("note.md", src.encode("utf-8"))
    texts = [b.text for b in doc.blocks]
    assert "- 一号事项\n- 二号事项" in texts
    assert any("print(1)" in t for t in texts)


def test_markdown_title_fallback_to_filename():
    doc = parse_document_file("无名文档.md", "没有标题的正文。\n".encode())
    assert doc.title == "无名文档"
    assert doc.blocks[0].headings == []


def test_html_structure():
    doc = parse_document_file("manual.html", HTML.encode("utf-8"))
    # 标题优先取 <title>
    assert doc.title == "运维手册"
    paths = [b.headings for b in doc.blocks]
    assert ["运维手册"] in paths
    assert ["运维手册", "巡检"] in paths
    assert ["运维手册", "巡检", "告警"] in paths
    # 表格块
    tables = [b for b in doc.blocks if b.is_table]
    assert len(tables) == 1
    assert "| 项目 | 频率 |" in tables[0].text
    assert "| 备份 | 每日 |" in tables[0].text
    # 段落块
    paras = [b for b in doc.blocks if not b.is_table]
    assert any(b.text == "系统概述文本。" for b in paras)


def test_plain_text_single_block():
    doc = parse_document_file("制度.txt", "第一条规定。\n第二条规定。\n".encode())
    assert doc.title == "制度"
    assert len(doc.blocks) == 1
    block = doc.blocks[0]
    assert block.headings == [] and not block.is_table and block.page_no is None
    assert "第一条规定。" in block.text and "第二条规定。" in block.text


def test_plain_text_gbk_fallback():
    # 含「文」等字符的 GBK 字节流不是合法 utf-8，应回退 gb18030 解码
    text = "中文文件内容管理规范"
    doc = parse_document_file("legacy.txt", text.encode("gb18030"))
    assert doc.blocks[0].text == text


def test_pdf_dispatch_requires_mineru():
    # 本环境未安装 mineru（可选重依赖），应抛 RuntimeError 并给出安装指引
    with pytest.raises(RuntimeError, match="mineru"):
        parse_document_file("scan.pdf", b"%PDF-1.4 fake bytes")


class _FakeTikaResponse:
    status_code = 200
    text = "Tika 抽取的 Office 正文"

    def raise_for_status(self):
        return None


def test_office_dispatch_tika_success(monkeypatch):
    monkeypatch.setattr(httpx, "put", lambda *a, **k: _FakeTikaResponse())
    doc = parse_document_file("合同.docx", b"PK-zip-bytes")
    assert doc.title == "合同"
    assert len(doc.blocks) == 1
    assert doc.blocks[0].text == "Tika 抽取的 Office 正文"
    assert doc.blocks[0].headings == [] and not doc.blocks[0].is_table


def test_office_dispatch_tika_unreachable(monkeypatch):
    def _boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "put", _boom)
    with pytest.raises(RuntimeError, match="Tika"):
        parse_document_file("报表.xlsx", b"PK-zip-bytes")


def test_unknown_extension_rejected():
    with pytest.raises(ValueError, match="不支持的文件类型"):
        parse_document_file("file.rtf", b"x")
