from pathlib import Path

from pydantic import field_validator

from ..models import StrictModel
from ..source_types import SourceCommand, SourceContext
from ._content_limits import finalize_accumulated


class DocxExtractTextInput(StrictModel):
    """Input for docx.extract_text.

    Attributes:
        path: Workspace-relative DOCX path.
    """

    path: str

    @field_validator("path")
    @classmethod
    def require_docx_extension(cls, value: str) -> str:
        """Requires the path to target a DOCX file.

        Args:
            value: Workspace-relative path from verifier.json.

        Returns:
            The unchanged path when it has a DOCX extension.

        Raises:
            ValueError: If the path does not end in ``.docx``.
        """
        if Path(value).suffix.lower() != ".docx":
            raise ValueError("docx.extract_text requires a .docx file")
        return value


class DocxExtractTextOutput(StrictModel):
    """Output from docx.extract_text.

    Attributes:
        text: Extracted body paragraph and table text.
    """

    text: str


class ExtractText(SourceCommand):
    """Extracts text from DOCX body paragraphs and tables."""

    name = "extract_text"
    input_model = DocxExtractTextInput
    output_model = DocxExtractTextOutput

    def run(self, source_input: DocxExtractTextInput, context: SourceContext) -> DocxExtractTextOutput:
        """Runs DOCX text extraction.

        Args:
            source_input: Validated DOCX extraction input.
            context: Source runtime context.

        Returns:
            Extracted DOCX text.

        """
        resolved = context.resolve_path(source_input.path)
        return DocxExtractTextOutput(text=extract_docx(resolved, context.max_content_chars))


def _sdt_tags():
    """Returns the qualified XML tags used to detect and unwrap content controls.

    Returns:
        A tuple of ``(p, tbl, tr, tc, r, hyperlink, sdt, sdtContent)`` qualified tags.
    """
    from docx.oxml.ns import qn

    return (
        qn("w:p"),
        qn("w:tbl"),
        qn("w:tr"),
        qn("w:tc"),
        qn("w:r"),
        qn("w:hyperlink"),
        qn("w:sdt"),
        qn("w:sdtContent"),
    )


def _inline_text(element) -> str:
    """Extracts the visible text of a single run, hyperlink, or content control.

    ``<w:sdt>`` (structured document tag) elements implement Word content
    controls such as rich-text or plain-text placeholders, dropdowns, and date
    pickers. Their actual text lives one level deeper, inside
    ``<w:sdtContent>``. python-docx's own run/hyperlink/paragraph ``.text``
    properties only look at direct ``w:r``/``w:hyperlink`` children, so any
    text wrapped in an inline ``w:sdt`` is silently skipped. This recurses into
    ``w:sdtContent`` (and any further-nested ``w:sdt``) so that text is no
    longer invisible to a grader.

    Args:
        element: A ``w:r``, ``w:hyperlink``, or ``w:sdt`` element.

    Returns:
        The element's text, or an empty string for anything else.
    """
    p_tag, tbl_tag, tr_tag, tc_tag, r_tag, hyperlink_tag, sdt_tag, sdt_content_tag = _sdt_tags()

    tag = element.tag
    if tag == r_tag:
        return element.text or ""
    if tag == hyperlink_tag:
        return "".join(_inline_text(child) for child in element if child.tag in (r_tag, sdt_tag))
    if tag == sdt_tag:
        content = element.find(sdt_content_tag)
        if content is None:
            return ""
        return "".join(_inline_text(child) for child in content if child.tag in (r_tag, hyperlink_tag, sdt_tag))
    return ""


def _paragraph_text(p_element) -> str:
    """Returns a paragraph's full text, including text inside inline ``w:sdt`` wrappers.

    Args:
        p_element: A ``w:p`` element.

    Returns:
        The paragraph's text, equivalent to python-docx's ``Paragraph.text``
        except it also surfaces text nested inside content controls.
    """
    _p_tag, _tbl_tag, _tr_tag, _tc_tag, r_tag, hyperlink_tag, sdt_tag, _sdt_content_tag = _sdt_tags()
    return "".join(_inline_text(child) for child in p_element if child.tag in (r_tag, hyperlink_tag, sdt_tag))


def _iter_block_elements(container_element):
    """Yields ``w:p``/``w:tbl`` elements under ``container_element``, in document order.

    Mirrors python-docx's own ``paragraphs``/``tables`` accessors (which only
    look at direct children) but additionally descends into ``<w:sdt>``
    (content control) wrappers, whose ``<w:sdtContent>`` child can itself
    contain paragraphs, tables, or further-nested ``<w:sdt>`` elements.
    Without this, a whole paragraph or table wrapped in a content control is
    silently skipped, even though Word renders it normally.

    Args:
        container_element: The lxml element whose children to walk (e.g. a
            ``CT_Body`` or a ``w:tc`` cell element).

    Yields:
        ``w:p`` and ``w:tbl`` elements in document order.
    """
    p_tag, tbl_tag, _tr_tag, _tc_tag, _r_tag, _hyperlink_tag, sdt_tag, sdt_content_tag = _sdt_tags()

    for child in container_element:
        tag = child.tag
        if tag in (p_tag, tbl_tag):
            yield child
        elif tag == sdt_tag:
            content = child.find(sdt_content_tag)
            if content is not None:
                yield from _iter_block_elements(content)


def _cell_text(tc_element) -> str:
    """Returns a table cell's text, one paragraph per line.

    Args:
        tc_element: A ``w:tc`` cell element.

    Returns:
        Newline-joined paragraph text, equivalent to python-docx's
        ``_Cell.text`` except it also surfaces text nested inside content
        controls (block-level or inline).
    """
    return "\n".join(_paragraph_text(p) for p in _iter_block_elements(tc_element))


def extract_docx(path: Path, limit: int) -> str:
    """Extracts body paragraph and table text from a DOCX file.

    Args:
        path: Resolved DOCX file path.
        limit: Maximum number of characters to return.

    Returns:
        Extracted text capped to ``limit`` characters. If the document is longer than the cap,
        a truncation notice is appended so the cut is visible to the grader instead of silent.
    """
    from docx import Document
    from docx.table import Table

    p_tag, tbl_tag, *_ = _sdt_tags()

    doc = Document(path)
    blocks = list(_iter_block_elements(doc.element.body))
    paragraphs = [block for block in blocks if block.tag == p_tag]
    tables = [Table(block, doc._body) for block in blocks if block.tag == tbl_tag]

    parts: list[str] = []
    total = 0
    truncated = False
    for p_element in paragraphs:
        text = _paragraph_text(p_element)
        if text.strip():
            total = append_limited(parts, text, total, limit)
            if total >= limit:
                truncated = True
                break
    if not truncated:
        for table in tables:
            # Use python-docx's row/cell iteration (it already de-duplicates merged
            # cells correctly), but compute each cell's text ourselves so text
            # nested inside content controls within the cell is not lost.
            for row in table.rows:
                cells = [text for cell in row.cells if (text := _cell_text(cell._element).strip())]
                if cells:
                    total = append_limited(parts, " | ".join(cells), total, limit)
                    if total >= limit:
                        truncated = True
                        break
            if truncated:
                break
    return finalize_accumulated(parts, limit, truncated)


def append_limited(parts: list[str], text: str, total: int, limit: int) -> int:
    """Appends text and returns the updated character count.

    Args:
        parts: Text fragments collected so far.
        text: Text fragment to append.
        total: Current approximate character count.
        limit: Maximum desired character count.

    Returns:
        Updated approximate character count.
    """
    parts.append(text)
    return total + len(text) + 1


COMMANDS = (ExtractText(),)
