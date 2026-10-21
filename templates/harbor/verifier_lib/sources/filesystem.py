from pathlib import Path

from ..models import StrictModel
from ..source_types import SourceCommand, SourceContext


class FilesystemCheckPathExistsInput(StrictModel):
    """Input for filesystem.check_path_exists.

    Attributes:
        path: Workspace-relative path to check.
    """

    path: str


class FilesystemCheckPathExistsOutput(StrictModel):
    """Output from filesystem.check_path_exists.

    Attributes:
        exists: Whether the path exists.
        is_file: Whether the path exists and is a regular file.
        is_directory: Whether the path exists and is a directory.
    """

    exists: bool
    is_file: bool
    is_directory: bool


class CheckPathExists(SourceCommand):
    """Checks whether a workspace path exists."""

    name = "check_path_exists"
    input_model = FilesystemCheckPathExistsInput
    output_model = FilesystemCheckPathExistsOutput

    def run(self, source_input: FilesystemCheckPathExistsInput, context: SourceContext) -> FilesystemCheckPathExistsOutput:
        """Runs the path existence check.

        Args:
            source_input: Validated path check input.
            context: Source runtime context.

        Returns:
            Path existence and type output.
        """
        resolved = context.resolve_path(source_input.path)
        return FilesystemCheckPathExistsOutput(
            exists=resolved.exists(),
            is_file=resolved.is_file(),
            is_directory=resolved.is_dir(),
        )


class FilesystemExtractManyInput(StrictModel):
    """Input for extracting several heterogeneous files into one source."""

    paths: list[str]


class FilesystemExtractManyOutput(StrictModel):
    """Labeled text from every requested file, under one aggregate cap."""

    text: str
    paths: list[str]


def _hit_cap(text: str, limit: int) -> bool:
    return len(text) >= limit or "[TRUNCATED:" in text[-400:]


def _extract_one(resolved: Path, limit: int) -> str:
    """Text of one file under ``limit`` characters, by format."""
    suffix = resolved.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        from .xlsx import extract_xlsx

        return extract_xlsx(resolved, limit)
    if suffix == ".docx":
        from .docx import extract_docx

        return extract_docx(resolved, limit)
    if suffix == ".pdf":
        from .pdf import extract_pdf_text

        return extract_pdf_text(resolved, limit)
    if suffix == ".pptx":
        from .pptx import extract_pptx_text

        return extract_pptx_text(resolved, limit)
    from ._content_limits import cap_with_notice

    return cap_with_notice(resolved.read_text(encoding="utf-8", errors="replace"), limit)


class ExtractMany(SourceCommand):
    """Extracts DOCX, XLSX/XLSM, PDF and PPTX files as one labeled source."""

    name = "extract_many"
    input_model = FilesystemExtractManyInput
    output_model = FilesystemExtractManyOutput

    def run(
        self, source_input: FilesystemExtractManyInput, context: SourceContext
    ) -> FilesystemExtractManyOutput:
        if not source_input.paths:
            return FilesystemExtractManyOutput(text="", paths=[])

        per_file_limit = max(1, context.max_content_chars // len(source_input.paths))
        resolved_paths = [context.resolve_path(p) for p in source_input.paths]
        extracted = [_extract_one(path, per_file_limit) for path in resolved_paths]

        # Even split is the FLOOR (no file ever gets less than before). Budget that short files
        # did not use is handed to the files that hit their cap, so a small memo beside a large
        # workbook no longer truncates the workbook while most of the budget sits unused.
        overflowed = [i for i, text in enumerate(extracted) if _hit_cap(text, per_file_limit)]
        if overflowed:
            spare = sum(per_file_limit - len(t) for i, t in enumerate(extracted) if i not in overflowed)
            if spare > 0:
                bonus = spare // len(overflowed)
                for i in overflowed:
                    extracted[i] = _extract_one(resolved_paths[i], per_file_limit + bonus)

        parts = [f"[File: {rel}]\n{text}" for rel, text in zip(source_input.paths, extracted)]

        return FilesystemExtractManyOutput(
            text="\n\n".join(parts)[: context.max_content_chars],
            paths=list(source_input.paths),
        )


COMMANDS = (CheckPathExists(), ExtractMany())
