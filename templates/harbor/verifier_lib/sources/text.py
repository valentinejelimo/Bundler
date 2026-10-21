from ..models import StrictModel
from ..source_types import SourceCommand, SourceContext
from ._content_limits import cap_with_notice


class TextExtractTextInput(StrictModel):
    """Input for text.extract_text.

    Attributes:
        path: Workspace-relative text path.
    """

    path: str


class TextExtractTextOutput(StrictModel):
    """Output from text.extract_text.

    Attributes:
        text: Raw UTF-8 text.
    """

    text: str


class ExtractText(SourceCommand):
    """Reads any workspace file as raw UTF-8 text.

    This command intentionally does not validate file extension. It is the
    generic plain-text escape hatch for authors. If the target cannot be read as
    UTF-8, the verifier reports that as malformed agent output instead of
    rejecting the source definition.
    """

    name = "extract_text"
    input_model = TextExtractTextInput
    output_model = TextExtractTextOutput

    def run(self, source_input: TextExtractTextInput, context: SourceContext) -> TextExtractTextOutput:
        """Reads UTF-8 text from a workspace file.

        Args:
            source_input: Validated text extraction input.
            context: Source runtime context.

        Returns:
            Raw text capped to the configured source content limit. If the file is longer than
            the cap, a truncation notice is appended so the cut is visible to the grader instead
            of silent.

        """
        resolved = context.resolve_path(source_input.path)
        raw = resolved.read_text(encoding="utf-8")
        return TextExtractTextOutput(text=cap_with_notice(raw, context.max_content_chars))


COMMANDS = (ExtractText(),)
