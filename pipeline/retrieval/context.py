"""The trust boundary between retrieved chunks, model text and validation.

``model_text`` is untrusted presentation: document text can contain anything,
including strings that look exactly like passage handles or citation headers.
``passages`` is provenance created directly from the retrieved chunk records.
Validation must use the latter and must never reconstruct it by parsing the
former.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """One retrieved passage and the provenance assigned by the pipeline."""

    handle: int
    page: int | None
    text: str
    citation: str


@dataclass(frozen=True)
class RagContext:
    """One indivisible model-input/provenance pair.

    Keeping both representations in the same value prevents a caller from
    validating an answer against a context other than the one shown to the
    model.  The tuple and frozen records also stop later mutation from changing
    what a handle means after generation.
    """

    passages: tuple[Passage, ...]
    numbered: bool

    def __post_init__(self):
        if type(self.passages) is not tuple:
            raise TypeError("passages must be an immutable tuple")
        if type(self.numbered) is not bool:
            raise TypeError("numbered must be a boolean")
        if any(not isinstance(passage, Passage) for passage in self.passages):
            raise TypeError("passages must contain Passage records")

        handles = [passage.handle for passage in self.passages]
        if any(type(handle) is not int or handle < 1 for handle in handles):
            raise ValueError("passage handles must be positive integers")
        if len(handles) != len(set(handles)):
            raise ValueError("passage handles must be unique")
        if any(not isinstance(passage.text, str) for passage in self.passages):
            raise TypeError("passage text must be a string")
        if any(not isinstance(passage.citation, str) for passage in self.passages):
            raise TypeError("passage citations must be strings")
        if any(
            passage.page is not None
            and (type(passage.page) is not int or passage.page < 1)
            for passage in self.passages
        ):
            raise ValueError("passage pages must be positive integers or None")

    @property
    def model_text(self) -> str:
        """Render the model view; provenance is never reconstructed from it."""
        blocks = []
        for passage in self.passages:
            head = (
                f"[P{passage.handle}] {passage.citation}"
                if self.numbered
                else passage.citation
            )
            blocks.append(f"{head}\n{passage.text}")
        return "\n\n---\n\n".join(blocks)

    def by_handle(self) -> dict[int, Passage]:
        return {passage.handle: passage for passage in self.passages}
