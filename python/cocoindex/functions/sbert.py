"""SentenceTransformer embedding functionality."""

from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .. import op
from ..typing import Vector


class SentenceTransformerEmbed(op.FunctionSpec):
    """
    `SentenceTransformerEmbed` embeds a text into a vector space using the [SentenceTransformer](https://huggingface.co/sentence-transformers) library.

    Args:

        model: The name of the SentenceTransformer model to use.
        args: Additional arguments to pass to the SentenceTransformer constructor. e.g. {"trust_remote_code": True}

    Note:
        This function requires the optional sentence-transformers dependency.
        Install it with: pip install 'cocoindex[embeddings]'
    """

    model: str
    args: dict[str, Any] | None = None
    flow_name: str = "default"


@op.executor_class(
    gpu=True,
    cache=True,
    batching=True,
    max_batch_size=512,
    behavior_version=1,
    arg_relationship=(op.ArgRelationship.EMBEDDING_ORIGIN_TEXT, "text"),
)
class SentenceTransformerEmbedExecutor:
    """Executor for SentenceTransformerEmbed."""

    spec: SentenceTransformerEmbed
    _model: Any | None = None

    def analyze(self) -> type:
        try:
            # Only import sentence_transformers locally when it's needed, as its import is very slow.
            import sentence_transformers  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                "sentence_transformers is required for SentenceTransformerEmbed function. "
                "Install it with one of these commands:\n"
                "  pip install 'cocoindex[embeddings]'\n"
                "  pip install sentence-transformers"
            ) from e

        args = self.spec.args or {}
        self._model = sentence_transformers.SentenceTransformer(self.spec.model, **args)
        dim = self._model.get_sentence_embedding_dimension()
        return Vector[np.float32, Literal[dim]]  # type: ignore

    def __call__(self, text: list[str]) -> list[NDArray[np.float32]]:
        assert self._model is not None

        # Sort the text by length to minimize the number of padding tokens.
        text_with_idx = [(idx, t) for idx, t in enumerate(text)]
        text_with_idx.sort(key=lambda x: len(x[1]))
        sorted_texts = [t for _, t in text_with_idx]

        # Create progress reporter (file-based IPC for cross-process progress)
        from ..progress import ProgressReporter
        progress = ProgressReporter("embedding", len(text), flow_name=self.spec.flow_name)

        # Process in smaller batches for frequent progress updates and better memory management
        batch_size = min(4, len(sorted_texts))  # Process 4 items at a time to reduce VRAM usage
        all_results: list[NDArray[np.float32]] = []

        for i in range(0, len(sorted_texts), batch_size):
            batch_texts = sorted_texts[i:i + batch_size]
            batch_results = self._model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False  # Disable internal progress bar
            )
            all_results.extend(batch_results)

            # Free intermediate memory immediately
            del batch_results

            # Clear GPU cache after each sub-batch to prevent VRAM buildup
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # torch not available, skip cache clearing

            # Report progress to file (visible to main process)
            progress.update(len(batch_texts), f"batch {i//batch_size + 1}")

        # Restore original order
        final_results: list[NDArray[np.float32] | None] = [
            None for _ in range(len(text))
        ]
        for (idx, _), result in zip(text_with_idx, all_results):
            final_results[idx] = result

        # Report completion (triggers LLM extraction detection for code flow)
        progress.complete("Embedding complete")

        return cast(list[NDArray[np.float32]], final_results)
