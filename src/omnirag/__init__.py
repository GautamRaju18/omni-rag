"""OmniRAG: multimodal graph-based retrieval augmented generation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("omnirag")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"
