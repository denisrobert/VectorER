"""Framework for embedding-and-vector-based entity resolution.

Two pipelines are provided:

* :class:`~vectorer.incremental.IncrementalPipeline` - incremental/online ER:
  ``parse -> embed -> vector search blocking (top-k) -> Fellegi-Sunter
  scoring -> classify`` for one incoming record against a reference store;
* :class:`~vectorer.batch.BatchPipeline` - batch/offline ER:
  ``parse -> embed -> canopy blocking -> Fellegi-Sunter scoring of canopy
  pairs -> Swoosh clustering`` over a whole dataset.

The Fellegi-Sunter comparison set (:mod:`vectorer.comparisons`) is extensible
and covers every comparison option currently available in Splink.
"""

from .records import (
    DictParser,
    JsonLinesParser,
    JsonParser,
    Parser,
    RecordSchema,
    embed_text,
    to_record_dict,
)
from .embeddings import CharacterHashingEmbedding, EmbeddingModel, SentenceTransformerEmbedding
from .vectorstores import FlatIndex, InMemoryVectorDatabase, IndexingStrategy, VectorDatabase
from .blocking import BlockedCandidate, CanopyIndex, VectorBlocker, canopy_blocking
from .comparisons import (
    Comparison,
    ComparisonSpec,
    ComparisonRegistry,
    PairValues,
    REGISTRY,
    Level,
    available_comparisons,
    comparison_catalog,
    comparison_fields,
    comparison_set,
    comparison_to_dict,
    make_comparison,
    make_comparisons,
    register_comparison,
)
from .scoring import DEFAULT_PRIOR, DEFAULT_THRESHOLD, FellegiSunterScorer, WeightTable
from .classification import (
    Classifier,
    Decision,
    MatchResult,
    ScoredCandidate,
    ThresholdClassifier,
)
from .clustering import (
    Cluster,
    ClusterAssignment,
    ScoredPair,
    SwooshClusterer,
    connected_components,
    gswoosh,
    select_representative,
)
from .incremental import IncrementalPipeline, Resolution
from .batch import BatchPipeline, BatchResult
from .pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION

__all__ = [
    # records / parsing
    "RecordSchema",
    "Parser",
    "DictParser",
    "JsonParser",
    "JsonLinesParser",
    "embed_text",
    "to_record_dict",
    # embeddings
    "EmbeddingModel",
    "SentenceTransformerEmbedding",
    "CharacterHashingEmbedding",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_MODEL_REVISION",
    # vector stores
    "IndexingStrategy",
    "FlatIndex",
    "VectorDatabase",
    "InMemoryVectorDatabase",
    # blocking
    "BlockedCandidate",
    "VectorBlocker",
    "CanopyIndex",
    "canopy_blocking",
    # comparisons
    "Comparison",
    "ComparisonSpec",
    "ComparisonRegistry",
    "REGISTRY",
    "Level",
    "PairValues",
    "make_comparison",
    "make_comparisons",
    "comparison_to_dict",
    "comparison_fields",
    "comparison_set",
    "available_comparisons",
    "comparison_catalog",
    "register_comparison",
    # scoring
    "FellegiSunterScorer",
    "WeightTable",
    "DEFAULT_PRIOR",
    "DEFAULT_THRESHOLD",
    # classification
    "Decision",
    "Classifier",
    "ThresholdClassifier",
    "MatchResult",
    "ScoredCandidate",
    # clustering (Swoosh)
    "ScoredPair",
    "Cluster",
    "ClusterAssignment",
    "SwooshClusterer",
    "gswoosh",
    "connected_components",
    "select_representative",
    # pipelines
    "IncrementalPipeline",
    "Resolution",
    "BatchPipeline",
    "BatchResult",
]