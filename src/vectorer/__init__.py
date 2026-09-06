"""Framework for embedding-and-vector-based entity resolution.

Three pipeline modes are provided:

* :class:`~vectorer.incremental.IncrementalPipeline` - incremental/online ER:
  ``parse -> embed -> vector search blocking (top-k) -> Fellegi-Sunter
  scoring -> classify`` for one incoming record against a reference store;
* :class:`~vectorer.batch.BatchPipeline` - batch/offline ER:
  ``parse -> embed -> canopy blocking -> Fellegi-Sunter scoring of canopy
  pairs -> Swoosh clustering`` over a whole dataset;
* :class:`~vectorer.link.RecordLinker` - two-database record linkage:
  canonical projection of each side -> (directed) index one DB and resolve the
  other, or (symmetric) cross-DB canopy pairs -> Fellegi-Sunter scoring -> link
  edges (never merging the two stores).

The Fellegi-Sunter comparison set (:mod:`vectorer.comparisons`) is extensible
and spans the standard attribute-comparison families of record linkage.
"""

__version__ = "0.5.1"

from .records import (
    DictParser,
    JsonLinesParser,
    JsonParser,
    Parser,
    RecordSchema,
    embed_text,
    to_record_dict,
)
from .embeddings import (
    CharacterHashingEmbedding,
    EmbeddingModel,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)
from .vectorstores import FlatIndex, InMemoryVectorDatabase, IndexingStrategy, VectorDatabase
from .vectorstore_adapters import QdrantVectorDatabase
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
    time_decay_wrapper,
    time_decayed_comparison_builder,
)
from .scoring import DEFAULT_PRIOR, DEFAULT_THRESHOLD, FellegiSunterScorer, WeightTable, import_splink_scorer
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
    latest_merge,
    select_representative,
    union_merge,
)
from .incremental import IncrementalPipeline, Resolution
from .batch import BatchPipeline, BatchResult
from .link import FieldMap, LinkEdge, LinkTable, RecordLinker
from .distributed import (
    RayExecutor,
    build_global_tf_tables,
    create_executor,
    distributed_batch_er,
    distributed_closure,
    distributed_closure_reduce,
    distributed_score_and_reduce,
    distributed_score_pairs,
    hash_pair,
    merge_tf_counters,
    streaming_distributed_closure,
)
from .pins import EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION

__all__ = [
    "__version__",
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
    "OpenAIEmbedding",
    "CharacterHashingEmbedding",
    "EMBEDDING_MODEL_ID",
    "EMBEDDING_MODEL_REVISION",
    # vector stores
    "IndexingStrategy",
    "FlatIndex",
    "VectorDatabase",
    "InMemoryVectorDatabase",
    "QdrantVectorDatabase",
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
    "time_decay_wrapper",
    "time_decayed_comparison_builder",
    # scoring
    "FellegiSunterScorer",
    "WeightTable",
    "import_splink_scorer",
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
    "union_merge",
    "latest_merge",
    # pipelines
    "IncrementalPipeline",
    "Resolution",
    "BatchPipeline",
    "BatchResult",
    # record linkage (two databases)
    "RecordLinker",
    "LinkTable",
    "LinkEdge",
    "FieldMap",
    # distributed batch ER
    "distributed_batch_er",
    "distributed_closure",
    "distributed_closure_reduce",
    "distributed_score_pairs",
    "distributed_score_and_reduce",
    "streaming_distributed_closure",
    "create_executor",
    "RayExecutor",
    "merge_tf_counters",
    "build_global_tf_tables",
    "hash_pair",
]