"""Pinned embedding-model identifiers for reproducible runs.

Kept in one place so every embedding construction site (pipeline defaults,
examples, persistence metadata) references the same model.
"""

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Exact Hugging Face hub commit used for the reported experiments.  Passing it
# as ``revision=`` to the model loader forces the hub to serve the snapshot,
# so re-runs embed with the same weights.
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"