"""Record modelling and the *parsing* stage of the pipeline.

The rest of the framework treats a record as a plain mapping (``dict``), so
anything from a survey row to a streaming JSON payload can flow through the
same blocking/scoring machinery.  This module provides:

* :func:`to_record_dict` - coerce dicts, dataclasses and ``to_dict()`` objects
  into the canonical mapping shape used everywhere else;
* :class:`RecordSchema` - the ordered list of fields of interest (plus an
  optional id column) that drives text serialization for embedding and pair
  frame construction;
* :class:`Parser` and a set of concrete parsers (:class:`DictParser`,
  :class:`JsonParser`, :class:`JsonLinesParser`) for the first stage of both
  pipelines; and
* :func:`embed_text` - render a record to the string that is fed to the
  embedding model.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, is_dataclass, asdict
from typing import Any, Hashable, Iterable, Mapping, Optional, Sequence

Record = Mapping[str, Any]
RecordDict = dict[str, Any]


def to_record_dict(value: Any) -> RecordDict:
    """Coerce ``value`` into a plain ``dict[str, Any]`` record.

    Accepts mappings (returned as a copy), dataclass instances (via ``asdict``)
    and objects that expose ``to_dict()`` (used by ``Person``-style models in
    the reference implementation).
    """
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return dict(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, Mapping):
            return dict(out)
    raise TypeError(
        "records must be mappings, dataclasses, or expose to_dict(); got "
        f"{type(value).__name__}"
    )


@dataclass(frozen=True)
class RecordSchema:
    """Declarative description of the fields a pipeline cares about.

    Parameters
    ----------
    fields:
        Ordered names of comparison / embedding fields.  This is the schema
        that the Fellegi-Sunter comparison set and the text serializer consult.
    id_column:
        Optional column holding the record's identity.  Used by the batch
        pipeline to report cluster membership in terms of user-facing ids
        rather than positional indices.
    text_fields:
        Field names included in the embedding text.  Defaults to ``fields``;
        supply explicitly when the embedding text should differ from the set of
        compared fields (e.g. richer context for the vector index).
    """

    fields: tuple[str, ...]
    id_column: Optional[str] = None
    text_fields: Optional[tuple[str, ...]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        if self.text_fields is not None:
            object.__setattr__(self, "text_fields", tuple(self.text_fields))

    @property
    def embedding_fields(self) -> tuple[str, ...]:
        return self.text_fields or self.fields

    def id_of(self, record: Record) -> Optional[Hashable]:
        """Return the record's id from :attr:`id_column` (or ``None``)."""
        if self.id_column is None:
            return None
        return record.get(self.id_column)

    def subset(self, record: Record) -> RecordDict:
        """Return ``record`` limited to the schema fields."""
        return {name: record.get(name) for name in self.fields}


def embed_text(
    record: Record,
    fields: Optional[Sequence[str]] = None,
    template: Optional[str] = None,
) -> str:
    """Render ``record`` to the string that embeddings are computed over.

    When ``template`` is given it is formatted with the record as keyword
    arguments (``"Name: {first_name}"``).  Otherwise each field is emitted as
    ``"<Field>: <value>"`` lines, skipping ``None``/empty values, and missing
    fields are printed as ``None``.  Field order is deterministic (schema
    order), which keeps the embedding text stable across records.
    """
    if template is not None:
        return template.format_map(record)

    names = list(fields) if fields is not None else list(record)
    parts: list[str] = []
    for name in names:
        value = record.get(name)
        if value is None:
            continue
        parts.append(f"{name}: {value}")
    return "\n".join(parts)


class Parser(abc.ABC):
    """Abstract first stage: turn an inbound payload into a record mapping."""

    @abc.abstractmethod
    def parse(self, payload: Any) -> RecordDict:
        """Parse a single payload into a record mapping."""

    def parse_many(self, payloads: Iterable[Any]) -> list[RecordDict]:
        return [self.parse(p) for p in payloads]


class DictParser(Parser):
    """Pass-through parser: ``payload`` must already be a record mapping."""

    def parse(self, payload: Any) -> RecordDict:
        return to_record_dict(payload)


class JsonParser(Parser):
    """Parse a JSON object (string or bytes) into a record mapping."""

    def __init__(self, schema: Optional[RecordSchema] = None) -> None:
        self.schema = schema

    def parse(self, payload: Any) -> RecordDict:
        if isinstance(payload, (str, bytes)):
            payload = json.loads(payload)
        record = to_record_dict(payload)
        return record


class JsonLinesParser(Parser):
    """Parse a stream of newline-delimited JSON objects."""

    def __init__(self, schema: Optional[RecordSchema] = None) -> None:
        self.schema = schema

    def parse(self, payload: Any) -> RecordDict:
        payload = payload.strip() if isinstance(payload, str) else payload
        return json.loads(payload)

    def parse_many(self, payloads: Iterable[Any]) -> list[RecordDict]:
        out: list[RecordDict] = []
        for blob in payloads:
            if isinstance(blob, str):
                for line in blob.splitlines():
                    stripped = line.strip()
                    if stripped:
                        out.append(self.parse(stripped))
            else:
                out.append(self.parse(blob))
        return out