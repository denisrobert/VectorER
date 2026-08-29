"""Tests for the parsing / record-modelling stage."""

from dataclasses import dataclass

import pytest

from vectorer.records import (
    DictParser,
    JsonLinesParser,
    JsonParser,
    RecordSchema,
    embed_text,
    to_record_dict,
)


@dataclass
class Dummy:
    first_name: str
    last_name: str

    def to_dict(self):
        return {"first_name": self.first_name, "last_name": self.last_name}


def test_to_record_dict_from_mapping():
    assert to_record_dict({"a": 1}) == {"a": 1}


def test_to_record_dict_from_dataclass_with_to_dict():
    assert to_record_dict(Dummy("john", "smith")) == {
        "first_name": "john",
        "last_name": "smith",
    }


def test_to_record_dict_rejects_non_records():
    with pytest.raises(TypeError):
        to_record_dict(123)


def test_embed_text_skips_none_and_uses_schema_order():
    record = {"first_name": "john", "last_name": None, "email": "j@x.com"}
    text = embed_text(record, fields=["first_name", "last_name", "email"])
    assert "first_name: john" in text
    assert "last_name:" not in text
    assert "email: j@x.com" in text


def test_embed_text_template():
    record = {"first_name": "john", "last_name": "smith"}
    assert embed_text(record, template="{first_name} {last_name}") == "john smith"


def test_dict_parser_passthrough():
    assert DictParser().parse({"a": 1}) == {"a": 1}


def test_json_parser_string_and_object():
    assert JsonParser().parse('{"a": 1}') == {"a": 1}
    assert JsonParser().parse({"a": 1}) == {"a": 1}


def test_json_lines_parser():
    parser = JsonLinesParser()
    blob = '{"a": 1}\n\n{"a": 2}'
    assert parser.parse_many([blob]) == [{"a": 1}, {"a": 2}]


def test_record_schema_id_and_fields():
    schema = RecordSchema(("first_name", "last_name"), id_column="id")
    record = {"id": 7, "first_name": "john", "last_name": "smith", "z": 1}
    assert schema.id_of(record) == 7
    assert schema.subset(record) == {"first_name": "john", "last_name": "smith"}