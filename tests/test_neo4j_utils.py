"""Unit tests for the pure Neo4j sanitization helpers."""
import neo4j_utils


class TestSanitizeRelType:
    def test_spaces_and_symbols_become_underscored_upper(self):
        assert neo4j_utils.sanitize_rel_type("is related to!!") == "IS_RELATED_TO"

    def test_empty_and_none_default_to_related_to(self):
        assert neo4j_utils.sanitize_rel_type("") == "RELATED_TO"
        assert neo4j_utils.sanitize_rel_type(None) == "RELATED_TO"

    def test_leading_digit_defaults_to_related_to(self):
        assert neo4j_utils.sanitize_rel_type("123") == "RELATED_TO"

    def test_collapses_repeated_underscores(self):
        assert neo4j_utils.sanitize_rel_type("a---b") == "A_B"

    def test_truncated_to_50_chars(self):
        assert len(neo4j_utils.sanitize_rel_type("x" * 80)) == 50


class TestSanitizeNodeLabel:
    def test_symbols_become_underscored(self):
        assert neo4j_utils.sanitize_node_label("Foo-Bar Baz") == "Foo_Bar_Baz"

    def test_empty_and_none_default_to_entity(self):
        assert neo4j_utils.sanitize_node_label("") == "Entity"
        assert neo4j_utils.sanitize_node_label(None) == "Entity"

    def test_leading_digit_is_prefixed(self):
        assert neo4j_utils.sanitize_node_label("123 Foo").startswith("Entity_")

    def test_truncated_to_50_chars(self):
        assert len(neo4j_utils.sanitize_node_label("y" * 80)) == 50


class TestCleanProps:
    def test_primitives_pass_through(self):
        out = neo4j_utils._clean_props({"a": 1, "b": "x", "c": 1.5, "d": True})
        assert out == {"a": 1, "b": "x", "c": 1.5, "d": True}

    def test_non_primitives_are_stringified(self):
        out = neo4j_utils._clean_props({"lst": [1, 2], "dct": {"k": "v"}})
        assert out["lst"] == "[1, 2]"
        assert out["dct"] == "{'k': 'v'}"


def test_insert_and_clear_return_false_without_driver():
    assert neo4j_utils.insert_graph_to_neo4j(None, [], []) is False
    assert neo4j_utils.clear_neo4j_database(None) is False
