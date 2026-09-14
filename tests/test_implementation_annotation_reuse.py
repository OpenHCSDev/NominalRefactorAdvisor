"""Annotation syntax reuse must not reuse another declaration's resolved globals."""

import ast
from types import ModuleType

import pytest

from nominal_refactor_advisor.implementation_identity import (
    _annotation_dependencies,
    _annotation_reference_names,
    implementation_module_names,
)
from nominal_refactor_advisor.scan_cache import ScanCache


def function_in(module_name, dependency_name):
    module = ModuleType(module_name)
    exec(
        "class Payload: pass\n"
        "def action(value: 'list[Payload]') -> 'Payload': return value\n",
        vars(module),
    )
    module.Payload.__module__ = dependency_name
    return module


def test_shared_annotation_syntax_keeps_each_functions_own_globals(monkeypatch):
    left = function_in("left", "left_dependency")
    right = function_in("right", "right_dependency")
    original = ast.parse
    parsed = []

    def parse(source, *args, **kwargs):
        parsed.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse)
    with ScanCache.scope():
        left_names = implementation_module_names((left.action,))
        right_names = implementation_module_names((right.action,))
        assert parsed == ["list[Payload]", "Payload"]
    assert "left_dependency" in left_names
    assert "right_dependency" not in left_names
    assert "right_dependency" in right_names
    assert "left_dependency" not in right_names


def test_reused_syntax_does_not_cache_binding_presence_or_value():
    module = function_in("source", "initial_dependency")
    function = module.action
    original = module.Payload
    with ScanCache.scope():
        assert _annotation_dependencies(function, vars(module)) == (original, original)
        del module.Payload
        assert _annotation_dependencies(function, vars(module)) == ()
        replacement = type("Replacement", (), {})
        module.Payload = replacement
        assert _annotation_dependencies(function, vars(module)) == (
            replacement,
            replacement,
        )
        function.__annotations__ = {"value": "Another"}
        module.Another = original
        assert _annotation_dependencies(function, vars(module)) == (original,)


@pytest.mark.parametrize("annotation", ("list[Payload]", "not valid ["))
def test_annotation_syntax_cache_ends_with_the_scan(monkeypatch, annotation):
    original = ast.parse
    parsed = []

    def parse(source, *args, **kwargs):
        parsed.append(source)
        return original(source, *args, **kwargs)

    monkeypatch.setattr(ast, "parse", parse)
    _annotation_reference_names(annotation)
    _annotation_reference_names(annotation)
    assert len(parsed) == 2
    with ScanCache.scope():
        expected = _annotation_reference_names(annotation)
        assert _annotation_reference_names(annotation) == expected
        assert len(parsed) == 3
    with ScanCache.scope():
        assert _annotation_reference_names(annotation) == expected
        assert len(parsed) == 4


def test_string_annotation_discovery_does_not_evaluate_expressions():
    def forbidden():
        raise AssertionError("Annotation discovery must not invoke a source function")

    def action():
        pass

    action.__annotations__ = {"value": "forbidden()"}
    with ScanCache.scope():
        assert _annotation_dependencies(action, {"forbidden": forbidden}) == (
            forbidden,
        )
        assert _annotation_dependencies(action, {}) == ()


def test_live_annotation_objects_remain_original_objects():
    marker = object()

    def action():
        pass

    action.__annotations__ = {"value": marker}
    with ScanCache.scope():
        assert _annotation_dependencies(action, {})[0] is marker
