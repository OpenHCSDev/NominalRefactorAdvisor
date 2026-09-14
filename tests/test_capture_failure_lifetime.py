"""Cached failure evidence preserves causes without retaining execution frames."""

import gc
from weakref import ref

import pytest

from nominal_refactor_advisor.captured_reference import (
    CapturedReferenceViolation,
    OpenCapturedReference,
)


class Payload:
    pass


def with_frame(error):
    payload = Payload()
    witness = ref(payload)
    try:
        raise error
    except BaseException as caught:
        return caught, witness


@pytest.mark.parametrize("link", ("cause", "context", "suppressed", "cycle", "group"))
def test_capture_releases_the_whole_exception_graph_without_replacing_causes(link):
    root, root_frame = with_frame(ValueError("original rejection"))
    inner, inner_frame = with_frame(KeyError("original dependency"))
    witnesses = [root_frame, inner_frame]
    errors = [root, inner]
    if link == "group":
        other, other_frame = with_frame(TypeError("another branch"))
        group, group_frame = with_frame(ExceptionGroup("dependencies", [inner, other]))
        root.__cause__ = group
        witnesses.extend((other_frame, group_frame))
        errors.extend((other, group))
    elif link == "cause":
        root.__cause__ = inner
    elif link == "cycle":
        root.__cause__ = inner
        inner.__context__ = root
    else:
        root.__context__ = inner
        root.__suppress_context__ = link == "suppressed"
    links = [
        (error.__cause__, error.__context__, error.__suppress_context__)
        for error in errors
    ]
    assert all(witness() is not None for witness in witnesses)
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=root
    )
    gc.collect()
    assert all(witness() is None for witness in witnesses)
    assert capture.cause is root
    assert all(error.__traceback__ is None for error in errors)
    for error, (cause, context, suppressed) in zip(errors, links, strict=True):
        assert error.__cause__ is cause
        assert error.__context__ is context
        assert error.__suppress_context__ is suppressed
    with pytest.raises(ValueError) as raised:
        capture.require_closed()
    assert raised.value.__cause__ is root


def test_capture_cleanup_does_not_invoke_an_exception_override():
    class Rejection(ValueError):
        def with_traceback(self, tb):
            raise AssertionError("Cleanup must not invoke an exception override")

    error, witness = with_frame(Rejection("original"))
    capture = OpenCapturedReference(
        CapturedReferenceViolation.UNPROVED_EFFECTS, cause=error
    )
    gc.collect()
    assert witness() is None
    assert capture.cause is error
