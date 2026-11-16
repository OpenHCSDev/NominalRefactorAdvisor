"""A check that stops a fix names the kind of input it needs, and that kind
survives every wrapper between the check and the synthesis record."""

from __future__ import annotations

from nominal_refactor_advisor.needed_input import NeededInput, NeedsInput


def _wrapped(error: BaseException) -> ValueError:
    """`error` as a caller sees it: raised again inside a broader failure."""

    try:
        try:
            raise error
        except ValueError as cause:
            raise ValueError("the operation's preflight stopped") from cause
    except ValueError as wrapped:
        return wrapped


def test_a_wrapped_check_still_names_its_need() -> None:
    error = _wrapped(NeedsInput("a picker must name the owner", NeededInput.OWNER_PICK))

    assert NeedsInput.needs_of(error) == (NeededInput.OWNER_PICK,)


def test_a_free_text_check_reads_as_untyped() -> None:
    assert NeedsInput.needs_of(_wrapped(ValueError("no reason typed"))) == (
        NeededInput.UNTYPED,
    )


def test_each_kind_appears_once_in_declaration_order() -> None:
    assert NeededInput.distinct(
        (NeededInput.OPERATION, NeededInput.OWNER_PICK, NeededInput.OPERATION)
    ) == (NeededInput.OWNER_PICK, NeededInput.OPERATION)

