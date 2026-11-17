"""The evidence kinds a class-overlap verdict weighs, below the class index that cites them."""

from __future__ import annotations

from nominal_refactor_advisor.receipts import ReceiptKind
from nominal_refactor_advisor.needed_input import NeededInput


class OverlapReceiptKind(ReceiptKind):
    """One kind of evidence: a concept receipt with its fixed log-odds weight, or a
    move guard that names the input the move needs and weighs nothing in the
    concept score."""

    SHARED_METHOD = "shared_method", 1.5, False
    SHARED_VALUE = "shared_value", 1.0, False
    SHARED_FIELD = "shared_field", 0.5, False
    EXISTING_OWNER = "existing_owner", 1.0, False
    FAMILY_AXIS_VALUE = "family_axis_value", 0.0, False
    FEW_DECLARATIONS = "few_declarations", -1.5, False
    KEPT_DIFFERENCE = "kept_difference", 0.0, False
    SHARED_CONSUMER = "shared_consumer", 1.0, False
    NO_SHARED_CONSUMER = "no_shared_consumer", 0.0, False
    SAME_FAMILY = "same_family", 0.5, False
    SEPARATE_FAMILIES = "separate_families", 0.0, False
    SAME_CONSTRUCTION = "same_construction", 0.0, False
    DIFFERENT_CONSTRUCTION = "different_construction", 0.0, True, NeededInput.SOURCE_CHANGE
    IDENTITY_USED = "identity_used", 0.0, True, NeededInput.SOURCE_CHANGE
    IDENTITY_DECLARED = "identity_declared", 0.0, True, NeededInput.PROOF
    POSITIONAL_CONSTRUCTION = "positional_construction", 0.0, True, NeededInput.SOURCE_CHANGE
    INHERITED_DECLARATION = "inherited_declaration", 0.0, True, NeededInput.OPERATION
    UNWANTED_SUPPLY = "unwanted_supply", 0.0, True, NeededInput.OPERATION
    DECLARATION_DIFFERS = "declaration_differs", 0.0, True, NeededInput.SOURCE_CHANGE
