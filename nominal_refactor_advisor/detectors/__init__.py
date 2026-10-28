"""Detector registry, substrate, and concrete implementations.

Importing `_implementations` registers every concrete detector; the names
below are the ones code and tests read through this package.
"""

from . import _implementations as _implementations
from ._base import (
    CompactClassRepositoryContext,
    CompactFindingStream,
    CompactModuleProjectionDetectorMixin,
    CompactMultiModuleProjectionDetectorMixin,
    CompactProjectionContextBuilder,
    CompactProjectionGroupContextBuilder,
    CompactProjectionGroups,
    ContextualGlobalCacheContract,
    ContextualModuleIssueDetector,
    CrossModuleCandidateDetector,
    DetectorCacheGranularity,
    DetectorConfig,
    IssueDetector,
    PerModuleIssueDetector,
    SemanticDescentGraphIssueDetector,
    SemanticMirrorIssueDetector,
    SourceLocalIssueDetectorMixin,
    SsotAuthorityBoundaryDetector,
)
from ._carrier_collapse import (
    ClosedParameterConveyorDetector,
    DeclaredCarrierExpansionDetector,
)
from ._class_overlap import (
    DuplicateClassDetector,
    MissingMixinDetector,
)
from ._environment import (
    EnvironmentBooleanAuthorityDriftDetector,
)
from ._record_checks import (
    RedundantTypeCheckDetector,
)
from ._registration_owners import (
    HandRolledRegistrationDetector,
)
from ._relocations import (
    ConstantSubtypeAnswerDetector,
    LambdaFoldDetector,
    OwnStateBranchAnswerDetector,
    ThreadedAbsenceDetector,
    ZeroOneTupleDetector,
)
from ._restated_facts import (
    DeadModuleDetector,
    ExactKeySetDetector,
    ForeignAbsenceProbeDetector,
    ForwardingFacadeDetector,
    LegacyPathMarkerDetector,
    NamedAttributeDetector,
    StringKindDispatchDetector,
)
from ._restated_identity import (
    RestatedIdentityDetector,
)
from ._runtime import (
    ManualClassRegistrationDetector,
    NumericLiteralDispatchDetector,
)
from ._semantic_descent import (
    SemanticMirrorWithoutDescentDetector,
    UnmodeledRecordShapeDetector,
)
from ._structural import (
    ExactLeafMethodAncestorPromotionDetector,
)
from ._surface import (
    default_detectors,
)
from ._systemic import (
    AutoRegisterExplicitPriorityOrderingDetector,
    DeclarativeDetectorClassDetector,
    DirectBuildFindingRendererDetector,
    ExternalEnumCaseRecoveryDetector,
    InheritedAutoRegisterConfigBoilerplateDetector,
    ResidualClosedAxisIndirectionDetector,
)

__all__ = (
    "AutoRegisterExplicitPriorityOrderingDetector",
    "ClosedParameterConveyorDetector",
    "CompactClassRepositoryContext",
    "CompactFindingStream",
    "CompactModuleProjectionDetectorMixin",
    "CompactMultiModuleProjectionDetectorMixin",
    "CompactProjectionContextBuilder",
    "CompactProjectionGroupContextBuilder",
    "CompactProjectionGroups",
    "ContextualGlobalCacheContract",
    "ContextualModuleIssueDetector",
    "CrossModuleCandidateDetector",
    "DeadModuleDetector",
    "DeclarativeDetectorClassDetector",
    "DeclaredCarrierExpansionDetector",
    "default_detectors",
    "DetectorCacheGranularity",
    "DetectorConfig",
    "DirectBuildFindingRendererDetector",
    "DuplicateClassDetector",
    "EnvironmentBooleanAuthorityDriftDetector",
    "ExactKeySetDetector",
    "HandRolledRegistrationDetector",
    "ExactLeafMethodAncestorPromotionDetector",
    "ExternalEnumCaseRecoveryDetector",
    "ForeignAbsenceProbeDetector",
    "ForwardingFacadeDetector",
    "InheritedAutoRegisterConfigBoilerplateDetector",
    "IssueDetector",
    "LegacyPathMarkerDetector",
    "ManualClassRegistrationDetector",
    "MissingMixinDetector",
    "NamedAttributeDetector",
    "NumericLiteralDispatchDetector",
    "PerModuleIssueDetector",
    "RedundantTypeCheckDetector",
    "ResidualClosedAxisIndirectionDetector",
    "RestatedIdentityDetector",
    "ConstantSubtypeAnswerDetector",
    "LambdaFoldDetector",
    "OwnStateBranchAnswerDetector",
    "ThreadedAbsenceDetector",
    "ZeroOneTupleDetector",
    "SemanticDescentGraphIssueDetector",
    "SemanticMirrorIssueDetector",
    "SemanticMirrorWithoutDescentDetector",
    "SourceLocalIssueDetectorMixin",
    "SsotAuthorityBoundaryDetector",
    "StringKindDispatchDetector",
    "UnmodeledRecordShapeDetector",
)
