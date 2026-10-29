"""Detector registry, substrate, and concrete implementations.

Importing `_implementations` registers every concrete detector; the names
below are the ones code and tests read through this package.
"""

from .. import export_tools as _export_tools
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
from ._false_distinctions import (
    IdenticalArmsDetector,
    NameOnlySubclassDetector,
    UndistinguishedEnumMembersDetector,
    UndistinguishedTypeTestDetector,
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
    OwnTypeTestDetector,
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

__all__ = _export_tools.derive_public_exports(
    globals(),
    _export_tools.PublicExportPolicy(
        module_name=__name__,
        types_only=False,
        allow_callables=True,
        include_enums=True,
        package_reexports=True,
    ),
)
