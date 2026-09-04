"""Source-backed context for codemod selection and authority resolution."""

from __future__ import annotations

import ast
from abc import (
    ABC,
    abstractmethod,
)
from collections.abc import (
    Iterable,
    Mapping,
)
from dataclasses import (
    dataclass,
    field,
)
from functools import cached_property
from typing import TYPE_CHECKING, Self

from .assignment_projection import AssignmentStatementNameProjection
from .ast_tools import (
    AstKeywordSourceProjection,
    ParsedModule,
    SourceModule,
)
from .class_index import (
    ClassFamilyIndex,
    ModuleClassReferenceResolver,
)
from .codemod_declaration_source import DirectClassDeclarationAuthority
from .codemod_import_graph import SourceModuleImportGraph
from .codemod_paths import (
    SourcePathCandidateSet,
    SourcePathResolutionAuthority,
)
from .codemod_selector_models import SourceRewriteTarget
from .models import SourceLocation
from .source_geometry import SourceLineSegmentAuthority
from .source_index import (
    AstTargetDigest,
    AstTargetNode,
    AstTargetNodeIndex,
    AstTargetNodeKind,
    IndexedSourceAuthority,
)

if TYPE_CHECKING:
    from .codemod_runtime import CodemodSourceSnapshot


@dataclass(frozen=True)
class CodemodSelectorContext(IndexedSourceAuthority, ABC):
    """Shared semantic selection context for recipe synthesis."""

    class_family_index: ClassFamilyIndex | None = None
    module_node_cache: Mapping[str, ast.Module] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    ast_target_node_cache: Mapping[str, AstTargetNode] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    module_import_graph_cache: SourceModuleImportGraph | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _direct_class_declaration_indexes_by_file_path: dict[
        str, "ClassDirectDeclarationIndex"
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _class_reference_resolvers_by_file_path: dict[str, ModuleClassReferenceResolver] = (
        field(default_factory=dict, init=False, repr=False, compare=False)
    )
    _parsed_modules_by_file_path: dict[str, ParsedModule] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    @cached_property
    def source_file_paths(self) -> tuple[str, ...]:
        return self.source_index.target_file_paths

    def resolve_source_paths(self, file_paths: Iterable[str]) -> frozenset[str]:
        return frozenset(
            SourcePathResolutionAuthority(
                requested_path=file_path,
                candidate_set=SourcePathCandidateSet.from_paths(self.source_file_paths),
            ).required_path()
            for file_path in file_paths
        )

    @abstractmethod
    def execution_snapshot(self) -> "CodemodSourceSnapshot":
        """Return the concrete source authority used to execute rewrites."""

        raise NotImplementedError

    @property
    def required_class_family_index(self) -> ClassFamilyIndex:
        if self.class_family_index is None:
            raise ValueError("Class-family selector requires ClassFamilyIndex")
        return self.class_family_index

    @cached_property
    def ast_target_nodes_by_id(
        self,
    ) -> Mapping[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
        if self.ast_target_node_cache is not None:
            return self.ast_target_node_cache
        return AstTargetNodeIndex.from_source_mapping(
            self.source_index,
            self.sources_by_file_path,
        ).nodes_by_target_id

    @cached_property
    def module_nodes_by_file_path(self) -> Mapping[str, ast.Module]:
        if self.module_node_cache is not None:
            return self.module_node_cache
        return {
            file_path: ast.parse(source, filename=file_path)
            for file_path, source in self.sources_by_file_path.items()
        }

    @cached_property
    def module_import_graph(self) -> SourceModuleImportGraph:
        if self.module_import_graph_cache is not None:
            return self.module_import_graph_cache
        return SourceModuleImportGraph(
            source_index=self.source_index,
            module_nodes_by_file_path=self.module_nodes_by_file_path,
        )

    def direct_class_declaration_index_for_file(
        self,
        file_path: str,
    ) -> "ClassDirectDeclarationIndex":
        cache = self._direct_class_declaration_indexes_by_file_path
        if file_path not in cache:
            cache[file_path] = ClassDirectDeclarationIndex.from_context_file(
                self,
                file_path,
            )
        return cache[file_path]

    def module_node_for_source_path(self, source_path: str) -> ast.Module | None:
        resolved_path = SourcePathResolutionAuthority.from_source_index(
            source_path,
            self.source_index,
        ).optional_path()
        if resolved_path is None:
            return None
        return self.module_nodes_by_file_path.get(resolved_path)

    def parsed_module_for_source_path(self, source_path: str) -> ParsedModule:
        """Resolve one current module with its canonical source identity."""

        source_file = self.module_import_graph.source_file_for_path(source_path)
        if source_file is None:
            raise ValueError(f"Source module {source_path!r} is unavailable")
        cache = self._parsed_modules_by_file_path
        if source_file.file_path in cache:
            return cache[source_file.file_path]
        module = self.module_nodes_by_file_path.get(source_file.file_path)
        source = self.sources_by_file_path.get(source_file.file_path)
        if module is None or source is None:
            raise ValueError(f"Source module {source_path!r} is unavailable")
        cache[source_file.file_path] = SourceModule.from_path_identity(
            source_file.module_path_identity,
            source,
        ).parsed_module(
            module,
        )
        return cache[source_file.file_path]

    def class_reference_resolver_for_source_path(
        self,
        source_path: str,
    ) -> ModuleClassReferenceResolver:
        """Resolve class expressions against the current nominal class index."""

        parsed_module = self.parsed_module_for_source_path(source_path)
        cache = self._class_reference_resolvers_by_file_path
        if parsed_module.file_path not in cache:
            cache[parsed_module.file_path] = ModuleClassReferenceResolver(
                parsed_module,
                self.required_class_family_index,
            )
        return cache[parsed_module.file_path]

    def module_assignment_statement(
        self,
        source_path: str,
        assignment_name: str,
    ) -> ast.Assign | ast.AnnAssign | None:
        module = self.module_node_for_source_path(source_path)
        if module is None:
            return None
        matching_statements = tuple(
            statement
            for statement in module.body
            if assignment_name in AssignmentStatementNameProjection(statement).names
        )
        if len(matching_statements) != 1:
            return None
        statement = matching_statements[0]
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            return statement
        return None

    def target_node_for_rewrite_target(
        self,
        target: SourceRewriteTarget,
    ) -> tuple[str, AstTargetDigest, AstTargetNode]:
        target_identifier = target.required_target_id(self.source_index)
        node = self.ast_target_nodes_by_id.get(target_identifier)
        if node is None:
            raise ValueError(
                f"Exact source target {target_identifier!r} is absent from current "
                "source"
            )
        return (
            target_identifier,
            self.source_index.target_by_id[target_identifier],
            node,
        )

    def required_class_target_for_authority_evidence(
        self,
        evidence: SourceLocation,
    ) -> AstTargetDigest:
        """Resolve a class authority by declaration identity, not stale geometry."""

        return self.required_target_for_evidence(
            evidence,
            node_kind=AstTargetNodeKind.CLASS,
        )

    def required_target_for_evidence(
        self,
        evidence: SourceLocation,
        *,
        node_kind: AstTargetNodeKind,
    ) -> AstTargetDigest:
        """Resolve one exact source target from repository-symbol evidence."""

        source_paths = self.resolve_source_paths((evidence.file_path,))
        targets = tuple(
            target
            for target in self.source_index.targets_matching_repository_symbol(
                evidence.symbol
            )
            if target.node_kind is node_kind and target.file_path in source_paths
        )
        if len(targets) != 1:
            raise ValueError(
                f"{node_kind.value} evidence {evidence.symbol!r} resolves to "
                f"{len(targets)} source targets"
            )
        return targets[0]


@dataclass(frozen=True)
class ResolvedClassTarget:
    """Resolved source-index target paired with its class AST node."""

    target: AstTargetDigest
    node: ast.ClassDef

    @classmethod
    def from_rewrite_target(
        cls,
        context: CodemodSelectorContext,
        target_reference: SourceRewriteTarget,
    ) -> Self:
        """Resolve one exact class identity from a recipe target."""

        _target_id, target, node = context.target_node_for_rewrite_target(
            target_reference
        )
        if not target.is_class or not isinstance(node, ast.ClassDef):
            raise ValueError("Source rewrite target must identify one class")
        return cls(target=target, node=node)

    @property
    def file_path(self) -> str:
        return self.target.file_path

    @property
    def qualname(self) -> str:
        return self.target.qualname

    @property
    def name(self) -> str:
        return self.target.name

    @property
    def line(self) -> int:
        return self.target.line

    def symbol(self, context: CodemodSelectorContext) -> str | None:
        """Project this resolved source class into the repository class graph."""

        return context.required_class_family_index.symbol_for(
            file_path=self.file_path,
            qualname=self.qualname,
        )

    def required_symbol(self, context: CodemodSelectorContext) -> str:
        symbol = self.symbol(context)
        if symbol is None:
            raise ValueError(f"Class {self.qualname!r} is absent from the family index")
        return symbol

    @property
    def dataclass_argument_sources(self) -> tuple[str, ...] | None:
        for decorator in self.node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Name):
                target_name = target.id
            elif isinstance(target, ast.Attribute):
                target_name = target.attr
            else:
                continue
            if target_name != "dataclass":
                continue
            if not isinstance(decorator, ast.Call):
                return ()
            return (
                *(ast.unparse(argument) for argument in decorator.args),
                *(
                    AstKeywordSourceProjection(keyword).source()
                    for keyword in decorator.keywords
                ),
            )
        return None


@dataclass(frozen=True)
class ClassDirectDeclarationIndex:
    """Direct class field declarations keyed by source-index target id."""

    declarations_by_target_id: Mapping[str, Mapping[str, str]]

    @classmethod
    def from_context_file(
        cls,
        context: CodemodSelectorContext,
        file_path: str,
    ) -> "ClassDirectDeclarationIndex":
        targets_by_file = context.source_index.targets_by_file
        if not targets_by_file.contains_file(file_path):
            return cls(declarations_by_target_id={})
        source = context.sources_by_file_path.get(file_path)
        if source is None:
            return cls(declarations_by_target_id={})
        source_segments = SourceLineSegmentAuthority(source)
        declarations_by_target_id: dict[str, Mapping[str, str]] = {}
        nodes_by_target_id = context.ast_target_nodes_by_id
        for target in targets_by_file[file_path]:
            if not target.is_class:
                continue
            node = nodes_by_target_id.get(target.target_id)
            if not isinstance(node, ast.ClassDef):
                continue
            declarations_by_target_id[target.target_id] = (
                DirectClassDeclarationAuthority(
                    source_segments=source_segments,
                    node=node,
                ).declarations_by_name()
            )
        return cls(declarations_by_target_id=declarations_by_target_id)
