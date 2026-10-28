# Extracted from c5b955e:nominal_refactor_advisor/cli.py


@dataclass(frozen=True)
class ColdScanAnalysis:
    """The scan's analysis when the cache serves nothing: focused, compact or full."""

    args: argparse.Namespace
    roots: tuple[Path, ...]
    config: DetectorConfig
    path_scope: AnalysisPathScope
    parse_cache_dir: Path | None
    source_policy: PythonSourcePathPolicy
    analysis_cache_dir: Path | None
    json_payload_profile: JsonPayloadProfile
    preparse_cache_policy: JsonSummaryPreparseCachePolicy
    scan_analysis_required: bool
    codemod_requested: bool
    focused_loop_cold_policy: FocusedLoopColdAnalysisPolicy
    semantic_descent_cache_context: SemanticDescentGraphCacheContext
    semantic_descent_analysis_source: SemanticDescentGraphAnalysisSource

    def outcome(
        self, cached_semantic_descent_graph: SemanticDescentGraph | None
    ) -> ScanAnalysisOutcome:
        """Analyse cold, keeping a graph the fast path already loaded."""
        if self.focused_loop_cold_policy.enabled:
            return self._focused_local(cached_semantic_descent_graph)
        if (
            self.args.json
            and self.json_payload_profile.sections.compact_analysis_compatible
            and not self.preparse_cache_policy.parsed_modules_required
            and self.scan_analysis_required
            and not self.codemod_requested
            and not DetectorTypePartition(
                default_detector_types_for_analysis()
            ).ast_retaining_context_detector_types
        ):
            return self._compact_global(cached_semantic_descent_graph)
        return self._full(cached_semantic_descent_graph)

    def _focused_local(
        self, cached_semantic_descent_graph: SemanticDescentGraph | None
    ) -> ScanAnalysisOutcome:
        detector_types = default_detector_types_for_analysis()
        detector_partition = DetectorTypePartition(detector_types)
        local_detector_types = detector_partition.detector_types_for(
            DetectorCacheGranularity.PER_MODULE
        )
        parse_elapsed = 0.0
        analysis_elapsed = 0.0
        findings = []
        module_cache_statuses: list[type[AnalysisCacheStatus]] = []
        seen_report_paths: set[Path] = set()
        for report_root in self.path_scope.report_roots:
            normalized_report_path = report_root.resolve()
            if normalized_report_path in seen_report_paths:
                continue
            seen_report_paths.add(normalized_report_path)
            started = perf_counter()
            local_modules = parse_python_module_roots(
                (report_root,),
                cache_dir=self.parse_cache_dir,
                use_parse_cache=self.args.use_parse_cache,
                parse_workers=self.args.parse_workers,
                source_policy=self.source_policy,
            )
            parse_elapsed += perf_counter() - started
            started = perf_counter()
            for local_module in local_modules:
                module_result = analyze_module_detector_types_with_cache(
                    local_module,
                    self.config,
                    detector_types=local_detector_types,
                    presentation_roots=self.roots,
                    analysis_cache_dir=self.analysis_cache_dir,
                )
                findings.extend(module_result.findings)
                module_cache_statuses.append(module_result.cache_status)
            analysis_elapsed += perf_counter() - started
            del local_modules
        return ScanAnalysisOutcome(
            modules=ParsedModuleSet(()),
            findings=SortedFindingsAuthority.sort(findings),
            parse_seconds=round(parse_elapsed, 3),
            analysis_seconds=round(analysis_elapsed, 3),
            analysis_cache_status=AnalysisCacheStatus.combine(module_cache_statuses),
            analysis_cache_identity=None,
            scan_status=JsonScanStatus(
                mode=JsonScanMode.focused_local_partial,
                analyzed_detector_count=len(local_detector_types),
                omitted_detector_count=(
                    len(detector_types) - len(local_detector_types)
                ),
            ),
            cached_semantic_descent_graph=cached_semantic_descent_graph,
        )

    def _compact_global(
        self, cached_semantic_descent_graph: SemanticDescentGraph | None
    ) -> ScanAnalysisOutcome:
        requested_compact_semantic_graph = (
            self.json_payload_profile.sections.semantic_descent_graph
        )
        if requested_compact_semantic_graph:
            cached_semantic_descent_graph = (
                self.semantic_descent_cache_context.cached_graph()
            )
        compact_result = analyze_compact_roots_with_cache(
            self.roots,
            self.config,
            cache_dir=self.parse_cache_dir,
            analysis_cache_dir=self.analysis_cache_dir,
            use_parse_cache=self.args.use_parse_cache,
            parse_workers=self.args.parse_workers,
            source_policy=self.source_policy,
            report_scope=self.path_scope,
            include_semantic_descent_graph=(
                requested_compact_semantic_graph
                and cached_semantic_descent_graph is None
            ),
        )
        built_semantic_descent_graph = compact_result.semantic_descent_graph
        if built_semantic_descent_graph is not None:
            cached_semantic_descent_graph = built_semantic_descent_graph
            self.semantic_descent_cache_context.store_exact_graph(
                built_semantic_descent_graph
            )
        return ScanAnalysisOutcome(
            modules=ParsedModuleSet(()),
            findings=compact_result.findings,
            parse_seconds=round(compact_result.preparation_seconds, 3),
            analysis_seconds=round(compact_result.analysis_seconds, 3),
            analysis_cache_status=compact_result.cache_status,
            analysis_cache_identity=compact_result.cache_identity,
            scan_status=JsonScanStatus.exact_compact_global(
                len(default_detector_types_for_analysis()),
            ),
            cached_semantic_descent_graph=cached_semantic_descent_graph,
        )

    def _full(
        self, cached_semantic_descent_graph: SemanticDescentGraph | None
    ) -> ScanAnalysisOutcome:
        started = perf_counter()
        modules = parse_python_module_roots(
            self.roots,
            cache_dir=self.parse_cache_dir,
            use_parse_cache=self.args.use_parse_cache,
            parse_workers=self.args.parse_workers,
            source_policy=self.source_policy,
        )
        parse_seconds = round(perf_counter() - started, 3)
        if not self.scan_analysis_required:
            return ScanAnalysisOutcome(
                modules=modules,
                findings=[],
                parse_seconds=parse_seconds,
                analysis_seconds=0.0,
                analysis_cache_status=None,
                analysis_cache_identity=None,
                scan_status=None,
                cached_semantic_descent_graph=cached_semantic_descent_graph,
            )
        started = perf_counter()
        analysis_result = analyze_modules_with_cache(
            self.roots,
            modules,
            self.config,
            analysis_cache_dir=self.analysis_cache_dir,
            analysis_workers=self.args.analysis_workers,
            source_policy=self.source_policy,
            semantic_descent_source=self.semantic_descent_analysis_source,
            report_scope=self.path_scope,
        )
        return ScanAnalysisOutcome(
            modules=modules,
            findings=self.path_scope.filter_findings(analysis_result.findings),
            parse_seconds=parse_seconds,
            analysis_seconds=round(perf_counter() - started, 3),
            analysis_cache_status=analysis_result.cache_status,
            analysis_cache_identity=analysis_result.cache_identity,
            scan_status=None,
            cached_semantic_descent_graph=cached_semantic_descent_graph,
        )


@dataclass(frozen=True)
class FastCacheScan(FastCacheQuestion[ScanAnalysisOutcome]):
    """The scan's analysis from what the fast cache path served.

    A served result still needs the semantic-descent graph and, when snapshots
    come from cache, the codemod source context; without either the scan
    analyses cold.
    """

    cold: ColdScanAnalysis
    started: float
    requires_semantic_descent_cache: bool
    uses_evidence_local_partial_reuse: bool
    latest_semantic_descent_graph: SemanticDescentGraph | None
    source_context_cache_lookup_enabled: bool

    def exact(self, result: ExactCacheHit) -> ScanAnalysisOutcome:
        return self._served(result, JsonScanStatus.exact_cache(result.cache_identity))

    def partial(self, result: PartialCacheReuse) -> ScanAnalysisOutcome:
        detector_types = default_detector_types_for_analysis()
        analyzed_detector_count = len(
            EvidenceLocalPartialDetectorSelection.from_detector_types(
                detector_types
            ).rerun_detector_family
        )
        return self._served(
            result,
            JsonScanStatus(
                mode=JsonScanMode.focused_cache_partial,
                analyzed_detector_count=analyzed_detector_count,
                omitted_detector_count=len(detector_types) - analyzed_detector_count,
            ),
        )

    def miss(self) -> ScanAnalysisOutcome:
        return self.cold.outcome(None)

    def _served(
        self, result: ReusedCacheResult, scan_status: JsonScanStatus
    ) -> ScanAnalysisOutcome:
        cached_semantic_descent_graph = None
        if self.requires_semantic_descent_cache:
            cached_semantic_descent_graph = (
                self.cold.semantic_descent_cache_context.cached_graph()
            )
            if (
                cached_semantic_descent_graph is None
                and self.uses_evidence_local_partial_reuse
            ):
                cached_semantic_descent_graph = self.latest_semantic_descent_graph
            if cached_semantic_descent_graph is None:
                return self.cold.outcome(None)
        cached_source_context = None
        if self.source_context_cache_lookup_enabled:
            source_context_lookup = CodemodSourceContextCache(
                self.cold.analysis_cache_dir
            ).load(result.cache_identity)
            if not source_context_lookup.status.is_hit:
                return self.cold.outcome(cached_semantic_descent_graph)
            cached_source_context = source_context_lookup.context
        return ScanAnalysisOutcome(
            modules=ParsedModuleSet(()),
            findings=self.cold.path_scope.filter_findings(result.findings),
            parse_seconds=0.0,
            analysis_seconds=round(perf_counter() - self.started, 3),
            analysis_cache_status=result.cache_status,
            analysis_cache_identity=result.cache_identity,
            scan_status=scan_status,
            cached_semantic_descent_graph=cached_semantic_descent_graph,
            cached_source_context=cached_source_context,
        )
