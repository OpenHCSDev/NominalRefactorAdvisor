"""Regression tests for silent missing-path fallbacks, EAFP/broad exception
swallowers, BOUND-5 embedded foreign code literals, and cloned DeclaredFamily roots."""

from __future__ import annotations

from pathlib import Path

from nominal_refactor_advisor.ast_tools import (
    PythonSourcePathPolicy,
    parse_python_modules,
)
from nominal_refactor_advisor.detectors._base import DetectorConfig
from nominal_refactor_advisor.detectors._silent_fallbacks import (
    ClonedDeclaredFamilyDetector,
    EmbeddedCodeLiteralDetector,
    SilentExceptionFallbackDetector,
    SilentMissingPathFallbackDetector,
)
from nominal_refactor_advisor.patterns import PatternId
from nominal_refactor_advisor.scan_cache import ScanCache


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def _detect(detector_cls, root: Path):
    with ScanCache.scope():
        modules = parse_python_modules(
            root,
            use_parse_cache=False,
            source_policy=PythonSourcePathPolicy(include_tests=False),
        )
        return detector_cls().detect(modules, DetectorConfig())


def test_new_pattern_ids_continue_stable_sequence() -> None:
    assert PatternId.EMBEDDED_CODE_LITERAL == 60
    assert PatternId.SILENT_MISSING_PATH_FALLBACK == 61
    assert PatternId.SILENT_EXCEPTION_FALLBACK == 62
    assert PatternId.CLONED_DECLARED_FAMILY == 63


def test_embedded_code_literal_flags_large_html_js_and_ignores_docstrings(
    tmp_path: Path,
) -> None:
    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head><style>body { color: #fff; }</style></head>",
        "<body><div id='app'></div>",
        "<script>",
        "const root = document.getElementById('app');",
        "function render(rows) { root.innerHTML = rows.join(''); }",
        *(f"// line {i}" for i in range(20)),
        "</script></body></html>",
    ]
    blob = "\n".join(html_lines)
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/gallery.py": (
                f'"""\n{blob}\n"""\n\n'
                f'_GALLERY_HTML = """\n{blob}\n"""\n'
                'SHORT = "<html><script>1</script></html>"\n'
            ),
        },
    )
    findings = _detect(EmbeddedCodeLiteralDetector, tmp_path)
    assert len(findings) == 1
    assert findings[0].pattern_id is PatternId.EMBEDDED_CODE_LITERAL
    assert "HTML/CSS/JS" in findings[0].summary
    assert "BOUND-5" in findings[0].summary


def test_silent_missing_path_fallback_catches_filters_ternaries_and_default_or(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/pipeline.py": (
                "from pathlib import Path\n\n"
                "def merge_shards(shards: list[Path]) -> list[Path]:\n"
                "    return [p for p in shards if p.is_file()]\n\n"
                "def load_bench(path: Path, limit: int) -> tuple[str, ...]:\n"
                "    return build_cards(path) if limit > 0 and path.is_file() else ()\n\n"
                "def choose_source(path: Path):\n"
                "    return JsonlSource(path) if path.is_file() else DefaultCalibrationTexts()\n\n"
                "def load_rows(rows: list[str]):\n"
                "    return rows or DefaultCalibrationTexts().load_rows(10)\n\n"
                "def fail_loud_missing(paths: list[Path]) -> list[Path]:\n"
                "    return [p for p in paths if not p.exists()]\n\n"
                "def discover_files(root: Path) -> list[Path]:\n"
                "    return [p for p in root.iterdir() if p.is_file()]\n"
            ),
        },
    )
    findings = _detect(SilentMissingPathFallbackDetector, tmp_path)
    symbols = {f.primary_evidence.symbol for f in findings}
    assert symbols == {"merge_shards", "load_bench", "choose_source", "load_rows"}
    assert all(f.pattern_id is PatternId.SILENT_MISSING_PATH_FALLBACK for f in findings)


def test_silent_exception_fallback_catches_eafp_attribute_and_broad_swallows(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/training.py": (
                "import pickle\n\n"
                "def add_loss(loss, outputs):\n"
                "    try:\n"
                "        return outputs.logits\n"
                "    except (AttributeError, IndexError, TypeError):\n"
                "        return loss\n\n"
                "def get_head(trainer):\n"
                "    try:\n"
                "        conf_head = trainer.vocab_loss_policy.confidence_head\n"
                "    except AttributeError:\n"
                "        conf_head = None\n"
                "    return conf_head\n\n"
                "def load_paper(path):\n"
                "    try:\n"
                "        return path.read_text()\n"
                "    except Exception:\n"
                "        return ''\n\n"
                "def load_cache(raw: bytes):\n"
                "    try:\n"
                "        return pickle.loads(raw)\n"
                "    except (OSError, pickle.PickleError, EOFError, AttributeError):\n"
                "        return None\n\n"
                "def capture_error(fn):\n"
                "    error = None\n"
                "    try:\n"
                "        fn()\n"
                "    except BaseException as caught:\n"
                "        error = caught\n"
                "    return error\n"
            ),
        },
    )
    findings = _detect(SilentExceptionFallbackDetector, tmp_path)
    symbols = {f.primary_evidence.symbol for f in findings}
    assert symbols == {"add_loss", "get_head", "load_paper"}
    assert all(f.pattern_id is PatternId.SILENT_EXCEPTION_FALLBACK for f in findings)


def test_cloned_declared_family_flags_parallel_roots_and_allows_subclasses(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        {
            "pkg/__init__.py": "",
            "pkg/nominal.py": "class DeclaredFamily:\n    pass\n",
            "pkg/pipeline.py": (
                "from pkg.nominal import DeclaredFamily\n\n"
                "class PipelineStage(DeclaredFamily):\n    pass\n\n"
                "class Stage1(PipelineStage):\n    pass\n"
            ),
            "pkg/gallery.py": (
                "from pkg.nominal import DeclaredFamily\n\n"
                "class PipelineStage(DeclaredFamily):\n    pass\n\n"
                "class Stage1(PipelineStage):\n    pass\n"
            ),
            "pkg/unique.py": (
                "from pkg.nominal import DeclaredFamily\n\n"
                "class UniqueFamily(DeclaredFamily):\n    pass\n"
            ),
        },
    )
    findings = _detect(ClonedDeclaredFamilyDetector, tmp_path)
    assert len(findings) == 1
    assert findings[0].pattern_id is PatternId.CLONED_DECLARED_FAMILY
    assert "PipelineStage" in findings[0].summary
    assert len(findings[0].evidence) == 2
