"""Every diffusion catalog family is loaded by a CI lane or marked experimental.

A family counts as loaded when one of its catalog model IDs is the model of a
case, or a string in a test file, that a registered CUDA diffusion lane runs.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import msgspec
import pytest

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.ci.ci_register import HWBackend, ut_parse_one_file

_REPO_ROOT = Path(__file__).resolve().parents[5]
_CATALOG_PATH = _REPO_ROOT / "docs/src/snippets/diffusion/model-catalog.jsx"
_BRIDGE_DIR = _REPO_ROOT / "test/registered/e2e/diffusion"
# Unit tests build models from random or tiny weights, never a catalog checkpoint.
_NON_LOADING_SUITES = frozenset({"unit"})

_FAMILY_RE = re.compile(r'\{\s*name:\s*"(?P<name>[^"]+)",(?P<body>.*?)\n\s*\},', re.S)
_MODEL_IDS_RE = re.compile(r"modelIds:\s*\[(?P<ids>.*?)\]", re.S)
_EXPERIMENTAL_RE = re.compile(r'experimental:\s*"(?P<reason>[^"]+)"')
_SUITE_CALL_RE = re.compile(r'run_diffusion_suite\(\s*"(?P<suite>[^"]+)"')

pytestmark = [
    pytest.mark.skipif(
        not _CATALOG_PATH.is_file() or not _BRIDGE_DIR.is_dir(),
        reason="needs a source checkout with docs/ and test/registered/",
    ),
    # gpu_cases.py trims its lists per platform; the lanes checked here are CUDA.
    pytest.mark.skipif(
        not current_platform.is_cuda(), reason="the diffusion lanes are CUDA lanes"
    ),
]


class CatalogFamily(msgspec.Struct, frozen=True):
    name: str
    model_ids: tuple[str, ...]
    experimental: str | None


class LoadedModels(msgspec.Struct, frozen=True):
    exact: frozenset[str]
    # f-string model paths, with each replacement field matching anything
    patterns: tuple[re.Pattern, ...]

    def loads(self, model_id: str) -> bool:
        return model_id in self.exact or any(
            pattern.fullmatch(model_id) for pattern in self.patterns
        )


def _parse_catalog(text: str) -> list[CatalogFamily]:
    families = []
    for match in _FAMILY_RE.finditer(text):
        body = match["body"]
        ids = _MODEL_IDS_RE.search(body)
        experimental = _EXPERIMENTAL_RE.search(body)
        families.append(
            CatalogFamily(
                name=match["name"],
                model_ids=tuple(re.findall(r'"([^"]+)"', ids["ids"])) if ids else (),
                experimental=experimental["reason"] if experimental else None,
            )
        )
    return families


def _uncovered(families: list[CatalogFamily], loaded: LoadedModels) -> list[str]:
    return [
        family.name
        for family in families
        if family.experimental is None
        and not any(loaded.loads(model_id) for model_id in family.model_ids)
    ]


def _lane_registered_files() -> list[Path]:
    files = []
    for path in sorted(_BRIDGE_DIR.glob("test_*.py")):
        registries, _ = ut_parse_one_file(str(path))
        if any(r.backend == HWBackend.CUDA and r.disabled is None for r in registries):
            files.append(path)
    return files


def _strings_in_file(path: Path, gpu_cases, test_utils) -> LoadedModels:
    exact: set[str] = set()
    patterns: list[re.Pattern] = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            exact.add(node.value)
        elif isinstance(node, ast.JoinedStr):
            # Only a template that starts with a literal "org/" names a repo.
            head = node.values[0] if node.values else None
            if not (isinstance(head, ast.Constant) and "/" in head.value):
                continue
            parts = [
                re.escape(v.value) if isinstance(v, ast.Constant) else ".+"
                for v in node.values
            ]
            patterns.append(re.compile("".join(parts)))
        elif isinstance(node, ast.Name) and node.id.endswith("_MODEL_NAME_FOR_TEST"):
            value = getattr(test_utils, node.id, None)
            if isinstance(value, str):
                exact.add(value)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.endswith(".gpu_cases")
        ):
            for alias in node.names:
                exact.update(
                    _case_models(getattr(gpu_cases, alias.name, None), gpu_cases)
                )
    return LoadedModels(exact=frozenset(exact), patterns=tuple(patterns))


def _case_models(value, gpu_cases) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        case.server_args.model_path
        for case in value
        if isinstance(case, gpu_cases.DiffusionTestCase)
    }


def _lane_loaded_models() -> LoadedModels:
    from sglang.multimodal_gen.test import test_utils
    from sglang.multimodal_gen.test.server import gpu_cases

    server_dir = Path(gpu_cases.__file__).resolve().parent
    files: list[Path] = []
    exact: set[str] = set()
    for registered in _lane_registered_files():
        suites = _SUITE_CALL_RE.findall(registered.read_text())
        if not suites:
            files.append(registered)
        for suite in set(suites) - _NON_LOADING_SUITES:
            for _, cases in gpu_cases.PARAMETRIZED_CASE_GROUPS.get(suite, []):
                exact.update(case.server_args.model_path for case in cases)
            for rel in gpu_cases.FILE_SUITES.get(
                suite, []
            ) + gpu_cases.STANDALONE_FILES.get(suite, []):
                if not rel.startswith("../unit/"):
                    files.append((server_dir / rel).resolve())
    patterns: list[re.Pattern] = []
    for path in files:
        loaded = _strings_in_file(path, gpu_cases, test_utils)
        exact |= loaded.exact
        patterns.extend(loaded.patterns)
    return LoadedModels(exact=frozenset(exact), patterns=tuple(patterns))


@pytest.fixture(scope="module")
def families() -> list[CatalogFamily]:
    parsed = _parse_catalog(_CATALOG_PATH.read_text())
    assert parsed and all(family.model_ids for family in parsed), (
        "model-catalog.jsx no longer parses into families with modelIds"
    )
    return parsed


@pytest.fixture(scope="module")
def loaded() -> LoadedModels:
    return _lane_loaded_models()


def test_every_catalog_family_is_ci_loaded_or_experimental(families, loaded):
    uncovered = _uncovered(families, loaded)
    assert not uncovered, (
        f"Catalog families with no CI-loaded modelId: {uncovered}. Add a case to a "
        "lane that loads one of its modelIds, or give the entry an "
        '`experimental: "<reason>"` field in model-catalog.jsx.'
    )


def test_experimental_family_is_not_ci_loaded(families, loaded):
    stale = [
        family.name
        for family in families
        if family.experimental is not None
        and any(loaded.loads(model_id) for model_id in family.model_ids)
    ]
    assert not stale, (
        f"Catalog families marked experimental that a CI lane now loads: {stale}. "
        "Drop their `experimental` field in model-catalog.jsx."
    )


def test_new_unmarked_family_is_reported():
    catalog = """
    {
      name: "Loaded",
      modelIds: ["org/loaded"],
    },
    {
      name: "Marked",
      modelIds: ["org/marked"],
      experimental: "No CI lane loads it.",
    },
    {
      name: "New",
      modelIds: [
        "org/new-a",
        "org/new-b",
      ],
      cookbook: "/cookbook/new",
    },
    {
      name: "Templated",
      modelIds: ["org/variant-1"],
    },
"""
    loaded = LoadedModels(
        exact=frozenset({"org/loaded"}),
        patterns=(re.compile(re.escape("org/variant-") + ".+"),),
    )
    assert _uncovered(_parse_catalog(catalog), loaded) == ["New"]
