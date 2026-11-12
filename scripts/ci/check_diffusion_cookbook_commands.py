#!/usr/bin/env python3
"""Check that the diffusion Cookbook's launch commands parse and validate.

Every `sglang serve` / `sglang generate` command in docs/cookbook/diffusion is
parsed with the parser those subcommands build for diffusion, then passed to
`ServerArgs.from_cli_args` on a mocked idle GPU. Nothing is launched and no
weights are read. Static code blocks run on an H100 80 GB; builder commands
(dumped by docs/scripts/dump_diffusion_commands.mjs) run on the hardware they
were rendered for.

Hugging Face runs offline and ModelScope is disabled, so a model resolves only
from the local HF cache.
`--fetch-model-index` first downloads model_index.json and the small JSON
configs the pipeline configs read, one `hf_hub_download` per named file.

Each command ends in one class:
  ok                the arguments parse and ServerArgs accepts them
  parse_error       the CLI parser rejects an argument
  validation_error  ServerArgs rejects the combination
  skipped           validation is not meaningful here (the reason is listed)

Usage:
  python3 scripts/ci/check_diffusion_cookbook_commands.py
  node docs/scripts/dump_diffusion_commands.mjs > gen.json
  python3 scripts/ci/check_diffusion_cookbook_commands.py --generated gen.json \\
      --fetch-model-index --markdown-out report.md

Exits 1 when any command has a parse or validation error.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec

REPO_ROOT = Path(__file__).resolve().parents[2]
COOKBOOK_DIR = REPO_ROOT / "docs" / "cookbook" / "diffusion"

OK = "ok"
PARSE_ERROR = "parse_error"
VALIDATION_ERROR = "validation_error"
SKIPPED = "skipped"
CLASSES = (OK, PARSE_ERROR, VALIDATION_ERROR, SKIPPED)


class Hardware(msgspec.Struct, frozen=True):
    # "cuda" runs on the mocked GPU, "cpu" on the real CPU platform, and "npu"
    # is skipped (there is no NPU platform mock).
    platform: str
    vendor: str
    name: str
    capability: tuple[int, int] | None
    memory_gb: float
    device_count: int = 8
    shares_host_memory: bool = False


# Builder hardware ids (docs/src/snippets/_deployment.jsx catalog plus the
# per-config `hardware` lists). Unknown ids fall back on the dumped vendor.
HARDWARE = {
    "b300": Hardware("cuda", "nvidia", "NVIDIA B300 SXM6 AC", (10, 3), 288),
    "gb300": Hardware("cuda", "nvidia", "NVIDIA GB300", (10, 3), 288, 4),
    "b200": Hardware("cuda", "nvidia", "NVIDIA B200", (10, 0), 192),
    "gb200": Hardware("cuda", "nvidia", "NVIDIA GB200", (10, 0), 192, 4),
    "dgx-spark": Hardware("cuda", "nvidia", "NVIDIA GB10", (12, 1), 128, 1, True),
    "h200": Hardware("cuda", "nvidia", "NVIDIA H200", (9, 0), 141),
    "h100": Hardware("cuda", "nvidia", "NVIDIA H100 80GB HBM3", (9, 0), 80),
    "h20-3e": Hardware("cuda", "nvidia", "NVIDIA H20", (9, 0), 141),
    "h800": Hardware("cuda", "nvidia", "NVIDIA H800", (9, 0), 80),
    "a100": Hardware("cuda", "nvidia", "NVIDIA A100-SXM4-80GB", (8, 0), 80),
    "cuda": Hardware("cuda", "nvidia", "NVIDIA H100 80GB HBM3", (9, 0), 80),
    "rtxpro6000": Hardware(
        "cuda", "nvidia", "NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0), 96
    ),
    "rtx6000pro": Hardware(
        "cuda", "nvidia", "NVIDIA RTX PRO 6000 Blackwell Server Edition", (12, 0), 96
    ),
    "rtx6000ada": Hardware(
        "cuda", "nvidia", "NVIDIA RTX 6000 Ada Generation", (8, 9), 48
    ),
    "rtx5090": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 5090", (12, 0), 32),
    "rtx5080": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 5080", (12, 0), 16),
    "rtx5070ti": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 5070 Ti", (12, 0), 16),
    "rtx5070": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 5070", (12, 0), 12),
    "rtx4090": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 4090", (8, 9), 24),
    "rtx4080": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 4080", (8, 9), 16),
    "rtx4070": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 4070", (8, 9), 12),
    "rtx4060ti": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 4060 Ti", (8, 9), 16),
    "rtx3090": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 3090", (8, 6), 24),
    "rtx3060": Hardware("cuda", "nvidia", "NVIDIA GeForce RTX 3060", (8, 6), 12),
    # AMD and Intel GPUs validate on the mocked CUDA platform; failures that
    # come from their own backends are skipped (see _classify_failure).
    "mi300x": Hardware("cuda", "amd", "AMD Instinct MI300X", (9, 4), 192),
    "mi325x": Hardware("cuda", "amd", "AMD Instinct MI325X", (9, 4), 256),
    "mi350x": Hardware("cuda", "amd", "AMD Instinct MI350X", (9, 5), 288),
    "mi355x": Hardware("cuda", "amd", "AMD Instinct MI355X", (9, 5), 288),
    "rocm": Hardware("cuda", "amd", "AMD Instinct MI300X", (9, 4), 192),
    "arc_b": Hardware("cuda", "xpu", "Intel Arc B580", (8, 0), 12),
    "xeon": Hardware("cpu", "cpu", "Intel Xeon", None, 0),
    "a2": Hardware("npu", "npu", "Ascend A2", None, 64),
    "a3": Hardware("npu", "npu", "Ascend A3", None, 64),
    "a5": Hardware("npu", "npu", "Ascend 950", None, 128),
}
STATIC_HW = "h100"
# Vendor groups of the builder hardware catalog (_deployment.jsx and config.hardware).
VENDOR_FALLBACK = {
    "blackwell": "b200",
    "hopper": "h100",
    "consumer": "rtx4090",
    "amd": "mi300x",
    "npu": "a3",
    "cpu": "xeon",
    "xpu": "arc_b",
}


def hardware_for(hw_id: str, vendor: str | None = None) -> Hardware:
    if hw_id in HARDWARE:
        return HARDWARE[hw_id]
    return HARDWARE[VENDOR_FALLBACK.get(vendor or "", STATIC_HW)]


class Command(msgspec.Struct, kw_only=True):
    source: str  # "docs/...mdx:57" (static) or "docs/...mdx#<selection>" (generated)
    kind: str  # "static" | "generated"
    subcommand: str  # "serve" | "generate"
    argv: list[str]
    env: dict[str, str] = {}
    hw: str = STATIC_HW
    vendor: str | None = None
    # Config files the page writes (heredocs and titled code blocks).
    files: dict[str, str] = {}
    # Fills a `--model-path` placeholder: the model the page documents.
    default_model: str | None = None

    @property
    def text(self) -> str:
        env = [f"{key}={shlex.quote(value)}" for key, value in self.env.items()]
        return " ".join(env + [shlex.join(["sglang", self.subcommand, *self.argv])])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*)$")
TAB_OPEN_RE = re.compile(r"<Tab\b[^>]*\btitle=[\"']([^\"']*)[\"']")
TAB_CLOSE_RE = re.compile(r"</Tab>")
SHELL_LANGS = {"", "bash", "sh", "shell", "zsh", "console", "terminal"}
CONFIG_TITLE_RE = re.compile(r"^[\w./-]+\.(?:ya?ml|json)$")

HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")
ARRAY_START_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=\((.*)$", re.S)
ASSIGN_TOKEN_RE = re.compile(r"^([A-Za-z_]\w*)=(.*)$", re.S)
ARRAY_REF_RE = re.compile(r"^\$\{(\w+)\[[@*]\]\}$")
VAR_REF_RE = re.compile(r"\$\{(\w+)\}|\$([A-Za-z_]\w*)")
# `<NAME>` placeholders are protected from shell tokenizing (they would
# otherwise read as redirections) and restored afterwards.
ANGLE_PLACEHOLDER_RE = re.compile(r"<([A-Za-z][\w.-]*)>")
PLACEHOLDER_RE = re.compile(
    r"<[A-Za-z][\w.-]*>|\{\{\w+\}\}|\$\{\w+\}|\$[A-Za-z_]\w*|\$\([^)]*\)?"
)
_LT, _GT = "\x02", "\x03"
SEPARATORS = {";", "&&", "||", "|", "&", ";;", "|&"}
REDIRECTIONS = {">", ">>", "<", ">&", "&>", "&>>", "<>", ">|"}
CONTEXT_PLATFORMS = (
    (re.compile(r"\bAscend\b|\bNPUs?\b|\bA[23] Series\b"), "a3"),
    (re.compile(r"\bAMD\b|\bROCm\b|\bMI3\d\dX?\b|\baiter\b", re.I), "mi300x"),
    (re.compile(r"\bXeon\b"), "xeon"),
)


class CodeBlock(msgspec.Struct, frozen=True):
    lang: str
    title: str
    lines: list[str]
    start_line: int  # 1-based line of the first body line
    context: str  # tab title, heading and paragraph before the block


def iter_code_blocks(text: str) -> list[CodeBlock]:
    lines = text.splitlines()
    blocks: list[CodeBlock] = []
    tabs: list[str] = []
    heading = ""
    paragraph: list[str] = []
    after_blank = False
    i = 0
    while i < len(lines):
        line = lines[i]
        fence = FENCE_RE.match(line)
        if not fence:
            for title in TAB_OPEN_RE.findall(line):
                tabs.append(title)
            if TAB_CLOSE_RE.search(line) and tabs:
                tabs.pop()
            heading_match = HEADING_RE.match(line)
            if heading_match:
                heading, paragraph = heading_match.group(1), []
            elif not line.strip():
                after_blank = True
            else:
                # Keep the last paragraph until the next one starts.
                paragraph = [*([] if after_blank else paragraph), line.strip()]
                after_blank = False
            i += 1
            continue
        indent, marker, info = fence.groups()
        lang, _, title = info.strip().partition(" ")
        body: list[str] = []
        j = i + 1
        while j < len(lines):
            close = FENCE_RE.match(lines[j])
            if (
                close
                and close.group(2).startswith(marker)
                and not close.group(3).strip()
            ):
                break
            body.append(
                lines[j][len(indent) :] if lines[j].startswith(indent) else lines[j]
            )
            j += 1
        context = " | ".join(filter(None, [*tabs, heading, " ".join(paragraph), title]))
        blocks.append(CodeBlock(lang.lower(), title.strip(), body, i + 2, context))
        paragraph = []
        i = j + 1
    return blocks


def _protect(text: str) -> str:
    return ANGLE_PLACEHOLDER_RE.sub(lambda m: f"{_LT}{m.group(1)}{_GT}", text)


def _restore(token: str) -> str:
    return token.replace(_LT, "<").replace(_GT, ">")


def tokenize(text: str) -> list[str]:
    """Split one logical shell line into words and operator tokens."""
    # Drop file-descriptor numbers glued to a redirection (`2>&1`, `2> err`).
    text = re.sub(r"(?<=\s)\d+(?=>)", "", " " + _protect(text))
    lexer = shlex.shlex(text, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return [_restore(token) for token in lexer]


def _split_segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in SEPARATORS:
            segments.append([])
        elif token in REDIRECTIONS or token.startswith(">"):
            skip_next = True  # the redirection target
        else:
            segments[-1].append(token)
    return [segment for segment in segments if segment]


class ShellState:
    """Shell variables, arrays and exported env a code block defines."""

    def __init__(self):
        self.variables: dict[str, str | list[str] | None] = {}
        self.exported: dict[str, str] = {}

    def expand(self, token: str) -> list[str]:
        array = ARRAY_REF_RE.match(token)
        if array:
            value = self.variables.get(array.group(1))
            if isinstance(value, list):
                return list(value)
            return [token] if value is None else [value]
        whole = VAR_REF_RE.fullmatch(token)
        if whole:
            value = self.variables.get(whole.group(1) or whole.group(2))
            if isinstance(value, list):
                return list(value)

        def scalar(match: re.Match) -> str:
            name = match.group(1) or match.group(2)
            value = self.variables.get(name)
            if value is None:
                return match.group(0)  # unresolved: substituted later
            return " ".join(value) if isinstance(value, list) else value

        return [VAR_REF_RE.sub(scalar, token)]

    def expand_all(self, tokens: list[str]) -> list[str]:
        return [word for token in tokens for word in self.expand(token)]


def _logical_lines(lines: list[str], state: ShellState, files: dict[str, str]):
    """Yield (line_offset, text) for each logical shell line.

    Joins backslash continuations, records heredoc bodies written with `cat`
    as files, and collects multi-line array assignments into `state`.
    """
    i = 0
    while i < len(lines):
        line = lines[i]
        heredoc = HEREDOC_RE.search(line)
        if heredoc and not line.lstrip().startswith("#"):
            delimiter = heredoc.group(2)
            body: list[str] = []
            j = i + 1
            while j < len(lines) and lines[j].strip() != delimiter:
                body.append(lines[j])
                j += 1
            head = line[: heredoc.start()] + line[heredoc.end() :]
            target = re.search(r"\bcat\b[^>]*>\s*(\S+)", head)
            if target:
                files[target.group(1).strip("'\"")] = "\n".join(body) + "\n"
            substitution = re.match(r"^\s*(?:export\s+)?([A-Za-z_]\w*)=\$\(", line)
            if substitution:
                # VAR=$(cmd <<EOF ... EOF\n): the value is computed at run time.
                state.variables[substitution.group(1)] = None
            i = j + 1
            if substitution and i < len(lines) and lines[i].strip() == ")":
                i += 1
            continue

        start = i
        text = line
        while text.rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            text = text.rstrip()[:-1] + " " + lines[i]

        array = ARRAY_START_RE.match(text)
        if array:
            content = array.group(2)
            while True:
                try:
                    lexer = shlex.shlex(
                        _protect(content), posix=True, punctuation_chars="()"
                    )
                    lexer.whitespace_split = True
                    lexer.commenters = "#"
                    words = list(lexer)
                except ValueError:  # an unclosed quote spans the next line
                    words = None
                if words is not None and ")" in words:
                    words = words[: words.index(")")]
                    state.variables[array.group(1)] = state.expand_all(
                        [_restore(word) for word in words]
                    )
                    break
                if i + 1 >= len(lines):
                    break
                i += 1
                content += "\n" + lines[i]
            i += 1
            continue

        yield start, text
        i += 1


def parse_shell(
    lines: list[str],
    *,
    first_line: int = 1,
    state: ShellState | None = None,
    files: dict[str, str] | None = None,
) -> list[tuple[int, str, list[str], dict[str, str]]]:
    """Return (line, subcommand, argv, env) for each `sglang serve|generate`."""
    state = state or ShellState()
    files = {} if files is None else files
    commands = []
    for offset, text in _logical_lines(lines, state, files):
        stripped = text.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("$ "):  # console prompt
            stripped = stripped[2:]
        try:
            tokens = tokenize(stripped)
        except ValueError:
            continue
        for segment in _split_segments(tokens):
            exported = segment[0] == "export"
            words = segment[1:] if exported else segment
            assignments: dict[str, str] = {}
            k = 0
            while k < len(words) and ASSIGN_TOKEN_RE.match(words[k]):
                name, value = ASSIGN_TOKEN_RE.match(words[k]).groups()
                assignments[name] = " ".join(state.expand(value))
                k += 1
            if k == len(words):
                for name, value in assignments.items():
                    if "$(" in value:
                        state.variables[name] = None
                    else:
                        state.variables[name] = value
                    if exported:
                        state.exported[name] = value
                continue
            for index in range(k, len(words) - 1):
                if words[index] == "sglang" and words[index + 1] in (
                    "serve",
                    "generate",
                ):
                    env = dict(state.exported)
                    if index == k:
                        env.update(assignments)
                    argv = state.expand_all(words[index + 2 :])
                    commands.append((first_line + offset, words[index + 1], argv, env))
                    break
    return commands


def _option_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return None


def model_path_of(argv: list[str]) -> str | None:
    value = _option_value(argv, "--model-path")
    if value is None and argv and not argv[0].startswith("-"):
        value = argv[0]
    return value


def _static_hardware(context: str, env: dict[str, str]) -> str:
    if env.get("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "").lower() == "cpu":
        return "xeon"
    if any(key.startswith("ASCEND_") for key in env):
        return "a3"
    if env.get("SGLANG_USE_AITER") == "1":
        return "mi300x"
    for pattern, hw in CONTEXT_PLATFORMS:
        if pattern.search(context):
            return hw
    return STATIC_HW


def extract_page(path: Path, root: Path = REPO_ROOT) -> list[Command]:
    text = path.read_text(encoding="utf-8")
    try:
        relative = path.resolve().relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    files: dict[str, str] = {}
    found = []
    for block in iter_code_blocks(text):
        if block.lang in SHELL_LANGS:
            for line, subcommand, argv, env in parse_shell(
                block.lines, first_line=block.start_line, files=files
            ):
                found.append((line, subcommand, argv, env, block.context))
        elif CONFIG_TITLE_RE.match(block.title):
            files[block.title] = "\n".join(block.lines) + "\n"

    models = [model_path_of(argv) for _, _, argv, _, _ in found]
    models = [m for m in models if m and not PLACEHOLDER_RE.search(m)]
    default_model = collections.Counter(models).most_common(1)[0][0] if models else None
    commands = []
    for line, subcommand, argv, env, context in found:
        model_id = _option_value(argv, "--model-id")
        commands.append(
            Command(
                source=f"{relative}:{line}",
                kind="static",
                subcommand=subcommand,
                argv=argv,
                env=env,
                hw=_static_hardware(context, env),
                files=dict(files),
                default_model=(
                    model_id if model_id and "/" in model_id else default_model
                ),
            )
        )
    return commands


def load_generated(path: Path) -> list[Command]:
    """Read the builder commands docs/scripts/dump_diffusion_commands.mjs wrote."""
    commands = []
    for entry in json.loads(path.read_text(encoding="utf-8")):
        parsed = parse_shell(entry["command"].splitlines())
        if len(parsed) != 1:
            raise ValueError(
                f"{entry['source']}: expected one sglang command, found {len(parsed)}"
            )
        _, subcommand, argv, env = parsed[0]
        commands.append(
            Command(
                source=entry["source"],
                kind="generated",
                subcommand=subcommand,
                argv=argv,
                env=env,
                hw=entry.get("hw") or STATIC_HW,
                vendor=entry.get("vendor"),
                default_model=entry.get("model"),
            )
        )
    return commands


# ---------------------------------------------------------------------------
# --fetch-model-index
# ---------------------------------------------------------------------------

REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
PACKAGE_CONFIGS = ("config.json", "manifest.json", "hf_quant_config.json")


def model_index_files(model_index: dict | None, variants: set[str]) -> list[str]:
    """The small JSON configs a pipeline config reads for this repo."""
    if model_index is None:
        # Native packages without model_index.json (FLUX.3 and similar).
        names = list(PACKAGE_CONFIGS)
        for variant in sorted(variants):
            names += [f"variants/{variant}/{name}" for name in PACKAGE_CONFIGS]
        return names
    names = []
    for component, spec in model_index.items():
        if component.startswith("_") or not isinstance(spec, list) or not spec[0]:
            continue
        if component == "scheduler":
            names.append("scheduler/scheduler_config.json")
        elif not component.startswith(("tokenizer", "processor", "feature_extractor")):
            names.append(f"{component}/config.json")
    return names


def fetch_model_indices(commands: list[Command], hf_hub_download=None) -> list[str]:
    """Download only model_index.json and small JSON configs; never weights."""
    if hf_hub_download is None:
        from huggingface_hub import hf_hub_download
    repos: dict[str, set[str]] = collections.defaultdict(set)
    for command in commands:
        for repo in (
            model_path_of(command.argv),
            _option_value(command.argv, "--model-id"),
        ):
            if repo and REPO_ID_RE.match(repo) and not os.path.exists(repo):
                variant = _option_value(command.argv, "--model-variant")
                repos[repo].update([variant] if variant else [])
    log = []

    def fetch(repo: str, filename: str) -> str | None:
        if not filename.endswith(".json"):
            raise ValueError(f"refusing to fetch non-JSON file {repo}/{filename}")
        try:
            return hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:  # 404, gated repo, network
            log.append(f"{repo}/{filename}: {type(exc).__name__}")
            return None

    for repo, variants in sorted(repos.items()):
        index_path = fetch(repo, "model_index.json")
        model_index = None
        if index_path:
            with open(index_path, encoding="utf-8") as f:
                model_index = json.load(f)
        for filename in model_index_files(model_index, variants):
            fetch(repo, filename)
    return log


# ---------------------------------------------------------------------------
# Validation (runs in one worker process per hardware profile)
# ---------------------------------------------------------------------------

HF_CACHE_MARKERS = (
    "LocalEntryNotFoundError",
    "OfflineModeIsEnabled",
    "Cannot find an appropriate cached snapshot",
    "Cannot find the requested files in the disk cache",
    "outgoing traffic has been disabled",
    "GatedRepoError",
    "failed to download from Hugging Face",
    "Could not find model at",
    "could not fetch",
    "Failed to download or parse model_index.json",
)
OPTIONAL_DEPENDENCY_RE = re.compile(
    r"ModuleNotFoundError|No module named|is not installed|Install \S+ with `pip install"
)
PLATFORM_BACKEND_RE = {
    "amd": re.compile(r"aiter|rocm|\bhip\b", re.I),
    "xpu": re.compile(r"\bxpu\b|intel", re.I),
}
DUMMY_VALUES = {
    "host": "127.0.0.1",
    "port": "30000",
    "node_rank": "0",
    "dist_init_addr": "127.0.0.1:20000",
    "nnodes": "1",
}
# Builder placeholders that have one obvious local value.
KNOWN_PLACEHOLDERS = {
    "NODE_RANK": "0",
    "NODE0_IP": "127.0.0.1",
    "HOST_IP": "127.0.0.1",
    "PORT": "30000",
}
MEDIA_SUFFIXES = (("image", ".png"), ("video", ".mp4"), ("audio", ".wav"))


class CommandParseError(Exception):
    pass


def _exception_text(exc: BaseException) -> str:
    parts = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        parts.append(f"{type(exc).__name__}: {exc}")
        exc = exc.__cause__ or exc.__context__
    return " <- ".join(parts)


def _classify_failure(text: str, hw: Hardware) -> tuple[str, str]:
    summary = text.split(" <- ")[0]
    pattern = PLATFORM_BACKEND_RE.get(hw.vendor)
    if pattern is not None and pattern.search(text):
        return (
            SKIPPED,
            f"{hw.vendor} command fails on the mocked CUDA platform: {summary}",
        )
    if any(marker in text for marker in HF_CACHE_MARKERS):
        return SKIPPED, f"validation needs model files not in the HF cache: {summary}"
    if OPTIONAL_DEPENDENCY_RE.search(text):
        return SKIPPED, f"validation needs an optional dependency: {summary}"
    return VALIDATION_ERROR, summary


def _substitute_placeholders(
    *,
    argv: list[str],
    parser: argparse.ArgumentParser,
    workdir: str,
    default_model: str | None,
) -> tuple[list[str], list[str]]:
    """Replace <X>, {{X}} and unresolved $X with dummies typed by their flag."""
    actions = {
        opt: action for action in parser._actions for opt in action.option_strings
    }
    notes = []

    def dummy(action, placeholder: str) -> str:
        dest = action.dest if action is not None else "model_path"
        if dest == "model_path":
            return default_model or "dummy-org/dummy-model"
        if dest in DUMMY_VALUES:
            return DUMMY_VALUES[dest]
        if action.choices:
            return str(next(iter(action.choices)))
        if action.type is int:
            return "1"
        if action.type is float:
            return "1.0"
        if re.search(r"(_dir|^output_path)$", dest):
            return workdir
        if re.search(r"path|file|dir", dest):
            name = re.sub(r"\W+", "_", placeholder).strip("_") or "placeholder"
            suffix = next((s for key, s in MEDIA_SUFFIXES if key in dest), "")
            path = os.path.join(workdir, name + suffix)
            Path(path).touch()
            return path
        return "dummy"

    def known(match: re.Match) -> str:
        name = match.group(0).strip("{}<>$")
        return KNOWN_PLACEHOLDERS.get(name, match.group(0))

    out = []
    for index, arg in enumerate(argv):
        arg = PLACEHOLDER_RE.sub(known, arg)
        if not PLACEHOLDER_RE.search(arg):
            out.append(arg)
            continue
        flag, value = (
            arg.split("=", 1) if arg.startswith("--") and "=" in arg else (None, arg)
        )
        if flag is None and index > 0 and argv[index - 1] in actions:
            flag = argv[index - 1]
        action = actions.get(flag) if flag else None
        if action is None and flag is None and index > 0:
            out.append(PLACEHOLDER_RE.sub("dummy", arg))
            notes.append(f"{arg} -> dummy")
            continue
        if PLACEHOLDER_RE.fullmatch(value):
            replacement = dummy(action, value)
        else:  # a placeholder inside a larger value, e.g. "$DIR/model.safetensors"
            replacement = PLACEHOLDER_RE.sub(
                workdir if "/" in value else "dummy", value
            )
        notes.append(f"{value} -> {replacement}")
        out.append(
            f"{flag}={replacement}"
            if arg.startswith("--") and "=" in arg
            else replacement
        )
    return out, notes


def _relocate(arg: str, written: dict[str, str]) -> str:
    """Point an argument naming a file the page writes at its scratch copy."""
    flag, sep, value = arg.partition("=") if arg.startswith("--") else ("", "", arg)
    if value in written:
        return f"{flag}{sep}{written[value]}"
    return arg


def _uncached_model_index(model_path: str | None) -> str | None:
    """Why the registry cannot resolve this Hub model offline, or None.

    Without a cached model_index.json the registry silently falls back to the
    generic diffusers config, so the validation result would not describe the
    model the page documents.
    """
    from sglang.multimodal_gen.registry import get_non_diffusers_pipeline_name
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        maybe_download_model_index,
    )

    if not model_path or os.path.exists(model_path) or not REPO_ID_RE.match(model_path):
        return None
    if get_non_diffusers_pipeline_name(model_path):
        return None
    try:
        maybe_download_model_index(model_path)
    except Exception as exc:
        return (
            "validation needs model files not in the HF cache: model_index.json: "
            + _exception_text(exc).split(" <- ")[0]
        )
    return None


class _RaisingParser(argparse.ArgumentParser):
    def error(self, message):
        raise CommandParseError(message)

    def exit(self, status=0, message=None):
        raise CommandParseError(message or f"exit {status}")


def _parse_and_validate(command: Command, argv: list[str], workdir: str) -> dict:
    """Run the CLI parsing `sglang serve|generate` does, then ServerArgs."""
    from sglang.cli.serve import (
        _extract_model_type_override,
        _normalize_positional_model_path,
    )
    from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
        add_multimodal_gen_generate_args,
    )
    from sglang.multimodal_gen.runtime.entrypoints.cli.serve import (
        add_multimodal_gen_serve_args,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    parser = _RaisingParser(prog=f"sglang {command.subcommand}")
    positional = False
    default_args = None
    if command.subcommand == "serve":
        try:
            backend, argv = _extract_model_type_override(argv)
        except ValueError as exc:
            raise CommandParseError(str(exc)) from exc
        if backend not in ("auto", "diffusion"):
            return {
                "status": SKIPPED,
                "reason": f"--model-type {backend} (not the diffusion CLI)",
            }
        argv, positional = _normalize_positional_model_path(argv)
        add_multimodal_gen_serve_args(parser)
        default_args = {"warmup_mode": "server"}
    else:
        add_multimodal_gen_generate_args(parser)
    argv, notes = _substitute_placeholders(
        argv=argv, parser=parser, workdir=workdir, default_model=command.default_model
    )
    parsed, unknown = parser.parse_known_args(argv)
    if positional:
        parsed._sglang_explicit_arg_names = {"model_path"}
    if command.subcommand == "generate":
        parsed.request_id = "cookbook-check"
    result = {"status": OK, "reason": "", "notes": notes}
    try:
        ServerArgs.from_cli_args(parsed, unknown, default_args, argv=argv)
    except SystemExit as exc:
        if "unrecognized arguments" in str(exc):
            raise CommandParseError(str(exc).removeprefix("error: ")) from exc
        raise RuntimeError(f"SystemExit: {exc}") from exc
    return result


def _worker_env(hw: Hardware) -> dict[str, str]:
    """Env a worker runs with: no visible GPU and every model hub offline."""
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": hw.platform,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        # ModelScope ignores HF_HUB_OFFLINE, and the registry's snapshot fallback
        # would download a whole repo from it.
        "SGLANG_USE_MODELSCOPE": "false",
    }


def _command_env(env: dict[str, str]) -> dict[str, str]:
    """The page's env prefix, minus anything that would undo _worker_env."""
    return {
        key: PLACEHOLDER_RE.sub("dummy", value)
        for key, value in env.items()
        if key not in _worker_env(HARDWARE[STATIC_HW]) and not key.startswith("HF_")
    }


def _check_one(command: Command, hw: Hardware) -> dict:
    argv = [arg for arg in command.argv if arg != "--dry-run"]
    env = _command_env(command.env)
    cwd = os.getcwd()
    saved_env = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="cookbook-check-") as workdir:
        written = {}
        for name, content in command.files.items():
            path = Path(workdir, name.lstrip("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            written[name] = str(path)
        argv = [_relocate(arg, written) for arg in argv]
        os.chdir(workdir)
        os.environ.update(env)
        try:
            result = _parse_and_validate(command, argv, workdir)
        except CommandParseError as exc:
            result = {"status": PARSE_ERROR, "reason": f"error: {exc}"}
        except Exception as exc:  # every other failure is a validation result
            status, reason = _classify_failure(_exception_text(exc), hw)
            result = {"status": status, "reason": reason}
        finally:
            os.chdir(cwd)
            os.environ.clear()
            os.environ.update(saved_env)
    if result["status"] in (OK, VALIDATION_ERROR):
        model_path = model_path_of(argv)
        if model_path and PLACEHOLDER_RE.search(model_path):
            model_path = command.default_model
        missing = _uncached_model_index(model_path)
        if missing:
            result.update(status=SKIPPED, reason=missing)
    return result


def _install_hf_guard() -> None:
    """Make every Hub download in this process read the local cache only."""
    import huggingface_hub
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True
    originals = {
        "snapshot_download": huggingface_hub.snapshot_download,
        "hf_hub_download": huggingface_hub.hf_hub_download,
    }

    def cache_only(fn):
        def wrapper(*args, **kwargs):
            kwargs["local_files_only"] = True
            return fn(*args, **kwargs)

        return wrapper

    guarded = {name: cache_only(fn) for name, fn in originals.items()}
    for module in list(sys.modules.values()):
        for name, fn in originals.items():
            if isinstance(module, type(sys)) and vars(module).get(name) is fn:
                setattr(module, name, guarded[name])


def _mock_gpu(hw: Hardware) -> None:
    """Make torch and current_platform report one idle GPU of this hardware."""
    from types import SimpleNamespace
    from unittest import mock

    import torch

    total = int(hw.memory_gb * 1024**3)
    props = SimpleNamespace(
        name=hw.name,
        total_memory=total,
        major=hw.capability[0],
        minor=hw.capability[1],
        multi_processor_count=132,
        is_integrated=hw.shares_host_memory,
        gcnArchName="",
    )
    for name, value in {
        "is_available": lambda: True,
        "device_count": lambda: hw.device_count,
        "current_device": lambda: 0,
        "get_device_capability": lambda device=None: hw.capability,
        "get_device_name": lambda device=None: hw.name,
        "get_device_properties": lambda device=None: props,
        "mem_get_info": lambda device=None: (total, total),
        "empty_cache": lambda: None,
        "synchronize": lambda device=None: None,
        "set_device": lambda device: None,
    }.items():
        mock.patch.object(torch.cuda, name, value).start()

    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability

    capability = DeviceCapability(*hw.capability)
    platform_cls = type(current_platform)
    for name, fn in {
        "get_device_capability": lambda cls, device_id=0: capability,
        "get_device_name": lambda cls, device_id=0: hw.name,
        "get_device_uuid": lambda cls, device_id=0: f"GPU-mock-{device_id}",
        "get_device_total_memory": lambda cls, device_id=0: total,
        "get_available_gpu_memory": lambda cls, *args, **kwargs: hw.memory_gb,
        "device_shares_host_memory": lambda cls: hw.shares_host_memory,
        "is_full_nvlink": lambda cls, device_ids: True,
        "log_warnings": lambda cls: None,
    }.items():
        mock.patch.object(platform_cls, name, classmethod(fn)).start()
    # Capability probes are lru-cached per class; start them from the mock.
    for klass in platform_cls.__mro__:
        for attr in vars(klass).values():
            func = getattr(attr, "__func__", attr)
            if hasattr(func, "cache_clear"):
                func.cache_clear()


class _WorkerPayload(msgspec.Struct):
    hardware: Hardware
    commands: list[Command]


def run_worker(in_path: str, out_path: str) -> None:
    payload = msgspec.json.decode(Path(in_path).read_bytes(), type=_WorkerPayload)
    hw, commands = payload.hardware, payload.commands

    import logging

    if hw.platform == "cuda":
        _mock_gpu(hw)
    # Import the CLI modules (and the registry) before guarding the Hub calls.
    import sglang.cli.serve  # noqa: F401
    import sglang.multimodal_gen.registry  # noqa: F401
    import sglang.multimodal_gen.runtime.entrypoints.cli.generate  # noqa: F401
    import sglang.multimodal_gen.runtime.entrypoints.cli.serve  # noqa: F401

    _install_hf_guard()
    logging.disable(logging.WARNING)
    results = [_check_one(command, hw) for command in commands]
    Path(out_path).write_text(json.dumps(results), encoding="utf-8")


def _run_group(
    hw_id: str, hw: Hardware, commands: list[Command], scratch: str
) -> list[dict]:
    in_path = os.path.join(scratch, f"{hw_id}.in.json")
    out_path = os.path.join(scratch, f"{hw_id}.out.json")
    log_path = os.path.join(scratch, f"{hw_id}.log")
    payload = _WorkerPayload(hardware=hw, commands=commands)
    Path(in_path).write_bytes(msgspec.json.encode(payload))
    env = dict(os.environ)
    env.update(
        _worker_env(hw),
        PYTHONPATH=os.pathsep.join(
            filter(None, [str(REPO_ROOT / "python"), os.environ.get("PYTHONPATH")])
        ),
    )
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker", in_path, out_path],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=scratch,
        )
    if process.returncode != 0 or not os.path.exists(out_path):
        tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"worker for {hw_id} failed (exit {process.returncode}):\n{tail}"
        )
    return json.loads(Path(out_path).read_text(encoding="utf-8"))


def check_commands(commands: list[Command], jobs: int) -> list[dict]:
    results: list[dict | None] = [None] * len(commands)
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, command in enumerate(commands):
        hw = hardware_for(command.hw, command.vendor)
        if hw.platform == "npu":
            results[index] = {
                "status": SKIPPED,
                "reason": "Ascend NPU command; no NPU platform mock",
                "notes": [],
            }
        else:
            key = (
                command.hw
                if command.hw in HARDWARE
                else VENDOR_FALLBACK.get(command.vendor or "", STATIC_HW)
            )
            groups[key].append(index)

    with tempfile.TemporaryDirectory(prefix="cookbook-check-workers-") as scratch:

        def run(item):
            hw_id, indices = item
            return indices, _run_group(
                hw_id, HARDWARE[hw_id], [commands[i] for i in indices], scratch
            )

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for indices, group_results in pool.map(run, sorted(groups.items())):
                for index, result in zip(indices, group_results):
                    results[index] = result
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(commands: list[Command], results: list[dict]) -> str:
    counts = collections.Counter(
        (c.kind, r["status"]) for c, r in zip(commands, results)
    )
    lines = [
        "# Diffusion Cookbook command check",
        "",
        "| source | " + " | ".join(CLASSES) + " |",
        "|---|" + "---|" * len(CLASSES),
    ]
    for kind in ("static", "generated"):
        if any(c.kind == kind for c in commands):
            lines.append(
                f"| {kind} | "
                + " | ".join(str(counts[(kind, s)]) for s in CLASSES)
                + " |"
            )
    for status in CLASSES[1:]:
        rows = [(c, r) for c, r in zip(commands, results) if r["status"] == status]
        if not rows:
            continue
        lines += ["", f"## {status} ({len(rows)})", ""]
        for command, result in rows:
            lines.append(f"- `{command.source}` ({command.kind}): {result['reason']}")
            lines.append(f"  - `{command.text}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pages",
        nargs="*",
        help="Cookbook pages to scan (default: docs/cookbook/diffusion/**/*.mdx).",
    )
    parser.add_argument(
        "--no-static", action="store_true", help="Skip the code blocks."
    )
    parser.add_argument(
        "--generated",
        type=Path,
        help="Builder commands from docs/scripts/dump_diffusion_commands.mjs.",
    )
    parser.add_argument(
        "--fetch-model-index",
        action="store_true",
        help="Download model_index.json and small JSON configs first (no weights).",
    )
    parser.add_argument("--markdown-out", type=Path, help="Write the report here.")
    parser.add_argument("--json-out", type=Path, help="Write per-command results here.")
    parser.add_argument(
        "--list", action="store_true", help="Print the extracted commands and exit."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Worker processes.",
    )
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        run_worker(*args.worker)
        return 0

    commands: list[Command] = []
    if not args.no_static:
        pages = (
            [Path(page) for page in args.pages]
            if args.pages
            else sorted(COOKBOOK_DIR.rglob("*.mdx"))
        )
        for page in pages:
            commands += extract_page(page)
    if args.generated:
        commands += load_generated(args.generated)
    if args.list:
        for command in commands:
            print(f"{command.source} [{command.hw}] {command.text}")
        return 0
    if args.fetch_model_index:
        for line in fetch_model_indices(commands):
            print(f"fetch: {line}", file=sys.stderr)

    started = time.monotonic()
    results = check_commands(commands, args.jobs)
    report = render_markdown(commands, results)
    print(report.split("\n## ")[0].rstrip())
    print(f"\n{len(commands)} commands checked in {time.monotonic() - started:.0f} s")
    if args.markdown_out:
        args.markdown_out.write_text(report, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(
                [
                    {"source": c.source, "kind": c.kind, "command": c.text, **r}
                    for c, r in zip(commands, results)
                ],
                indent=1,
            ),
            encoding="utf-8",
        )
    for command, result in zip(commands, results):
        if result["status"] in (PARSE_ERROR, VALIDATION_ERROR):
            print(
                f"{result['status']}: {command.source}: {result['reason']}\n  {command.text}"
            )
    return (
        1 if any(r["status"] in (PARSE_ERROR, VALIDATION_ERROR) for r in results) else 0
    )


if __name__ == "__main__":
    sys.exit(main())
