"""
Golden-snapshot equivalence suite for the v0.6.0 CLI refactor (Task 1 / Plan A.1).

Captured AGAINST THE UNTOUCHED MONOLITH (cli.py as of v0.5.9, pre-refactor).
The refactor's guarantee is byte-identical output; any diff after a refactor
step is a regression, not formatting.

How it works
------------
Each case spawns `python -m roampal.cli <args>` as a real subprocess with a
fully isolated environment (temp HOME/APPDATA/USERPROFILE/XDG_CONFIG_HOME,
NO_COLOR=1, unreachable fixed ports), captures
    stdout + stderr + exit code
and compares it against a golden file under `tests/unit/golden/`.

Regeneration (deliberate, explicit):
    set ROAMPAL_REGENERATE_GOLDEN=1  (or pytest env) to recapture all goldens
    and rewrite the fixture files. Goldens are regenerated, never hand-edited.
One deliberate diff is expected during Task 4 (epilog becomes parser-generated,
additive only) and later for the Task 15 version bump; see
dev/docs/releases/v0.6.0/RELEASE_NOTES.md Test Plan A.

Non-determinism control: every absolute path from the temp env is normalized
to <ROOT>, so snapshots are stable across machines and OSes. Server-dependent
cases point at a fixed unoccupied port (29191) so the result is always the
"deterministic connection-refused path". Config seeding writes to every
platform's _config_dir() candidate (Windows APPDATA plus XDG/HOME config
dirs) — without that, the same case yields different output on Linux CI vs
Windows dev because _config_dir() is OS-dependent.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
REPO_ROOT = Path(__file__).resolve().parents[6]  # repo root, not the roampal package dir
TESTS_DIR = Path(__file__).resolve().parent
GOLDEN_DIR = TESTS_DIR / "golden"
PY_EXE = sys.executable

REGEN = os.environ.get("ROAMPAL_REGENERATE_GOLDEN") == "1"

CASES = [
    # (id, args, initial_state)
    # Byte-golden cases only: commands whose output is machine-independent.
    # doctor and reembed BOTH load the real ONNX models (HF stderr warnings,
    # model-dep section output), so they are covered by structural tests below
    # instead of byte snapshots — see test_plan note in the module docstring.
    ("help", ["--help"], {}),
    ("version", ["--version"], {}),
    ("status_json_stopped", ["status", "--json", "--port", "29191"], {}),
    ("stats_json_stopped", ["stats", "--json", "--port", "29191"], {}),
    ("profile_list_empty", ["profile", "list"], {}),
    ("profile_list_three", ["profile", "list"], {"profiles": ["alpha", "beta", "gamma"]}),
    ("profile_show_default", ["profile", "show"], {}),
    ("profile_show_persisted_unregistered", ["profile", "show"], {"use": "main"}),
    ("profile_show_persisted_registered", ["profile", "show"], {"use": "alpha", "profiles": ["alpha", "beta", "gamma"]}),
    ("ingest_help", ["ingest", "--help"], {}),
    ("sidecar_status_unconfigured", ["sidecar", "status"], {}),
]


def _register_profiles(root: Path, names: list[str]) -> None:
    """Seed a profiles.json registry in ProfileRegistry's format at every
    platform config-dir candidate.

    The loader iterates TOP-LEVEL keys (name -> Profile dict); a nested
    "profiles" key would be interpreted as a profile named 'profiles'.
    `profile_manager._config_dir()` is OS-dependent (Windows APPDATA/Roampal;
    Linux/macOS XDG_CONFIG_HOME or ~/.config + /roampal lowercase), so we
    seed all of them — whichever one resolves for the child env is hit.
    """
    import json

    import roampal.profile_manager as pm

    payload = json.dumps(
        {
            n: {"name": n, "slug": pm.profile_slug(n), "path": None}
            for n in names
        }
    )
    candidates = [
        root / "Roampal",                     # win32: APPDATA
        root / "xdg" / "roampal",             # XDG_CONFIG_HOME
        root.parent / ".config" / "roampal",  # HOME fallback
    ]
    for d in candidates:
        d.mkdir(parents=True, exist_ok=True)
        (d / "profiles.json").write_text(payload, encoding="utf-8")


def _normalize(text: str, root: Path) -> str:
    """Normalize machine-dependent bits to stable placeholders."""
    text = text.replace(str(root), "<ROOT>")
    # Normalize backslashes of temp-root-derived subpaths (e.g. <ROOT>\Roampal\data)
    text = text.replace("<ROOT>\\", "<ROOT>/")
    # Normalize the mktemp("home") counter suffix — it depends on how many
    # tests in the session created a "home" temp dir before this one, so a
    # golden captured in a full-suite run (home12) would mismatch a
    # standalone run (home0). Only the segment right after <ROOT> is
    # affected; anything deeper is inside a different tree.
    text = re.sub(r"(<ROOT>/home)\d+", r"\1N", text)
    # Task 27 (cross-platform goldens): the checkout path itself never
    # belongs in output — on CI it is /home/runner/work/<repo>/roampal-core,
    # a machine-varying absolute path. Map it (either separator form).
    text = text.replace(str(REPO_ROOT), "<REPOROOT>")
    text = text.replace(str(REPO_ROOT).replace("\\", "/"), "<REPOROOT>")
    # Absolute source paths from tracebacks (site-packages / editable install)
    py_dir = str(Path(PY_EXE).parent)
    if py_dir in text:
        text = text.replace(py_dir, "<PYDIR>")
    py = PY_EXE.replace("\\\\", "\\")
    if py in text:
        text = text.replace(py, "<PYEXE>")
    site = str(REPO_ROOT / "roampal")
    if site in text:
        text = text.replace(site, "<PKGEROOT>")
    # Task 27: site/dist-packages dirs NOT under the interpreter's dir are
    # machine-specific (a venv borrowing another interpreter's packages
    # leaks a foreign site-packages path — observed as a home-dir path that
    # failed <REPOROOT>/<PYDIR> masking in test_doctor /
    # test_reembed cases). Mask everything up to and including the
    # site/dist-packages component, either direction, if present.
    text = re.sub(
        r"[^\s:\"']*?[/\\](?:site|dist)-packages[/\\]",
        "<SITESP>/",
        text,
    )
    framed = re.sub(r"roampal-cli\.py:\d+", "roampal-cli.py:LINE", text)
    # Task 42 (Python 3.13): argparse's help layout width is version-dependent
    # (3.13 lays the command column one slot wider, so `summarize` no longer
    # wraps onto a second line and every row gains a space). Same text,
    # different spacing. Join WRAPPED rows back together and collapse the
    # column gap so the comparison is independent of argparse's version:
    #   - a command/option token alone on a line, its description on the next
    #     indented line (one or more spaces + "  " minimum) → one row
    #   - runs of 2+ spaces between the token and its description → exactly
    #     2 spaces
    # The text itself is still compared exactly; only the layout varies.
    return _normalize_help_layout(framed)


def _normalize_help_layout(text: str) -> str:
    """Join argparse's wrapped help rows and collapse column gaps (Task 42).

    argparse's wrapped-row shape (both versions, whenever the description
    column cannot fit on the token's line):
        `    summarize`                     <- token alone, indent N
        `              Summarize existing`  <- description, indent > N
    Merge such pairs into `    summarize  Summarize existing` and collapse
    every run of 2+ spaces between a token and its description (indented
    lines only) to exactly 2 spaces. A continuation that does NOT look like
    a description (usage lines, section headers, blank lines, unindented
    text) ends the pair instead of merging. The text itself is still
    compared exactly; only the layout varies.
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        nxt_indent = len(nxt) - len(nxt.lstrip()) if nxt is not None else 0
        line_indent = len(line) - len(line.lstrip())
        # A bare-token line's spaces are ONLY its leading indent — any run
        # of 2+ spaces AFTER the first non-blank character means a
        # token+description row (or a deeper-indented header).
        # Section headers (`Setup:`, `Memory:`) and metavar placeholders
        # (`<command>`) are NOT token rows: they end with `:` or start
        # with `<`, and their next line is a SIBLING (same-or-shallower
        # indent for headers; same indent for `<command>`), so the
        # deeper-next-line check mostly excludes them — but a header
        # directly followed by its items is deeper, hence the explicit
        # trailing-`:` rule.
        stripped = line.lstrip()
        after_indent = line[len(line) - len(stripped):]
        is_bare_token = (
            line.startswith(" ")
            and stripped
            and not re.search(r"\S\s\s", after_indent)
            and not stripped.startswith("usage:")
            and not stripped.startswith("-")
            and not stripped.startswith("<")  # metavar placeholders
            and not stripped.endswith(":")  # section headers
        )
        is_desc_next = (
            nxt is not None
            and nxt.strip()
            and nxt_indent > line_indent
            # a description never looks like a new token row/usage/header
            and not nxt.lstrip().startswith("usage:")
        )
        if is_bare_token and is_desc_next:
            out.append(f"{line.rstrip()}  {nxt.strip()}")
            i += 2
            continue
        out.append(line)
        i += 1

    # Collapse column gaps: 2+ spaces between a command/option token and its
    # description → exactly 2 spaces. Only inside indented lines (never the
    # `usage:` line or unindented banner text). `    summarize  Summarize`
    # and 3.13's wider `    summarize   Summarize` both become the same row.
    collapsed = []
    for line in out:
        if line.startswith(" "):
            collapsed.append(re.sub(r"(?<=\S)   +(?=\S)", "  ", line))
        else:
            collapsed.append(line)
    return "\n".join(collapsed)


def _run_case(args: list[str], state: dict, tmp_path_factory) -> tuple[str, int]:
    home = tmp_path_factory.mktemp("home")
    appdata = home / "AppDataRoaming"
    appdata.mkdir()
    xdg = home / "xdg"
    xdg.mkdir()
    temp_root = home.parent  # pytest tmp-tree root, used by normalization

    if state.get("profiles"):
        _register_profiles(appdata, state["profiles"])
    if state.get("use"):
        # Persisted active profile: env isolation happens inside the child.
        # Write via a helper subprocess that runs WITH the same isolated env.
        base_env = _base_env(appdata, xdg)
        code = (
            "import roampal.profile_manager as pm; "
            f"pm.write_active_profile_file({state['use']!r})"
        )
        subprocess.run(
            [PY_EXE, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=base_env,
            timeout=60,
        )

    env = _base_env(appdata, xdg)
    completed = subprocess.run(
        [PY_EXE, "-m", "roampal.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",  # pinned decoding side (see _base_env note)
        errors="replace",
        cwd=str(REPO_ROOT),
        env=env,
        timeout=120,
    )
    return completed.stdout + completed.stderr, completed.returncode, temp_root


def test_reembed_dry_run_offline(tmp_path_factory, unsandboxed_home):
    """reembed --dry-run is structural, not golden.

    Excluded from the byte-level goldens on purpose: the run loads the real
    ONNX embedder, and the surrounding stderr (HF symlink/token warnings,
    cache paths) is machine- and network-dependent. The contract asserted
    here is the deterministic slice: exit 0 and the exact success line.
    Model loads are skipped when no local HF cache exists (fresh CI runners).

    Task 26: the HF-cache skip-check is the SANCTIONED unsandboxed opt-in —
    it must read the process's REAL cache location to decide skip-vs-run on
    this machine (`unsandboxed_home` restores it just for the check; the
    child CLI itself still runs under its own fully isolated _base_env)."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.exists() or not any(hf_cache.glob("models--*")):
        pytest.skip("no local HF cache; reembed would download models in this test")

    stdout, exit_code, temp_root = _run_case(
        ["reembed", "--dry-run"], {}, tmp_path_factory
    )
    assert exit_code == 0, stdout
    assert "Dry run complete for profile 'default': 0 record(s) would be re-embedded" in stdout
    normalized = _normalize(stdout, temp_root)
    assert str(Path.home()) not in normalized


def test_doctor_offline_clean(tmp_path_factory):
    """doctor is structural, not golden.

    cmd_doctor initializes the real memory system (loads ONNX models) and its
    diagnostics depend on the host env (deps versions, torch presence, HF
    warning interleaving in stderr). Byte-exactness is not achievable across
    machines; the deterministic contract asserted here is: exit 0, banner,
    mode line, and the final all-checks summary. Doctor makes no network calls
    (verified: `NET: []` over its body in the task-1 recon) — the banner cats
    PyPI update checks only print in other commands via `print_update_notice`.
    """
    stdout, exit_code, temp_root = _run_case(["doctor"], {}, tmp_path_factory)
    assert exit_code == 0, stdout
    assert "Roampal Doctor - Diagnostics" in stdout
    assert "Mode:" in stdout
    normalized = _normalize(stdout, temp_root)
    assert str(Path.home()) not in normalized
    assert str(Path.cwd()) not in normalized


def _base_env(appdata: Path, xdg: Path) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in {
            "APPDATA",
            "HOME",
            "USERPROFILE",
            "XDG_CONFIG_HOME",
            "ROAMPAL_PROFILE",
            "ROAMPAL_DATA_PATH",
            "ROAMPAL_EMBED_MODEL",
        }
    }
    env["APPDATA"] = str(appdata)
    env["HOME"] = str(appdata.parent)
    env["USERPROFILE"] = str(appdata.parent)
    env["XDG_CONFIG_HOME"] = str(xdg)
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    env["COLUMNS"] = "80"  # stable argparse help wrapping under non-tty
    # v0.6.0 Task 15: pin the child's stdout encoding. Without this, whether
    # the child writes UTF-8 or cp1252 depends on the ESCAPING CONSOLE of
    # whoever runs pytest; em-dashes in output then mismatch goldens that
    # were captured under a different console. text=True in subprocess.run
    # decodes with the parent's locale — child UTF-8 + parent cp1252
    # produces mojibake. Both sides now pinned UTF-8.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def pytest_generate_tests(metafunc):
    if "golden_case" in metafunc.fixturenames:
        metafunc.parametrize(
            "golden_case", CASES, ids=[c[0] for c in CASES]
        )


def test_cli_golden(golden_case, tmp_path_factory):
    case_id, args, state = golden_case
    stdout, exit_code, temp_root = _run_case(args, state, tmp_path_factory)
    captured = f"### exit: {exit_code}\n--- stdout+stderr ---\n{stdout}"
    normalized = _normalize(captured, temp_root)

    golden_file = GOLDEN_DIR / f"{case_id}.golden.txt"

    if REGEN:
        # Deliberate recapture (e.g. after the Task 4 epilog change). Writes
        # the new golden and passes; CI never runs with ROAMPAL_REGENERATE_GOLDEN
        # set, so a rewritten-without-NEED golden must be reviewed in the PR.
        golden_file.write_text(normalized, encoding="utf-8", newline="\n")
        return

    if not golden_file.exists():
        golden_file.write_text(normalized, encoding="utf-8", newline="\n")
        pytest.fail(
            f"No golden file for {case_id!r}; created it. Run the suite a second "
            "time (without ROAMPAL_REGENERATE_GOLDEN) to verify determinism: "
            "this pass now compares against the file it just wrote."
        )

    expected = golden_file.read_text(encoding="utf-8")
    if normalized != expected:
        diff_lines = [
            f"-{a!r}\n+{b!r}"
            for a, b in zip(expected.splitlines(), normalized.splitlines())
            if a != b
        ]
        pytest.fail(
            f"Golden mismatch for {case_id!r} (first differing lines shown; "
            f"full detail: diff {golden_file} against current output).\n"
            + "\n".join(diff_lines[:20])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
