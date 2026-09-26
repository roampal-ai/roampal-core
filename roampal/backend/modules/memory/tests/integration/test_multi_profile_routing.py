"""
Many-profiles-at-once routing — Round 2 Item 6/7 (v0.6.0, Task 21).

DESIGNED-RED UNTIL TASKS 17-19 LAND (see IMPLEMENTATION_TASKS.md Round 2).
Audit finding F1: a request that does not name its profile is resolved by
the SHARED SERVER from the server process's OWN cwd and env — i.e. from
whichever MCP process or hook happened to spawn (or last respawn) it. With
a binding in play, another project's memories can land in the server's
bound profile. Written first per the "red before green" cadence so the
failures below are the proof the suite covers F1, the same pattern as
Task 2's deliberately-red duplicate scan and Task 11's dedup negative
controls.

The requests are the TARGET CONTRACT the fixed tree must serve (what the
real clients will send after Tasks 17-19):

- hook seam, unbound project B, no env  -> X-Roampal-Profile: "default"
  (Task 18: clients ALWAYS name their profile, explicit default included)
- OpenCode seam, bound project C        -> X-Roampal-Cwd: <dir C> only
  (Task 19: plugin's own resolution empty -> cwd header; the SERVER
  resolves binding_for_cwd(cwd=<header>) for that request)
- env-var client D                      -> X-Roampal-Profile: "env-d"
  (env var resolves CLIENT-side; the header carries it)
- legacy/headerless client              -> NO headers at all
  (Task 17: old installed plugin copies keep working — the server
  resolves persisted `use` -> `default`, never its own cwd binding,
  never a spawner's leaked ROAMPAL_PROFILE)

Scenario (one real server, cwd INSIDE bound project A, run twice — param
`leaked_server_env`):
  - run 1: server env ROAMPAL_PROFILE unset — F1's cwd-binding half
  - run 2: server env ROAMPAL_PROFILE=work-a — F1's env-leak half
    (a spawner that handed its own project env to the shared server)

Every client stores (memory-bank fact; the hook client additionally a
full exchange through /api/hooks/stop) and searches, all gathered
concurrently. Contract: each store + search lands ONLY in its own
profile — no cross-reads, no cross-writes. Because the shared server was
spawned from bound project A, target profile "work-a" must receive
NOTHING at all (asserted on disk: its data dir is never created).

Current-tree red demonstrates as:
  - client C's and the headerless client's stores both land in work-a
    (cwd header ignored; headerless resolves the server's own binding /
    leaked env), co-locating foreign markers in one wrong profile —
    caught by the cross-contamination check
  - their TARGET profiles (proj-c, main) are never even created; the
    server's own bound profile (work-a) materializes instead — caught
    by the disk-level checks
Tasks 17-19 flip every assertion green without touching this file.
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", "..")))

import httpx
import pytest

import roampal.profile_manager as pm
import roampal.server.main as srv
from roampal.backend.modules.memory.search_service import SearchService


class _FakeEmbeddingService:
    """Deterministic md5-seeded embedder (768-dim real cosine geometry).

    Same fixture shape as test_dedup_regression.py: NO ONNX model load,
    distinct texts embed far apart, stable across processes. Injected into
    the server's _shared_embed_service port so every request-path UMS
    shares it and stays deterministic/local.
    """

    DIM = 768

    def _token_vector(self, text: str):
        import hashlib
        import random

        seed = int.from_bytes(
            hashlib.md5(text.encode("utf-8")).digest()[:4], "big"
        )
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.DIM)]
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    async def embed_text(self, text, role="passage"):
        return self._token_vector(text)

    async def prewarm(self):
        return None


MARKER_FACT_B = "MULTI_ROUTE_B_fact: hook seam profile marker for unbound project B"
MARKER_FACT_C = "MULTI_ROUTE_C_fact: opencode cwd header marker for bound project C"
MARKER_FACT_D = "MULTI_ROUTE_D_fact: env var client marker for profile env-d"
MARKER_FACT_H = "MULTI_ROUTE_H_fact: headerless client marker lands in persisted default"

EXCHANGE_USER_B = "MULTI_ROUTE_B_exchange user message: how does routing behave?"
EXCHANGE_ASSISTANT_B = "MULTI_ROUTE_B_exchange reply: it is still being fixed"

# (dir A holds the server cwd; dir B is the unbound hook project; dir C is the
# bound OpenCode project; dir D is the env-var client's project.)
PROFILES_SEEDED = ("work-a", "proj-c", "env-d", "main")


@pytest.fixture(autouse=True)
async def _cancel_warmup_tasks():
    """Cancel UMS v0.5.9 background warmup / reembed tasks after each test.

    Standard integration-suite leak-prevention pattern (same as
    test_phantom_cleanup_safety.py / test_dedup_regression.py): orphaned
    asyncio.to_thread workers hang the next initialize() on Python 3.10.
    The fake embed service + neutralized CE make these trivial here, but
    per-profile UMS instances on the request paths still schedule them.
    """
    yield
    try:
        for task in asyncio.all_tasks():
            if task.get_name() in ("warmup_ce", "warmup_embedding", "reembed") and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
    except RuntimeError:
        pass


@pytest.fixture
def routing_env(tmp_path, monkeypatch, request):
    """Isolated registry + data root + server cwd inside bound project A.

    Parametrized on leaked_server_env: True adds ROAMPAL_PROFILE=work-a
    to the server environment (F1's env half), False leaves it unset
    (F1's cwd-binding half). Auto-locates every seeded profile under the
    temp APPDATA base, so a leaked profile's data dir is direct evidence.
    """
    leaked = request.param

    tmp = tmp_path
    # Config/data roots on every supported platform driven into tmp.
    monkeypatch.setenv("APPDATA", str(tmp))
    xdg_conf = tmp / "xdg_config"
    xdg_data = tmp / "xdg_data"
    xdg_conf.mkdir()
    xdg_data.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_conf))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))
    monkeypatch.setenv("HOME", str(tmp))
    monkeypatch.delenv("ROAMPAL_DATA_PATH", raising=False)
    monkeypatch.delenv("ROAMPAL_DEV", raising=False)
    if leaked:
        monkeypatch.setenv("ROAMPAL_PROFILE", "work-a")
    else:
        monkeypatch.delenv("ROAMPAL_PROFILE", raising=False)

    dir_a = tmp / "projectA"
    dir_b = tmp / "projectB"
    dir_c = tmp / "projectC"
    for d in (dir_a, dir_b, dir_c):
        d.mkdir()

    reg_path = pm._registry_path()
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: None for name in PROFILES_SEEDED}
    payload["bindings"] = {str(dir_a): "work-a", str(dir_c): "proj-c"}
    reg_path.write_text(json.dumps(payload), encoding="utf-8")

    # Persisted `profile use` = main: the headerless fallback target.
    pm.write_active_profile_file("main")

    # THE server-side F1 condition: shared server launched with cwd in
    # bound project A (real cwd, restored by monkeypatch after the test).
    monkeypatch.chdir(dir_a)

    fake = _FakeEmbeddingService()
    monkeypatch.setattr(srv, "_shared_embed_service", fake)
    # Cross-encoder is neutralized (cosine-only fallback, _rerank_skipped)
    # so ranking is deterministic and free of ONNX loads.
    monkeypatch.setattr(SearchService, "_load_ce", lambda self: False)

    # Freeze server profile state between the two parametrized runs.
    srv._memory_by_profile.clear()
    srv._session_manager_by_profile.clear()
    srv._search_cache.clear()
    srv._injection_map.clear()
    # Fresh asyncio primitives per run (module globals bind to a loop).
    monkeypatch.setattr(srv, "_init_lock", asyncio.Lock())

    yield {
        "leaked_server_env": leaked,
        "dir_a": dir_a,
        "dir_b": dir_b,
        "dir_c": dir_c,
    }

    srv._memory_by_profile.clear()
    srv._session_manager_by_profile.clear()
    srv._search_cache.clear()
    srv._injection_map.clear()


def _data_base() -> "os.PathLike":
    return pm.system_default_data_path()


def _result_texts(resp_json):
    out = []
    for r in resp_json.get("results", []):
        text = r.get("text")
        if not text:
            text = r.get("content") or r.get("metadata", {}).get("text", "")
        out.append(text or "")
    return out


def _probe_prefix(marker: str) -> str:
    """Unique stable prefix identifying exactly one client's marker text."""
    return marker.split("_fact")[0]


async def _search_own(client, headers, own_marker):
    return await client.post(
        "/api/search",
        json={"query": own_marker, "limit": 8},
        headers=headers,
    )


@pytest.mark.parametrize("routing_env", [False, True], indirect=True)
async def test_unregistered_profile_name_fails_explicit(routing_env):
    """Routing contract (Item 6 / Task 12 acceptance): a request naming an
    UNREGISTERED profile — via header or via a binding — must fail
    explicitly (404 + create hint), never silently write into the default
    store (the UMS ProfileNotFoundError fallback would cross-profile-bleed
    now that headers/bindings are the routing source)."""
    app = srv.create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp_header = await client.post(
            "/api/memory-bank/add",
            headers={"X-Roampal-Profile": "not-a-profile"},
            json={"content": "UNIT_TEST_MARKER unregistered header", "tags": ["identity"]},
        )
        assert resp_header.status_code == 404
        assert "not registered" in resp_header.text
        assert "roampal profile create" in resp_header.text

        resp_cwd = await client.post(
            "/api/search",
            # dir B is UNBOUND — the cwd header falls through to the
            # persisted `use` (main, registered); the header never names
            # the unregistered profile.
            headers={"X-Roampal-Cwd": str(routing_env["dir_b"])},
            json={"query": "UNIT_TEST_MARKER unregistered cwd probe", "limit": 3},
        )
        assert resp_cwd.status_code == 200  # unbound dir -> use -> main (registered)  # unbound dir -> use -> main (registered)

    base = _data_base()
    # The only UMS buckets that may exist: persisted-use target (main) and
    # default. NO store for the unregistered name, NOT even the root bleed.
    for slug in ("not_a_profile", "not-a-profile"):
        assert not (base / slug).exists(), (
            f"unregistered profile silently wrote into {slug!r} store"
        )
    leak_dir = base / "work_a"
    assert not leak_dir.exists()


@pytest.mark.parametrize("routing_env", [False, True], indirect=True)
async def test_many_profiles_at_once_route_independently(routing_env):
    """One server, parallel clients on different profiles — every store and
    search lands only in its own profile. Written first as the designed RED
    (F1 proof); green since Tasks 17-19.
    """
    app = srv.create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        h_hook_default = {"X-Roampal-Profile": "default"}
        h_opencode_cwd = {"X-Roampal-Cwd": str(routing_env["dir_c"])}
        h_env_client = {"X-Roampal-Profile": "env-d"}
        h_headerless = {}

        # ---- Parallel in-profile STORES ----
        stop_b, add_b, add_c, add_d, add_h = await asyncio.gather(
            client.post(
                "/api/hooks/stop",
                headers=h_hook_default,
                json={
                    "conversation_id": "sess-b",
                    "user_message": EXCHANGE_USER_B,
                    "assistant_response": EXCHANGE_ASSISTANT_B,
                    "lifecycle_only": False,
                    "noun_tags": ["routing", "hook"],
                },
            ),
            client.post(
                "/api/memory-bank/add",
                headers=h_hook_default,
                json={
                    "content": MARKER_FACT_B,
                    "tags": ["identity"],
                    "noun_tags": ["routing"],
                },
            ),
            client.post(
                "/api/memory-bank/add",
                headers=h_opencode_cwd,
                json={
                    "content": MARKER_FACT_C,
                    "tags": ["identity"],
                    "noun_tags": ["routing"],
                },
            ),
            client.post(
                "/api/memory-bank/add",
                headers=h_env_client,
                json={
                    "content": MARKER_FACT_D,
                    "tags": ["identity"],
                    "noun_tags": ["routing"],
                },
            ),
            client.post(
                "/api/memory-bank/add",
                headers=h_headerless,
                json={
                    "content": MARKER_FACT_H,
                    "tags": ["identity"],
                    "noun_tags": ["routing"],
                },
            ),
        )

        for label, resp in (("hook-exchange", stop_b), ("hook-fact", add_b),
                            ("opencode-fact", add_c), ("env-fact", add_d),
                            ("headerless-fact", add_h)):
            assert resp.status_code == 200, f"[{label}] store failed: {resp.text}"

        # ---- Parallel searches: own-profile + foreign-probe per client ----
        s_b, s_c, s_d, s_h = await asyncio.gather(
            _search_own(client, h_hook_default, MARKER_FACT_B),
            _search_own(client, h_opencode_cwd, MARKER_FACT_C),
            _search_own(client, h_env_client, MARKER_FACT_D),
            _search_own(client, h_headerless, MARKER_FACT_H),
        )
        for label, resp in (("hook/default", s_b), ("opencode/proj-c", s_c),
                            ("env/env-d", s_d), ("headerless/main", s_h)):
            assert resp.status_code == 200, f"[{label}] search failed: {resp.text}"
        texts_b, texts_c, texts_d, texts_h = (
            _result_texts(r.json()) for r in (s_b, s_c, s_d, s_h)
        )

        all_markers = (MARKER_FACT_B, MARKER_FACT_C, MARKER_FACT_D, MARKER_FACT_H)

        # --- Own marker must exist in the client's own target profile.
        # On the F1 tree these technically PASS for C/headerless (their
        # stores were co-routed into work-a, and the search follows them
        # there) — the designed red fires in the cross-contamination and
        # disk-level checks below. These rows stay as the contract: after
        # 17-19 the search targets the true profile, so a future regression
        # that DIVERTS a client to a different wrong profile fails here.
        for label, texts, own in (
            ("hook/default", texts_b, MARKER_FACT_B),
            ("opencode/proj-c", texts_c, MARKER_FACT_C),
            ("env/env-d", texts_d, MARKER_FACT_D),
            ("headerless/main", texts_h, MARKER_FACT_H),
        ):
            assert any(own in t for t in texts), (
                f"[{label}] OWN MARKER NOT FOUND in its own profile — the fact "
                f"was routed elsewhere (F1 repro, "
                f"server_env_leaked={routing_env['leaked_server_env']}): {texts}"
            )

        # --- No cross-reads: a client's search must never surface another
        # client's marker in its profile.
        for label, texts, own in (
            ("hook/default", texts_b, MARKER_FACT_B),
            ("opencode/proj-c", texts_c, MARKER_FACT_C),
            ("env/env-d", texts_d, MARKER_FACT_D),
            ("headerless/main", texts_h, MARKER_FACT_H),
        ):
            foreign_hits = [
                t for t in texts
                for m in all_markers
                if m is not own and _probe_prefix(m) in t
            ]
            assert not foreign_hits, (
                f"[{label}] CROSS-CONTAMINATION: foreign markers surfaced in a "
                f"foreign profile's search: {foreign_hits}"
            )

        # The exchange went through the hook seam: it must be retrievable in
        # the hook profile's working collection (same profile as its fact).
        exch = await client.post(
            "/api/search",
            headers=h_hook_default,
            json={"query": EXCHANGE_USER_B, "limit": 8},
        )
        assert exch.status_code == 200
        assert any("MULTI_ROUTE_B_exchange" in t for t in _result_texts(exch.json())), (
            "hook-seam exchange not retrievable in the hook profile"
        )

        # ---- Disk-level routing verification ----
        # Each target profile must have been materialized (its UMS data dir
        # created), and the server's own bound profile must NOT exist.
        base = _data_base()
        for slug, label in (("main", "headerless/main"),
                            ("proj_c", "opencode/proj-c"),
                            ("env_d", "env/env-d")):
            assert (base / slug).exists(), (
                f"[{label}] target profile store was never created — the fact "
                f"was routed elsewhere (F1 repro, "
                f"server_env_leaked={routing_env['leaked_server_env']})"
            )
        leak_dir = base / "work_a"  # slug(name) for "work-a"
        assert not leak_dir.exists(), (
            f"PROFILE LEAK: a request without identity was routed to the "
            f"shared server's own cwd binding ({routing_env['leaked_server_env']=}) and "
            f"materialized data in {leak_dir}"
        )
