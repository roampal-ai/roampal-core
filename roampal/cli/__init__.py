"""roampal.cli — v0.6.0 Task 3/5 scaffold.

Entry point for `roampal <command>` (console script `roampal = roampal.cli:main`)
and attribute access to the CLI surface. During Tasks 5-10 the monolith lives
at `roampal/cli/_monolith_impl.py` (moved verbatim); this package `__init__`
re-exports `main()` and proxies attribute access to the impl module PLUS every
moved group module (`update_check`, `setup`, ...), so each group move keeps
existing `from roampal.cli import X` call sites and `patch("roampal.cli.X")`
targets working unchanged.

Proxy design (mock-compatible PEP 562 style):
- Names are MATERIALIZED into this module's own ``__dict__`` the first time
  they resolve through the proxy. ``mock.patch`` decides restore-vs-delete
  from ``attribute in target.__dict__`` — without materialization, teardown
  would delete the impl's attribute instead of restoring the original.
- Writes rebind the value in EVERY ``roampal.cli.*`` module holding the same
  object (Task 25 / Item 8: from-imports share one object across group
  modules, so a single-owner rebind would leak the un-patched original into
  the other call-sites) AND update the package mirror (so patch's
  get_original/restore round-trip stays in sync). Deletes remove from all
  holders and the mirror.
- Dunders and names local to this module bypass the proxy — the ``__class__``
  swap itself recursed in the first Task 3 variant (documented there).
- Lookup order: impl first, then loaded ``roampal.cli.*`` group modules via
  sys.modules vars-dicts (never getattr on siblings — that re-enters the
  proxy and recursed).
"""

import sys
from types import ModuleType

from roampal.cli import _monolith_impl as _impl_module
from roampal.cli._monolith_impl import main  # noqa: F401 re-export entry


def _import_all_group_modules():
    """Import every roampal.cli.* group module eagerly so the proxy lookup,
    dir(), and from-imports resolve immediately (mirrors the monolith's
    single-import behavior). New group modules added by Tasks 6-10 are picked
    up automatically from the package directory listing."""
    pkg = sys.modules[__name__]
    try:
        import pkgutil
        import importlib

        for spec in pkgutil.iter_modules(pkg.__path__):
            name = f"{__name__}.{spec.name}"
            if name in sys.modules:
                continue
            # __main__ must NOT be pre-imported: under `python -m roampal.cli`,
            # runpy expects to execute it itself; pre-importing triggers a
            # RuntimeWarning on stderr (byte-golden pollution).
            if spec.name == "__main__":
                continue
            importlib.import_module(name)
    except Exception as e:  # pragma: no cover - diagnostics only
        import logging

        logging.getLogger("roampal.cli").warning(
            "roampal.cli group import failed: %s", e
        )


def _iter_cli_modules():
    """Yield attribute-dict backends in lookup order: impl, then group modules."""
    yield _impl_module
    pkg = sys.modules.get("roampal.cli")
    if pkg is None:
        return
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return
    import pkgutil

    for spec in pkgutil.iter_modules(pkg_path):
        mod = sys.modules.get("roampal.cli." + spec.name)
        if mod is None or mod is _impl_module or mod is pkg:
            continue
        yield mod


def _lookup(name):
    """Return (owning_module, value). impl first, then loaded group modules."""
    impl_vars = vars(_impl_module)
    if name in impl_vars:
        return _impl_module, impl_vars[name]
    pkg = sys.modules.get("roampal.cli")
    if pkg is not None and getattr(pkg, "__path__", None):
        import pkgutil

        for spec in pkgutil.iter_modules(pkg.__path__):
            mod = sys.modules.get("roampal.cli." + spec.name)
            if mod is None or mod is _impl_module:
                continue
            mod_vars = vars(mod)
            if name in mod_vars:
                return mod, mod_vars[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _holders(name):
    """Task 25 / Item 8: every roampal.cli.* module holding ``name``.

    From-imports share one object across modules (``memory_cmds.py``, ``setup.py``
    etc. each hold the same ``GREEN``), so a single-owner rebind would leave
    the other call-sites seeing the original after ``patch("roampal.cli.X")``.
    Only identity-equal holders are rebound (which on restore grounds the
    result: each holder gets back the one original)."""
    try:
        _, original = _lookup(name)
    except AttributeError:
        return (_impl_module,)  # unknown name (mock.create) → impl owns it
    return tuple(
        mod
        for mod in _iter_cli_modules()
        if name in vars(mod) and vars(mod)[name] is original
    )


class _CliPackage(ModuleType):
    """Module proxy: forwards missing names via impl + group modules."""

    def __getattr__(self, name):
        owner, value = _lookup(name)
        # Materialize into our own dict so mock.patch sees local=True and
        # its teardown restores the original value instead of deleting it.
        vars(self)[name] = value
        return value

    def __setattr__(self, name, value):
        if not name.startswith("__"):
            # Task 25: rebind the value in EVERY holder of the same object,
            # plus the package mirror, so patch() stays visible to call-sites
            # in all group modules and restore puts the original back everywhere.
            for mod in _holders(name):
                setattr(mod, name, value)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith("__"):
            return super().__delattr__(name)
        for mod in _holders(name):
            del mod.__dict__[name]
        vars(self).pop(name, None)
        return None

    def __dir__(self):
        # F6: this WAS a stray module-level function (dir() called it
        # unbound -> TypeError "missing 1 required positional argument").
        # As a real class method it reports the package mirror plus every
        # group/impl module's names — what patchers and tab-completion want.
        base = set(super().__dir__())
        for mod in _iter_cli_modules():
            base |= set(vars(mod))
        return sorted(base)


sys.modules[__name__].__class__ = _CliPackage

# Group modules moved out of the impl (Task 5+) load here, after the __class__
# swap so anything they import from roampal.cli (colors/logger/state) sees the
# proxy normally.
_import_all_group_modules()

# Task 25 / Item 8: EAGERLY materialize every proxy-resolvable name into this
# module's __dict__. mock.patch's get_original() reads target.__dict__ DIRECTLY
# (bypassing __getattr__): a not-yet-materialized name snapshots is_local=False,
# and its teardown then goes through the delattr branch — wiping the (rebuilt)
# mock from every holder class-side instead of restoring the original. With the
# all-holders rebind this would strand the mock in NOTHING and lose the real
# values. Materializing up front guarantees patch always takes the local=True
# setattr path, whose exit runs through __setattr__ → _holders() → restore-all.
def _materialize_all():
    own = vars(sys.modules[__name__])
    for mod in _iter_cli_modules():
        for key, value in vars(mod).items():
            if key.startswith("__") or key in own:
                continue
            own[key] = value

_materialize_all()
