"""Profile group (v0.6.0 Task 10/13): cmd_profile.

Moved verbatim from the pre-refactor roampal/cli.py as part of the Task 10
split (list/show/use/unuse/switch/create/register/delete). Shared helpers
(colors, no_input accessor) live in roampal.cli._common; profile-manager
imports stay function-local as in the monolith. Task 13 extends this
module with the bind/unbind subcommands, the show binding report ("which
binding is active in cwd and why"), and the list bindings section —
additive; the legacy subcommand paths stay byte-identical. Round 2 Item 7
/ Task 24: `switch` no longer kills the server (use + note).
"""

import os
from pathlib import Path

from roampal.cli._common import (
    BLUE,
    GREEN,
    RED,
    RESET,
    YELLOW,
    no_input as _no_input_flag,
)

# F6 (Task 32): the cwd-binding shadow check — a directory binding
# outranks the persisted `use` in the client resolution chain
# (env > config-env > binding > use > default), so setting/switching the
# persisted profile from inside a bound directory LOOKS successful while
# sessions there keep resolving to the binding's profile. Use/switch print
# this warning whenever the cwd (or an ancestor) has a binding — whatever
# name it points at, including the same one (the binding still wins).
def _warn_binding_override(reg, name):
    from roampal.profile_manager import binding_for_cwd

    matched = binding_for_cwd(registry=reg)
    if matched is None:
        return
    bound_key, bound_name = matched
    if bound_name == name:
        print(
            f"{YELLOW}Note: {bound_key} is directory-bound to {bound_name!r}, so sessions in it "
            f"keep resolving to {bound_name!r} (bindings outrank the persisted 'use').{RESET}"
        )
    else:
        print(
            f"{YELLOW}Note: {bound_key} is DIRECTORY-BOUND to {bound_name!r} — sessions opened "
            f"under it keep resolving there, NOT the persisted {name!r} you just set.{RESET}"
            f"\n  Unbind if intended: roampal profile unbind --path {bound_key}"
        )


def _print_bindings(reg):
    """Bindings section for `profile list` (only reached when bindings exist)."""
    from roampal.profile_manager import DEFAULT_PROFILE

    bindings = reg.bindings()
    print(f"Directory bindings ({len(bindings)}):")
    for d, pname in sorted(bindings.items()):
        note = ""
        # F6 follow-up: literal `default` is a VALID binding target (the
        # registry never stores it as a profile, so exists() is False for
        # it) — the not-registered annotation must not fire for it.
        if pname != DEFAULT_PROFILE and not reg.exists(pname):
            note = " [profile not registered!]"
        print(f"  {d}")
        print(f"    -> {pname}{note}")


def cmd_profile(args):
    """Manage named memory profiles (v0.5.1).

    Subcommands:
      list       — show all registered profiles and resolved paths
      create     — create a new profile (auto-located under default base)
      register   — register an existing directory as a named profile
      delete     — remove a profile from the registry (optional --destroy-data)
      show       — print the active profile and its resolved path (+ cwd binding)
    """
    from roampal.profile_manager import (
        DEFAULT_PROFILE,
        InvalidProfileNameError,
        ProfileAlreadyExistsError,
        ProfileError,
        ProfileNotFoundError,
        ProfileRegistry,
        active_profile_name,
        active_profile_source,
        binding_for_cwd,
        clear_active_profile_file,
        resolve_data_path,
        system_default_data_path,
        write_active_profile_file,
    )

    sub = getattr(args, "profile_command", None)
    reg = ProfileRegistry()

    if sub == "list" or sub is None:
        profiles = reg.list()
        print(f"Default profile: {system_default_data_path()}")
        if not profiles:
            print("No named profiles registered.")
            print(f"Create one: {BLUE}roampal profile create <name>{RESET}")
            bindings = reg.bindings()
            if bindings:
                print()
                _print_bindings(reg)
            return 0
        print()
        print(f"Registered profiles ({len(profiles)}):")
        for p in profiles:
            resolved = reg.resolve(profile=p)
            suffix = " (custom path)" if p.path else ""
            print(f"  {p.name}{suffix}")
            print(f"    slug: {p.slug}")
            print(f"    path: {resolved}")
        bindings = reg.bindings()
        if bindings:
            print()
            _print_bindings(reg)
        return 0

    if sub == "show":
        active = active_profile_name()
        source = active_profile_source()
        try:
            resolved = resolve_data_path(active, registry=reg)
        except ProfileNotFoundError:
            print(
                f"{YELLOW}Active profile {active!r} is not registered.{RESET}"
            )
            print(f"Register or create it: {BLUE}roampal profile create {active}{RESET}")
            return 1
        source_label = {
            "env": "ROAMPAL_PROFILE env var",
            "binding": "directory binding for this cwd",
            "file": "persisted via 'roampal profile use'",
            "default": "fallback (no env var, no persisted file)",
        }.get(source, source)
        print(f"Active profile: {active}  ({source_label})")
        print(f"Data path:      {resolved}")
        # v0.6.0 review fix 5, round 2: surface any pinned running server —
        # a pin whose owner was hard-killed lingers in the pin file, and
        # previously nothing revealed it. Visibility fixes the stuck case.
        is_dev = os.environ.get("ROAMPAL_DEV", "").lower() in ("1", "true", "yes")
        from roampal.cli._common import PROD_PORT, DEV_PORT
        from roampal.profile_manager import read_server_pin

        pin_port = DEV_PORT if is_dev else PROD_PORT
        pinned = read_server_pin(pin_port)
        if pinned and pinned != active:
            print(
                f"{YELLOW}Note: a server on port {pin_port} was launched with"
                f" 'roampal start --profile {pinned}' — sessions with no"
                f" explicit config route there.\n"
                f"  Clear it with: roampal stop, or a bare roampal start.{RESET}"
            )
        elif pinned == active:
            print(
                f"{YELLOW}Note: the running server on port {pin_port} is pinned to"
                f" this profile (launched via 'roampal start --profile {pinned}').{RESET}"
            )
        if os.environ.get("ROAMPAL_DATA_PATH"):
            print(
                f"{YELLOW}Note: ROAMPAL_DATA_PATH env var is set and overrides profile resolution.{RESET}"
            )
        # v0.6.0 Task 13: report the binding active in cwd — and WHY (which
        # ancestor directory matched). Only prints when a binding exists.
        matched = binding_for_cwd(registry=reg)
        if matched is not None:
            bound_key, bound_name = matched
            if bound_name == active:
                print(f"Bound directory: {bound_key}")
                print(
                    f"  (cwd ancestry match — all directories under it resolve to"
                    f" {bound_name!r} unless ROAMPAL_PROFILE is set)"
                )
            else:
                print(
                    f"{YELLOW}Bound directory: {bound_key} -> {bound_name!r}"
                    f" (shadowed by another resolution source this cwd could see){RESET}"
                )
        return 0

    if sub == "use":
        name = args.name
        if name == DEFAULT_PROFILE:
            # Using "default" is the same as unuse — clear the file.
            removed = clear_active_profile_file()
            if removed:
                print(f"{GREEN}Reverted persisted active profile to {DEFAULT_PROFILE}{RESET}")
            else:
                print(f"Active profile is already {DEFAULT_PROFILE}")
            # F6: a cwd binding still outranks even a reverted/default
            # persisted state — warn the operator here too.
            _warn_binding_override(reg, name)
            return 0
        # Verify it's registered before persisting.
        if not reg.exists(name):
            print(f"{RED}Error:{RESET} profile {name!r} is not registered")
            print(f"Create it first: {BLUE}roampal profile create {name}{RESET}")
            return 1
        write_active_profile_file(name)
        resolved = reg.resolve(name)
        print(f"{GREEN}Active profile set to {name!r} (persistent){RESET}")
        print(f"  Data path: {resolved}")
        if os.environ.get("ROAMPAL_PROFILE", "").strip():
            print(
                f"{YELLOW}Note: ROAMPAL_PROFILE env var is set in this shell and will override the persisted value for this session.{RESET}"
            )
        # F6: a cwd binding SHADOWS the persisted `use` for every session
        # opened under it — previously `use`/`switch` were silent about
        # that and the user's change never took effect in this cwd.
        _warn_binding_override(reg, name)
        return 0

    if sub == "unuse":
        removed = clear_active_profile_file()
        if removed:
            print(
                f"{GREEN}Cleared persisted active profile. Falling back to ROAMPAL_PROFILE env var or '{DEFAULT_PROFILE}'.{RESET}"
            )
        else:
            print("No persisted active profile was set.")
        return 0

    if sub == "switch":
        # Round 2 Item 7 / Task 24: switch == use + a note. The server is
        # NEVER killed: per-request profile routing (v0.6.0 Item 6 —
        # clients name their profiles, MCP re-resolves per call, hooks
        # send headers per post, and headerless resolution reads the
        # persisted `use` at request time) means every session picks the
        # new profile up on its next exchange. The old kill interrupted
        # every OTHER running session and made the respawner's env/cwd
        # the routing source (F2 incident observed 2026-09-18).
        name = args.name

        # Step 1: set the persisted active profile (same as 'use')
        if name == DEFAULT_PROFILE:
            removed = clear_active_profile_file()
            if removed:
                print(f"{GREEN}Reverted persisted active profile to {DEFAULT_PROFILE}{RESET}")
            else:
                print(f"Active profile is already {DEFAULT_PROFILE}")
            # F6: a cwd binding outranks the default resolved state — warn.
            _warn_binding_override(reg, name)
        else:
            if not reg.exists(name):
                print(f"{RED}Error:{RESET} profile {name!r} is not registered")
                print(f"Create it first: {BLUE}roampal profile create {name}{RESET}")
                return 1
            write_active_profile_file(name)
            resolved = reg.resolve(name)
            print(f"{GREEN}Active profile set to {name!r} (persistent){RESET}")
            print(f"  Data path: {resolved}")

        print()
        print(
            f"{GREEN}Ready. Every session picks this up on its next exchange — the running "
            f"server is left alone (other sessions uninterrupted).{RESET}"
        )
        if os.environ.get("ROAMPAL_PROFILE", "").strip():
            print(
                f"{YELLOW}Note: ROAMPAL_PROFILE env var is set in this shell and will override the persisted value for commands run here.{RESET}"
            )
        # F6: same cwd-binding shadow warning as `use`.
        _warn_binding_override(reg, name)
        return 0

    if sub == "create":
        name = args.name
        try:
            p = reg.create(name)
        except (ProfileAlreadyExistsError, InvalidProfileNameError) as e:
            print(f"{RED}Error:{RESET} {e}")
            return 1
        resolved = reg.resolve(profile=p)
        print(f"{GREEN}Created profile {name!r}{RESET}")
        print(f"  slug: {p.slug}")
        print(f"  path: {resolved}")
        print()
        print("Use it for a project folder (and everything under it):")
        print(f"  {BLUE}roampal profile bind {name} --path <dir>{RESET}")
        print("Or make it your default everywhere else:")
        print(f"  {BLUE}roampal profile use {name}{RESET}")
        return 0

    if sub == "register":
        name = args.name
        path = args.path
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        try:
            p = reg.register(name, path)
        except (ProfileAlreadyExistsError, InvalidProfileNameError, ProfileError) as e:
            print(f"{RED}Error:{RESET} {e}")
            return 1
        print(f"{GREEN}Registered profile {name!r}{RESET}")
        print(f"  slug: {p.slug}")
        print(f"  path: {p.path}")
        if not Path(p.path).exists():
            print(
                f"{YELLOW}Note: path does not exist yet — will be created on first use.{RESET}"
            )
        return 0

    if sub == "delete":
        name = args.name
        if name == DEFAULT_PROFILE:
            print(f"{RED}Error:{RESET} cannot delete the default profile")
            return 1
        try:
            profile = reg.get(name)
            if profile is None:
                print(f"{RED}Error:{RESET} profile {name!r} not registered")
                return 1
            resolved = reg.resolve(profile=profile)
        except ProfileError as e:
            print(f"{RED}Error:{RESET} {e}")
            return 1

        destroy = bool(getattr(args, "destroy_data", False))
        if destroy and not _no_input_flag():
            confirm = input(
                f"{YELLOW}Destroy data at {resolved}? (yes/no): {RESET}"
            ).strip().lower()
            if confirm != "yes":
                print("Cancelled.")
                return 1

        # v0.6.0 review fix 8: report what delete() cascades — bound
        # directories would otherwise 404 silently.
        bound_dirs = sorted(d for d, pname in reg.bindings().items() if pname == name)

        try:
            reg.delete(name, destroy_data=destroy)
        except ProfileError as e:
            print(f"{RED}Error:{RESET} {e}")
            return 1
        if destroy:
            print(f"{GREEN}Deleted profile {name!r} and destroyed data at {resolved}{RESET}")
        else:
            print(f"{GREEN}Unregistered profile {name!r}{RESET}")
            print(f"  (data at {resolved} was NOT deleted — remove manually if desired)")
        if bound_dirs:
            print(
                f"  {YELLOW}Unbound {len(bound_dirs)} director"
                f"{'y' if len(bound_dirs) == 1 else 'ies'} that pointed at {name!r} "
                f"— each now resolves like any unbound folder:{RESET}"
            )
            # Say where each folder actually goes now (env > parent binding >
            # `profile use` > default), not a blanket "default" — with a
            # `profile use` set, that is where they land.
            from roampal.profile_manager import active_profile_name, active_profile_source

            source_label = {
                "env": "ROAMPAL_PROFILE env var",
                "binding": "binding on a parent folder",
                "file": "profile use",
                "default": "default",
            }
            for d in bound_dirs:
                now = active_profile_name(cwd=Path(d), registry=reg)
                src = source_label.get(active_profile_source(cwd=Path(d), registry=reg), "")
                print(f"    - {d} -> {now} ({src})")
            print(
                f"  Re-bind after re-creating: {BLUE}roampal profile bind <name> --path <dir>{RESET}"
            )
        return 0

    if sub == "bind":
        name = args.name
        directory = getattr(args, "path", None) or os.getcwd()
        try:
            key = reg.bind(name, directory)
        except ProfileNotFoundError as e:
            print(f"{RED}Error:{RESET} {e}")
            print(f"Create it first: {BLUE}roampal profile create {name}{RESET}")
            return 1
        except ProfileError as e:
            print(f"{RED}Error:{RESET} {e}")
            return 1
        print(f"{GREEN}Bound profile {name!r} to directory {key}{RESET}")
        print(
            f"  (subdirectories of {key} resolve to {name!r} unless"
            f" ROAMPAL_PROFILE is set)"
        )
        return 0

    if sub == "unbind":
        directory = getattr(args, "path", None) or os.getcwd()
        verify_name = getattr(args, "name", None)
        from roampal.profile_manager import _norm_binding_dir

        key = _norm_binding_dir(directory)
        existing = reg.bindings().get(key)
        if existing is None:
            print(f"No binding for that directory (no registry change).")
            return 0
        if verify_name is not None and existing != verify_name:
            print(
                f"{RED}Error:{RESET} {key} is bound to {existing!r}, not {verify_name!r}"
            )
            return 1
        reg.unbind(directory)
        print(f"{GREEN}Unbound profile {existing!r} from directory {key}{RESET}")
        return 0

    print(f"{RED}Unknown profile subcommand: {sub!r}{RESET}")
    return 1
