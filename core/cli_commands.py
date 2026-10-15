"""Headless subcommands: the features that used to be TUI-only slash commands or loose scripts,
reachable from a shell or CI with machine-readable output.

    motion skill list|show|candidates|promote|reject|rollback|ab-test
    motion eval list|run|memory
    motion trajectory compare BEFORE AFTER
    motion sessions list|show
    motion memory search|stats
    motion tracking on|off|status
    motion check [PROVIDER] [--all]

Conventions shared by every subcommand:
  --json       one JSON document on stdout (human-readable progress, if any, goes to stderr)
  --workspace  the project whose `.motion/` state is read or changed (default: current directory)
  exit codes   0 ok, 1 the check/operation did not succeed (regression, failed check, nothing to
               promote), 2 usage error (unknown name, bad file, missing argument)

Nothing here promotes a skill on its own: `skill ab-test` reports a verdict, and `skill promote`
is a separate, explicit call - the same gate the TUI has (issue #16).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2

COMMANDS = ("skill", "eval", "trajectory", "sessions", "memory", "tracking", "check")


def _common_parent() -> argparse.ArgumentParser:
    """Options every subcommand takes. `default=SUPPRESS` matters: a subparser's defaults are
    written over values the top-level parser already parsed, so without it `motion --provider x
    skill list` would have its provider reset to None by the subparser's own default."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--workspace", default=argparse.SUPPRESS, help="project directory (default: current directory)")
    p.add_argument("--provider", default=argparse.SUPPRESS, help="provider/model id (default: config.yml's default)")
    p.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS,
                   help="machine-readable output on stdout")
    return p


def add_subcommands(sub: "argparse._SubParsersAction") -> None:
    common = [_common_parent()]

    skill = sub.add_parser("skill", parents=common, help="Review and manage learned skills")
    s = skill.add_subparsers(dest="skill_action", required=True)
    p = s.add_parser("list", parents=common, help="List active skills")
    p.add_argument("--all", action="store_true", help="include candidates and rejected skills")
    p = s.add_parser("show", parents=common, help="Print a skill (candidates included)")
    p.add_argument("name")
    s.add_parser("candidates", parents=common, help="Skills awaiting review")
    for action, text in (("promote", "Make a candidate active and recallable"),
                         ("reject", "Reject a candidate (kept on disk for provenance)"),
                         ("rollback", "Restore a skill's previous version")):
        p = s.add_parser(action, parents=common, help=text)
        p.add_argument("name")
    p = s.add_parser("ab-test", parents=common, help="A/B a candidate skill against the eval baseline (billed)")
    p.add_argument("--skill", required=True, help="path to the candidate skill's .md file")
    p.add_argument("--tasks", default=None, help="comma-separated eval task ids (default: all non-held-out scriptable)")
    p.add_argument("--repeat", type=int, default=1, help="attempts per task per variant (default 1)")

    ev = sub.add_parser("eval", parents=common, help="Run the evaluation baselines")
    e = ev.add_subparsers(dest="eval_action", required=True)
    p = e.add_parser("list", parents=common, help="List eval tasks (no provider calls)")
    p.add_argument("--include-held-out", action="store_true")
    p = e.add_parser("run", parents=common, help="Run the task baseline (billed)")
    p.add_argument("--task", action="append", dest="tasks", help="only this task id (repeatable)")
    p.add_argument("--include-held-out", action="store_true")
    p.add_argument("--keep-workdirs", action="store_true")
    p.add_argument("--out", default=None, help="report path (default: evals/results/<timestamp>.json)")
    p = e.add_parser("memory", parents=common, help="Run the memory-quality scenarios (billed)")
    p.add_argument("--scenario", default=None, help="only this scenario (default: all)")
    p.add_argument("--out", default=None, help="report path (default: a temp file)")

    tr = sub.add_parser("trajectory", parents=common, help="Inspect saved trajectories")
    t = tr.add_subparsers(dest="trajectory_action", required=True)
    t.add_parser("list", parents=common, help="Trajectories saved in this workspace")
    p = t.add_parser("compare", parents=common, help="Diff two trajectory files")
    p.add_argument("before")
    p.add_argument("after")

    se = sub.add_parser("sessions", parents=common, help="Saved session transcripts (need /tracking on)")
    ss = se.add_subparsers(dest="sessions_action", required=True)
    p = ss.add_parser("list", parents=common, help="Saved sessions, newest first")
    p.add_argument("--limit", type=int, default=20)
    p = ss.add_parser("show", parents=common, help="Print one session's turns")
    p.add_argument("session_id")

    me = sub.add_parser("memory", parents=common, help="Query this workspace's long-term memory")
    mm = me.add_subparsers(dest="memory_action", required=True)
    p = mm.add_parser("search", parents=common, help="Recall memories for a query")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    mm.add_parser("stats", parents=common, help="Where the memory DB is and how much is in it")

    tk = sub.add_parser("tracking", parents=common, help="Save session transcripts locally (on|off|status)")
    tk.add_argument("mode", nargs="?", choices=("on", "off", "status"), default="status")

    ck = sub.add_parser("check", parents=common, help="Live wire-format check of a provider (billed, a few hundred tokens)")
    ck.add_argument("check_provider", nargs="?", metavar="PROVIDER", help="provider id (default: the configured default)")
    ck.add_argument("--all", action="store_true", help="check every provider that has an API key")


# ── helpers ──────────────────────────────────────────────────────────────────

def _emit(args: argparse.Namespace, data: Any, text: str = "") -> None:
    if getattr(args, "json_output", False):
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif text:
        print(text)


def _err(message: str) -> int:
    print(f"motion: {message}", file=sys.stderr)
    return EXIT_USAGE


def _workspace(args: argparse.Namespace) -> str:
    return os.path.abspath(getattr(args, "workspace", None) or os.getenv("MOTION_WORKSPACE") or os.getcwd())


def _script_args(args: argparse.Namespace, **extra: Any) -> argparse.Namespace:
    ns = argparse.Namespace(provider=getattr(args, "provider", None), json_output=getattr(args, "json_output", False))
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def _skill_row(name: str, desc: str, path: Optional[Path]) -> Dict[str, Any]:
    from core.skill_state import load_meta

    meta = load_meta(path) if path else None
    prov = (meta or {}).get("provenance", {}) or {}
    return {
        "name": name,
        "description": desc,
        "status": (meta or {}).get("status", "active"),
        "saved_by_hand": meta is None,
        "version": (meta or {}).get("version"),
        "verified": bool(prov.get("verified")) if meta else None,
        "from_prompt": prov.get("prompt", "")[:120] if meta else None,
        "path": str(path) if path else None,
    }


# ── skill ────────────────────────────────────────────────────────────────────

async def _skill_lifecycle(args: argparse.Namespace, path: Path) -> bool:
    """promote/reject/rollback need the agent's synthesizer (it owns the MemoryDB index entry that
    makes a promoted skill recallable and removes it again on reject/rollback)."""
    from core import headless

    agent = headless.build_agent(getattr(args, "provider", None), _workspace(args))
    try:
        synth = agent.synthesizer
        if args.skill_action == "promote":
            return await synth.promote(str(path))
        if args.skill_action == "reject":
            return synth.reject(str(path))
        return synth.rollback(str(path))
    finally:
        try:
            agent.memory.close()
        except Exception:
            pass


def _cmd_skill(args: argparse.Namespace) -> int:
    from core.skills import SkillLibrary

    action = args.skill_action
    if action == "ab-test":
        from scripts import skill_ab_test

        return asyncio.run(skill_ab_test.main_async(_script_args(
            args, skill=args.skill, tasks=args.tasks, repeat=args.repeat)))

    lib = SkillLibrary.for_workspace(_workspace(args))
    if action == "list":
        files = lib._files(include_inactive=args.all)
        rows = [_skill_row(n, lib._describe(p), p) for n, p in files.items()]
        _emit(args, {"skills": rows, "awaiting_review": len(lib.pending())},
              "\n".join(f"{r['name']}\t{r['status']}\t{r['description']}" for r in rows) or "No skills.")
        return EXIT_OK
    if action == "candidates":
        rows = [_skill_row(n, d, lib.resolve_path(n, include_inactive=True)) for n, d, _meta in lib.pending()]
        _emit(args, {"candidates": rows},
              "\n".join(f"{r['name']}\t{r['status']} v{r['version']}\t{'verified' if r['verified'] else 'unverified'}\t{r['description']}"
                        for r in rows) or "No skills awaiting review.")
        return EXIT_OK

    path = lib.resolve_path(args.name, include_inactive=True)
    if path is None:
        return _err(f"skill not found: {args.name}")
    if action == "show":
        row = _skill_row(args.name, lib._describe(path), path)
        content = path.read_text(encoding="utf-8", errors="replace")
        _emit(args, {**row, "content": content}, content)
        return EXIT_OK

    try:
        ok = asyncio.run(_skill_lifecycle(args, path))
    except Exception as exc:
        from core.headless import HeadlessUsageError

        if isinstance(exc, HeadlessUsageError):
            return _err(str(exc))
        raise
    reason = {
        "promote": "has no candidate state to promote (already a plain saved skill)",
        "reject": "has no candidate state to reject",
        "rollback": "has no previous version to roll back to",
    }[action]
    _emit(args, {"skill": args.name, "action": action, "ok": ok, **({} if ok else {"reason": reason})},
          f"{action}: {args.name} ok" if ok else f"{args.name} {reason}")
    return EXIT_OK if ok else EXIT_FAIL


# ── eval ─────────────────────────────────────────────────────────────────────

def _cmd_eval(args: argparse.Namespace) -> int:
    action = args.eval_action
    if action == "memory":
        from evals.memory_quality import SCENARIOS
        from scripts import eval_memory_quality

        if args.scenario and args.scenario not in SCENARIOS:
            return _err(f"unknown scenario {args.scenario!r}; choose from {', '.join(SCENARIOS)}")
        return asyncio.run(eval_memory_quality.main_async(_script_args(args, scenario=args.scenario, out=args.out)))

    from scripts import run_evals

    ns = _script_args(
        args, tasks=getattr(args, "tasks", None), include_held_out=args.include_held_out,
        keep_workdirs=getattr(args, "keep_workdirs", False), list=(action == "list"), out=getattr(args, "out", None))
    return asyncio.run(run_evals.main_async(ns))


# ── trajectory ───────────────────────────────────────────────────────────────

def _cmd_trajectory(args: argparse.Namespace) -> int:
    from core.session import state_dir

    if args.trajectory_action == "list":
        directory = state_dir(_workspace(args), "trajectories", create=False)
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if directory.is_dir() else []
        rows = [{"file": str(p), "bytes": p.stat().st_size} for p in files]
        _emit(args, {"trajectories": rows}, "\n".join(r["file"] for r in rows) or "No saved trajectories.")
        return EXIT_OK

    from scripts import compare_trajectories as ct

    docs = []
    for label, path in (("before", args.before), ("after", args.after)):
        try:
            docs.append(ct.load(path))
        except (OSError, json.JSONDecodeError) as exc:
            return _err(f"cannot read {label} trajectory {path}: {exc}")
    _emit(args, ct.compare_data(*docs), ct.compare(*docs))
    return EXIT_OK


# ── sessions ─────────────────────────────────────────────────────────────────

def _cmd_sessions(args: argparse.Namespace) -> int:
    from core.session import SessionStore

    ws = _workspace(args)
    if args.sessions_action == "list":
        rows = SessionStore.list_sessions(ws, limit=args.limit)
        _emit(args, {"sessions": rows},
              "\n".join(f"{r['id']}\t{r['modified']}\t{r['turns']} turn(s)\t{r['first_prompt']}" for r in rows)
              or "No saved sessions (turn on /tracking to record them).")
        return EXIT_OK
    turns = SessionStore.load(ws, args.session_id)
    if not turns:
        return _err(f"no session named {args.session_id!r} in {ws}")
    text = "\n\n".join(f"> {t.get('prompt', '')}\n{t.get('response', '')}" for t in turns)
    _emit(args, {"id": Path(args.session_id).stem, "turns": turns}, text)
    return EXIT_OK


# ── memory ───────────────────────────────────────────────────────────────────

def _memory_path(workspace: str) -> str:
    from core.config import ConfigManager
    from main import _default_memory_path

    return ConfigManager().get("memory_path") or _default_memory_path(workspace)


class _KeywordOnly:
    """Embedder stand-in for when no provider can be built: HybridRetriever then does FTS5 only."""
    semantic_available = False

    async def get_embedding(self, text: str) -> list:
        return []


def _cmd_memory(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    path = _memory_path(ws)
    exists = os.path.exists(path)
    if args.memory_action == "stats":
        count = None
        if exists:
            from memory.db import MemoryDB

            db = MemoryDB(path)
            try:
                count = db.count()
            finally:
                db.close()
        data = {"path": path, "exists": exists, "memories": count, "bytes": os.path.getsize(path) if exists else 0}
        _emit(args, data, f"{path}\n{count if count is not None else 0} memories" + ("" if exists else " (no database yet)"))
        return EXIT_OK

    if not exists:  # never create an empty DB just to search it
        _emit(args, {"query": args.query, "semantic": False, "results": [], "path": path}, "No memories (no database yet).")
        return EXIT_OK
    return asyncio.run(_memory_search(args, path))


async def _memory_search(args: argparse.Namespace, path: str) -> int:
    from memory.db import MemoryDB
    from memory.retriever import HybridRetriever

    embedder: Any = _KeywordOnly()
    agent = None
    try:  # semantic recall when a provider can be built; keyword-only otherwise
        from core import headless

        agent = headless.build_agent(getattr(args, "provider", None), _workspace(args))
        embedder = agent
        db = agent.memory
    except Exception:
        db = MemoryDB(path)
    try:
        retriever = HybridRetriever(db, embedder)
        try:
            results = await asyncio.wait_for(retriever.retrieve(args.query, top_k=args.top_k), 15)
        except Exception:
            embedder = _KeywordOnly()
            results = await HybridRetriever(db, embedder).retrieve(args.query, top_k=args.top_k)
        semantic = bool(getattr(embedder, "semantic_available", False))
    finally:
        db.close()
    rows = [{"content": r["content"], "score": r["score"], "raw_score": r.get("raw_score"),
             "type": r["type"], "age": r.get("age"),
             "weak": r.get("raw_score") is not None and r["raw_score"] < HybridRetriever.WEAK_SEMANTIC_SCORE}
            for r in results]

    def line(r: Dict[str, Any]) -> str:
        tags = [t for t in (r["type"], r["age"], None if r["raw_score"] is None else f"cosine {r['raw_score']:.2f}" + (" WEAK" if r["weak"] else "")) if t]
        return f"[{', '.join(tags)}] {r['content']}"

    _emit(args, {"query": args.query, "semantic": semantic, "results": rows, "path": path},
          "\n".join(line(r) for r in rows) or "No matching memories.")
    return EXIT_OK


# ── tracking ─────────────────────────────────────────────────────────────────

def _cmd_tracking(args: argparse.Namespace) -> int:
    from core.config import ConfigManager

    cm = ConfigManager()
    if args.mode != "status":
        try:
            cm.set("track_interactions", args.mode == "on")
        except Exception as exc:
            return _err(f"could not save the setting: {exc}")
    enabled = args.mode == "on" or (args.mode == "status" and bool(cm.get("track_interactions")))
    _emit(args, {"tracking": enabled, "sessions_dir": str(Path(_workspace(args)) / ".motion" / "sessions")},
          f"Interaction tracking is {'ON' if enabled else 'OFF'}.")
    return EXIT_OK


# ── check ────────────────────────────────────────────────────────────────────

def _cmd_check(args: argparse.Namespace) -> int:
    from scripts import live_check

    return asyncio.run(live_check.main_async(_script_args(
        args, provider=args.check_provider or getattr(args, "provider", None), all=args.all)))


_HANDLERS = {
    "skill": _cmd_skill, "eval": _cmd_eval, "trajectory": _cmd_trajectory, "sessions": _cmd_sessions,
    "memory": _cmd_memory, "tracking": _cmd_tracking, "check": _cmd_check,
}


def run(args: argparse.Namespace) -> int:
    """Dispatch a parsed subcommand; unexpected failures become a one-line error + exit 1, not a traceback."""
    try:
        return _HANDLERS[args.command](args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"motion: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAIL
