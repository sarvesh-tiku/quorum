#!/usr/bin/env python3
"""QUORUM demo: baseline-vs-QUORUM on the IDENTICAL injected faults.

Act 1 — the baseline dies: single-agent migration with the gate OFF and faults ON.
        The agent reports "migration complete". The independent oracle says FAIL.
Act 2 — QUORUM survives: same world, same faults, gate ON. The jury overrules the
        brain on the silent bug, the system rewinds + fixes, and the oracle says PASS.

Run offline (mock Claude) by default:        python demo/run_demo.py
Force a live run (needs ANTHROPIC_API_KEY):   python demo/run_demo.py --live
Emit a JSON event trace for the web UI:       python demo/run_demo.py --json-out web/trace.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.brain import Event, Orchestrator  # noqa: E402
from quorum.jurors import ConsensusGate  # noqa: E402
from quorum.llm import MockClient, MockConfig, make_client  # noqa: E402
from quorum.world import FaultConfig, MigrationWorld, summary_json  # noqa: E402


# ANSI colors for the terminal feed.
class C:
    R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"; B = "\033[94m"
    M = "\033[95m"; CY = "\033[96m"; GR = "\033[90m"; BOLD = "\033[1m"; END = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


COLOR = _supports_color()


def col(s: str, c: str) -> str:
    return f"{c}{s}{C.END}" if COLOR else s


def make_feed(events: list, slow: float = 0.0):
    """A printing event sink that also records every event (for the web trace)."""
    def sink(e: Event):
        events.append({"kind": e.kind, "payload": e.payload})
        _print_event(e)
        if slow:
            time.sleep(slow)
    return sink


def _print_event(e: Event) -> None:
    p = e.payload
    if e.kind == "step_start":
        print(col(f"\n▶ STEP {p['step']} ", C.BOLD + C.B)
              + col(f"[{p['tool']}]", C.CY)
              + (col(f"  (attempt {p['attempt']})", C.GR) if p.get("attempt", 1) > 1 else ""))
        if p.get("thought"):
            print(col(f"   brain: {p['thought']}", C.GR))
    elif e.kind == "claim":
        print(col(f"   claim: {p['claim']}", C.Y))
    elif e.kind == "vote":
        if p["red_flagged"]:
            print(col(f"     · juror#{p['juror_id']:>2} RED-FLAGGED ({p['red_flag_reason']})", C.GR))
        else:
            mark = col("PASS", C.G) if p["vote"] == "PASS" else col("FAIL", C.R)
            ev = (p["evidence"][:88] + "…") if len(p["evidence"]) > 89 else p["evidence"]
            print(f"     · juror#{p['juror_id']:>2} {mark}  {col(ev, C.GR)}")
    elif e.kind == "gate_result":
        d = p["decision"]
        color = C.G if d == "PASS" else C.R
        print(col(f"   ⚖ GATE {d}", C.BOLD + color)
              + col(f"  (PASS {p['pass_votes']} / FAIL {p['fail_votes']} / "
                    f"red-flag {p['red_flagged']} / polled {p['jurors_polled']})", C.GR))
        if p.get("root_cause"):
            print(col(f"   root cause: {p['root_cause']}", C.M))
    elif e.kind == "commit":
        tag = "gated ✓" if p.get("gated") else "ungated"
        print(col(f"   ✅ COMMIT {p['tool']} ({tag})", C.G))
    elif e.kind == "rollback":
        print(col(f"   ↩  ROLLBACK {p['tool']} — rewinding to last good snapshot", C.R + C.BOLD))
    elif e.kind == "replan":
        print(col(f"   ⟳ RE-PLAN with juror evidence", C.M))
    elif e.kind == "finish":
        rec = p["reconciliation"]
        print(col(f"\n   brain reports: \"{p['brain_claim']}\"", C.Y))
        verdict = "PASS ✅" if rec["passed"] else "FAIL ❌"
        vcolor = C.G if rec["passed"] else C.R
        print(col(f"   INDEPENDENT RECONCILIATION ORACLE → {verdict}", C.BOLD + vcolor))
        print(col(f"     source_count={rec['source_count']} target_count={rec['target_count']} "
                  f"missing={rec['n_missing']} mismatched={rec['n_mismatched']}", C.GR))
        print(col(f"     source_revenue={rec['source_revenue']} target_revenue={rec['target_revenue']} "
                  f"(delta {rec['revenue_delta']})", C.GR))


def banner(title: str, color: str) -> None:
    line = "═" * 74
    print(col(f"\n{line}", color))
    print(col(f"  {title}", C.BOLD + color))
    print(col(line, color))


def run_one(name: str, gate_enabled: bool, force_mock: bool, slow: float,
            mock_cfg: MockConfig, k: int, max_jurors: int) -> dict:
    client = make_client(force_mock=force_mock, mock_cfg=mock_cfg)
    world = MigrationWorld(seed=7, n_rows=1000, faults=FaultConfig.hard_mode())
    events: list = []
    sink = make_feed(events, slow=slow)
    gate = ConsensusGate(client, k=k, max_jurors=max_jurors,
                         on_vote=lambda v: sink(Event("vote", _vote(v))))
    orch = Orchestrator(client, world, gate_enabled=gate_enabled, gate=gate, sink=sink)
    result = orch.run()
    return {"name": name, "gate_enabled": gate_enabled, "result": result, "events": events,
            "live": client.is_live}


def _vote(v):
    return {"juror_id": v.juror_id, "vote": v.vote, "evidence": v.evidence,
            "confidence": v.confidence, "red_flagged": v.red_flagged,
            "red_flag_reason": v.red_flag_reason}


def main() -> None:
    ap = argparse.ArgumentParser(description="QUORUM baseline-vs-QUORUM demo")
    ap.add_argument("--live", action="store_true", help="use the real Claude API (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--slow", type=float, default=0.0, help="seconds to pause between feed events (drama)")
    ap.add_argument("--k", type=int, default=3, help="first-to-ahead-by-K threshold")
    ap.add_argument("--max-jurors", type=int, default=24, help="max jurors per gate")
    ap.add_argument("--noise", type=float, default=0.12, help="(mock) fraction of jurors that misfire")
    ap.add_argument("--redflag", type=float, default=0.10, help="(mock) fraction of malformed votes")
    ap.add_argument("--json-out", type=str, default="", help="write the full event trace as JSON for the web UI")
    ap.add_argument("--only", choices=["baseline", "quorum"], help="run just one act")
    args = ap.parse_args()

    force_mock = not args.live
    if args.live and not os.environ.get("ANTHROPIC_API_KEY"):
        print(col("⚠ --live requested but ANTHROPIC_API_KEY is not set; running on the MOCK client.", C.Y))
        force_mock = True

    mock_cfg = MockConfig(juror_noise=args.noise, juror_redflag_rate=args.redflag)
    mode = "LIVE (Claude API)" if not force_mock else "MOCK (offline fixtures)"
    print(col(f"\nQUORUM demo — client: {mode}", C.BOLD + C.CY))
    print(col("Task: migrate `orders` + Postgres, zero data loss / zero downtime. "
              "Faults injected: NUMERIC→INT truncation + 12 dropped CDC rows.", C.GR))

    runs = {}
    if args.only != "quorum":
        banner("ACT 1 — BASELINE  (single agent, NO consensus gate, faults ON)", C.R)
        runs["baseline"] = run_one("baseline", False, force_mock, args.slow, mock_cfg, args.k, args.max_jurors)
    if args.only != "baseline":
        banner("ACT 2 — QUORUM  (consensus gate ON, identical faults)", C.G)
        runs["quorum"] = run_one("quorum", True, force_mock, args.slow, mock_cfg, args.k, args.max_jurors)

    # Side-by-side verdict.
    banner("SIDE-BY-SIDE VERDICT  (independent reconciliation oracle)", C.CY)
    for key in ("baseline", "quorum"):
        if key not in runs:
            continue
        r = runs[key]["result"]
        rec = r["reconciliation"]
        st = r["stats"]
        verdict = "PASS ✅" if rec["passed"] else "FAIL ❌"
        vcolor = C.G if rec["passed"] else C.R
        label = "BASELINE" if key == "baseline" else "QUORUM  "
        print(col(f"  {label} → {verdict}", C.BOLD + vcolor)
              + col(f"   missing={rec['n_missing']} mismatched={rec['n_mismatched']} "
                    f"revenue_delta={rec['revenue_delta']}", C.GR))
        print(col(f"            jurors_polled={st['jurors_polled_total']} "
                  f"red_flagged={st['votes_red_flagged']} gate_blocks={st['gate_blocks']} "
                  f"rollbacks={st['rollbacks']}", C.GR))

    if "baseline" in runs and "quorum" in runs:
        b = runs["baseline"]["result"]["reconciliation"]["passed"]
        q = runs["quorum"]["result"]["reconciliation"]["passed"]
        if (not b) and q:
            print(col("\n  ★ The jury overruled the brain on a silent bug the baseline shipped, "
                      "rewound, and fixed it.", C.BOLD + C.M))

    if args.json_out:
        out = {"runs": runs, "params": {"k": args.k, "max_jurors": args.max_jurors,
               "noise": args.noise, "redflag": args.redflag, "live": not force_mock}}
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=str))
        print(col(f"\n  wrote event trace → {args.json_out}", C.GR))


if __name__ == "__main__":
    main()
