#!/usr/bin/env python3
"""Reliability benchmark: the pass^k-style curve QUORUM is designed to flatten.

Runs baseline and QUORUM N times each (offline, varying juror noise/seed) and reports
the fraction of runs whose INDEPENDENT reconciliation oracle says PASS. This is the
killer chart: baseline collapses under injected faults; QUORUM stays near 1.0, and the
cost is a dial (jurors_polled), not model size.

    python demo/reliability.py --trials 40
    python demo/reliability.py --trials 40 --json-out web/reliability.json
"""

import argparse
import json

from quorum.brain import Orchestrator
from quorum.jurors import ConsensusGate
from quorum.llm import MockClient, MockConfig
from quorum.world import FaultConfig, MigrationWorld


def one_run(gate_enabled: bool, seed: int, k: int, max_jurors: int,
            noise: float, redflag: float) -> dict:
    client = MockClient(MockConfig(juror_noise=noise, juror_redflag_rate=redflag, seed=seed))
    world = MigrationWorld(seed=seed, n_rows=400, faults=FaultConfig.hard_mode())
    gate = ConsensusGate(client, k=k, max_jurors=max_jurors)
    orch = Orchestrator(client, world, gate_enabled=gate_enabled, gate=gate)
    result = orch.run()
    return {
        "passed": result["reconciliation"]["passed"],
        "revenue_delta": result["reconciliation"]["revenue_delta"],
        "jurors_polled": result["stats"]["jurors_polled_total"],
        "gate_blocks": result["stats"]["gate_blocks"],
    }


def sweep(label: str, gate_enabled: bool, trials: int, k: int, max_jurors: int,
          noise: float, redflag: float) -> dict:
    runs = [one_run(gate_enabled, seed=100 + i, k=k, max_jurors=max_jurors,
                    noise=noise, redflag=redflag) for i in range(trials)]
    n_pass = sum(1 for r in runs if r["passed"])
    avg_jurors = sum(r["jurors_polled"] for r in runs) / max(1, trials)
    return {
        "label": label,
        "trials": trials,
        "pass_rate": n_pass / max(1, trials),
        "n_pass": n_pass,
        "avg_jurors_polled": avg_jurors,
        "runs": runs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QUORUM reliability benchmark (pass^k)")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--max-jurors", type=int, default=24)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--redflag", type=float, default=0.10)
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    baseline = sweep("baseline", False, args.trials, args.k, args.max_jurors, args.noise, args.redflag)
    quorum = sweep("quorum", True, args.trials, args.k, args.max_jurors, args.noise, args.redflag)

    print(f"\nRELIABILITY over {args.trials} trials each "
          f"(faults ON; juror noise={args.noise}, red-flag rate={args.redflag}, K={args.k})\n")
    print(f"  {'mode':<10} {'pass_rate':>10} {'passed/total':>14} {'avg_jurors':>12}")
    print("  " + "-" * 48)
    for s in (baseline, quorum):
        print(f"  {s['label']:<10} {s['pass_rate']*100:>9.1f}% "
              f"{s['n_pass']:>6}/{s['trials']:<7} {s['avg_jurors_polled']:>12.1f}")

    print(f"\n  → Baseline pass^1 ≈ {baseline['pass_rate']*100:.0f}% (collapses on the silent bug).")
    print(f"  → QUORUM   pass^1 ≈ {quorum['pass_rate']*100:.0f}% "
          f"(verification compute buys reliability; brain cost flat).")

    # ASCII bar.
    print("\n  pass-rate:")
    for s in (baseline, quorum):
        bar = "█" * int(round(s["pass_rate"] * 40))
        print(f"    {s['label']:<10} |{bar:<40}| {s['pass_rate']*100:.0f}%")

    if args.json_out:
        out = {"baseline": baseline, "quorum": quorum,
               "params": {"trials": args.trials, "k": args.k, "max_jurors": args.max_jurors,
                          "noise": args.noise, "redflag": args.redflag}}
        Path(args.json_out).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n  wrote → {args.json_out}")


if __name__ == "__main__":
    main()
