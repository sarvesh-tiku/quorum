"""Offline tests for QUORUM: world, red-flag filter, gate, and end-to-end recovery.

All tests use the MockClient — no network. Run with: python -m pytest -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quorum.brain import Orchestrator
from quorum.jurors import ConsensusGate, red_flag
from quorum.llm import MockClient, MockConfig
from quorum.tools import IRREVERSIBLE
from quorum.world import FaultConfig, MigrationWorld


# --------------------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------------------

def test_clean_migration_reconciles_pass():
    w = MigrationWorld(seed=1, n_rows=200, faults=FaultConfig.clean())
    w.provision_target("numeric")
    w.enable_dual_write()
    w.backfill_all()
    w.replay_cdc()
    w.cutover()
    rec = w.reconcile()
    assert rec["passed"], rec


def test_truncation_fault_drops_revenue_silently():
    w = MigrationWorld(seed=1, n_rows=200, faults=FaultConfig(truncate_total_to_int=True))
    w.provision_target("numeric")  # brain asked for numeric, fault forces int
    assert w.target_total_is_int  # the silent bug
    w.backfill_all()
    rec = w.reconcile()
    assert not rec["passed"]
    assert rec["source_revenue"] != rec["target_revenue"]  # money silently dropped


def test_cdc_drop_loses_rows():
    w = MigrationWorld(seed=1, n_rows=100, faults=FaultConfig(drop_rows_during_cdc=12))
    w.provision_target("numeric")
    w.enable_dual_write()
    for _ in range(40):
        w.emit_live_write()
    w.backfill_all()
    w.replay_cdc()  # drops 12
    rec = w.reconcile()
    assert not rec["passed"]
    assert rec["n_missing"] == 12, rec


def test_snapshot_restore_roundtrip():
    w = MigrationWorld(seed=1, n_rows=50)
    w.provision_target("numeric")
    snap = w.snapshot()
    w.backfill_all()
    assert w.target_count() == 50
    w.restore(snap)
    assert w.target_count() == 0  # rewound


# --------------------------------------------------------------------------------------
# Red-flag filter
# --------------------------------------------------------------------------------------

def test_red_flag_rejects_malformed():
    assert red_flag("{not json")[0] is None
    assert red_flag("just prose, no json")[0] is None
    assert red_flag('{"vote": "PASS"}')[0] is None  # no evidence
    assert red_flag('{"vote": "MAYBE", "evidence": "long enough evidence here"}')[0] is None
    assert red_flag("")[0] is None
    assert red_flag("x" * 5000)[0] is None  # over-long


def test_red_flag_accepts_well_formed():
    obj, reason = red_flag('{"vote": "FAIL", "evidence": "target_count=988 != source_count=1000", "confidence": 0.9}')
    assert obj is not None and reason == ""
    assert obj["vote"] == "FAIL"


# --------------------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------------------

def test_gate_passes_clean_state():
    client = MockClient(MockConfig(juror_noise=0.0, juror_redflag_rate=0.0))
    w = MigrationWorld(seed=2, n_rows=100, faults=FaultConfig.clean())
    w.provision_target("numeric")
    w.backfill_all()
    gate = ConsensusGate(client, k=3, max_jurors=20)
    res = gate.evaluate(w, {"tool": "backfill_all", "args": {}},
                        "target count == source count; checksums match", "backfill")
    assert res.decision == "PASS", res
    assert res.pass_votes - res.fail_votes >= 3


def test_gate_blocks_truncation_bug():
    client = MockClient(MockConfig(juror_noise=0.0, juror_redflag_rate=0.0))
    w = MigrationWorld(seed=2, n_rows=100, faults=FaultConfig(truncate_total_to_int=True))
    w.provision_target("numeric")
    w.backfill_all()
    gate = ConsensusGate(client, k=3, max_jurors=20)
    res = gate.evaluate(w, {"tool": "backfill_all", "args": {}},
                        "target count == source count; per-row checksums match", "backfill")
    assert res.decision == "FAIL", res
    assert res.fail_votes - res.pass_votes >= 3
    assert "INTEGER" in res.root_cause or "revenue" in res.root_cause.lower()


def test_gate_washes_out_noise_and_redflags():
    # Even with 15% noise and 12% red-flag junk, the gate should reach the right call.
    client = MockClient(MockConfig(juror_noise=0.15, juror_redflag_rate=0.12, seed=5))
    w = MigrationWorld(seed=3, n_rows=100, faults=FaultConfig(truncate_total_to_int=True))
    w.provision_target("numeric")
    w.backfill_all()
    gate = ConsensusGate(client, k=3, max_jurors=40)
    res = gate.evaluate(w, {"tool": "backfill_all", "args": {}},
                        "per-row checksums match", "backfill")
    assert res.decision == "FAIL", res
    assert res.red_flagged >= 0  # red-flagged votes were excluded from the tally


# --------------------------------------------------------------------------------------
# End-to-end orchestrator
# --------------------------------------------------------------------------------------

def test_baseline_ships_the_bug():
    client = MockClient(MockConfig())
    w = MigrationWorld(seed=7, n_rows=500, faults=FaultConfig.hard_mode())
    orch = Orchestrator(client, w, gate_enabled=False)
    result = orch.run()
    assert not result["reconciliation"]["passed"]  # baseline FAILS silently
    assert result["stats"]["gate_blocks"] == 0


def test_quorum_catches_and_recovers():
    client = MockClient(MockConfig())
    w = MigrationWorld(seed=7, n_rows=500, faults=FaultConfig.hard_mode())
    orch = Orchestrator(client, w, gate_enabled=True)
    result = orch.run()
    assert result["reconciliation"]["passed"], result["reconciliation"]
    assert result["stats"]["gate_blocks"] >= 1   # the jury overruled the brain at least once
    assert result["stats"]["rollbacks"] >= 1     # and a rewind happened


def test_irreversible_tagging():
    assert "cutover" in IRREVERSIBLE
    assert "backfill_all" in IRREVERSIBLE
    assert "finish" not in IRREVERSIBLE


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
