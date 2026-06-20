"""The migration world: a sandboxed, snapshot-able stand-in for a real cloud/DB migration.

We model the headline task from the design doc:

    "Migrate the `orders` service and its Postgres database from the legacy stack
     to the new target stack, with zero data loss and zero downtime, and prove it."

To keep the demo fully offline and deterministic (no real Docker/Postgres needed for
the loop), the "databases" are in-memory tables of rows. But every operation that a
real migration performs has a faithful analogue here:

  * provision target schema           -> create target table with a column type spec
  * backfill source -> target          -> copy rows in chunks (lossy if faults injected)
  * dual-write / CDC during migration  -> capture live writes, replay to target
  * cutover                            -> flip the "live" pointer to target
  * validate                           -> row-count + per-row checksum reconciliation

The crucial property: a wrong cast (NUMERIC(10,2) -> INT dropping cents) or a dropped
row is PERMANENT and SILENT. The agent's own "migration complete" report will not catch
it. Only an independent reconciliation oracle — or a jury that re-runs the queries —
will.

This module is pure Python with no Claude dependency, so it can be unit-tested and run
offline. The brain drives it through a tool surface (see tools.py); the jurors inspect
it read-only (see jurors.py).
"""

import copy
import hashlib
import json
import random
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any, Optional


# --------------------------------------------------------------------------------------
# Fault injection — the "hard mode" that kills the baseline agent.
# --------------------------------------------------------------------------------------

@dataclass
class FaultConfig:
    """Silent corruptions the world will inject during a migration.

    Each is the kind of thing that "looks done" but fails reconciliation. They are
    SILENT: no exception, no log line. The naive agent reports success.
    """

    # Backfill casts target `total` column to integer cents-dropping (NUMERIC -> INT).
    truncate_total_to_int: bool = False
    # Drop this many rows during one CDC/dual-write window (writes that landed only on
    # the old DB).
    drop_rows_during_cdc: int = 0
    # Skip applying the dual-write shim entirely (so writes during migration are lost).
    skip_dual_write: bool = False

    @classmethod
    def hard_mode(cls) -> "FaultConfig":
        """The demo's hard mode: the bugs the baseline ships and the jury catches."""
        return cls(truncate_total_to_int=True, drop_rows_during_cdc=12, skip_dual_write=False)

    @classmethod
    def clean(cls) -> "FaultConfig":
        return cls()

    def any_active(self) -> bool:
        return self.truncate_total_to_int or self.drop_rows_during_cdc > 0 or self.skip_dual_write


# --------------------------------------------------------------------------------------
# The world state.
# --------------------------------------------------------------------------------------

@dataclass
class MigrationWorld:
    """A snapshot-able world holding a source DB, a target DB, and live write traffic.

    Rows are dicts: {"id": int, "total": Decimal, "status": str, "customer": str}.
    `total` is a money value; the migration must preserve it exactly (cents matter).
    """

    seed: int = 7
    n_rows: int = 1000
    faults: FaultConfig = field(default_factory=FaultConfig.clean)

    # Internal state
    source: dict[int, dict] = field(default_factory=dict)
    target: dict[int, dict] = field(default_factory=dict)
    target_provisioned: bool = False
    target_total_is_int: bool = False  # True if target column was created as INT (bug)
    dual_write_active: bool = False
    cdc_buffer: list[dict] = field(default_factory=list)  # writes captured during migration
    cdc_only_ids: set = field(default_factory=set)        # ids that arrived after backfill froze
    live_points_to_target: bool = False  # cutover flips this
    migrated_manifest: Optional[dict] = None  # frozen source at cutover (the "as-migrated" set)
    _next_id: int = 0
    _rng: random.Random = field(default_factory=lambda: random.Random())
    log: list[str] = field(default_factory=list)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        self._seed_source()

    # ---- seeding -----------------------------------------------------------------

    def _seed_source(self) -> None:
        statuses = ["pending", "paid", "shipped", "refunded", "cancelled"]
        customers = [f"cust_{i:04d}" for i in range(200)]
        for i in range(1, self.n_rows + 1):
            # Money with real cents (so an int-truncation bug is detectable).
            dollars = self._rng.randint(5, 950)
            cents = self._rng.choice([0, 25, 49, 50, 75, 99])
            total = Decimal(f"{dollars}.{cents:02d}")
            self.source[i] = {
                "id": i,
                "total": total,
                "status": self._rng.choice(statuses),
                "customer": self._rng.choice(customers),
            }
        self._next_id = self.n_rows + 1
        self.log.append(f"seeded source with {self.n_rows} orders")

    # ---- live write traffic (the "zero downtime" pressure) -----------------------

    def emit_live_write(self) -> dict:
        """A new order arrives. If dual-write is active it should hit BOTH stores."""
        oid = self._next_id
        self._next_id += 1
        dollars = self._rng.randint(5, 950)
        cents = self._rng.choice([0, 25, 49, 50, 75, 99])
        row = {
            "id": oid,
            "total": Decimal(f"{dollars}.{cents:02d}"),
            "status": "pending",
            "customer": self._rng.choice([f"cust_{i:04d}" for i in range(200)]),
        }
        # The write always lands on whichever store is "live"...
        if self.live_points_to_target:
            self._apply_write_to_target(row)
        else:
            self.source[oid] = dict(row)
            # ...and, during a migration window, a write that arrives after the backfill
            # snapshot was frozen only reaches target via CDC/dual-write. We mark it
            # cdc-only so backfill won't also copy it (modeling a point-in-time backfill).
            if self.dual_write_active:
                self.cdc_only_ids.add(oid)
                if not self.faults.skip_dual_write:
                    self.cdc_buffer.append(dict(row))
        return row

    # ---- migration operations (the brain drives these via tools) -----------------

    def provision_target(self, total_column_type: str = "numeric") -> str:
        """Create the target schema. A 'integer' type for `total` is the truncation bug."""
        self.target = {}
        self.target_provisioned = True
        self.target_total_is_int = (total_column_type.lower() in ("int", "integer"))
        # The fault injector can FORCE the bug regardless of what the brain asked for —
        # this models a silent infra/driver default that truncates.
        if self.faults.truncate_total_to_int:
            self.target_total_is_int = True
        self.log.append(
            f"provisioned target (total column type="
            f"{'INTEGER' if self.target_total_is_int else 'NUMERIC(10,2)'})"
        )
        return "target provisioned"

    def enable_dual_write(self) -> str:
        self.dual_write_active = True
        self.log.append("dual-write / CDC enabled")
        return "dual-write enabled"

    def _apply_write_to_target(self, row: dict) -> None:
        stored = dict(row)
        if self.target_total_is_int:
            stored["total"] = Decimal(int(stored["total"]))  # drop cents silently
        self.target[stored["id"]] = stored

    def backfill_chunk(self, start_id: int, count: int) -> str:
        """Copy a chunk of source rows into target. Truncation applies per the schema."""
        if not self.target_provisioned:
            raise RuntimeError("backfill before provisioning target")
        copied = 0
        for oid in range(start_id, start_id + count):
            # Rows that arrived after the backfill snapshot froze are CDC-only — they
            # must reach target through replay, not through backfill.
            if oid in self.source and oid not in self.cdc_only_ids:
                self._apply_write_to_target(self.source[oid])
                copied += 1
        self.log.append(f"backfilled rows [{start_id}, {start_id + count}) -> {copied} copied")
        return f"backfilled {copied} rows"

    def backfill_all(self, chunk_size: int = 250) -> str:
        ids = sorted(self.source.keys())
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]
            self.backfill_chunk(chunk[0], chunk[-1] - chunk[0] + 1)
        return f"backfilled all {len(ids)} source rows"

    def replay_cdc(self) -> str:
        """Apply the buffered live writes to target. Faults can drop some silently."""
        buffer = self.cdc_buffer
        self.cdc_buffer = []
        to_drop = self.faults.drop_rows_during_cdc
        applied = 0
        for idx, row in enumerate(buffer):
            # Silently drop the first `to_drop` rows (writes that "landed only on old DB").
            if idx < to_drop:
                # The row exists in source (it was a live write to source) but never
                # reaches target -> data loss that reconciliation will expose.
                continue
            self._apply_write_to_target(row)
            applied += 1
        self.log.append(
            f"replayed CDC buffer: {applied} applied"
            + (f", {min(to_drop, len(buffer))} SILENTLY DROPPED" if to_drop else "")
        )
        return f"replayed {applied} CDC writes"

    def cutover(self) -> str:
        """Atomic cutover: drain any remaining CDC buffer, then flip live traffic.

        A real cutover briefly freezes writes, drains the final CDC tail to target, and
        flips the pointer. If the CDC drop fault is active, this final drain still loses
        rows — which the cutover gate then catches.
        """
        self.replay_cdc()  # final drain of in-flight writes
        # Freeze the "as-migrated" manifest: the source dataset at the instant of cutover.
        # Reconciliation validates THIS migrated against target (post-cutover live writes
        # legitimately go only to target and are not part of the migration's correctness).
        self.migrated_manifest = copy.deepcopy(self.source)
        self.live_points_to_target = True
        self.dual_write_active = False
        self.log.append("CUTOVER: drained CDC tail; live traffic now points to target")
        return "cutover complete"

    # ---- read-only inspection (jurors + reconciliation oracle use these) ---------

    def source_count(self) -> int:
        return len(self.source)

    def target_count(self) -> int:
        return len(self.target)

    def source_revenue(self) -> Decimal:
        return sum((r["total"] for r in self.source.values()), Decimal("0"))

    def target_revenue(self) -> Decimal:
        return sum((r["total"] for r in self.target.values()), Decimal("0"))

    def _row_checksum(self, row: dict) -> str:
        # Stable per-row checksum over the load-bearing fields, money normalized to cents.
        total_cents = int((row["total"] * 100).to_integral_value(rounding=ROUND_DOWN))
        payload = f"{row['id']}|{total_cents}|{row['status']}|{row['customer']}"
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def table_checksum(self, which: str) -> str:
        table = self.source if which == "source" else self.target
        h = hashlib.sha256()
        for oid in sorted(table.keys()):
            h.update(self._row_checksum(table[oid]).encode())
        return h.hexdigest()[:16]

    def copied_rows_intact(self) -> dict:
        """For rows present in BOTH stores, are the values intact (no truncation)?

        This is the right check after BACKFILL, where target legitimately lags source by
        the in-flight (CDC-only) writes. It isolates the truncation bug from the expected
        count gap: it compares per-row checksums on the intersection only.
        """
        common = sorted(set(self.source) & set(self.target))
        mismatched = [oid for oid in common
                      if self._row_checksum(self.source[oid]) != self._row_checksum(self.target[oid])]
        return {
            "common_rows": len(common),
            "mismatched_rows": mismatched,
            "n_mismatched": len(mismatched),
            "intact": len(mismatched) == 0 and len(common) > 0,
        }

    def reconcile(self) -> dict:
        """The independent ground-truth oracle. Hard PASS/FAIL, no subjectivity.

        Compares the as-migrated dataset vs target on row count, per-row checksums, and
        revenue total. After cutover it uses the frozen migration manifest (source at
        cutover) so post-cutover live writes — which legitimately go only to target —
        don't register as spurious extra rows.

        This is what judges watch: baseline=FAIL, QUORUM=PASS on identical faults.
        """
        reference = self.migrated_manifest if self.migrated_manifest is not None else self.source
        if self.migrated_manifest is not None:
            target_view = {oid: self.target[oid] for oid in self.target if oid in reference}
        else:
            target_view = self.target
        missing = sorted(set(reference) - set(target_view))
        extra = sorted(set(target_view) - set(reference))
        mismatched = []
        for oid in sorted(set(reference) & set(target_view)):
            if self._row_checksum(reference[oid]) != self._row_checksum(target_view[oid]):
                mismatched.append(oid)
        src_rev = sum((r["total"] for r in reference.values()), Decimal("0"))
        tgt_rev = sum((target_view[oid]["total"] for oid in target_view), Decimal("0"))
        passed = not missing and not extra and not mismatched and src_rev == tgt_rev
        return {
            "passed": passed,
            "source_count": len(reference),
            "target_count": len(target_view),
            "missing_rows": missing,
            "extra_rows": extra,
            "mismatched_rows": mismatched,
            "source_revenue": str(src_rev),
            "target_revenue": str(tgt_rev),
            "revenue_delta": str(tgt_rev - src_rev),
            "n_missing": len(missing),
            "n_mismatched": len(mismatched),
        }

    # ---- snapshot / restore (the rewind machinery) -------------------------------

    def snapshot(self) -> dict:
        """A deep, restorable snapshot of the entire world state."""
        return {
            "source": copy.deepcopy(self.source),
            "target": copy.deepcopy(self.target),
            "target_provisioned": self.target_provisioned,
            "target_total_is_int": self.target_total_is_int,
            "dual_write_active": self.dual_write_active,
            "cdc_buffer": copy.deepcopy(self.cdc_buffer),
            "cdc_only_ids": set(self.cdc_only_ids),
            "live_points_to_target": self.live_points_to_target,
            "migrated_manifest": copy.deepcopy(self.migrated_manifest),
            "_next_id": self._next_id,
            "rng_state": self._rng.getstate(),
            "log_len": len(self.log),
        }

    def restore(self, snap: dict) -> None:
        self.source = copy.deepcopy(snap["source"])
        self.target = copy.deepcopy(snap["target"])
        self.target_provisioned = snap["target_provisioned"]
        self.target_total_is_int = snap["target_total_is_int"]
        self.dual_write_active = snap["dual_write_active"]
        self.cdc_buffer = copy.deepcopy(snap["cdc_buffer"])
        self.cdc_only_ids = set(snap["cdc_only_ids"])
        self.live_points_to_target = snap["live_points_to_target"]
        self.migrated_manifest = copy.deepcopy(snap["migrated_manifest"])
        self._next_id = snap["_next_id"]
        self._rng.setstate(snap["rng_state"])
        # Trim the log back to the snapshot point and note the rewind.
        self.log = self.log[: snap["log_len"]]
        self.log.append("<< ROLLED BACK to last good snapshot >>")

    # ---- state summary handed to brain + jurors ----------------------------------

    def state_summary(self) -> dict:
        """A compact, read-only summary of the observable world (no fault flags!).

        Jurors and the brain see this. They do NOT see `self.faults` — they must
        discover corruption by inspecting state, exactly like a real verifier.
        """
        intact = self.copied_rows_intact()
        return {
            "source_count": self.source_count(),
            "target_count": self.target_count(),
            "target_provisioned": self.target_provisioned,
            "target_total_column_type": "INTEGER" if self.target_total_is_int else "NUMERIC(10,2)",
            "dual_write_active": self.dual_write_active,
            "cdc_buffer_pending": len(self.cdc_buffer),
            "live_points_to_target": self.live_points_to_target,
            "source_revenue": str(self.source_revenue()),
            "target_revenue": str(self.target_revenue()),
            "source_checksum": self.table_checksum("source"),
            "target_checksum": self.table_checksum("target"),
            # For the backfill check: are the rows already copied intact (cents preserved)?
            "copied_rows_common": intact["common_rows"],
            "copied_rows_mismatched": intact["n_mismatched"],
            "copied_rows_intact": intact["intact"],
            # As-migrated reconciliation (manifest-aware), for the cutover check.
            "as_migrated_missing": self.reconcile()["n_missing"],
            "as_migrated_mismatched": self.reconcile()["n_mismatched"],
        }


def _decimal_default(o: Any):
    if isinstance(o, Decimal):
        return str(o)
    raise TypeError


def summary_json(d: dict) -> str:
    return json.dumps(d, indent=2, default=_decimal_default, sort_keys=True)
