"""
Pipeline state machine — BRD §3.

Full gated states (Phase 3+):
  IDLE → PRE_SENSING → G1_PROMOTIONS_BLOCKED → RUNNING_SENSE →
  FIELD_COLLECTION → G2_CONSENSUS_BLOCKED → RUNNING_POST →
  G3_RETRAIN_BLOCKED → CYCLE_COMPLETE

Phase 2 intra-cycle step states (D-017 — no gates yet):
  INGESTING → BASELINING → LOADING_SIGNALS → SENSING → SCORING
  → CYCLE_COMPLETE

These granular states give Phase 3 well-defined insertion points for
the three gate states (G1/G2/G3) without restructuring the runner.
"""
from enum import Enum


class PipelineState(str, Enum):
    # ── Phase 3+ gated states (BRD §3) ────────────────────────────────────
    IDLE                    = "IDLE"
    PRE_SENSING             = "PRE_SENSING"
    G1_PROMOTIONS_BLOCKED   = "G1_PROMOTIONS_BLOCKED"
    RUNNING_SENSE           = "RUNNING_SENSE"
    FIELD_COLLECTION        = "FIELD_COLLECTION"
    G2_CONSENSUS_BLOCKED    = "G2_CONSENSUS_BLOCKED"
    RUNNING_POST            = "RUNNING_POST"
    G3_RETRAIN_BLOCKED      = "G3_RETRAIN_BLOCKED"
    # ── Phase 2 step states (D-017) ────────────────────────────────────────
    INGESTING               = "INGESTING"
    BASELINING              = "BASELINING"
    LOADING_SIGNALS         = "LOADING_SIGNALS"
    SENSING                 = "SENSING"
    SCORING                 = "SCORING"
    # ── Terminal states ────────────────────────────────────────────────────
    CYCLE_COMPLETE          = "CYCLE_COMPLETE"
    ERROR                   = "ERROR"
