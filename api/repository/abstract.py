"""
AbstractRepository — BRD §5.0

All persistence flows through this interface.
NO other file may import sqlite3, open a file, or write SQL.
"""
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any


class AbstractRepository(ABC):

    # ── Generic CRUD ────────────────────────────────────────────────────────
    @abstractmethod
    def get(self, entity: str, key: dict) -> dict | None:
        """Return a single row matching key dict, or None."""

    @abstractmethod
    def query(self, entity: str, filters: dict = None,
               order_by: list[str] = None) -> list[dict]:
        """Return all rows matching filters (ANDed equality)."""

    @abstractmethod
    def upsert(self, entity: str, rows: list[dict]) -> int:
        """Insert or replace rows; return affected count."""

    @abstractmethod
    def delete(self, entity: str, filters: dict) -> int:
        """Delete rows matching filters; return deleted count."""

    # ── Bulk frame I/O ──────────────────────────────────────────────────────
    @abstractmethod
    def read_frame(self, entity: str, filters: dict = None) -> "pd.DataFrame":
        """Return table (or filtered subset) as a pandas DataFrame."""

    @abstractmethod
    def write_frame(self, entity: str, df: "pd.DataFrame", mode: str = "upsert") -> int:
        """Write a DataFrame to entity; mode='upsert' or 'replace'."""

    # ── Transaction context manager ─────────────────────────────────────────
    @abstractmethod
    @contextmanager
    def transaction(self):
        """Context manager for atomic transactions.
        Usage: with repo.transaction(): ...
        """
        yield  # pragma: no cover

    # ── Gate operations ─────────────────────────────────────────────────────
    @abstractmethod
    def get_gate_status(self, gate_id: str, cycle_id: str) -> str:
        """Return current gate status string (pending|blocked|approved)."""

    @abstractmethod
    def set_gate_status(self, gate_id: str, cycle_id: str,
                        status: str, actor: str) -> None:
        """Update gate status AND append audit_log row in the same transaction."""

    # ── Pipeline state machine ───────────────────────────────────────────────
    @abstractmethod
    def get_pipeline_state(self, cycle_id: str) -> dict | None:
        """Return pipeline_state row for cycle_id, or None."""

    @abstractmethod
    def set_pipeline_state(self, cycle_id: str, state: str,
                           meta: dict, actor: str) -> None:
        """Persist pipeline state AND append audit_log row in the same transaction."""

    # ── Human-facing import/export ───────────────────────────────────────────
    @abstractmethod
    def import_csv(self, entity: str, path: str) -> int:
        """Load a CSV file into entity table; return row count inserted.
        CSV/file I/O is ONLY allowed here."""

    @abstractmethod
    def export_excel(self, entity: str, path: str, filters: dict = None) -> str:
        """Export entity (optionally filtered) to an Excel file at path;
        return the file path. File I/O is ONLY allowed here."""

    @abstractmethod
    def read_csv_raw(self, path: str) -> "pd.DataFrame":
        """Read a CSV file as a raw DataFrame without persisting it.
        Used by ingestion to validate data before calling upsert().
        File I/O is ONLY allowed here — callers must not open files directly."""
