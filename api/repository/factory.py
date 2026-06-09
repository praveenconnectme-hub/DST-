"""RepositoryFactory — BRD §5.0.

Usage:
    repo = RepositoryFactory.create(config)

config dict must contain 'type' key.  v1 supports 'sqlite' only.
"""
from .abstract import AbstractRepository
from .sqlite_repo import SQLiteRepository


class RepositoryFactory:
    @staticmethod
    def create(config: dict) -> AbstractRepository:
        repo_type = config.get("type", "sqlite")
        if repo_type == "sqlite":
            db_path = config.get("db_path", "/data/dst.db")
            return SQLiteRepository(db_path=db_path)
        raise ValueError(f"Unknown repository type: {repo_type!r}. Supported: 'sqlite'")
