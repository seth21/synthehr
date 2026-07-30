import sqlite3
from typing import Any
from contextlib import contextmanager

class Database:

    def __init__(self, db_path: str = "synthetic_ehr.db"):
        self.db_path = str(db_path)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        # Enforce foreign keys
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(
        self,
        sql: str,
        params: list[tuple[Any, ...]],
    ) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params)

    def fetch_one(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> sqlite3.Row | None:
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetch_all(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[sqlite3.Row]:
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def insert(
        self,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        cursor = self.execute(sql, params)
        self.commit()
        return cursor.lastrowid

    def update(
        self,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        cursor = self.execute(sql, params)
        self.commit()
        return cursor.rowcount

    def delete(
        self,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        cursor = self.execute(sql, params)
        self.commit()
        return cursor.rowcount

    def executescript(self, script: str):
        self.conn.executescript(script)
        self.commit()

    def table_exists(self, table: str) -> bool:
        row = self.fetch_one(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name=?
            """,
            (table,),
        )
        return row is not None

    @contextmanager
    def transaction(self):
        try:
            yield
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

db = Database()


