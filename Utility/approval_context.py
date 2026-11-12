"""Postgres store for human-in-the-loop approvals. Table: approvals(id, date_requested, request, approved, rejected)."""
from dataclasses import dataclass
from datetime import date

import psycopg

from Utility.config import postgres_conninfo


@dataclass
class Approval:
    id: int
    date_requested: date
    request: str
    approved: bool
    rejected: bool


class ApprovalContext:
    def __init__(self):
        """Connect and create the table if it doesn't exist."""
        self.conn = psycopg.connect(**postgres_conninfo(), autocommit=True)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS approvals ("
            "id serial PRIMARY KEY, "
            "date_requested date NOT NULL DEFAULT CURRENT_DATE, "
            "request text NOT NULL, "
            "approved boolean NOT NULL DEFAULT false)"
        )
        # added after the table first shipped, so tables created earlier get the column too
        self.conn.execute("ALTER TABLE approvals ADD COLUMN IF NOT EXISTS rejected boolean NOT NULL DEFAULT false")

    def create(self, request: str) -> int:
        """Add a request, not approved yet, dated today. Returns its id."""
        row = self.conn.execute("INSERT INTO approvals (request) VALUES (%s) RETURNING id", (request,)).fetchone()
        return row[0]

    def update(self, id: int, approved: bool | None = None, rejected: bool | None = None, request: str | None = None) -> bool:
        """Change the approval, rejection and/or text of a request; fields left as None are kept.

        A request is never approved and rejected at once: setting one to True clears the other (unless you set both,
        which raises ValueError). False is returned if the id doesn't exist.
        """
        if approved and rejected:
            raise ValueError("A request cannot be approved and rejected at the same time.")
        if approved and rejected is None:
            rejected = False
        if rejected and approved is None:
            approved = False
        cur = self.conn.execute(
            "UPDATE approvals SET approved = COALESCE(%s, approved), rejected = COALESCE(%s, rejected), "
            "request = COALESCE(%s, request) WHERE id = %s",
            (approved, rejected, request, id),
        )
        return cur.rowcount > 0

    def delete(self, id: int) -> bool:
        """Delete a request. False if the id doesn't exist."""
        return self.conn.execute("DELETE FROM approvals WHERE id = %s", (id,)).rowcount > 0

    def getall(self) -> list[Approval]:
        """Every request, oldest first."""
        rows = self.conn.execute("SELECT id, date_requested, request, approved, rejected FROM approvals ORDER BY id").fetchall()
        return [Approval(*row) for row in rows]

    def close(self) -> None:
        self.conn.close()
