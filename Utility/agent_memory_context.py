"""Postgres store for the agent's memory. Table: agent_memory(last_update, memory), a single row for now."""
from datetime import date

import psycopg

from Utility.config import postgres_conninfo


class AgentMemoryContext:
    def __init__(self):
        """Connect and create the table if it doesn't exist."""
        self.conn = psycopg.connect(**postgres_conninfo(), autocommit=True)
        # The CHECK on id keeps the table to a single row.
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS agent_memory ("
            "id integer PRIMARY KEY DEFAULT 1 CHECK (id = 1), "
            "last_update date NOT NULL DEFAULT CURRENT_DATE, "
            "memory text NOT NULL)"
        )

    def save(self, memory: str) -> None:
        """Store the memory, replacing the existing one; last_update is set to today."""
        self.conn.execute(
            "INSERT INTO agent_memory (id, last_update, memory) VALUES (1, CURRENT_DATE, %s) "
            "ON CONFLICT (id) DO UPDATE SET last_update = CURRENT_DATE, memory = EXCLUDED.memory",
            (memory,),
        )

    def update(self, memory: str) -> bool:
        """Change the stored memory and set last_update to today. Returns False if no memory is stored yet."""
        cur = self.conn.execute("UPDATE agent_memory SET last_update = CURRENT_DATE, memory = %s WHERE id = 1", (memory,))
        return cur.rowcount > 0

    def delete(self) -> None:
        self.conn.execute("DELETE FROM agent_memory WHERE id = 1")

    def get(self) -> tuple[date, str] | None:
        """(last_update, memory), or None if there is no memory."""
        row = self.conn.execute("SELECT last_update, memory FROM agent_memory WHERE id = 1").fetchone()
        return (row[0], row[1]) if row else None

    def close(self) -> None:
        self.conn.close()
