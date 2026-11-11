"""Postgres + pgvector store for chunk vectors. Table: chunk_vectors(filename, vector)."""
import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from Utility.config import EMBED_DIM, postgres_conninfo


class VectorDbContext:
    def __init__(self, dim: int = EMBED_DIM):
        """Connect and create the pgvector extension and the table if they don't exist."""
        self.conn = psycopg.connect(**postgres_conninfo(), autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(f"CREATE TABLE IF NOT EXISTS chunk_vectors (filename text PRIMARY KEY, vector vector({int(dim)}) NOT NULL)")

    def save(self, filename: str, vector: list[float]) -> None:
        """Insert the chunk's vector, or replace it if the filename is already stored."""
        self.conn.execute(
            "INSERT INTO chunk_vectors (filename, vector) VALUES (%s, %s) "
            "ON CONFLICT (filename) DO UPDATE SET vector = EXCLUDED.vector",
            (filename, Vector(vector)),
        )

    def delete(self, source: str) -> None:
        """Delete every chunk of a source PDF, i.e. filenames starting with '{source}.' (e.g. 'Doc.chunk_1.md')."""
        self.conn.execute("DELETE FROM chunk_vectors WHERE starts_with(filename, %s)", (f"{source}.",))

    def delete_all(self) -> None:
        self.conn.execute("DELETE FROM chunk_vectors")

    def top_n(self, vector: list[float], n: int) -> list[str]:
        """Filenames of the n chunks closest to the vector (cosine distance), closest first."""
        rows = self.conn.execute(
            "SELECT filename FROM chunk_vectors ORDER BY vector <=> %s LIMIT %s", (Vector(vector), n)
        ).fetchall()
        return [row[0] for row in rows]

    def close(self) -> None:
        self.conn.close()
