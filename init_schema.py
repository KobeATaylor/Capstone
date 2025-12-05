import sqlite3
from pathlib import Path

# Create database folder and connect
db_path = Path("database/typing.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path)

# Define schema
schema = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sentences (
  id   INTEGER PRIMARY KEY,
  text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ngrams (
  id         INTEGER PRIMARY KEY,
  n          INTEGER NOT NULL,
  context    TEXT NOT NULL,
  next_word  TEXT NOT NULL,
  count      INTEGER NOT NULL DEFAULT 1,
  UNIQUE (n, context, next_word)
);

CREATE INDEX IF NOT EXISTS idx_ngrams_n_context ON ngrams(n, context);
"""

# Execute and save
conn.executescript(schema)
conn.commit()

print("Schema created successfully at:", db_path)
conn.close()