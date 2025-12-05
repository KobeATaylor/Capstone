import sqlite3
from pathlib import Path

DB_PATH = Path("database/typing.db")

SENTENCES = [
    "I like pizza",
    "I like coding",
    "You like pizza",
    "I am happy",
    "I am learning"
]

def main():
    if not DB_PATH.exists():
        raise SystemExit("Database not found")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # Avoid inserting duplicates
    existing = {r["text"] for r in conn.execute("SELECT text FROM sentences")}
    to_add = [(s,) for s in SENTENCES if s not in existing]

    if to_add:
        conn.executemany("INSERT INTO sentences(text) VALUES (?)", to_add)
        conn.commit()
        print(f"Inserted {len(to_add)} new sentences.")
    else:
        print("No new sentences")

    rows = conn.execute("SELECT id, text FROM sentences ORDER BY id DESC LIMIT 10").fetchall()
    print("\nRecent sentences:")
    for r in rows:
        print(f"  #{r['id']:>3}  {r['text']}")

    conn.close()

if __name__ == "__main__":
    main()