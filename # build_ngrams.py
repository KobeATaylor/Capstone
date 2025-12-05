import sqlite3, re
from collections import Counter
from pathlib import Path

DB_PATH = Path("database/typing.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def tokenize(text: str):
    """Lowercase, remove punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]

def build_and_store():
    if not DB_PATH.exists():
        raise SystemExit("Database not found")

    # Load sentences
    with get_conn() as conn:
        rows = conn.execute("SELECT text FROM sentences").fetchall()

    # Count n-grams
    uni, bi, tri = Counter(), Counter(), Counter()

    for r in rows:
        words = tokenize(r["text"])
        if not words:
            continue

        # Unigrams (context = "")
        for i in range(len(words)):
            uni[("", words[i])] += 1

        # Bigrams (context = previous 1 word)
        for i in range(len(words) - 1):
            ctx, nxt = words[i], words[i + 1]
            bi[(ctx, nxt)] += 1

        # Trigrams (context = previous 2 words)
        for i in range(len(words) - 2):
            ctx = f"{words[i]} {words[i + 1]}"
            nxt = words[i + 2]
            tri[(ctx, nxt)] += 1

    # Insert or update counts in DB
    with get_conn() as conn:
        def upsert(n, ctx, nxt, c):
            conn.execute(
                """
                INSERT INTO ngrams(n, context, next_word, count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(n, context, next_word)
                DO UPDATE SET count = count + excluded.count
                """,
                (n, ctx, nxt, c),
            )

        for (ctx, nxt), c in uni.items(): upsert(1, ctx, nxt, c)
        for (ctx, nxt), c in bi.items():  upsert(2, ctx, nxt, c)
        for (ctx, nxt), c in tri.items(): upsert(3, ctx, nxt, c)
        conn.commit()

def preview():
    """Show quick preview of n-gram counts."""
    with get_conn() as conn:
        print("\n🔹 Top Unigrams:")
        for r in conn.execute("""
            SELECT next_word, count FROM ngrams
            WHERE n=1 ORDER BY count DESC, next_word LIMIT 10
        """):
            print(f"  {r['next_word']:10} {r['count']}")

        print("\n🔹 Sample Bigrams (context='i'):")
        for r in conn.execute("""
            SELECT next_word, count FROM ngrams
            WHERE n=2 AND context='i' ORDER BY count DESC
        """):
            print(f"  i → {r['next_word']:10} {r['count']}")

        print("\n🔹 Sample Trigrams (context='i like'):")
        for r in conn.execute("""
            SELECT next_word, count FROM ngrams
            WHERE n=3 AND context='i like' ORDER BY count DESC
        """):
            print(f"  i like → {r['next_word']:10} {r['count']}")

if __name__ == "__main__":
    build_and_store()
    preview()
    print("\n N-grams built and stored in", DB_PATH)