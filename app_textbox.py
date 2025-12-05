import re
import sqlite3
from pathlib import Path
from collections import Counter
import streamlit as st

DB_PATH = Path("database/typing.db")

def get_conn():
    if not DB_PATH.exists():
        st.error("Database not found. Run init_schema.py first.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def insert_sentence(text: str) -> int:
    with get_conn() as conn:
        cur = conn.execute("INSERT INTO sentences(text) VALUES (?)", (text,))
        conn.commit()
        return cur.lastrowid

def tokenize(text: str):
    text = text.lower()
    # Keep letters/digits/underscore/space/apostrophe
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]

def upsert_ngram(conn, n: int, context: str, next_word: str, c: int):
    conn.execute(
        """
        INSERT INTO ngrams(n, context, next_word, count)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(n, context, next_word)
        DO UPDATE SET count = count + excluded.count
        """,
        (n, context, next_word, c),
    )

def update_ngrams_for_sentence(text: str):
    words = tokenize(text)
    if not words:
        return 0, 0, 0

    uni, bi, tri = Counter(), Counter(), Counter()

    # Unigrams
    for i in range(len(words)):
        uni[("", words[i])] += 1
    # Bigrams
    for i in range(len(words) - 1):
        bi[(words[i], words[i + 1])] += 1
    # Trigrams
    for i in range(len(words) - 2):
        ctx = f"{words[i]} {words[i + 1]}"
        tri[(ctx, words[i + 2])] += 1

    with get_conn() as conn:
        for (ctx, nxt), c in uni.items():
            upsert_ngram(conn, 1, ctx, nxt, c)
        for (ctx, nxt), c in bi.items():
            upsert_ngram(conn, 2, ctx, nxt, c)
        for (ctx, nxt), c in tri.items():
            upsert_ngram(conn, 3, ctx, nxt, c)
        conn.commit()

    return len(uni), len(bi), len(tri)

def recent_sentences(limit=10):
    with get_conn() as conn:
        return conn.execute(
            "SELECT id, text FROM sentences ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

def vocab_size(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(DISTINCT next_word) AS v FROM ngrams WHERE n=1"
    ).fetchone()
    return int(row["v"] or 0)

def context_total(conn: sqlite3.Connection, n: int, context: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(count),0) AS s FROM ngrams WHERE n=? AND context=?",
        (n, context),
    ).fetchone()
    return int(row["s"] or 0)

def fetch_candidates(conn: sqlite3.Connection, n: int, context: str):
    rows = conn.execute(
        "SELECT next_word, count FROM ngrams WHERE n=? AND context=?",
        (n, context),
    ).fetchall()
    return [(r["next_word"], int(r["count"])) for r in rows]

def laplace_scores(cands, total: int, V: int, alpha: float = 1.0):
    denom = total + alpha * V
    return [(w, (c + alpha) / denom) for (w, c) in cands]

def predict_next(prefix: str, k: int = 5, alpha: float = 1.0):
    words = tokenize(prefix)

    with get_conn() as conn:
        V = vocab_size(conn)
        if V == 0:
            return []

        def top_k_from(n: int, context: str):
            total = context_total(conn, n, context)
            if total == 0:
                return []
            cands = fetch_candidates(conn, n, context)
            scores = laplace_scores(cands, total, V, alpha)
            scores.sort(key=lambda x: x[1], reverse=True)
            return [(w, p, n) for (w, p) in scores[:k]]

        # try trigram
        if len(words) >= 2:
            ctx = f"{words[-2]} {words[-1]}"
            out = top_k_from(3, ctx)
            if out:
                return out

        # backoff to bigram
        if len(words) >= 1:
            ctx = words[-1]
            out = top_k_from(2, ctx)
            if out:
                return out

        # backoff to unigram (context="")
        out = top_k_from(1, "")
        return out

st.set_page_config(page_title="Sentence Feeder", layout="centered")
st.title("Sentence Feeder")
st.write(
    "Type any sentence(s) below and click Submit to store them"
)

with st.form("typing_form", clear_on_submit=True):
    text = st.text_area(
        "Your text",
        height=120,
        placeholder="Example: I like pizza\nYou like pizza",
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        do_update = st.checkbox("Update N-grams for this sentence", value=True)
    with col2:
        submitted = st.form_submit_button("Submit")

if submitted:
    cleaned = text.strip()
    if not cleaned:
        st.warning("Please type something before submitting.")
    else:
        # You can choose to split by lines or treat the whole box as one sentence
        # Here we split by newlines and store each non-empty line as a sentence.
        lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
        inserted = 0
        for ln in lines:
            sid = insert_sentence(ln)
            inserted += 1
            if do_update:
                u, b, t = update_ngrams_for_sentence(ln)
        st.success(f"Saved {inserted} sentence(s).")
        if do_update:
            st.info("N-grams updated for the submitted text.")

st.subheader("Recent sentences")
rows = recent_sentences(10)
if rows:
    for r in rows:
        st.write(f"**#{r['id']:>3}** — {r['text']}")
else:
    st.caption("No sentences.")

st.divider()
st.caption("Tip: Use one sentence per line if you want each to be stored separately.")

st.subheader("Next-word suggestions")

col_a, col_b = st.columns([3, 1])
with col_a:
    prefix = st.text_input("Type a prefix (e.g., 'I like')", value="")
with col_b:
    k = st.number_input("Top-k", min_value=1, max_value=10, value=3, step=1)

if prefix.strip():
    try:
        results = predict_next(prefix, k=k, alpha=1.0)
        if results:
            level_names = {1: "unigram", 2: "bigram", 3: "trigram"}
            used_level = level_names.get(results[0][2], "?")
            st.caption(f"Backoff level used: **{used_level}**")
            for w, p, n in results:
                st.write(f"- **{w}**  —  P≈ `{p:.3f}` ({level_names.get(n)})")
        else:
            st.info("No suggestions yet")
    except Exception as e:
        st.error(f"Prediction error: {e}")
else:
    st.caption("Enter a few words above to see suggestions.")