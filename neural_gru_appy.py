import re
import sqlite3
from pathlib import Path
from collections import Counter
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DB_PATH = Path("database/typing.db")

def get_conn():
    if not DB_PATH.exists():
        st.error("Database not found. Run init_schema.py first.")
        st.stop()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]

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

        if len(words) >= 2:
            ctx = f"{words[-2]} {words[-1]}"
            out = top_k_from(3, ctx)
            if out:
                return out

        if len(words) >= 1:
            ctx = words[-1]
            out = top_k_from(2, ctx)
            if out:
                return out

        return top_k_from(1, "")

class NextWordGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.norm(out[:, -1, :])
        return self.fc(out)

class TrigramDataset(Dataset):
    def __init__(self, conn, word_to_id):
        self.samples = []
        self.word_to_id = word_to_id

        rows = conn.execute(
            "SELECT context, next_word FROM ngrams WHERE n=3"
        ).fetchall()

        for r in rows:
            ctx_words = r["context"].split()
            if len(ctx_words) != 2:
                continue

            ctx_ids = [word_to_id.get(w, word_to_id["<unk>"]) for w in ctx_words]
            target_id = word_to_id.get(r["next_word"], word_to_id["<unk>"])

            self.samples.append((ctx_ids, target_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx, target = self.samples[idx]
        return torch.tensor(ctx), torch.tensor(target)

@st.cache_resource
def load_gru_model():
    ckpt_path = Path("gru_nextword.pt")
    if not ckpt_path.exists():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu")

    word_to_id = ckpt["word_to_id"]
    id_to_word = ckpt["id_to_word"]
    vocab_size = len(word_to_id)

    model = NextWordGRU(
        vocab_size=vocab_size,
        embed_dim=64,
        hidden_dim=128,
        pad_idx=word_to_id["<pad>"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, word_to_id, id_to_word

def predict_with_gru(prefix, k=5):
    model_pack = load_gru_model()
    if not model_pack:
        return []

    model, word_to_id, id_to_word = model_pack

    words = tokenize(prefix)
    if not words:
        return []

    ctx = words[-2:] if len(words) >= 2 else ["<pad>", words[-1]]
    ctx_ids = [word_to_id.get(w, word_to_id["<unk>"]) for w in ctx]

    x = torch.tensor([ctx_ids])
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0]

    topk = torch.topk(probs, k)
    return [(id_to_word[int(i)], float(p)) for i, p in zip(topk.indices, topk.values)]

st.title("Neural GRU Word Predicator")

prefix = st.text_input("Type something:", "")

k = st.number_input("Top-k", 1, 10, 5)

method = st.radio("Prediction method:", ["N-gram", "Neural (GRU)"])

if prefix.strip():
    if method == "N-gram":
        results = predict_next(prefix, k)
        st.write("### N-gram Predictions")
        for w, p, level in results:
            st.write(f"**{w}** — {p:.4f} ({level})")
    else:
        results = predict_with_gru(prefix, k)
        st.write("### GRU Predictions")
        for w, p in results:
            st.write(f"**{w}** — {p:.4f}")

st.divider()