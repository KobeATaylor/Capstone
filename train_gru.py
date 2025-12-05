import sqlite3
from pathlib import Path
from collections import Counter
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DB_PATH = Path("database/typing.db")

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return [w for w in text.split() if w]


def build_vocab(min_freq: int = 1):
    if not DB_PATH.exists():
        raise SystemExit("Database not found. Run init_schema.py first.")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    counter = Counter()
    rows = conn.execute("SELECT text FROM sentences").fetchall()
    for r in rows:
        words = tokenize(r["text"])
        counter.update(words)

    conn.close()

    word_to_id = {
        "<pad>": 0,
        "<unk>": 1,
    }

    for word, freq in counter.items():
        if freq >= min_freq:
            word_to_id[word] = len(word_to_id)

    id_to_word = {i: w for w, i in word_to_id.items()}
    print(f"Vocab size: {len(word_to_id)} words")

    return word_to_id, id_to_word


class TrigramDataset(Dataset):

    def __init__(self, db_path: Path, word_to_id: dict, context_len: int = 2):
        self.word_to_id = word_to_id
        self.context_len = context_len
        self.samples = []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            "SELECT context, next_word, count FROM ngrams WHERE n=3"
        ).fetchall()
        conn.close()

        for r in rows:
            ctx_text = r["context"]         
            next_word = r["next_word"]       
            count = r["count"]

            ctx_words = ctx_text.split()
            if len(ctx_words) != self.context_len:
                continue

            ctx_ids = [
                self.word_to_id.get(w, self.word_to_id["<unk>"])
                for w in ctx_words
            ]
            target_id = self.word_to_id.get(next_word, self.word_to_id["<unk>"])

            self.samples.append((ctx_ids, target_id))

        print(f"Loaded {len(self.samples)} trigram training samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ctx_ids, target_id = self.samples[idx]
        ctx_tensor = torch.tensor(ctx_ids, dtype=torch.long)   
        target_tensor = torch.tensor(target_id, dtype=torch.long) 
        return ctx_tensor, target_tensor


class NextWordGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        pad_idx: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)       
        output, _ = self.gru(embedded)        
        last_hidden = output[:, -1, :]       
        last_hidden = self.norm(last_hidden)  
        logits = self.fc_out(last_hidden)  
        return logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    count = 0

    for ctx, target in loader:
        ctx = ctx.to(device)       
        target = target.to(device) 

        optimizer.zero_grad()
        logits = model(ctx)        
        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * ctx.size(0)
        count += ctx.size(0)

    return total_loss / max(1, count)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    word_to_id, id_to_word = build_vocab(min_freq=1)
    vocab_size = len(word_to_id)
    pad_idx = word_to_id["<pad>"]

    dataset = TrigramDataset(DB_PATH, word_to_id, context_len=2)
    if len(dataset) == 0:
        print("No trigram data found. Add sentences and rebuild ngrams first.")
        return

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NextWordGRU(vocab_size=vocab_size, embed_dim=64, hidden_dim=128, pad_idx=pad_idx)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    epochs = 5
    for epoch in range(1, epochs + 1):
        avg_loss = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
    }, "gru_nextword.pt")
    print("Saved model to gru_nextword.pt")


if __name__ == "__main__":
    main()
