import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ---------- CONFIG ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 64
batch_size = 32
max_iters = 10000
learning_rate = 3e-4

n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.3

# ---------- LOAD DATA ----------
with open("conversation_data_1000.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------- TRAIN TOKENIZER ----------
with open("temp.txt", "w", encoding="utf-8") as f:
    f.write(text)

spm.SentencePieceTrainer.train(
    input='temp.txt',
    model_prefix='spm',
    vocab_size=800,
    character_coverage=1.0,
    hard_vocab_limit=False   # 🔥 prevents crash
)

sp = spm.SentencePieceProcessor()
sp.load("spm.model")

vocab_size = sp.get_piece_size()

# ---------- ENCODE ----------
data = torch.tensor(sp.encode(text), dtype=torch.long)

split = int(0.9 * len(data))
train_data = data[:split]
val_data = data[split:]

# ---------- BATCH ----------
def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data_) - block_size - 1, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ---------- MODEL ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        tok = self.token_embedding(idx)
        pos = self.pos_embedding(torch.arange(T, device=device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B*T, vocab_size)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=150):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)

            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

# ---------- TRAIN ----------
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# ---------- SAVE ----------
torch.save(model.state_dict(), "word_model.pth")
# Removed: sp.save("spm.model")

print("✅ Word-level model trained!")