import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 64
n_embd = 256
n_head = 8
n_layer = 4
dropout = 0.3

# ---------- LOAD TOKENIZER ----------
sp = spm.SentencePieceProcessor()
sp.load("spm.model")

vocab_size = sp.get_piece_size()

# ---------- MODEL ----------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v

class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
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

    def forward(self, idx):
        B,T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.pos_embedding(torch.arange(T, device=device))
        x = tok + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, idx, max_new_tokens=150):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)

            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)

            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

# ---------- LOAD MODEL ----------
model = GPT().to(device)
model.load_state_dict(torch.load("word_model.pth", map_location=device))
model.eval()

print("🤖 Word-level Chatbot Ready!\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    prompt = f"User: {user_input}\nBot:"
    context = torch.tensor([sp.encode(prompt)], dtype=torch.long).to(device)

    output = model.generate(context)
    text_out = sp.decode(output[0].tolist())

    reply = text_out[len(prompt):].split("User:")[0]
    print("Bot:", reply.strip())