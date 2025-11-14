# train.py
import math, torch
from torch.optim import AdamW
from pathlib import Path
from model import MiniGPT                 # ← 从 model.py 导入
from tokenizer import CharTokenizer       # ← 从 tokenizer.py 导入

# ---- 超参数 ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256
batch_size = 64
n_layer, n_head, n_embd = 4, 4, 256
dropout = 0.1
max_steps = 2000
eval_interval = 200
lr = 3e-4

# ---- 数据与分词（字符级）----
text = Path('input.txt').read_text(encoding='utf-8')
tok = CharTokenizer(text)
ids = [tok.stoi[c] for c in text]
split = int(0.9 * len(ids))
train_ids, val_ids = ids[:split], ids[split:]

def get_batch(split_ids, block_size, batch_size, device):
    import torch
    L = len(split_ids)
    ix = torch.randint(0, L - block_size - 1, (batch_size,))
    x = torch.stack([torch.tensor(split_ids[i:i+block_size]) for i in ix]).to(device)
    y = torch.stack([torch.tensor(split_ids[i+1:i+block_size+1]) for i in ix]).to(device)
    return x, y

# ---- 模型与优化器 ----
model = MiniGPT(tok.vocab_size, block_size, n_layer, n_head, n_embd, dropout).to(device)
opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

def estimate_ppl(split_ids, steps=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(steps):
            x, y = get_batch(split_ids, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return math.exp(sum(losses)/len(losses))

# ---- 训练循环 ----
model.train()
for step in range(1, max_steps + 1):
    x, y = get_batch(train_ids, block_size, batch_size, device)
    _, loss = model(x, y)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % eval_interval == 0:
        tr_ppl = estimate_ppl(train_ids, steps=10)
        va_ppl = estimate_ppl(val_ids, steps=10)
        print(f"step {step}: loss {loss.item():.3f} | train PPL {tr_ppl:.2f} | val PPL {va_ppl:.2f}")

# ---- 保存权重与词表 ----
torch.save({'model': model.state_dict(), 'tok_vocab': tok.vocab}, 'mini_gpt.ckpt')
print("Saved to mini_gpt.ckpt")

# ---- 推理 Demo（两种：无缓存 + 带缓存），方便你对比 ----
@torch.no_grad()
def demo_generate(prompt="Once upon a time", max_new_tokens=200, use_kv=True, temperature=1.0, top_k=64):
    ckpt = torch.load('mini_gpt.ckpt', map_location=device)
    tok_vocab = ckpt['tok_vocab']
    # 复现 tokenizer（保证一致性）
    from tokenizer import CharTokenizer
    # 用保存的 vocab 重建一个 tokenizer
    t2 = CharTokenizer(''.join(tok_vocab))  # 只为还原 stoi/itos
    t2.vocab = tok_vocab
    t2.stoi = {ch: i for i, ch in enumerate(tok_vocab)}
    t2.itos = {i: ch for i, ch in enumerate(tok_vocab)}

    m = MiniGPT(len(tok_vocab), block_size, n_layer, n_head, n_embd, dropout).to(device)
    m.load_state_dict(ckpt['model'])
    m.eval()

    idx = torch.tensor([t2.encode(prompt)], device=device)
    if use_kv:
        out = m.generate_with_kv(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    else:
        out = m.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    print(t2.decode(out[0].tolist()))

# 你可以把下面两行其中一行注释掉，只跑一个
print("\n==== Generation w/ KV-Cache ====")
demo_generate(use_kv=True)
print("\n==== Generation w/o KV-Cache ====")
demo_generate(use_kv=False)
