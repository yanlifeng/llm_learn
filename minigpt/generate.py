# --- generate.py ---
import torch
from model import MiniGPT                 # ← 从 model.py 导入

ckpt = torch.load('mini_gpt.ckpt', map_location='cpu')
model = MiniGPT(vocab_size=len(ckpt['tok_vocab'])).eval()
model.load_state_dict(ckpt['model'])

# 重新构造 tokenizer
class T:
    def __init__(self, vocab): 
        self.vocab = vocab
        self.stoi = {ch:i for i,ch in enumerate(vocab)}
        self.itos = {i:ch for ch,i in self.stoi.items()}
    def encode(self,s): return [self.stoi[c] for c in s]
    def decode(self,ids): return ''.join(self.itos[i] for i in ids)
tok2 = T(ckpt['tok_vocab'])

prompt = "ROMEO:\n"
idx = torch.tensor([tok2.encode(prompt)], dtype=torch.long)
out = model.generate(idx, max_new_tokens=400, temperature=0.9, top_k=50)
print(tok2.decode(out[0].tolist()))

