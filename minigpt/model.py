# model.py
import math, torch, torch.nn as nn, torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # 预先构造下三角因果 mask（[1,1,T,T] 便于广播）
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.size()
        qkv = self.qkv(x)                          # (B,T,3C)
        q, k, v = qkv.split(C, dim=2)              # (B,T,C) each
        # 多头
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,h,T,d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # KV Cache：训练时 None；推理时为 dict{'k','v'}，形状 (B,h,t_cached,d)
        if kv_cache is not None:
            if 'k' in kv_cache:
                k = torch.cat([kv_cache['k'], k], dim=2)
                v = torch.cat([kv_cache['v'], v], dim=2)
            kv_cache['k'], kv_cache['v'] = k, v  # 持久化

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B,h,T,T')

        # 因果 mask
        # - 训练/无缓存：T'=T，直接用 [:,:,:T,:T]
        # - 增量推理/有缓存：当前查询在全局位置 cur_len-1，用 [:,:,cur_len-1:cur_len,:cur_len]
        if kv_cache is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        else:
            cur_len = k.size(2)               # 历史+当前
            row = cur_len - 1                 # 当前全局位置
            att = att.masked_fill(self.mask[:, :, row:row+1, :cur_len] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                                # (B,h,T,d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)           # (B,T,C)
        y = self.resid_drop(self.proj(y))
        return y, kv_cache

class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, mult=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, mult * n_embd),
            nn.GELU(),
            nn.Linear(mult * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.fc(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, caches=None):
        # Pre-LN：先归一化再子层
        y, cache = self.attn(self.ln1(x), kv_cache=None if caches is None else caches)
        x = x + y
        y = self.mlp(self.ln2(x))
        x = x + y
        if caches is not None and cache is not None:
            # 直接把 attn 的 kv_cache 写回本层 cache（保持你原接口名 caches）
            caches.update(cache)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size=256, n_layer=4, n_head=4, n_embd=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # 可选权重共享：
        # self.head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)                       # 训练时不维护 cache
        x = self.ln_f(x)
        logits = self.head(x)                # (B,T,V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # 无 KV Cache 的简易生成（每次截到最后 block_size 个）
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1:]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @torch.no_grad()
    def generate_with_kv(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
        """
        # ============================================
        #  Transformer 解码器-only 推理（单头化，便于看形状）
        #  - Prefill: 一次性送入上下文 [B,S]，构建所有层的 KV cache
        #  - Decode: 逐 token 生成；每步用到历史 KV cache + 本步 k/v
        #  - 采用 Pre-LN + 简化两层 FFN（非门控），忽略 bias、多头及 RoPE 细节
        # ============================================
        （注：上面这段注释来自你早期的伪代码，这里保留不改；本实现是**多头**的 PyTorch 版本）
        """
        self.eval()
        B, S = idx.shape
        assert S <= self.block_size

        # ---- Prefill：把 prompt 全长过一遍并**建立每层 KV cache** ----
        # 每层一个 dict cache（键名与你代码一致：'k','v'）
        layer_caches = [dict() for _ in range(len(self.blocks))]
        pos = torch.arange(0, S, device=idx.device).unsqueeze(0)           # [1,S]
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for l, blk in enumerate(self.blocks):
            x = blk(x, caches=layer_caches[l])                             # 写入该层 KV
        x = self.ln_f(x)
        logits_last = self.head(x)[:, -1, :]                               # [B,V]

        # ---- 第一步采样（与伪代码中 p_last 相同）----
        logits_last = logits_last / max(1e-6, temperature)
        if top_k:
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[:, -1:]] = -float('Inf')
        probs = F.softmax(logits_last, dim=-1)
        t_next = torch.multinomial(probs, num_samples=1)                   # [B,1]

        generated = [idx]
        cur_len = S

        # ---- Decode：逐步生成 ----
        for _ in range(max_new_tokens):
            # 单步位置 index：cur_len
            pos_i = torch.full((B, 1), cur_len, device=idx.device, dtype=torch.long)  # [B,1]
            x_t = self.tok_emb(t_next) + self.pos_emb(pos_i)                           # [B,1,C]

            # 逐层用各自的 KV cache 推进（只计算最后一步）
            h_t = x_t
            for l, blk in enumerate(self.blocks):
                h_t = blk(h_t, caches=layer_caches[l])                                  # KV 追加

            h_t = self.ln_f(h_t)                                                       # [B,1,C]
            logits_t = self.head(h_t)[:, -1, :]                                         # [B,V]
            logits_t = logits_t / max(1e-6, temperature)
            if top_k:
                v, _ = torch.topk(logits_t, top_k)
                logits_t[logits_t < v[:, -1:]] = -float('Inf')
            probs_t = F.softmax(logits_t, dim=-1)
            t_next = torch.multinomial(probs_t, num_samples=1)                          # [B,1]

            generated.append(t_next)
            cur_len += 1
            if eos_id is not None and torch.all(t_next.squeeze(1) == eos_id):
                break

        out = torch.cat(generated, dim=1)  # [B, S + <=max_new_tokens]
        return out
