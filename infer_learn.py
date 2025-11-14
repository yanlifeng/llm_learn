# ============================================
#  Transformer 解码器-only 推理（单头化，便于看形状）
#  - Prefill: 一次性送入上下文 [B,S]，构建所有层的 KV cache
#  - Decode: 逐 token 生成；每步用到历史 KV cache + 本步 k/v
#  - 采用 Pre-LN + 简化两层 FFN（非门控），忽略 bias、多头及 RoPE 细节
# ============================================

import math
import numpy as np

# ---------- 工具函数 ----------

def layer_norm(x, eps=1e-5):
    """
    x: [B, T, H]   （T 为位置数：prefill 时 T=S；decode 时 T=1）
    这里写个最简 LN；真实实现会带可学习的 gamma/beta
    """
    mean = x.mean(axis=-1, keepdims=True)      # [B,T,1]
    var  = x.var(axis=-1, keepdims=True)       # [B,T,1]
    xhat = (x - mean) / np.sqrt(var + eps)     # [B,T,H]
    return xhat  # 简化：省略 gamma/beta

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)    # 数值稳定
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_mask_for_lastpos(L):
    """
    针对 decode 单步（查询长度=1）的因果 mask，允许看历史 0..L-1 和自己 L-1。
    返回形状 [1, 1, L]，被 broadcast 到 [B, 1, L]
    """
    # 单步时其实只需把未来（这里没有未来）设为 -inf；写成全0更直观
    return np.zeros((1, 1, L), dtype=np.float32)

def build_strict_lower_tri_mask(S):
    """
    Prefill 阶段的标准下三角掩码，形状 [1, S, S]，用于广播到 [B,S,S]
    允许位置 i 看 <= i 的所有位置
    """
    mask = np.triu(np.ones((S, S), dtype=np.float32), k=1)  # 上三角（i<j）为1
    mask = np.where(mask == 1, -1e9, 0.0)                   # 上三角置为 -inf
    return mask[np.newaxis, :, :]                           # [1,S,S]

def top_k_sampling(prob_row, k=50, temperature=1.0):
    """
    对单个样本的一行概率向量做 Top-k 采样。
    prob_row: [V]
    """
    if temperature != 1.0:
        # 温度缩放在softmax前做更常见，这里示意放在后也可（仅示意）
        logits = np.log(np.maximum(prob_row, 1e-20)) / temperature
        prob_row = softmax(logits, axis=-1)

    # 选出 top-k
    idx = np.argpartition(prob_row, -k)[-k:]
    sub = prob_row[idx]
    sub = sub / sub.sum()
    # 多项分布采样
    choice = np.random.choice(len(idx), p=sub)
    return idx[choice]


# ---------- 模型权重（单头化；真实实现是多头+分层对象） ----------

class Weights:
    def __init__(self, L, H, D_ff, V):
        self.L = L
        self.H = H
        self.D_ff = D_ff
        self.V = V

        # 词嵌入/输出矩阵（共享权重）
        self.E = np.random.randn(V, H).astype(np.float32) * 0.02  # [V,H]

        # 每层的投影与FFN权重
        self.W_Q = [np.random.randn(H, H).astype(np.float32) * 0.02 for _ in range(L)]
        self.W_K = [np.random.randn(H, H).astype(np.float32) * 0.02 for _ in range(L)]
        self.W_V = [np.random.randn(H, H).astype(np.float32) * 0.02 for _ in range(L)]
        self.W_O = [np.random.randn(H, H).astype(np.float32) * 0.02 for _ in range(L)]
        self.W_up   = [np.random.randn(H, D_ff).astype(np.float32) * 0.02 for _ in range(L)]
        self.W_down = [np.random.randn(D_ff, H).astype(np.float32) * 0.02 for _ in range(L)]
        # LN 的 gamma/beta 省略（不改变形状理解）

# ---------- 层前向（Prefill：T=S；Decode：T=1） ----------

def attn_block_prefill(x,   # [B,S,H]
                       WQ, WK, WV, WO, 
                       causal_mask):  # [1,S,S]
    """
    返回：
      y: [B,S,H]  （注意力子层输出，已过输出投影+残差）
      K, V: [B,S,H] （供缓存）
    """
    B, S, H = x.shape
    y_ln = layer_norm(x)                           # [B,S,H]
    Q = y_ln @ WQ                                  # [B,S,H]
    K = y_ln @ WK                                  # [B,S,H]
    V = y_ln @ WV                                  # [B,S,H]

    # 注意：这里的 Kᵀ 是指在序列维度上转置点积： [B,S,H] • [B,H,S] -> [B,S,S]
    scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(H)   # [B,S,S]
    scores = scores + causal_mask                        # broadcast: [B,S,S] + [1,S,S]
    A = softmax(scores, axis=-1)                         # [B,S,S]
    O = A @ V                                            # [B,S,H]
    O = O @ WO                                           # [B,S,H]

    y = x + O                                            # 残差
    return y, K, V

def ffn_block(x, W_up, W_down):
    """
    两层 FFN（非门控示意）；可替换为 SwiGLU（门控）版本
    """
    z = layer_norm(x)                  # [B,T,H]
    u = z @ W_up                       # [B,T,D_ff]
    g = np.maximum(0.0, u)             # ReLU；也可用 SiLU/GeLU
    f = g @ W_down                     # [B,T,H]
    return x + f                       # 残差

def attn_block_decode_step(x_t,           # [B,1,H]
                           WQ, WK, WV, WO,
                           K_cache, V_cache):  # [B,S_hist,H]
    """
    单步解码用的注意力子层。会用到历史 KV cache，并把本步的 k/v 也并入计算。
    返回：
      y_t: [B,1,H]（注意力子层输出，已过输出投影+残差）
      k_t, v_t: [B,1,H]（本步新生成，供追加进缓存）
    """
    B, _, H = x_t.shape
    S_hist = K_cache.shape[1]  # 历史长度

    y_ln = layer_norm(x_t)         # [B,1,H]
    q_t = y_ln @ WQ                # [B,1,H]
    k_t = y_ln @ WK                # [B,1,H]
    v_t = y_ln @ WV                # [B,1,H]

    # 与 历史 K_cache 以及当前 k_t 一起做注意力
    K_all = np.concatenate([K_cache, k_t], axis=1)  # [B,S_hist+1,H]
    V_all = np.concatenate([V_cache, v_t], axis=1)  # [B,S_hist+1,H]

    # q_t @ K_all^T -> [B,1,S_hist+1]
    scores = (q_t @ K_all.transpose(0, 2, 1)) / math.sqrt(H)  # [B,1,S_hist+1]
    scores = scores + causal_mask_for_lastpos(S_hist + 1)     # [1,1,S_hist+1] 广播到 [B,1,S_hist+1]
    alpha = softmax(scores, axis=-1)                          # [B,1,S_hist+1]
    o_t = alpha @ V_all                                       # [B,1,H]
    o_t = o_t @ WO                                            # [B,1,H]

    y_t = x_t + o_t                                           # 残差
    return y_t, k_t, v_t

# ---------- 主流程：prefill + decode ----------

def transformer_infer(token_ids,   # [B,S] int
                      weights: Weights,
                      max_new_tokens=64,
                      eos_id=None,
                      top_k=50,
                      temperature=1.0):
    """
    返回：
      generated: [B, S + <=max_new_tokens]  追加的新 token
    假设：
      - 词嵌入与输出投影共享：logits = LN(h_last) @ E^T
      - 忽略位置编码细节（需要可加 P 或对 Q/K 做 RoPE）
    """
    B, S = token_ids.shape
    H, L, D_ff, V = weights.H, weights.L, weights.D_ff, weights.V

    # ---- 嵌入 ----
    # X: [B,S,H]
    X = weights.E[token_ids]   # 查表；真实实现要考虑 padding/masking

    # ---- Prefill：构建每层的 KV cache，并得到末位置的 logits ----
    # 每层的 KV cache：列表长度 L；每层是 [B,S,H]
    K_caches = [None] * L
    V_caches = [None] * L

    causal_mask = build_strict_lower_tri_mask(S)  # [1,S,S]

    h = X  # [B,S,H]
    for l in range(L):
        # 注意力子层（整个序列）——会产出 K/V，存入 cache
        h, K, V_ = attn_block_prefill(h, weights.W_Q[l], weights.W_K[l],
                                      weights.W_V[l], weights.W_O[l],
                                      causal_mask)     # h: [B,S,H], K/V: [B,S,H]
        K_caches[l] = K
        V_caches[l] = V_

        # FFN 子层
        h = ffn_block(h, weights.W_up[l], weights.W_down[l])  # [B,S,H]

    # 最终 LN + 输出 logits（prefill 的最后一位）
    h_final = layer_norm(h)                    # [B,S,H]
    logits_last = h_final[:, -1, :] @ weights.E.T   # [B,V]
    p_last = softmax(logits_last / temperature, axis=-1)  # [B,V]

    # ---- 从 p_last 采样得到 t_next（第一步 decode 的输入）----
    t_next = np.zeros((B,), dtype=np.int32)
    for b in range(B):
        t_next[b] = top_k_sampling(p_last[b], k=top_k, temperature=temperature)
    generated = [token_ids.copy()]  # 记录全序列（含原 S）

    # ---- Decode：逐步生成，循环“很多轮”，每轮产生 1 个 token ----
    # 当前序列长度（含上下文）
    cur_len = S

    for step in range(max_new_tokens):   # 每一轮 decode 只处理一个新 token
        # 1) 新 token 的嵌入（作为一个“长度为 1”的序列）
        x_t = weights.E[t_next][:, np.newaxis, :]    # [B,1,H]

        # 2) 逐层推进（注意：每层都要用自己的 KV cache）
        h_t = x_t  # [B,1,H]
        for l in range(L):
            # 注意力子层（单步）：使用历史 K/V cache + 本步 k/v
            h_t, k_t, v_t = attn_block_decode_step(
                h_t,
                weights.W_Q[l], weights.W_K[l], weights.W_V[l], weights.W_O[l],
                K_caches[l], V_caches[l]
            )  # h_t: [B,1,H], k_t/v_t: [B,1,H]

            # 更新该层的 cache（长度 +1）
            K_caches[l] = np.concatenate([K_caches[l], k_t], axis=1)   # [B,cur_len+1,H]
            V_caches[l] = np.concatenate([V_caches[l], v_t], axis=1)   # [B,cur_len+1,H]

            # FFN 子层（单步）
            h_t = ffn_block(h_t, weights.W_up[l], weights.W_down[l])    # [B,1,H]

        # 3) 输出层：得到 [B,1,V] 的分布，并采样下一 token
        logits_t = layer_norm(h_t) @ weights.E.T    # [B,1,V]
        p_t = softmax(logits_t / temperature, axis=-1)  # [B,1,V]
        p_row = p_t[:, 0, :]                        # [B,V]  —— squeeze 掉长度为1的维

        t_next = np.zeros((B,), dtype=np.int32)
        for b in range(B):
            t_next[b] = top_k_sampling(p_row[b], k=top_k, temperature=temperature)

        # 4) 追加到已生成序列
        new_tokens = t_next[:, np.newaxis]          # [B,1]
        generated.append(new_tokens)
        cur_len += 1

        # 5) 提前终止（如果全体都预测到 EOS）
        if eos_id is not None:
            if np.all(t_next == eos_id):
                break

    # 拼接出最终序列（原 S + 新生成）
    out = np.concatenate(generated, axis=1)   # [B, S + <=max_new_tokens]
    return out

