"""
bitbyte_lm/model.py

A tiny, byte-level (tokenizer-free) transformer that can run in three weight
modes so you can run the exact ablation discussed:

    "fp"      -- normal full-precision nn.Linear (baseline)
    "ternary" -- BitNet b1.58 style {-1, 0, +1} weights (quantization-aware
                 training via a straight-through estimator)
    "binary"  -- pure {-1, +1} weights, so you can empirically confirm/refute
                 the "zero matters" claim yourself instead of taking the
                 literature's word for it

Input representation: raw UTF-8 bytes. Vocab size is fixed at 256 -- there is
no tokenizer, no BPE, no vocabulary file. This is the "no tokens" design
discussed in chat.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BitLinear: the core of the experiment
# ---------------------------------------------------------------------------

def _round_clip(x, lo, hi):
    # Straight-through: forward uses the rounded/clipped value,
    # backward pretends this was the identity function.
    return (x.round().clamp(lo, hi) - x).detach() + x


class BitLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that quantizes its weight to
    ternary or binary values on every forward pass, but keeps a full
    precision "shadow" weight for gradient updates (quantization-aware
    training, same recipe BitNet b1.58 uses).
    """

    def __init__(self, in_features, out_features, bias=True, mode="ternary"):
        super().__init__()
        assert mode in ("fp", "ternary", "binary")
        self.mode = mode
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # LayerNorm before quantization -- standard BitNet trick, keeps the
        # activation scale sane so quantization doesn't blow up.
        self.norm = nn.LayerNorm(in_features)

    def quantize_weight(self, w):
        if self.mode == "fp":
            return w
        # scale = mean abs weight (per BitNet b1.58 paper's gamma)
        gamma = w.abs().mean().clamp(min=1e-5)
        if self.mode == "ternary":
            w_scaled = w / gamma
            w_q = _round_clip(w_scaled, -1, 1)
        else:  # binary
            w_q = torch.sign(w)
            w_q = w_q + (w_q == 0).float()  # avoid literal zeros
            w_q = (w_q - w).detach() + w  # straight-through
        return w_q * gamma

    def forward(self, x):
        x = self.norm(x)
        w_q = self.quantize_weight(self.weight)
        return F.linear(x, w_q, self.bias)


def make_linear(in_f, out_f, mode, bias=True):
    if mode == "fp":
        return nn.Linear(in_f, out_f, bias=bias)
    return BitLinear(in_f, out_f, bias=bias, mode=mode)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, block_size, mode):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = make_linear(d_model, 3 * d_model, mode)
        self.proj = make_linear(d_model, d_model, mode)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model, mode):
        super().__init__()
        self.fc1 = make_linear(d_model, 4 * d_model, mode)
        self.fc2 = make_linear(4 * d_model, d_model, mode)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoEMLP(nn.Module):
    """N complete MLP experts (each a full BitLinear/Linear pair, matching
    `mode`) combined via a learned per-token softmax gate. Used for the
    Branch-Train-MiX "mix" step: each expert starts from a DIFFERENT
    domain-specialized branch checkpoint (not shared/random init like a
    from-scratch MoE would), so the gate has real, already-differentiated
    behavior to route between from step one -- this is what avoids the
    gate-collapse-into-redundancy failure mode that a from-scratch dense
    MoE hits (verified empirically at toy scale before building this)."""
    def __init__(self, d_model, mode, n_experts):
        super().__init__()
        self.n_experts = n_experts
        self.experts = nn.ModuleList([MLP(d_model, mode) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)  # always fp32, tiny, not quantized

    def forward(self, x):
        gate = F.softmax(self.gate(x), dim=-1)                       # (B,T,n_experts)
        outs = torch.stack([e(x) for e in self.experts], dim=-1)      # (B,T,D,n_experts)
        return (outs * gate.unsqueeze(-2)).sum(-1)


class Block(nn.Module):
    def __init__(self, d_model, n_head, block_size, mode, n_experts=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, mode)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mode) if n_experts == 1 else MoEMLP(d_model, mode, n_experts)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model: byte-level, no tokenizer
# ---------------------------------------------------------------------------

class ByteBitGPT(nn.Module):
    VOCAB_SIZE = 256  # every possible byte value -- this IS the vocabulary

    def __init__(self, d_model=256, n_layer=12, n_head=8, block_size=512, mode="ternary",
                 n_experts=1):
        super().__init__()
        self.block_size = block_size
        self.mode = mode
        self.n_experts = n_experts
        self.tok_emb = nn.Embedding(self.VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_head, block_size, mode, n_experts=n_experts) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        # Keep the output head full precision -- standard practice, this is
        # a tiny fraction of params and quantizing it hurts more than it saves.
        self.head = nn.Linear(d_model, self.VOCAB_SIZE, bias=False)

        n_params = sum(p.numel() for p in self.parameters())
        experts_str = f" n_experts={n_experts}" if n_experts > 1 else ""
        print(f"[ByteBitGPT] mode={mode} layers={n_layer} d_model={d_model} "
              f"heads={n_head} params={n_params/1e6:.2f}M{experts_str}")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_bytes=200, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_bytes):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_byte], dim=1)
        self.train()
        return idx

    @staticmethod
    def decode(byte_tensor):
        """Turn generated byte ids back into text. No tokenizer needed --
        just interpret the ids as UTF-8 bytes."""
        b = bytes(byte_tensor.tolist())
        return b.decode("utf-8", errors="replace")
