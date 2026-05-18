"""
Neural Structural Intelligence — Step 5 Transformer: Preemptive Curriculum Design

MLP testbed finding: activation saturation (~96% overlap) made compatibility scoring
undiscriminating. Directional conflict score partially worked (add53 separated) but
single-head output competition dominated, masking representational conflict entirely.
Multi-head eliminated output competition so completely that forgetting disappeared
and there was nothing for the curriculum to affect.

This experiment sits in the sweet spot:
  - 1-layer transformer, embed_dim=64, 4 attention heads, context_len=32
  - Single shared output head (vocabulary prediction)
  - Three tasks across two structural families

  Task A: arithmetic next-token ("3 + 5 = 8") — base grok
  Task B: reversed arithmetic ("8 = 5 + 3") — same structural family, different surface
          Prediction: LOW disruption score, groks fast after A, no bridging needed
  Task C: logical syllogisms ("if A then B, A, therefore B") — different structural family
          Prediction: HIGH disruption score, forgetting without bridging, preserved with bridging

The transformer's attention mechanism distributes representations across heads and
positions — activation overlap between arithmetic and logic tasks should fall well
below the MLP's 96% saturation. Vocabulary separation (arithmetic digits/operators
vs logical connectives) reduces output competition vs three mod-97 tasks. But single
output head means some competition remains — enough to produce measurable forgetting
that the curriculum can affect.

Passing criteria:
  1. Compatibility score ranks B < C (reversed arithmetic safer than logic)
  2. Bridging curriculum for Task C preserves Task A accuracy measurably better
     than direct Task C introduction (target: >80% bridged vs <60% direct)

Copyright © 2025-2026 Richard Benfield. All rights reserved.
"""

import os, csv, copy, math, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

torch.manual_seed(42)
np.random.seed(42)

# ── Architecture ───────────────────────────────────────────────────────────────
EMBED_DIM    = 64
N_HEADS      = 4
CTX_LEN      = 32     # max sequence length (padded)
FF_DIM       = 128    # feedforward hidden dim
DROPOUT      = 0.0    # no dropout — we want clean topology signal

# ── Training ───────────────────────────────────────────────────────────────────
LR           = 1e-3
WD           = 0.1
LR_HOSTILE   = 2e-3   # for Task C introduction — creates forgetting pressure
WD_HOSTILE   = 0.3
BATCH_SIZE   = 128
TASK_A_STEPS = 20000
TASK_B_STEPS = 15000
TASK_C_STEPS = 15000
BRIDGE_STEPS = 5000
PRINT_EVERY  = 2000

# ── Performance ────────────────────────────────────────────────────────────────
LAM2_EVERY   = 500
EVAL_EVERY   = 50

# ── Fiedler sensitivity ────────────────────────────────────────────────────────
SENSITIVITY_TOP_K   = 300
SENSITIVITY_EPSILON = 1e-3

# ── Bridging ───────────────────────────────────────────────────────────────────
BRIDGE_BLEND_START = 0.8   # 80% new task at start of bridge phase
BRIDGE_BLEND_END   = 0.2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Vocabulary ─────────────────────────────────────────────────────────────────
# Arithmetic tokens
DIGITS    = [str(i) for i in range(20)]   # 0-19 for manageable combinatorics
ARITH_OPS = ['+', '-', '=']
# Logic tokens
LOGIC_ATOMS   = ['A', 'B', 'C', 'D']
LOGIC_CONNECT = ['if', 'then', 'all', 'are', 'is', 'therefore', ',']
PAD_TOK = '<PAD>'
BOS_TOK = '<BOS>'
EOS_TOK = '<EOS>'

ALL_TOKENS = [PAD_TOK, BOS_TOK, EOS_TOK] + DIGITS + ARITH_OPS + LOGIC_ATOMS + LOGIC_CONNECT
VOCAB_SIZE = len(ALL_TOKENS)
tok2id = {t: i for i, t in enumerate(ALL_TOKENS)}
id2tok = {i: t for t, i in tok2id.items()}
PAD_ID = tok2id[PAD_TOK]
BOS_ID = tok2id[BOS_TOK]
EOS_ID = tok2id[EOS_TOK]


def encode(tokens, max_len=CTX_LEN):
    ids = [BOS_ID] + [tok2id[t] for t in tokens] + [EOS_ID]
    ids = ids[:max_len]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


# ── Dataset generators ─────────────────────────────────────────────────────────

def make_arithmetic_dataset(n_train=2000, n_test=500, op='+'):
    """Task A/B: arithmetic expressions. op='+' or '-'."""
    examples = []
    for a in range(15):
        for b in range(15):
            if op == '+':
                c = a + b
                if c >= 20: continue
                tokens_fwd = [str(a), '+', str(b), '=', str(c)]
                examples.append(tokens_fwd)
            elif op == '-':
                c = a - b
                if c < 0: continue
                tokens_fwd = [str(a), '-', str(b), '=', str(c)]
                examples.append(tokens_fwd)
    # duplicate to get reasonable dataset size
    examples = examples * max(1, 500 // len(examples))
    np.random.shuffle(examples)

    def build(exs):
        xs, ys = [], []
        for toks in exs:
            ids = encode(toks)
            # predict next token at each position (language modelling)
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:],  dtype=torch.long)
            xs.append(x); ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    split = int(len(examples) * 0.8)
    return build(examples[:split]), build(examples[split:])


def make_reversed_arithmetic_dataset(n_train=2000, n_test=500):
    """Task B: reversed arithmetic "c = a + b" — same structural family, different surface."""
    examples = []
    for a in range(15):
        for b in range(15):
            c = a + b
            if c >= 20: continue
            tokens_rev = [str(c), '=', str(a), '+', str(b)]
            examples.append(tokens_rev)
    examples = examples * max(1, 500 // len(examples))
    np.random.shuffle(examples)

    def build(exs):
        xs, ys = [], []
        for toks in exs:
            ids = encode(toks)
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:],  dtype=torch.long)
            xs.append(x); ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    split = int(len(examples) * 0.8)
    return build(examples[:split]), build(examples[split:])


def make_logic_dataset():
    """
    Task C: modus ponens syllogisms.
    "if A then B , A , therefore B"
    Uses logic atoms A-D and connectives — structurally different from arithmetic.
    """
    atoms = LOGIC_ATOMS  # A, B, C, D
    examples = []
    for p in atoms:
        for q in atoms:
            if p == q: continue
            # modus ponens: "if p then q , p , therefore q"
            toks = ['if', p, 'then', q, ',', p, ',', 'therefore', q]
            examples.append(toks)
            # universal: "all p are q , p is p , therefore p is q"
            toks2 = ['all', p, 'are', q, ',', p, 'is', p, ',', 'therefore', p, 'is', q]
            examples.append(toks2)
    # duplicate for reasonable size
    examples = examples * max(1, 200 // len(examples))

    np.random.shuffle(examples)

    def build(exs):
        xs, ys = [], []
        for toks in exs:
            ids = encode(toks)
            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:],  dtype=torch.long)
            xs.append(x); ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    split = int(len(examples) * 0.8)
    return build(examples[:split]), build(examples[split:])


# ── Transformer model ──────────────────────────────────────────────────────────

class TransformerLM(nn.Module):
    """
    1-layer transformer language model.
    Single shared output projection head (vocabulary prediction).
    Causal (autoregressive) attention mask.
    """
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 n_heads=N_HEADS, ff_dim=FF_DIM, ctx_len=CTX_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(ctx_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True,
                                          dropout=DROPOUT)
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.ff1 = nn.Linear(embed_dim, ff_dim)
        self.ff2 = nn.Linear(ff_dim, embed_dim)
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.out = nn.Linear(embed_dim, vocab_size)
        self.ctx_len = ctx_len

        # causal mask — registered as buffer, not parameter
        mask = torch.triu(torch.ones(ctx_len, ctx_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        # causal self-attention
        attn_out, _ = self.attn(h, h, h, attn_mask=self.causal_mask[:T, :T],
                                key_padding_mask=(x == PAD_ID))
        h = self.attn_norm(h + attn_out)
        # feedforward
        ff_out = self.ff2(torch.relu(self.ff1(h)))
        h = self.ff_norm(h + ff_out)
        return self.out(h)   # (B, T, vocab_size)


# ── Spectral metrics ───────────────────────────────────────────────────────────

def build_laplacian_transformer(model):
    """
    Build weight graph Laplacian from transformer's key weight matrices.
    Uses: embed (token→embed), attn Q/K/V projections, ff1, ff2, out projection.
    Nodes: vocab positions, embed dimensions, ff hidden dims.
    """
    # Extract key weight matrices
    W_embed = model.embed.weight.detach().cpu().numpy()   # (vocab, embed)
    # attn in_proj_weight is [3*embed, embed] — Q, K, V stacked
    W_qkv = model.attn.in_proj_weight.detach().cpu().numpy()  # (3*E, E)
    E = EMBED_DIM
    W_q = W_qkv[:E, :]       # (E, E)
    W_k = W_qkv[E:2*E, :]
    W_v = W_qkv[2*E:, :]
    W_ff1 = model.ff1.weight.detach().cpu().numpy()   # (ff_dim, E)
    W_ff2 = model.ff2.weight.detach().cpu().numpy()   # (E, ff_dim)

    # Node layout: [0..E-1]=embed dims, [E..E+ff_dim-1]=ff hidden dims
    n_embed = E
    n_ff    = FF_DIM
    n       = n_embed + n_ff

    rows, cols, data = [], [], []

    def add_edges(W, src_offset, dst_offset, src_size, dst_size):
        nonlocal rows, cols, data
        for i in range(min(W.shape[0], dst_size)):
            for j in range(min(W.shape[1], src_size)):
                w = abs(float(W[i, j]))
                if w == 0: continue
                u = src_offset + j
                v = dst_offset + i
                if u >= n or v >= n: continue
                rows += [u, v]; cols += [v, u]; data += [w, w]

    # embed → embed (via Q, K, V — within-embed connections)
    add_edges(W_q, 0, 0, n_embed, n_embed)
    add_edges(W_k, 0, 0, n_embed, n_embed)
    add_edges(W_v, 0, 0, n_embed, n_embed)
    # embed → ff hidden
    add_edges(W_ff1, 0, n_embed, n_embed, n_ff)
    # ff hidden → embed
    add_edges(W_ff2, n_embed, 0, n_ff, n_embed)

    if not data:
        return None

    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    d = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix((d, (np.arange(n), np.arange(n))), shape=(n, n))
    return D - A


def fiedler_value(L):
    if L is None: return 0.0
    try:
        vals = eigsh(L, k=3, which='SM', return_eigenvectors=False, tol=1e-6, maxiter=5000)
        vals = np.sort(np.real(vals))
        return float(vals[1]) if len(vals) > 1 else 0.0
    except Exception:
        return 0.0


EQ_ID = tok2id['=']

def eval_acc(model, X_te, y_te):
    """
    Answer-token accuracy: only score the token immediately after '=' in the input.
    This is the result digit — the meaningful signal, not padding or operator tokens.
    Falls back to all non-PAD tokens if no '=' found.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_te)      # (B, T, V)
        preds = logits.argmax(-1) # (B, T)
        # find position of '=' in input; prediction target is the next position
        eq_mask = (X_te == EQ_ID)                    # (B, T)
        # shift: if '=' is at position t in X, the answer is y at position t
        # (y[t] = X[t+1], so the answer token to predict is at the '=' position in y)
        has_eq = eq_mask.any(dim=1)
        if has_eq.all():
            correct = (preds[eq_mask] == y_te[eq_mask]).float()
            return correct.mean().item()
        # fallback
        mask = (y_te != PAD_ID)
        return (preds[mask] == y_te[mask]).float().mean().item()


# ── Fiedler sensitivity ────────────────────────────────────────────────────────

def compute_fiedler_sensitivity(model, top_k=SENSITIVITY_TOP_K, eps=SENSITIVITY_EPSILON):
    """∂λ₂/∂|w_ij| for top-K weights by magnitude across transformer weight matrices."""
    L_base = build_laplacian_transformer(model)
    base_lam2 = fiedler_value(L_base)

    # Collect all named scalar weights with their current signed values
    edges = []
    target_params = ['attn.in_proj_weight', 'ff1.weight', 'ff2.weight']
    for name, param in model.named_parameters():
        if not any(name == t for t in target_params): continue
        w = param.detach().cpu().numpy()
        for idx in np.ndindex(w.shape):
            edges.append((name, idx, abs(w[idx]), w[idx]))

    edges.sort(key=lambda x: -x[2])
    edges = edges[:top_k]

    sensitivities = {}
    temp_model = copy.deepcopy(model)

    for name, idx, w_abs, w_signed in edges:
        param = dict(temp_model.named_parameters())[name]
        with torch.no_grad():
            orig = param[idx].item()
            param[idx] = orig + eps

        L_pert = build_laplacian_transformer(temp_model)
        lam2_pert = fiedler_value(L_pert)
        sens = abs(lam2_pert - base_lam2) / eps
        sensitivities[(name, idx)] = {'sens': sens, 'w_sign': np.sign(w_signed)}

        with torch.no_grad():
            param[idx] = orig

    return sensitivities, base_lam2


# ── Directional compatibility score ───────────────────────────────────────────

def structural_compatibility_score(model, X_task, y_task, sensitivities, n_samples=128):
    """Sign-conflict stress weighted by Fiedler sensitivity."""
    model_copy = copy.deepcopy(model)
    opt_tmp = optim.SGD(model_copy.parameters(), lr=1.0)
    opt_tmp.zero_grad()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    idx = torch.randperm(X_task.shape[0])[:n_samples]
    xn, yn = X_task[idx], y_task[idx]

    logits = model_copy(xn)   # (B, T, V)
    B, T, V = logits.shape
    loss = criterion(logits.reshape(B*T, V), yn.reshape(B*T))
    loss.backward()

    conflict_scores = []
    target_params_copy = dict(model_copy.named_parameters())
    with torch.no_grad():
        for (name, idx_w), info in sensitivities.items():
            sens = info['sens']
            w_sign = info['w_sign']
            param = target_params_copy[name]
            g = param.grad[idx_w].item() if param.grad is not None else 0.0
            update_sign = -np.sign(g) if g != 0 else 0
            conflict = 1.0 if (w_sign != 0 and update_sign != 0 and update_sign != w_sign) else 0.0
            conflict_scores.append(abs(g) * sens * conflict)

    score = float(np.mean(conflict_scores)) if conflict_scores else 0.0
    conflict_frac = float(np.sum([s > 0 for s in conflict_scores]) / len(conflict_scores)) \
                    if conflict_scores else 0.0
    return {'score': score, 'conflict_fraction': conflict_frac}


# ── Training ───────────────────────────────────────────────────────────────────

def train_phase(model, X_tr, y_tr, X_te, y_te, label,
                X_protect_tr=None, y_protect_tr=None,
                X_protect_te=None, y_protect_te=None,
                n_steps=TASK_A_STEPS, replay_frac=0.0,
                lr_override=None, wd_override=None):
    """
    Train for n_steps. If protect data provided, mix in replay_frac of it per batch.
    Returns records list.
    """
    _lr = lr_override if lr_override is not None else LR
    _wd = wd_override if wd_override is not None else WD
    opt = optim.AdamW(model.parameters(), lr=_lr, weight_decay=_wd)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    records = []
    lam2 = 0.0; acc = 0.0; acc_protect = 0.0
    n_tr = X_tr.shape[0]

    print(f"\n{'='*60}")
    print(f"{label} | steps={n_steps} | replay={replay_frac:.0%}")
    print(f"{'='*60}")

    for step in range(n_steps):
        n_replay = max(0, int(BATCH_SIZE * replay_frac)) if X_protect_tr is not None else 0
        n_new = BATCH_SIZE - n_replay

        idx = torch.randint(0, n_tr, (n_new,))
        xb, yb = X_tr[idx], y_tr[idx]

        model.train()
        opt.zero_grad()
        B, T, V = *xb.shape, VOCAB_SIZE
        logits = model(xb)
        loss = criterion(logits.reshape(B*T, V), yb.reshape(B*T))

        if n_replay > 0:
            idx_r = torch.randint(0, X_protect_tr.shape[0], (n_replay,))
            logits_r = model(X_protect_tr[idx_r])
            Br = n_replay
            loss = loss + criterion(logits_r.reshape(Br*T, V),
                                    y_protect_tr[idx_r].reshape(Br*T))

        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc = eval_acc(model, X_te, y_te)
                if X_protect_te is not None:
                    acc_protect = eval_acc(model, X_protect_te, y_protect_te)

        if step % LAM2_EVERY == 0:
            L = build_laplacian_transformer(model)
            lam2 = fiedler_value(L)

        records.append({'step': step, 'phase': label, 'loss': loss.item(),
                        'lambda2': lam2, 'acc': acc, 'acc_protect': acc_protect})

        if (step + 1) % PRINT_EVERY == 0:
            protect_str = f" | protect={acc_protect:.3f}" if X_protect_te is not None else ""
            print(f"  step {step+1:5d} | loss={loss.item():.4f} | "
                  f"acc={acc:.3f}{protect_str} | λ₂={lam2:.4f}")

    return records


def train_bridged(model, X_new_tr, y_new_tr, X_new_te, y_new_te,
                  X_base_tr, y_base_tr, X_base_te, y_base_te, label):
    """
    Bridge phase: linearly blend base→new over BRIDGE_STEPS using gentle LR,
    then full new-task training with hostile LR (same pressure as direct condition).
    """
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    records = []
    lam2 = 0.0; acc_new = 0.0; acc_base = 0.0
    n_new = X_new_tr.shape[0]; n_base = X_base_tr.shape[0]

    print(f"\n{'='*60}")
    print(f"{label} BRIDGE | steps={BRIDGE_STEPS}")
    print(f"{'='*60}")

    for step in range(BRIDGE_STEPS):
        blend = BRIDGE_BLEND_START + (BRIDGE_BLEND_END - BRIDGE_BLEND_START) * (step / BRIDGE_STEPS)
        n_n = max(1, int(BATCH_SIZE * blend))
        n_b = max(1, BATCH_SIZE - n_n)

        idx_n = torch.randint(0, n_new, (n_n,))
        idx_b = torch.randint(0, n_base, (n_b,))
        xb = torch.cat([X_new_tr[idx_n], X_base_tr[idx_b]])
        yb = torch.cat([y_new_tr[idx_n], y_base_tr[idx_b]])

        model.train()
        opt.zero_grad()
        B, T = xb.shape
        logits = model(xb)
        loss = criterion(logits.reshape(B*T, VOCAB_SIZE), yb.reshape(B*T))
        loss.backward(); opt.step()

        if step % EVAL_EVERY == 0:
            model.eval()
            with torch.no_grad():
                acc_new  = eval_acc(model, X_new_te, y_new_te)
                acc_base = eval_acc(model, X_base_te, y_base_te)

        if step % LAM2_EVERY == 0:
            L = build_laplacian_transformer(model)
            lam2 = fiedler_value(L)

        records.append({'step': step, 'phase': f'{label}_bridge',
                        'loss': loss.item(), 'lambda2': lam2,
                        'acc': acc_new, 'acc_protect': acc_base})

        if (step + 1) % PRINT_EVERY == 0:
            print(f"  step {step+1:5d} | loss={loss.item():.4f} | "
                  f"new={acc_new:.3f} | base={acc_base:.3f} | "
                  f"λ₂={lam2:.4f} | blend={blend:.0%} new")

    # then full training with hostile LR + 10% replay (bridging advantage)
    task_records = train_phase(
        model, X_new_tr, y_new_tr, X_new_te, y_new_te,
        f"{label}_task",
        X_protect_tr=X_base_tr, y_protect_tr=y_base_tr,
        X_protect_te=X_base_te, y_protect_te=y_base_te,
        n_steps=TASK_C_STEPS, replay_frac=0.1,
        lr_override=LR_HOSTILE, wd_override=WD_HOSTILE)

    for r in task_records:
        r['step'] += BRIDGE_STEPS

    return records + task_records


# ── Main ────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('results', exist_ok=True)

    print(f"Vocabulary size: {VOCAB_SIZE}")
    print(f"Tokens: {ALL_TOKENS}")
    print(f"Device: {DEVICE}")

    # Build datasets
    (X_add_tr, y_add_tr), (X_add_te, y_add_te) = make_arithmetic_dataset(op='+')
    (X_sub_tr, y_sub_tr), (X_sub_te, y_sub_te) = make_arithmetic_dataset(op='-')
    (X_rev_tr, y_rev_tr), (X_rev_te, y_rev_te) = make_reversed_arithmetic_dataset()
    (X_log_tr, y_log_tr), (X_log_te, y_log_te) = make_logic_dataset()

    for t in [X_add_tr, y_add_tr, X_add_te, y_add_te,
              X_sub_tr, y_sub_tr, X_sub_te, y_sub_te,
              X_rev_tr, y_rev_tr, X_rev_te, y_rev_te,
              X_log_tr, y_log_tr, X_log_te, y_log_te]:
        t.data = t.to(DEVICE).data

    print(f"\nDataset sizes:")
    print(f"  Task A (addition):          train={X_add_tr.shape[0]}, test={X_add_te.shape[0]}")
    print(f"  Task B (reversed addition): train={X_rev_tr.shape[0]}, test={X_rev_te.shape[0]}")
    print(f"  Task C (logic):             train={X_log_tr.shape[0]}, test={X_log_te.shape[0]}")

    # ── Phase A: train on addition until grokked ───────────────────────────────
    print(f"\n── Phase A: train on addition ───────────────────────────────")
    model_A = TransformerLM().to(DEVICE)
    param_count = sum(p.numel() for p in model_A.parameters())
    print(f"Model parameters: {param_count:,}")

    phase_A_records = train_phase(
        model_A, X_add_tr, y_add_tr, X_add_te, y_add_te,
        'phase_A_addition', n_steps=TASK_A_STEPS)

    acc_A_final = phase_A_records[-1]['acc']
    lam2_A_final = phase_A_records[-1]['lambda2']
    print(f"\n  Phase A final: acc={acc_A_final:.3f} | λ₂={lam2_A_final:.4f}")

    # save checkpoint
    torch.save({'model_state_dict': model_A.state_dict(),
                'step': TASK_A_STEPS, 'acc': acc_A_final},
               'results/transformer_ckpt_A.pt')

    # ── Fiedler sensitivities ──────────────────────────────────────────────────
    print(f"\n── Computing Fiedler sensitivities (top-{SENSITIVITY_TOP_K} weights) ──")
    sensitivities, base_lam2 = compute_fiedler_sensitivity(model_A)
    sens_vals = [v['sens'] for v in sensitivities.values()]
    print(f"  Base λ₂: {base_lam2:.4f}")
    print(f"  Sensitivity range: {min(sens_vals):.6f} – {max(sens_vals):.6f}")
    print(f"  Mean: {np.mean(sens_vals):.6f} | Median: {np.median(sens_vals):.6f}")

    # ── Compatibility scores ───────────────────────────────────────────────────
    print(f"\n── Directional compatibility scores ─────────────────────────")
    candidates = [
        ('reversed_arith (B)', X_rev_tr, y_rev_tr, 'LOW disruption predicted'),
        ('logic (C)',          X_log_tr, y_log_tr, 'HIGH disruption predicted'),
    ]
    scores = {}
    for label, X_task, y_task, prediction in candidates:
        result = structural_compatibility_score(model_A, X_task, y_task, sensitivities)
        scores[label] = result
        print(f"  {label:25s}: score={result['score']:.6f} | "
              f"conflict_frac={result['conflict_fraction']:.3f} | {prediction}")

    score_B = scores['reversed_arith (B)']['score']
    score_C = scores['logic (C)']['score']
    ranking_correct = score_C > score_B
    print(f"\n  Ranking correct (logic > reversed_arith): {'YES ✓' if ranking_correct else 'NO ✗'}")
    print(f"  Score ratio C/B: {score_C / (score_B + 1e-12):.2f}×")

    # ── Condition 1: Direct Task C introduction (no bridging) ─────────────────
    print(f"\n── Condition 1: Direct logic introduction (no bridging) ─────")
    model_direct = TransformerLM().to(DEVICE)
    sd = torch.load('results/transformer_ckpt_A.pt', map_location=DEVICE)['model_state_dict']
    model_direct.load_state_dict(sd)

    direct_records = train_phase(
        model_direct, X_log_tr, y_log_tr, X_log_te, y_log_te,
        'direct_logic',
        X_protect_tr=X_add_tr, y_protect_tr=y_add_tr,
        X_protect_te=X_add_te, y_protect_te=y_add_te,
        n_steps=TASK_C_STEPS, replay_frac=0.0,
        lr_override=LR_HOSTILE, wd_override=WD_HOSTILE)

    acc_logic_direct = direct_records[-1]['acc']
    acc_add_direct = direct_records[-1]['acc_protect']
    lam2_direct = [r['lambda2'] for r in direct_records if r['lambda2'] > 0]
    lam2_dip_direct = min(lam2_direct) if lam2_direct else 0.0
    steps_to_grok_direct = next((r['step'] for r in direct_records if r['acc'] >= 0.85), None)

    print(f"\n  Direct result: add={acc_add_direct:.3f} | logic={acc_logic_direct:.3f} | "
          f"λ₂_dip={lam2_dip_direct:.4f} | steps_to_grok={steps_to_grok_direct}")

    # ── Condition 2: Bridged Task C introduction ───────────────────────────────
    print(f"\n── Condition 2: Bridged logic introduction ───────────────────")
    model_bridged = TransformerLM().to(DEVICE)
    model_bridged.load_state_dict(sd)

    bridged_records = train_bridged(
        model_bridged, X_log_tr, y_log_tr, X_log_te, y_log_te,
        X_add_tr, y_add_tr, X_add_te, y_add_te, 'bridged_logic')

    acc_logic_bridged = bridged_records[-1]['acc']
    acc_add_bridged = bridged_records[-1]['acc_protect']
    lam2_bridged_vals = [r['lambda2'] for r in bridged_records if r['lambda2'] > 0]
    lam2_dip_bridged = min(lam2_bridged_vals) if lam2_bridged_vals else 0.0
    steps_to_grok_bridged = next((r['step'] for r in bridged_records if r['acc'] >= 0.85), None)

    print(f"\n  Bridged result: add={acc_add_bridged:.3f} | logic={acc_logic_bridged:.3f} | "
          f"λ₂_dip={lam2_dip_bridged:.4f} | steps_to_grok={steps_to_grok_bridged}")

    # ── Analysis ───────────────────────────────────────────────────────────────
    print(f"\n── Step 5 Transformer analysis ──────────────────────────────")
    print(f"  Base addition accuracy (Phase A): {acc_A_final:.3f}")
    print(f"  Direct:  add={acc_add_direct:.3f} | logic={acc_logic_direct:.3f} | "
          f"λ₂_dip={lam2_dip_direct:.4f} | steps_to_grok={steps_to_grok_direct}")
    print(f"  Bridged: add={acc_add_bridged:.3f} | logic={acc_logic_bridged:.3f} | "
          f"λ₂_dip={lam2_dip_bridged:.4f} | steps_to_grok={steps_to_grok_bridged}")

    bridging_preserves = acc_add_bridged > acc_add_direct
    bridging_reduces_dip = lam2_dip_bridged > lam2_dip_direct
    bridging_faster = (steps_to_grok_bridged is not None and
                       (steps_to_grok_direct is None or
                        steps_to_grok_bridged < steps_to_grok_direct))
    target_met = acc_add_bridged >= 0.80 and acc_add_direct < 0.60

    print(f"\n  Score ranking correct (C > B):              {'YES ✓' if ranking_correct else 'NO ✗'}")
    print(f"  Bridging preserves add accuracy better:     {'YES ✓' if bridging_preserves else 'NO ✗'}")
    print(f"  Bridging reduces λ₂ dip:                    {'YES ✓' if bridging_reduces_dip else 'NO ✗'}")
    print(f"  Bridging accelerates logic grokking:        {'YES ✓' if bridging_faster else 'NO ✗'}")
    print(f"  Target met (≥80% bridged, <60% direct):     {'YES ✓' if target_met else 'NO ✗'}")

    step5_validated = ranking_correct and bridging_preserves
    print(f"\n  Step 5 VALIDATED: {'YES ✓' if step5_validated else 'NOT YET ✗'}")

    # ── Save ───────────────────────────────────────────────────────────────────
    scores_rows = [{'task': k, 'score': v['score'],
                    'conflict_fraction': v['conflict_fraction']}
                   for k, v in scores.items()]
    with open('results/transformer_step5_scores.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=scores_rows[0].keys())
        w.writeheader(); w.writerows(scores_rows)

    all_records = (
        [{**r, 'condition': 'phase_A'}  for r in phase_A_records] +
        [{**r, 'condition': 'direct'}   for r in direct_records] +
        [{**r, 'condition': 'bridged'}  for r in bridged_records])
    with open('results/transformer_step5_training.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_records[0].keys())
        w.writeheader(); w.writerows(all_records)

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(18, 16))

    ax = axes[0]
    task_labels = ['reversed_arith (B)', 'logic (C)']
    raw_scores  = [scores[l]['score'] for l in task_labels]
    cf          = [scores[l]['conflict_fraction'] for l in task_labels]
    max_s = max(raw_scores) + 1e-12
    norm_scores = [s / max_s for s in raw_scores]
    colours = ['#2ecc71', '#e74c3c']
    bars = ax.bar(task_labels, norm_scores, color=colours, alpha=0.8,
                  label='Directional conflict score (normalised)')
    ax.bar(task_labels, cf, color='#9b59b6', alpha=0.5, width=0.4,
           label='Conflict fraction')
    for bar, val in zip(bars, norm_scores):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01, f'{val:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1.3)
    ax.set_title('Neural Structural Intelligence — Step 5: Transformer Preemptive Curriculum\n'
                 'Directional conflict scores: should rank logic (C) > reversed arithmetic (B)',
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    d_steps = [r['step'] for r in direct_records]
    b_steps = [r['step'] for r in bridged_records]
    ax.plot(d_steps, [r['lambda2'] for r in direct_records],
            color='#e74c3c', linewidth=1.5, label='Direct')
    ax.plot(b_steps, [r['lambda2'] for r in bridged_records],
            color='#2ecc71', linewidth=1.5, label='Bridged')
    ax.axvline(BRIDGE_STEPS, color='grey', linestyle=':', linewidth=1, label='Bridge→task')
    ax.set_ylabel('λ₂ (Fiedler)'); ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('λ₂ during Task C (logic) introduction')

    ax = axes[2]
    ax.plot(d_steps, [r['acc_protect'] for r in direct_records],
            color='#e74c3c', linewidth=1.5, label='Task A acc (add) — direct')
    ax.plot(b_steps, [r['acc_protect'] for r in bridged_records],
            color='#2ecc71', linewidth=1.5, label='Task A acc (add) — bridged')
    ax.plot(d_steps, [r['acc'] for r in direct_records],
            color='#e74c3c', linewidth=1.0, linestyle='--', label='Task C acc (logic) — direct')
    ax.plot(b_steps, [r['acc'] for r in bridged_records],
            color='#2ecc71', linewidth=1.0, linestyle='--', label='Task C acc (logic) — bridged')
    ax.axhline(0.80, color='black', linestyle=':', linewidth=1, label='80% target (bridged)')
    ax.axhline(0.60, color='grey',  linestyle=':', linewidth=1, label='60% threshold (direct)')
    ax.axvline(BRIDGE_STEPS, color='grey', linestyle=':', linewidth=1)
    ax.set_ylabel('Token accuracy'); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Training step'); ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('results/transformer_step5.png', dpi=150)
    plt.close()
    print(f"\n  Saved: results/transformer_step5.png")
    print(f"         results/transformer_step5_scores.csv")
    print(f"         results/transformer_step5_training.csv")


if __name__ == '__main__':
    run()
