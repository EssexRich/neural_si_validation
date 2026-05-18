"""
Neural Structural Intelligence — Proof of Concept
Grokking detection via weight graph spectral topology analysis.

Copyright © 2025-2026 Richard Benfield. All rights reserved.
"""

import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Hyperparameters ────────────────────────────────────────────────────────────
P = 97                    # modulus
HIDDEN = 128
TRAIN_FRAC = 0.5
LR = 1e-3
WEIGHT_DECAY = 1.0
TOTAL_STEPS = 50_000
BATCH_SIZE = 512
WINDOW = 200              # rolling window for AR1 / VR
PRINT_EVERY = 1_000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ────────────────────────────────────────────────────────────────────

def make_dataset(p):
    """All p² pairs for (a+b) mod p, one-hot encoded inputs."""
    pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_FRAC)
    train_pairs, test_pairs = pairs[:split], pairs[split:]

    def encode(pairs_list):
        xs, ys = [], []
        for a, b in pairs_list:
            x = np.zeros(2 * p, dtype=np.float32)
            x[a] = 1.0
            x[p + b] = 1.0
            xs.append(x)
            ys.append((a + b) % p)
        return torch.tensor(np.array(xs)), torch.tensor(np.array(ys), dtype=torch.long)

    return encode(train_pairs), encode(test_pairs)


# ── Model ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ── Weight graph construction & spectral analysis ──────────────────────────────

def build_laplacian(model):
    """
    Build the full-network bipartite weight graph Laplacian.
    Nodes: all neurons across layers.
    Edges: |w_ij| between connected neurons.
    L = D - A.
    """
    W1 = model.fc1.weight.detach().cpu().numpy()   # (hidden, input)
    W2 = model.fc2.weight.detach().cpu().numpy()   # (output, hidden)

    n_in = W1.shape[1]
    n_h  = W1.shape[0]
    n_out = W2.shape[0]
    n = n_in + n_h + n_out

    rows, cols, data = [], [], []

    # Layer 1: input nodes [0..n_in) <-> hidden nodes [n_in..n_in+n_h)
    hi, hj = np.where(W1 != 0)
    for h_idx, i_idx in zip(hi, hj):
        u = i_idx
        v = n_in + h_idx
        w = abs(W1[h_idx, i_idx])
        rows += [u, v]; cols += [v, u]; data += [w, w]

    # Layer 2: hidden nodes [n_in..n_in+n_h) <-> output nodes [n_in+n_h..n)
    oi, hj2 = np.where(W2 != 0)
    for o_idx, h_idx in zip(oi, hj2):
        u = n_in + h_idx
        v = n_in + n_h + o_idx
        w = abs(W2[o_idx, h_idx])
        rows += [u, v]; cols += [v, u]; data += [w, w]

    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    d = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix((d, (np.arange(n), np.arange(n))), shape=(n, n))
    L = D - A
    return L


def fiedler_value(L):
    """Second-smallest eigenvalue of L (algebraic connectivity)."""
    n = L.shape[0]
    k = min(3, n - 1)
    try:
        vals = eigsh(L, k=k, which='SM', return_eigenvectors=False, tol=1e-6, maxiter=3000)
        vals = np.sort(np.real(vals))
        return float(vals[1]) if len(vals) > 1 else 0.0
    except Exception:
        return 0.0


def ar1(series):
    """Lag-1 Pearson autocorrelation of a 1-D array."""
    if len(series) < 4:
        return 0.0
    s = np.array(series, dtype=float)
    s -= s.mean()
    if s.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def variance_ratio(series):
    """Var(recent 20%) / Var(earliest 80%)."""
    s = np.array(series, dtype=float)
    n = len(s)
    if n < 5:
        return 1.0
    cut = max(1, int(n * 0.8))
    early = s[:cut]
    recent = s[cut:]
    v_early  = np.var(early)  if len(early)  > 1 else 1e-12
    v_recent = np.var(recent) if len(recent) > 1 else 1e-12
    if v_early < 1e-12:
        return 1.0
    return float(v_recent / v_early)


def shannon_entropy(update_magnitudes):
    """Shannon entropy of weight update magnitude distribution."""
    mags = np.abs(update_magnitudes).flatten()
    if mags.sum() < 1e-12:
        return 0.0
    # bin into 64 bins
    hist, _ = np.histogram(mags, bins=64, density=False)
    hist = hist.astype(float)
    hist += 1e-12
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def order_parameter(model, prev_params):
    """
    σ₁²/Σσ² computed per layer, then averaged.
    Uses the per-layer weight change matrix W(t) - W(t-1) for each Linear layer.
    A 1D flattened vector always gives order_p=1.0 by construction — wrong.
    Per-layer 2D matrices give a meaningful singular value spectrum.
    """
    results = []
    for (name, param), prev in zip(model.named_parameters(), prev_params):
        if param.dim() < 2:
            continue
        delta = param.detach().cpu().numpy() - prev
        if delta.shape[0] < 2 or delta.shape[1] < 2:
            continue
        try:
            sv = np.linalg.svd(delta, compute_uv=False)
            sv_sq = sv ** 2
            total = sv_sq.sum()
            if total < 1e-12:
                continue
            results.append(float(sv_sq[0] / total))
        except Exception:
            continue
    return float(np.mean(results)) if results else 0.0


# ── Training loop ──────────────────────────────────────────────────────────────

def train():
    os.makedirs('results', exist_ok=True)

    (X_train, y_train), (X_test, y_test) = make_dataset(P)
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test,  y_test  = X_test.to(DEVICE),  y_test.to(DEVICE)

    model = MLP(2 * P, HIDDEN, P).to(DEVICE)
    optimiser = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # snapshot weights for delta calculation
    prev_weights = np.concatenate([
        p.detach().cpu().numpy().flatten() for p in model.parameters()
    ])
    prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]

    # rolling λ₂ history for AR1/VR
    lambda2_history = []

    # records
    records = []  # list of dicts, one per step

    n_train = X_train.shape[0]

    print(f"Training on {DEVICE}. {n_train} train pairs, {X_test.shape[0]} test pairs.")
    print(f"Steps: {TOTAL_STEPS}, window: {WINDOW}, batch: {BATCH_SIZE}\n")

    for step in range(TOTAL_STEPS):
        model.train()

        # random batch
        idx = torch.randint(0, n_train, (BATCH_SIZE,))
        xb, yb = X_train[idx], y_train[idx]

        optimiser.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimiser.step()

        train_loss = loss.item()

        # ── structural metrics ─────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            # test accuracy
            test_logits = model(X_test)
            test_acc = (test_logits.argmax(dim=1) == y_test).float().mean().item()

            # weight delta (flat, for entropy; per-layer for order_p)
            curr_weights = np.concatenate([
                p.detach().cpu().numpy().flatten() for p in model.parameters()
            ])
            delta_W = curr_weights - prev_weights
            prev_weights = curr_weights.copy()

            # Fiedler
            L = build_laplacian(model)
            lam2 = fiedler_value(L)
            lambda2_history.append(lam2)

            # rolling window metrics
            window_slice = lambda2_history[-WINDOW:] if len(lambda2_history) >= WINDOW else lambda2_history[:]
            ar1_val = ar1(window_slice)
            vr_val  = variance_ratio(window_slice)

            # entropy of update magnitudes
            entropy_val = shannon_entropy(delta_W)

            # order parameter (per-layer SVD)
            order_p_val = order_parameter(model, prev_layer_params)
            prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]

        records.append({
            'step':      step,
            'train_loss': train_loss,
            'test_acc':   test_acc,
            'lambda2':    lam2,
            'ar1':        ar1_val,
            'vr':         vr_val,
            'entropy':    entropy_val,
            'order_p':    order_p_val,
        })

        if (step + 1) % PRINT_EVERY == 0:
            print(f"Step {step+1:6d} | loss={train_loss:.4f} | acc={test_acc:.3f} | "
                  f"λ₂={lam2:.4f} | AR1={ar1_val:.3f} | VR={vr_val:.3f} | "
                  f"S={entropy_val:.3f} | order_p={order_p_val:.3f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = 'results/grokking_spectral.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nData saved to {csv_path}")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    torch.save({
        'step': TOTAL_STEPS - 1,
        'model_state_dict': model.state_dict(),
        'lambda2': records[-1]['lambda2'],
        'test_acc': records[-1]['test_acc'],
    }, 'results/checkpoint_grokking_final.pt')
    print("Checkpoint saved: results/checkpoint_grokking_final.pt")

    # ── Plot ───────────────────────────────────────────────────────────────────
    steps     = [r['step']       for r in records]
    losses    = [r['train_loss'] for r in records]
    accs      = [r['test_acc']   for r in records]
    lam2s     = [r['lambda2']    for r in records]
    ar1s      = [r['ar1']        for r in records]
    vrs       = [r['vr']         for r in records]
    entropies = [r['entropy']    for r in records]
    order_ps  = [r['order_p']    for r in records]

    # detect grok onset: first step where test_acc crosses 0.9
    grok_step = None
    for r in records:
        if r['test_acc'] >= 0.9:
            grok_step = r['step']
            break

    fig, ax1 = plt.subplots(figsize=(16, 7))

    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Loss / Accuracy', color='black')
    ax1.plot(steps, losses, color='#e74c3c', linewidth=1.2, label='Train loss', alpha=0.85)
    ax1.plot(steps, accs,   color='#2ecc71', linewidth=1.8, label='Test accuracy')
    ax1.tick_params(axis='y')
    ax1.set_ylim(-0.05, max(max(losses) * 1.05, 1.05))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Structural metrics', color='#2c3e50')
    ax2.plot(steps, lam2s,     color='#3498db', linewidth=1.2, label='λ₂ (Fiedler)',  alpha=0.9)
    ax2.plot(steps, ar1s,      color='#9b59b6', linewidth=1.5, label='AR1',            alpha=0.9)
    ax2.plot(steps, vrs,       color='#f39c12', linewidth=1.2, label='VR',             alpha=0.9)
    ax2.plot(steps, entropies, color='#1abc9c', linewidth=1.0, label='Entropy (S)',    alpha=0.75, linestyle='--')
    ax2.plot(steps, order_ps,  color='#e67e22', linewidth=1.0, label='order_p',        alpha=0.75, linestyle=':')
    ax2.tick_params(axis='y', labelcolor='#2c3e50')

    if grok_step is not None:
        ax1.axvline(x=grok_step, color='black', linestyle='--', linewidth=1.5,
                    label=f'Grok onset (step {grok_step})')
        ax1.text(grok_step + TOTAL_STEPS * 0.005, ax1.get_ylim()[1] * 0.95,
                 f'grok @ {grok_step}', fontsize=9, color='black')

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, ncol=2)

    plt.title('Neural Structural Intelligence — Grokking PoC\n'
              '(a + b) mod 97 | 2-layer MLP | Weight graph spectral metrics',
              fontsize=11)
    plt.tight_layout()

    plot_path = 'results/grokking_spectral.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # ── Success criterion check ────────────────────────────────────────────────
    if grok_step is not None:
        # find earliest step where AR1 or VR began a sustained climb
        # "began climbing" = first step where value > mean of first 100 steps by > 0.1
        base_ar1 = np.mean(ar1s[:100])
        base_vr  = np.mean(vrs[:100])

        ar1_climb_step = next(
            (r['step'] for r in records if r['ar1'] > base_ar1 + 0.1), None)
        vr_climb_step  = next(
            (r['step'] for r in records if r['vr']  > base_vr  + 0.2), None)

        print(f"\n── Success criterion check ──────────────────────────────")
        print(f"  Grok onset (test_acc ≥ 0.9): step {grok_step}")
        print(f"  AR1 first climb above baseline: step {ar1_climb_step}")
        print(f"  VR  first climb above baseline: step {vr_climb_step}")

        for label, climb_step in [('AR1', ar1_climb_step), ('VR', vr_climb_step)]:
            if climb_step is not None:
                lead = grok_step - climb_step
                result = "PASS" if lead >= 100 else "partial"
                print(f"  {label} lead time: {lead} steps — {result}")
    else:
        print("\nNetwork did not grok within training run (test_acc never reached 0.9).")
        print("Consider increasing TOTAL_STEPS or checking hyperparameters.")


if __name__ == '__main__':
    train()
