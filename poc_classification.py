"""
Neural Structural Intelligence — Step 2: Classification
Can structural metrics distinguish an upward break (grokking) from a
downward break (catastrophic forgetting)?

Protocol:
  Phase 1 — train on (a+b) mod 97 to grokking (~21k steps).
  Phase 2 — fine-tune on (a*b) mod 97 (different structure) with high LR,
             which causes rapid catastrophic forgetting on the original task.
  Record all five structural metrics throughout both phases.
  Compare signatures: rising λ₂ vs falling λ₂, entropy direction, VR spikes.

Copyright © 2025-2026 Richard Benfield. All rights reserved.
"""

import os, csv
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

# ── Config ─────────────────────────────────────────────────────────────────────
P           = 97
HIDDEN      = 128
TRAIN_FRAC  = 0.5
LR_PHASE1   = 1e-3
WD_PHASE1   = 1.0
LR_PHASE2   = 5e-3   # higher LR to force forgetting quickly
WD_PHASE2   = 0.0
PHASE1_STEPS = 25000  # enough to fully grok
PHASE2_STEPS = 10000  # enough to observe forgetting
BATCH_SIZE  = 512
WINDOW      = 200
PRINT_EVERY = 1000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ────────────────────────────────────────────────────────────────────

def make_dataset(p, task='add'):
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
            if task == 'add':
                ys.append((a + b) % p)
            elif task == 'mul':
                ys.append((a * b) % p)
        return torch.tensor(np.array(xs)), torch.tensor(np.array(ys), dtype=torch.long)

    return encode(train_pairs), encode(test_pairs)


# ── Model ──────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2 * P, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, P)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# ── Spectral metrics (same as PoC) ─────────────────────────────────────────────

def build_laplacian(model):
    W1 = model.fc1.weight.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy()
    n_in, n_h, n_out = W1.shape[1], W1.shape[0], W2.shape[0]
    n = n_in + n_h + n_out
    rows, cols, data = [], [], []
    hi, hj = np.where(W1 != 0)
    for h_idx, i_idx in zip(hi, hj):
        u, v, w = i_idx, n_in + h_idx, abs(W1[h_idx, i_idx])
        rows += [u, v]; cols += [v, u]; data += [w, w]
    oi, hj2 = np.where(W2 != 0)
    for o_idx, h_idx in zip(oi, hj2):
        u, v, w = n_in + h_idx, n_in + n_h + o_idx, abs(W2[o_idx, h_idx])
        rows += [u, v]; cols += [v, u]; data += [w, w]
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    d = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix((d, (np.arange(n), np.arange(n))), shape=(n, n))
    return D - A

def fiedler_value(L):
    n = L.shape[0]
    k = min(3, n - 1)
    try:
        vals = eigsh(L, k=k, which='SM', return_eigenvectors=False, tol=1e-6, maxiter=3000)
        vals = np.sort(np.real(vals))
        return float(vals[1]) if len(vals) > 1 else 0.0
    except Exception:
        return 0.0

def ar1(series):
    if len(series) < 4: return 0.0
    s = np.array(series, dtype=float); s -= s.mean()
    if s.std() < 1e-12: return 0.0
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])

def variance_ratio(series):
    s = np.array(series, dtype=float); n = len(s)
    if n < 5: return 1.0
    cut = max(1, int(n * 0.8))
    v_early  = np.var(s[:cut])  if cut > 1        else 1e-12
    v_recent = np.var(s[cut:])  if n - cut > 1    else 1e-12
    return float(v_recent / v_early) if v_early > 1e-12 else 1.0

def shannon_entropy(update_magnitudes):
    mags = np.abs(update_magnitudes).flatten()
    if mags.sum() < 1e-12: return 0.0
    hist, _ = np.histogram(mags, bins=64, density=False)
    hist = hist.astype(float) + 1e-12
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))

def order_parameter(model, prev_layer_params):
    results = []
    for (name, param), prev in zip(model.named_parameters(), prev_layer_params):
        if param.dim() < 2: continue
        delta = param.detach().cpu().numpy() - prev
        if delta.shape[0] < 2 or delta.shape[1] < 2: continue
        try:
            sv = np.linalg.svd(delta, compute_uv=False)
            sv_sq = sv ** 2; total = sv_sq.sum()
            if total < 1e-12: continue
            results.append(float(sv_sq[0] / total))
        except Exception:
            continue
    return float(np.mean(results)) if results else 0.0


# ── One training step with metric collection ───────────────────────────────────

def training_step(model, optimiser, criterion, X_train, y_train,
                  X_test_add, y_test_add, X_test_mul, y_test_mul,
                  lambda2_history, prev_weights, prev_layer_params, step, phase):

    model.train()
    n = X_train.shape[0]
    idx = torch.randint(0, n, (BATCH_SIZE,))
    xb, yb = X_train[idx], y_train[idx]
    optimiser.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimiser.step()

    model.eval()
    with torch.no_grad():
        acc_add = (model(X_test_add).argmax(1) == y_test_add).float().mean().item()
        acc_mul = (model(X_test_mul).argmax(1) == y_test_mul).float().mean().item()

        curr_weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
        delta_W = curr_weights - prev_weights

        L = build_laplacian(model)
        lam2 = fiedler_value(L)
        lambda2_history.append(lam2)
        window_slice = lambda2_history[-WINDOW:]

        ar1_val     = ar1(window_slice)
        vr_val      = variance_ratio(window_slice)
        entropy_val = shannon_entropy(delta_W)
        order_p_val = order_parameter(model, prev_layer_params)

        new_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]

    return {
        'step': step, 'phase': phase,
        'train_loss': loss.item(),
        'acc_add': acc_add, 'acc_mul': acc_mul,
        'lambda2': lam2, 'ar1': ar1_val, 'vr': vr_val,
        'entropy': entropy_val, 'order_p': order_p_val,
    }, curr_weights, new_layer_params


# ── Main ────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('results', exist_ok=True)

    (X_add_tr, y_add_tr), (X_add_te, y_add_te) = make_dataset(P, 'add')
    (X_mul_tr, y_mul_tr), (X_mul_te, y_mul_te) = make_dataset(P, 'mul')
    X_add_tr, y_add_tr = X_add_tr.to(DEVICE), y_add_tr.to(DEVICE)
    X_add_te, y_add_te = X_add_te.to(DEVICE), y_add_te.to(DEVICE)
    X_mul_tr, y_mul_tr = X_mul_tr.to(DEVICE), y_mul_tr.to(DEVICE)
    X_mul_te, y_mul_te = X_mul_te.to(DEVICE), y_mul_te.to(DEVICE)

    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    prev_weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    lambda2_history = []
    records = []

    # ── Phase 1: train on addition to grokking ─────────────────────────────────
    print(f"Phase 1: training on (a+b) mod {P} for {PHASE1_STEPS} steps")
    opt1 = optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=WD_PHASE1)

    for step in range(PHASE1_STEPS):
        rec, prev_weights, prev_layer_params = training_step(
            model, opt1, criterion,
            X_add_tr, y_add_tr, X_add_te, y_add_te, X_mul_te, y_mul_te,
            lambda2_history, prev_weights, prev_layer_params, step, phase=1)
        records.append(rec)
        if (step + 1) % PRINT_EVERY == 0:
            print(f"  P1 step {step+1:6d} | loss={rec['train_loss']:.4f} | "
                  f"acc_add={rec['acc_add']:.3f} | acc_mul={rec['acc_mul']:.3f} | "
                  f"λ₂={rec['lambda2']:.3f} | AR1={rec['ar1']:.3f} | "
                  f"VR={rec['vr']:.3f} | S={rec['entropy']:.3f} | op={rec['order_p']:.3f}")

    add_acc_at_phase1_end = records[-1]['acc_add']
    print(f"\nPhase 1 complete. Addition accuracy: {add_acc_at_phase1_end:.3f}")
    print(f"  λ₂ at end of phase 1: {records[-1]['lambda2']:.3f}")

    # reset lambda2 history so AR1/VR reflect phase 2 dynamics cleanly
    lambda2_history.clear()

    # ── Phase 2: fine-tune on multiplication (causes forgetting) ───────────────
    print(f"\nPhase 2: fine-tuning on (a*b) mod {P} for {PHASE2_STEPS} steps (forgetting expected)")
    opt2 = optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WD_PHASE2)

    for step in range(PHASE2_STEPS):
        global_step = PHASE1_STEPS + step
        rec, prev_weights, prev_layer_params = training_step(
            model, opt2, criterion,
            X_mul_tr, y_mul_tr, X_add_te, y_add_te, X_mul_te, y_mul_te,
            lambda2_history, prev_weights, prev_layer_params, global_step, phase=2)
        records.append(rec)
        if (step + 1) % PRINT_EVERY == 0:
            print(f"  P2 step {step+1:6d} | loss={rec['train_loss']:.4f} | "
                  f"acc_add={rec['acc_add']:.3f} | acc_mul={rec['acc_mul']:.3f} | "
                  f"λ₂={rec['lambda2']:.3f} | AR1={rec['ar1']:.3f} | "
                  f"VR={rec['vr']:.3f} | S={rec['entropy']:.3f} | op={rec['order_p']:.3f}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = 'results/classification.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nData saved to {csv_path}")

    # ── Save checkpoints ───────────────────────────────────────────────────────
    torch.save({
        'step': PHASE1_STEPS - 1,
        'phase': 'phase1_grokked',
        'model_state_dict': model.state_dict(),
        'lambda2': [r['lambda2'] for r in records if r['phase']==1][-1],
        'acc_add': [r['acc_add'] for r in records if r['phase']==1][-1],
    }, 'results/checkpoint_classification_phase1.pt')
    print("Checkpoint saved: results/checkpoint_classification_phase1.pt")

    # ── Analysis ───────────────────────────────────────────────────────────────
    p1 = [r for r in records if r['phase'] == 1]
    p2 = [r for r in records if r['phase'] == 2]

    # grok onset in phase 1
    grok_step = next((r['step'] for r in p1 if r['acc_add'] >= 0.9), None)

    # forgetting onset in phase 2: first step where acc_add drops below 0.7
    forget_step = next((r['step'] for r in p2 if r['acc_add'] < 0.7), None)

    # λ₂ direction in phase 1 (upward) vs phase 2 (expected: downward)
    lam2_p1_start = p1[0]['lambda2']
    lam2_p1_end   = p1[-1]['lambda2']
    lam2_p2_start = p2[0]['lambda2'] if p2 else None
    lam2_p2_end   = p2[-1]['lambda2'] if p2 else None

    # entropy trend: mean of last 20% of each phase
    def mean_entropy_last20(phase_records):
        n = len(phase_records)
        return np.mean([r['entropy'] for r in phase_records[int(n*0.8):]])

    print(f"\n── Classification analysis ─────────────────────────────────────")
    print(f"  Phase 1 (upward break — grokking):")
    print(f"    λ₂ {lam2_p1_start:.3f} → {lam2_p1_end:.3f}  (direction: {'UP ✓' if lam2_p1_end > lam2_p1_start else 'DOWN ✗'})")
    print(f"    Entropy trend (last 20%): {mean_entropy_last20(p1):.3f}  (prediction: falling)")
    print(f"    Grok onset: step {grok_step}")
    print(f"  Phase 2 (downward break — catastrophic forgetting):")
    if p2:
        print(f"    λ₂ {lam2_p2_start:.3f} → {lam2_p2_end:.3f}  (direction: {'DOWN ✓' if lam2_p2_end < lam2_p2_start else 'UP — no forgetting yet'})")
        print(f"    Entropy trend (last 20%): {mean_entropy_last20(p2):.3f}  (prediction: rising)")
        print(f"    Forgetting onset (acc_add < 0.7): step {forget_step}")
        direction_correct = (lam2_p1_end > lam2_p1_start) and (lam2_p2_end < lam2_p2_start)
        print(f"\n  Direction classification by λ₂: {'PASS ✓' if direction_correct else 'FAIL ✗'}")
    else:
        print("    (no phase 2 data)")

    # ── Plot ───────────────────────────────────────────────────────────────────
    steps     = [r['step']      for r in records]
    acc_add   = [r['acc_add']   for r in records]
    acc_mul   = [r['acc_mul']   for r in records]
    losses    = [r['train_loss']for r in records]
    lam2s     = [r['lambda2']   for r in records]
    ar1s      = [r['ar1']       for r in records]
    vrs       = [r['vr']        for r in records]
    entropies = [r['entropy']   for r in records]
    order_ps  = [r['order_p']   for r in records]

    fig, ax1 = plt.subplots(figsize=(18, 7))
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Accuracy / Loss')

    ax1.plot(steps, acc_add, color='#2ecc71', linewidth=1.8, label='Acc: addition (task 1)')
    ax1.plot(steps, acc_mul, color='#e74c3c', linewidth=1.8, label='Acc: multiplication (task 2)')
    ax1.plot(steps, losses,  color='#bdc3c7', linewidth=0.8, label='Train loss', alpha=0.6)
    ax1.set_ylim(-0.05, max(max(losses)*1.05, 1.05))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Structural metrics')
    ax2.plot(steps, lam2s,     color='#3498db', linewidth=1.4, label='λ₂ (Fiedler)')
    ax2.plot(steps, ar1s,      color='#9b59b6', linewidth=1.2, label='AR1')
    ax2.plot(steps, vrs,       color='#f39c12', linewidth=0.9, label='VR', alpha=0.8)
    ax2.plot(steps, entropies, color='#1abc9c', linewidth=0.9, label='Entropy (S)', linestyle='--', alpha=0.8)
    ax2.plot(steps, order_ps,  color='#e67e22', linewidth=0.9, label='order_p', linestyle=':', alpha=0.8)

    # phase boundary
    ax1.axvline(x=PHASE1_STEPS, color='black', linestyle='--', linewidth=1.5)
    ax1.text(PHASE1_STEPS + 200, 0.95, 'Phase 2\n(fine-tune mul)', fontsize=8)

    if grok_step is not None:
        ax1.axvline(x=grok_step, color='#27ae60', linestyle=':', linewidth=1.2)
        ax1.text(grok_step + 200, 0.85, f'grok\n@{grok_step}', fontsize=7, color='#27ae60')

    if forget_step is not None:
        ax1.axvline(x=forget_step, color='#c0392b', linestyle=':', linewidth=1.2)
        ax1.text(forget_step + 200, 0.75, f'forget\n@{forget_step}', fontsize=7, color='#c0392b')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8, ncol=2)

    plt.title('Neural Structural Intelligence — Step 2: Classification\n'
              'Upward break (grokking) vs Downward break (catastrophic forgetting)',
              fontsize=11)
    plt.tight_layout()
    plot_path = 'results/classification.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    run()
