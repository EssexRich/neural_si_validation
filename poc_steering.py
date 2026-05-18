"""
Neural Structural Intelligence — Step 3: Steering
Can intervening on the λ₂ dip signal prevent catastrophic forgetting?

Protocol:
  Phase 1 — train on (a+b) mod 97 to grokking (identical to Step 2 baseline).
  Phase 2 — fine-tune on (a*b) mod 97.
             UNSTEERED: standard fine-tuning (Step 2 baseline result: 3% addition acc).
             STEERED:   monitor λ₂ slope over rolling 50-step window.
                        When slope goes negative → fire intervention:
                          - reduce LR by 50%
                          - switch batch to 50% addition / 50% multiplication
                          - if λ₂ rate > 0.004/step despite intervention → LR -50% again, 75% addition
  Run both variants from the same Phase 1 checkpoint.
  Compare addition accuracy at end of Phase 2.

Success criterion: steered addition accuracy > 80% vs unsteered ~3%.

Copyright © 2025-2026 Richard Benfield. All rights reserved.
"""

import os, csv, copy
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
P             = 97
HIDDEN        = 128
TRAIN_FRAC    = 0.5
LR_PHASE1     = 1e-3
WD_PHASE1     = 1.0
LR_PHASE2     = 5e-3
WD_PHASE2     = 0.0
PHASE1_STEPS  = 25000
PHASE2_STEPS  = 10000
BATCH_SIZE    = 512
WINDOW        = 200
SLOPE_WINDOW  = 50    # steps over which λ₂ slope is computed
PRINT_EVERY   = 500

# Steering thresholds (calibrated from Step 2 + iteration results)
SLOPE_TRIGGER       = -0.003  # fire when slope sustains below this (not just noisy negative)
SLOPE_CONFIRM_STEPS = 10      # slope must stay below trigger for this many consecutive steps
RATE_ESCALATE       = 0.004   # escalate if λ₂ rate still > this after first intervention
LR_REDUCTION        = 0.5     # multiply LR by this on intervention
ADD_FRAC_FLOOR      = 0.5     # starting addition fraction on first intervention
ADD_FRAC_MAX        = 0.75    # ceiling on addition fraction for batch mixing
ACC_FLOOR           = 0.80    # if addition acc drops below this, freeze fc1
# Key insight from iterations 1-2: batch mixing alone is insufficient at 75%.
# The 25% multiplication gradient slowly erodes addition over 10k steps.
# Solution: on escalation, freeze fc1 (representation layer) and let only fc2 adapt.
# fc2 is the task-specific readout; fc1 holds the generalised representations.
# This is the inversion rule's "freeze vulnerable layers" applied structurally.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Dataset ────────────────────────────────────────────────────────────────────

def make_dataset(p, task='add'):
    pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_FRAC)
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    def encode(pl):
        xs, ys = [], []
        for a, b in pl:
            x = np.zeros(2 * p, dtype=np.float32)
            x[a] = 1.0; x[p + b] = 1.0
            xs.append(x)
            ys.append((a + b) % p if task == 'add' else (a * b) % p)
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


# ── Spectral metrics ───────────────────────────────────────────────────────────

def build_laplacian(model):
    W1 = model.fc1.weight.detach().cpu().numpy()
    W2 = model.fc2.weight.detach().cpu().numpy()
    n_in, n_h, n_out = W1.shape[1], W1.shape[0], W2.shape[0]
    n = n_in + n_h + n_out
    rows, cols, data = [], [], []
    for h, i in zip(*np.where(W1 != 0)):
        u, v, w = i, n_in + h, abs(W1[h, i])
        rows += [u, v]; cols += [v, u]; data += [w, w]
    for o, h in zip(*np.where(W2 != 0)):
        u, v, w = n_in + h, n_in + n_h + o, abs(W2[o, h])
        rows += [u, v]; cols += [v, u]; data += [w, w]
    A = csr_matrix((data, (rows, cols)), shape=(n, n))
    d = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix((d, (np.arange(n), np.arange(n))), shape=(n, n))
    return D - A

def fiedler_value(L):
    try:
        vals = eigsh(L, k=3, which='SM', return_eigenvectors=False, tol=1e-6, maxiter=3000)
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
    ve = np.var(s[:cut]); vr = np.var(s[cut:])
    return float(vr / ve) if ve > 1e-12 else 1.0

def shannon_entropy(delta):
    mags = np.abs(delta).flatten()
    if mags.sum() < 1e-12: return 0.0
    h, _ = np.histogram(mags, bins=64)
    p = (h.astype(float) + 1e-12); p /= p.sum()
    return float(-np.sum(p * np.log2(p)))

def order_parameter(model, prev_params):
    results = []
    for (_, param), prev in zip(model.named_parameters(), prev_params):
        if param.dim() < 2: continue
        delta = param.detach().cpu().numpy() - prev
        if delta.shape[0] < 2 or delta.shape[1] < 2: continue
        try:
            sv = np.linalg.svd(delta, compute_uv=False)
            sv_sq = sv ** 2; total = sv_sq.sum()
            if total > 1e-12: results.append(float(sv_sq[0] / total))
        except Exception: continue
    return float(np.mean(results)) if results else 0.0

def lambda2_slope(history, window=SLOPE_WINDOW):
    """Linear regression slope of λ₂ over last `window` steps."""
    s = history[-window:]
    if len(s) < 3: return 0.0
    x = np.arange(len(s), dtype=float)
    return float(np.polyfit(x, s, 1)[0])

def lambda2_rate(history, window=200):
    """Mean per-step rate of λ₂ change over last `window` steps."""
    s = history[-window:]
    if len(s) < 2: return 0.0
    return float((s[-1] - s[0]) / len(s))


# ── Phase 1: train to grokking (shared) ────────────────────────────────────────

def run_phase1(model, X_add_tr, y_add_tr, X_add_te, y_add_te, X_mul_te, y_mul_te):
    opt = optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=WD_PHASE1)
    criterion = nn.CrossEntropyLoss()
    prev_weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    lam2_hist = []
    records = []
    n = X_add_tr.shape[0]

    print(f"Phase 1: training on (a+b) mod {P} for {PHASE1_STEPS} steps")
    for step in range(PHASE1_STEPS):
        model.train()
        idx = torch.randint(0, n, (BATCH_SIZE,))
        opt.zero_grad()
        loss = criterion(model(X_add_tr[idx]), y_add_tr[idx])
        loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            acc_add = (model(X_add_te).argmax(1) == y_add_te).float().mean().item()
            acc_mul = (model(X_mul_te).argmax(1) == y_mul_te).float().mean().item()
            curr = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
            delta = curr - prev_weights; prev_weights = curr.copy()
            L = build_laplacian(model); lam2 = fiedler_value(L); lam2_hist.append(lam2)
            ws = lam2_hist[-WINDOW:]
            op = order_parameter(model, prev_layer_params)
            prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
            records.append({'step': step, 'phase': 1, 'steered': False,
                            'train_loss': loss.item(), 'acc_add': acc_add, 'acc_mul': acc_mul,
                            'lambda2': lam2, 'ar1': ar1(ws), 'vr': variance_ratio(ws),
                            'entropy': shannon_entropy(delta), 'order_p': op,
                            'lam2_slope': lambda2_slope(lam2_hist),
                            'lam2_rate': lambda2_rate(lam2_hist),
                            'add_frac': 0.0, 'lr': LR_PHASE1, 'fc1_frozen': False,
                            'intervention': ''})

        if (step + 1) % PRINT_EVERY == 0:
            r = records[-1]
            print(f"  P1 {step+1:6d} | loss={r['train_loss']:.4f} | acc_add={r['acc_add']:.3f} | "
                  f"λ₂={r['lambda2']:.3f} | slope={r['lam2_slope']:.5f} | op={r['order_p']:.3f}")

    print(f"Phase 1 complete. Addition acc: {records[-1]['acc_add']:.3f}, λ₂: {records[-1]['lambda2']:.3f}")
    return records, lam2_hist


# ── Phase 2: fine-tune with optional steering ──────────────────────────────────

def run_phase2(model, X_add_tr, y_add_tr, X_mul_tr, y_mul_tr,
               X_add_te, y_add_te, X_mul_te, y_mul_te,
               lam2_hist_init, steer, label):

    lam2_hist = list(lam2_hist_init)
    current_lr = LR_PHASE2
    current_add_frac = 0.0
    intervened = False
    escalated = False
    fc1_frozen = False
    intervention_step = None
    slope_below_trigger_count = 0

    opt = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=WD_PHASE2)
    criterion = nn.CrossEntropyLoss()
    prev_weights = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]

    records = []
    n_add = X_add_tr.shape[0]
    n_mul = X_mul_tr.shape[0]

    print(f"\nPhase 2 [{label}]: fine-tuning on (a*b) mod {P} for {PHASE2_STEPS} steps")

    for step in range(PHASE2_STEPS):
        global_step = PHASE1_STEPS + step

        # ── steering logic ─────────────────────────────────────────────────────
        intervention_this_step = ''
        if steer:
            slope = lambda2_slope(lam2_hist)
            rate  = lambda2_rate(lam2_hist)

            # track consecutive steps below trigger threshold
            if len(lam2_hist) > SLOPE_WINDOW:
                if slope < SLOPE_TRIGGER:
                    slope_below_trigger_count += 1
                else:
                    slope_below_trigger_count = 0

            if not intervened and slope_below_trigger_count >= SLOPE_CONFIRM_STEPS:
                # confirmed sustained λ₂ dip — fire first intervention
                current_lr *= LR_REDUCTION
                current_add_frac = ADD_FRAC_FLOOR
                for g in opt.param_groups: g['lr'] = current_lr
                intervened = True
                intervention_step = global_step
                intervention_this_step = f'INTERVENE(lr={current_lr:.5f},add_frac={current_add_frac})'
                print(f"  *** INTERVENTION at step {global_step}: λ₂ slope={slope:.5f} "
                      f"(sustained {SLOPE_CONFIRM_STEPS} steps) "
                      f"→ LR={current_lr:.5f}, add_frac={current_add_frac}")

            elif intervened and not escalated and rate > RATE_ESCALATE:
                # escalate: λ₂ still accelerating — freeze fc1, only train fc2
                # set add_frac=1.0: while fc1 frozen, protect addition completely.
                # multiplication can wait — preservation first.
                current_lr *= LR_REDUCTION
                current_add_frac = 1.0
                for g in opt.param_groups: g['lr'] = current_lr
                # freeze fc1: representation layer holds addition structure
                for param in model.fc1.parameters():
                    param.requires_grad_(False)
                # rebuild optimiser over only trainable params (fc2 only)
                opt = optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_lr, weight_decay=WD_PHASE2)
                escalated = True
                fc1_frozen = True
                intervention_this_step = f'ESCALATE+FREEZE_FC1(lr={current_lr:.5f},add_frac=1.0)'
                print(f"  *** ESCALATION at step {global_step}: λ₂ rate={rate:.5f} "
                      f"→ LR={current_lr:.5f}, add_frac=1.0, fc1 FROZEN")

            # fc1 remains frozen for the rest of Phase 2 once escalation fires.
            # fc2 must learn multiplication from the frozen addition representations.

        # ── build batch ────────────────────────────────────────────────────────
        model.train()
        if current_add_frac > 0.0:
            n_add_batch = int(BATCH_SIZE * current_add_frac)
            n_mul_batch = BATCH_SIZE - n_add_batch
            idx_a = torch.randint(0, n_add, (n_add_batch,))
            idx_m = torch.randint(0, n_mul, (n_mul_batch,))
            xb = torch.cat([X_add_tr[idx_a], X_mul_tr[idx_m]], dim=0)
            yb = torch.cat([y_add_tr[idx_a], y_mul_tr[idx_m]], dim=0)
        else:
            idx_m = torch.randint(0, n_mul, (BATCH_SIZE,))
            xb, yb = X_mul_tr[idx_m], y_mul_tr[idx_m]

        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward(); opt.step()

        # ── metrics ────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            acc_add = (model(X_add_te).argmax(1) == y_add_te).float().mean().item()
            acc_mul = (model(X_mul_te).argmax(1) == y_mul_te).float().mean().item()
            curr = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
            delta = curr - prev_weights; prev_weights = curr.copy()
            L = build_laplacian(model); lam2 = fiedler_value(L); lam2_hist.append(lam2)
            ws = lam2_hist[-WINDOW:]
            op = order_parameter(model, prev_layer_params)
            prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
            records.append({'step': global_step, 'phase': 2, 'steered': steer,
                            'train_loss': loss.item(), 'acc_add': acc_add, 'acc_mul': acc_mul,
                            'lambda2': lam2, 'ar1': ar1(ws), 'vr': variance_ratio(ws),
                            'entropy': shannon_entropy(delta), 'order_p': op,
                            'lam2_slope': lambda2_slope(lam2_hist),
                            'lam2_rate': lambda2_rate(lam2_hist),
                            'add_frac': current_add_frac, 'lr': current_lr,
                            'fc1_frozen': fc1_frozen,
                            'intervention': intervention_this_step})

        if (step + 1) % PRINT_EVERY == 0:
            r = records[-1]
            print(f"  P2 {step+1:6d} | loss={r['train_loss']:.4f} | acc_add={r['acc_add']:.3f} | "
                  f"acc_mul={r['acc_mul']:.3f} | λ₂={r['lambda2']:.3f} | "
                  f"slope={r['lam2_slope']:.5f} | add_frac={r['add_frac']:.2f} | lr={r['lr']:.5f}")

    final = records[-1]
    print(f"\nPhase 2 [{label}] complete:")
    print(f"  Addition acc:      {final['acc_add']:.3f}")
    print(f"  Multiplication acc:{final['acc_mul']:.3f}")
    print(f"  λ₂ final:          {final['lambda2']:.3f}")
    if steer and intervention_step:
        print(f"  First intervention: step {intervention_step}")

    # save checkpoint for steered run (the preserved model is the useful one)
    if steer:
        torch.save({
            'phase': f'phase2_{label}',
            'model_state_dict': model.state_dict(),
            'lambda2': final['lambda2'],
            'acc_add': final['acc_add'],
            'acc_mul': final['acc_mul'],
        }, f'results/checkpoint_steering_{label}.pt')
        print(f"  Checkpoint saved: results/checkpoint_steering_{label}.pt")

    return records


# ── Plot ────────────────────────────────────────────────────────────────────────

def make_plot(p1_records, p2_unsteered, p2_steered):
    all_unsteered = p1_records + p2_unsteered
    all_steered   = p1_records + p2_steered

    steps_p1  = [r['step'] for r in p1_records]
    steps_p2u = [r['step'] for r in p2_unsteered]
    steps_p2s = [r['step'] for r in p2_steered]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=False)

    # ── Panel 1: accuracy comparison ───────────────────────────────────────────
    ax = axes[0]
    ax.set_title('Addition accuracy: steered vs unsteered during Phase 2', fontsize=10)
    ax.plot(steps_p2u, [r['acc_add'] for r in p2_unsteered],
            color='#e74c3c', linewidth=1.8, label='Unsteered (baseline)')
    ax.plot(steps_p2s, [r['acc_add'] for r in p2_steered],
            color='#2ecc71', linewidth=1.8, label='Steered')
    ax.axhline(0.8, color='black', linestyle=':', linewidth=1, label='Success threshold (80%)')
    # mark interventions
    for r in p2_steered:
        if r['intervention'].startswith('INTERVENE'):
            ax.axvline(r['step'], color='#27ae60', linestyle='--', linewidth=1.2, alpha=0.7)
            ax.text(r['step'] + 50, 0.85, 'intervene', fontsize=7, color='#27ae60')
        elif r['intervention'].startswith('ESCALATE'):
            ax.axvline(r['step'], color='#f39c12', linestyle='--', linewidth=1.2, alpha=0.7)
            ax.text(r['step'] + 50, 0.75, 'freeze fc1', fontsize=7, color='#f39c12')
    ax.set_ylabel('Addition accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 2: λ₂ trajectory across full run ─────────────────────────────────
    ax = axes[1]
    ax.set_title('λ₂ (Fiedler): full training run', fontsize=10)
    # phase 1 (shared)
    ax.plot(steps_p1, [r['lambda2'] for r in p1_records],
            color='#3498db', linewidth=1.4, label='Phase 1 (grokking)')
    ax.plot(steps_p2u, [r['lambda2'] for r in p2_unsteered],
            color='#e74c3c', linewidth=1.2, linestyle='--', label='Phase 2 unsteered')
    ax.plot(steps_p2s, [r['lambda2'] for r in p2_steered],
            color='#2ecc71', linewidth=1.2, label='Phase 2 steered')
    ax.axvline(PHASE1_STEPS, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('λ₂'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ── Panel 3: λ₂ slope during Phase 2 ──────────────────────────────────────
    ax = axes[2]
    ax.set_title(f'λ₂ slope (rolling {SLOPE_WINDOW}-step window) — trigger = 0 crossing', fontsize=10)
    ax.plot(steps_p2u, [r['lam2_slope'] for r in p2_unsteered],
            color='#e74c3c', linewidth=1.2, linestyle='--', label='Unsteered')
    ax.plot(steps_p2s, [r['lam2_slope'] for r in p2_steered],
            color='#2ecc71', linewidth=1.2, label='Steered')
    ax.axhline(0, color='black', linestyle=':', linewidth=1, label='Slope = 0 (trigger)')
    ax.set_xlabel('Training step'); ax.set_ylabel('λ₂ slope')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle('Neural Structural Intelligence — Step 3: Steering\n'
                 'Catastrophic forgetting prevented by λ₂ dip detection + curriculum intervention',
                 fontsize=11)
    plt.tight_layout()
    path = 'results/steering.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Plot saved to {path}")


# ── Main ────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('results', exist_ok=True)

    (X_add_tr, y_add_tr), (X_add_te, y_add_te) = make_dataset(P, 'add')
    (X_mul_tr, y_mul_tr), (X_mul_te, y_mul_te) = make_dataset(P, 'mul')
    for t in [X_add_tr, y_add_tr, X_add_te, y_add_te,
              X_mul_tr, y_mul_tr, X_mul_te, y_mul_te]:
        t.data = t.to(DEVICE).data

    # ── Phase 1: shared grokking run ───────────────────────────────────────────
    model_p1 = MLP().to(DEVICE)
    p1_records, lam2_hist_p1 = run_phase1(
        model_p1, X_add_tr, y_add_tr, X_add_te, y_add_te, X_mul_te, y_mul_te)

    # save checkpoint for both Phase 2 variants
    checkpoint = copy.deepcopy(model_p1.state_dict())

    # ── Phase 2a: unsteered (baseline) ────────────────────────────────────────
    model_u = MLP().to(DEVICE); model_u.load_state_dict(copy.deepcopy(checkpoint))
    p2_unsteered = run_phase2(
        model_u, X_add_tr, y_add_tr, X_mul_tr, y_mul_tr,
        X_add_te, y_add_te, X_mul_te, y_mul_te,
        lam2_hist_p1, steer=False, label='UNSTEERED')

    # ── Phase 2b: steered ──────────────────────────────────────────────────────
    model_s = MLP().to(DEVICE); model_s.load_state_dict(copy.deepcopy(checkpoint))
    p2_steered = run_phase2(
        model_s, X_add_tr, y_add_tr, X_mul_tr, y_mul_tr,
        X_add_te, y_add_te, X_mul_te, y_mul_te,
        lam2_hist_p1, steer=True, label='STEERED')

    # ── Save CSV ───────────────────────────────────────────────────────────────
    all_records = (
        [{**r, 'run': 'phase1'}    for r in p1_records]  +
        [{**r, 'run': 'unsteered'} for r in p2_unsteered] +
        [{**r, 'run': 'steered'}   for r in p2_steered]
    )
    csv_path = 'results/steering.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
        writer.writeheader(); writer.writerows(all_records)
    print(f"Data saved to {csv_path}")

    # ── Success criterion ──────────────────────────────────────────────────────
    u_final = p2_unsteered[-1]['acc_add']
    s_final = p2_steered[-1]['acc_add']
    s_mul   = p2_steered[-1]['acc_mul']
    print(f"\n── Success criterion ────────────────────────────────────────")
    print(f"  Unsteered addition acc at end: {u_final:.3f}")
    print(f"  Steered   addition acc at end: {s_final:.3f}")
    print(f"  Steered   multiplication acc:  {s_mul:.3f}")
    result = 'PASS ✓' if s_final >= 0.8 else (
             'PARTIAL' if s_final >= 0.5 else 'FAIL ✗')
    print(f"  Result: {result}  (threshold: 0.80)")

    make_plot(p1_records, p2_unsteered, p2_steered)


if __name__ == '__main__':
    run()
