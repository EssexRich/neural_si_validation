"""
Neural Structural Intelligence — Step 4 v6: Multi-Head Architecture
One structural change from v5: replace single fc2 with separate per-task heads.

Step 4 v1-v5 finding: the replay trap is caused by single-head output competition.
All three tasks share a 97-dimensional readout layer. No replay strategy resolves
this — tasks compete for the same output weights regardless of controller config,
floor setting, or network size.

v6 fix: multi-head MLP. Shared fc1 (representation layer). Separate fc2_A,
fc2_B, fc2_C (task-specific readout heads). Each task's output weights are
independent — no competition in the readout layer. The structural monitor
and accuracy-anchored controller are identical to v5. Only the architecture changes.

Hypothesis: with output competition eliminated, the accuracy-anchored controller
should be able to protect prior tasks while the new task learns. The shared fc1
can reorganise for each new task without destroying prior readout mappings because
those mappings live in separate heads.

Warm start: load fc1 weights from 512-unit Phase A checkpoint (grokked at step
4,869, add97=90.0%). Skip Phase A training entirely.

Performance optimisations from v5:
  LAM2_EVERY = 1000   (eigsh is expensive at 512 units)
  OP_EVERY   = 1000   (SVD likewise)
  EVAL_EVERY = 10     (test set eval every 10 steps)
  Heavy numpy ops (weight concat, entropy) only at OP_EVERY cadence

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

P1            = 97
HIDDEN        = 512
TRAIN_FRAC    = 0.5
LR_GROK       = 1e-3
WD_GROK       = 1.0
GROK_STEPS    = 25000
BATCH_SIZE    = 512
WINDOW        = 200
SLOPE_WINDOW  = 50
PRINT_EVERY   = 1000

# Accuracy-anchored controller
ACC_FLOOR     = 0.80   # protect prior tasks above this (no replay trap with multi-head)
BASE_REPLAY   = 0.10
MAX_REPLAY    = 0.90
DEFICIT_GAIN  = 4.0

# Performance: throttle expensive ops
LAM2_EVERY    = 1000
OP_EVERY      = 1000
EVAL_EVERY    = 10

PHASE_A_CKPT  = 'results/checkpoint_A_grok_grok4869.pt'  # 512-unit, add97=90%, step 4869

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset ────────────────────────────────────────────────────────────────────

def make_dataset(p, task='add97'):
    pairs = [(a, b) for a in range(p) for b in range(p)]
    np.random.shuffle(pairs)
    split = int(len(pairs) * TRAIN_FRAC)
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    def encode(pl):
        xs, ys = [], []
        for a, b in pl:
            x = np.zeros(2 * P1, dtype=np.float32)
            x[a % P1] = 1.0; x[P1 + b % P1] = 1.0
            xs.append(x)
            if task == 'add97':   ys.append((a + b) % 97)
            elif task == 'sub97': ys.append((a - b) % 97)
            elif task == 'add53': ys.append((a + b) % 53)
        return torch.tensor(np.array(xs)), torch.tensor(np.array(ys), dtype=torch.long)
    return encode(train_pairs), encode(test_pairs)


# ── Multi-head model ───────────────────────────────────────────────────────────

class MultiHeadMLP(nn.Module):
    """
    Shared fc1 representation layer. Separate fc2 readout head per task.
    Output competition eliminated — tasks share structure, not readout.
    """
    def __init__(self, hidden=HIDDEN, n_tasks=3, out_dims=None):
        super().__init__()
        if out_dims is None:
            out_dims = [P1, P1, 53]  # add97=97, sub97=97, add53=53
        self.fc1 = nn.Linear(2 * P1, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, d) for d in out_dims])

    def forward(self, x, head_idx):
        return self.heads[head_idx](torch.relu(self.fc1(x)))

    def forward_all(self, x):
        h = torch.relu(self.fc1(x))
        return [head(h) for head in self.heads]


# ── Spectral metrics ───────────────────────────────────────────────────────────

def build_laplacian_multihead(model):
    W1 = model.fc1.weight.detach().cpu().numpy()
    n_in, n_h = W1.shape[1], W1.shape[0]
    # use first head for spectral structure (shared representation is what matters)
    W2 = model.heads[0].weight.detach().cpu().numpy()
    n_out = W2.shape[0]
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
    named = [(n, p) for n, p in model.named_parameters()]
    for (_, param), prev in zip(named, prev_params):
        if param.dim() < 2: continue
        delta = param.detach().cpu().numpy() - prev
        if delta.shape[0] < 2 or delta.shape[1] < 2: continue
        try:
            sv = np.linalg.svd(delta, compute_uv=False)
            sv_sq = sv ** 2; total = sv_sq.sum()
            if total > 1e-12: results.append(float(sv_sq[0] / total))
        except Exception: continue
    return float(np.mean(results)) if results else 0.0

def lam2_slope(history):
    s = history[-SLOPE_WINDOW:]
    if len(s) < 3: return 0.0
    x = np.arange(len(s), dtype=float)
    return float(np.polyfit(x, s, 1)[0])


# ── Accuracy-anchored controller ───────────────────────────────────────────────

class AccuracyController:
    def __init__(self, protect_labels):
        self.protect_labels = protect_labels
        self.replay_frac = BASE_REPLAY
        self.deficit = 0.0
        self.worst_acc = 1.0

    def update(self, accs):
        if not self.protect_labels:
            self.replay_frac = 0.0
            return
        protected = [accs[l] for l in self.protect_labels if l in accs]
        if not protected: return
        self.worst_acc = min(protected)
        self.deficit = max(0.0, ACC_FLOOR - self.worst_acc)
        self.replay_frac = float(np.clip(
            BASE_REPLAY + self.deficit * DEFICIT_GAIN, BASE_REPLAY, MAX_REPLAY))


# ── Phase runner ───────────────────────────────────────────────────────────────

def run_phase(model, phase_name, task_label, head_idx,
              X_tr, y_tr,
              all_test_sets,       # list of (label, head_idx, X_te, y_te)
              protect_sets,        # list of (label, head_idx, X_tr, y_tr)
              protect_labels,
              n_steps, lr, wd,
              lam2_hist, global_offset,
              steer=False):

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    controller = AccuracyController(protect_labels) if steer else None

    prev_weights   = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
    prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    lam2_cached    = lam2_hist[-1] if lam2_hist else 0.0
    op_cached      = 0.0
    entropy_cached = 0.0
    accs_cached    = {label: 0.0 for label, _, _, _ in all_test_sets}

    n_tr = X_tr.shape[0]
    records = []
    grok_step = None
    primary_label = all_test_sets[0][0]

    print(f"\n{'='*60}")
    print(f"Phase: {phase_name} | Task: {task_label} | Head: {head_idx} | Steps: {n_steps} | steer={steer}")
    if steer:
        print(f"Protecting: {protect_labels} | ACC_FLOOR={ACC_FLOOR}")
    print(f"{'='*60}")

    for step in range(n_steps):
        global_step = global_offset + step

        # ── build batch ────────────────────────────────────────────────────────
        model.train()
        replay_frac = controller.replay_frac if controller else 0.0

        if replay_frac > 0.0 and protect_sets:
            # build list of (X_batch, y_batch, head_idx) segments — one per task
            segments = []
            n_replay_each = max(1, int(BATCH_SIZE * replay_frac) // len(protect_sets))
            n_new = max(1, BATCH_SIZE - n_replay_each * len(protect_sets))
            idx = torch.randint(0, n_tr, (n_new,))
            segments.append((X_tr[idx], y_tr[idx], head_idx))
            for plabel, ph_idx, pX, py in protect_sets:
                idx = torch.randint(0, pX.shape[0], (n_replay_each,))
                segments.append((pX[idx], py[idx], ph_idx))
            opt.zero_grad()
            total_loss = torch.tensor(0.0, requires_grad=True)
            for bx, by, bh in segments:
                logits = model(bx, bh)
                total_loss = total_loss + criterion(logits, by)
            total_loss.backward()
            opt.step()
            loss = total_loss
        else:
            idx = torch.randint(0, n_tr, (BATCH_SIZE,))
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            loss = criterion(model(xb, head_idx), yb)
            loss.backward()
            opt.step()

        # ── metrics ────────────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            if step % EVAL_EVERY == 0:
                for label, hidx, X_te, y_te in all_test_sets:
                    logits = model(X_te, hidx)
                    accs_cached[label] = (logits.argmax(1) == y_te).float().mean().item()
            accs = accs_cached

            if controller:
                controller.update(accs)

            if step % OP_EVERY == 0:
                curr = np.concatenate([p.detach().cpu().numpy().flatten() for p in model.parameters()])
                delta = curr - prev_weights; prev_weights = curr.copy()
                op_cached = order_parameter(model, prev_layer_params)
                prev_layer_params = [p.detach().cpu().numpy().copy() for p in model.parameters()]
                entropy_cached = shannon_entropy(delta)

            if step % LAM2_EVERY == 0:
                L = build_laplacian_multihead(model)
                lam2_cached = fiedler_value(L)
            lam2 = lam2_cached; lam2_hist.append(lam2)
            ws = lam2_hist[-WINDOW:]

            if grok_step is None and step > 100 and accs.get(primary_label, 0) >= 0.9:
                grok_step = global_step
                ckpt = f'results/checkpoint_{phase_name}_grok{global_step}.pt'
                torch.save({'step': global_step, 'phase': phase_name,
                            'model_state_dict': model.state_dict(),
                            'lambda2': lam2, 'accs': accs}, ckpt)
                print(f"  *** GROK step {global_step} ({step} into phase): "
                      f"acc_{primary_label}={accs[primary_label]:.3f} — checkpoint saved")

            deficit = controller.deficit if controller else 0.0
            rec = {
                'global_step': global_step, 'phase': phase_name,
                'train_loss': loss.item(), 'lambda2': lam2,
                'ar1': ar1(ws), 'vr': variance_ratio(ws),
                'entropy': entropy_cached, 'order_p': op_cached,
                'lam2_slope': lam2_slope(lam2_hist),
                'replay_frac': replay_frac, 'deficit': deficit,
            }
            for label, _, _, _ in all_test_sets:
                rec[f'acc_{label}'] = accs.get(label, 0.0)
            records.append(rec)

        if (step + 1) % PRINT_EVERY == 0:
            acc_str = ' | '.join(f'acc_{l}={accs.get(l,0):.3f}' for l,_,_,_ in all_test_sets)
            print(f"  step {step+1:6d} | loss={loss.item():.4f} | {acc_str} | "
                  f"λ₂={lam2:.3f} | deficit={deficit:+.3f} | replay={replay_frac:.0%}")

    final = records[-1]
    print(f"\nPhase '{phase_name}' complete:")
    for label, _, _, _ in all_test_sets:
        print(f"  acc_{label}: {final[f'acc_{label}']:.3f}")
    print(f"  λ₂: {final['lambda2']:.3f}")
    if grok_step is not None:
        print(f"  Grokked at global step {grok_step} ({grok_step - global_offset} steps into phase)")

    ckpt = f'results/checkpoint_{phase_name}_final.pt'
    torch.save({'step': global_offset + n_steps - 1, 'phase': phase_name,
                'model_state_dict': model.state_dict(),
                'lambda2': final['lambda2'],
                'accs': {l: final[f'acc_{l}'] for l,_,_,_ in all_test_sets},
                'grok_step': grok_step}, ckpt)
    print(f"  Checkpoint saved: {ckpt}")
    return records, lam2_hist, grok_step


# ── Main ────────────────────────────────────────────────────────────────────────

def run():
    os.makedirs('results', exist_ok=True)

    (X_add97_tr, y_add97_tr), (X_add97_te, y_add97_te) = make_dataset(97, 'add97')
    (X_sub97_tr, y_sub97_tr), (X_sub97_te, y_sub97_te) = make_dataset(97, 'sub97')
    (X_add53_tr, y_add53_tr), (X_add53_te, y_add53_te) = make_dataset(53, 'add53')
    for t in [X_add97_tr, y_add97_tr, X_add97_te, y_add97_te,
              X_sub97_tr, y_sub97_tr, X_sub97_te, y_sub97_te,
              X_add53_tr, y_add53_tr, X_add53_te, y_add53_te]:
        t.data = t.to(DEVICE).data

    # head indices: 0=add97, 1=sub97, 2=add53
    all_tests = [
        ('add97', 0, X_add97_te, y_add97_te),
        ('sub97', 1, X_sub97_te, y_sub97_te),
        ('add53', 2, X_add53_te, y_add53_te),
    ]

    model = MultiHeadMLP(hidden=HIDDEN, n_tasks=3, out_dims=[97, 97, 53]).to(DEVICE)
    lam2_hist = []
    all_records = []

    # ── Warm start: load fc1 from 512-unit Phase A checkpoint ─────────────────
    print(f"\n{'='*60}")
    print(f"Warm start: loading fc1 from {PHASE_A_CKPT}")
    ckpt_a = torch.load(PHASE_A_CKPT, map_location=DEVICE)
    # load fc1 weights only — head 0 (add97) also initialised from old fc2
    old_sd = ckpt_a['model_state_dict']
    model.fc1.weight.data.copy_(old_sd['fc1.weight'])
    model.fc1.bias.data.copy_(old_sd['fc1.bias'])
    model.heads[0].weight.data.copy_(old_sd['fc2.weight'])
    model.heads[0].bias.data.copy_(old_sd['fc2.bias'])
    grok_A = ckpt_a['step']
    lam2_after_A = ckpt_a['lambda2']
    # seed lam2_hist with Phase A final value repeated for WINDOW
    lam2_hist = [lam2_after_A] * WINDOW
    print(f"  fc1 + head_0 loaded. acc_add97={ckpt_a['accs']['add97']:.3f}, "
          f"λ₂={lam2_after_A:.3f}, grok_step={grok_A}")
    print(f"{'='*60}")

    # verify warm start accuracy
    model.eval()
    with torch.no_grad():
        logits = model(X_add97_te, 0)
        acc_check = (logits.argmax(1) == y_add97_te).float().mean().item()
    print(f"  Warm start add97 accuracy: {acc_check:.3f}")

    # ── Phase B — grok subtraction, head 1, protect add97 via head 0 ──────────
    recs, lam2_hist, grok_B = run_phase(
        model, 'B_grok', 'sub97', head_idx=1,
        X_tr=X_sub97_tr, y_tr=y_sub97_tr,
        all_test_sets=all_tests,
        protect_sets=[('add97', 0, X_add97_tr, y_add97_tr)],
        protect_labels=['add97'],
        n_steps=GROK_STEPS, lr=LR_GROK, wd=WD_GROK,
        lam2_hist=lam2_hist, global_offset=grok_A, steer=True)
    all_records.extend(recs)
    lam2_after_B = lam2_hist[-1]

    # ── Phase C — grok add53, head 2, protect add97 + sub97 ───────────────────
    recs, lam2_hist, grok_C = run_phase(
        model, 'C_grok', 'add53', head_idx=2,
        X_tr=X_add53_tr, y_tr=y_add53_tr,
        all_test_sets=all_tests,
        protect_sets=[('add97', 0, X_add97_tr, y_add97_tr),
                      ('sub97', 1, X_sub97_tr, y_sub97_tr)],
        protect_labels=['add97', 'sub97'],
        n_steps=GROK_STEPS, lr=LR_GROK, wd=WD_GROK,
        lam2_hist=lam2_hist, global_offset=grok_A + GROK_STEPS, steer=True)
    all_records.extend(recs)
    lam2_after_C = lam2_hist[-1]

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = 'results/compounding_v6_multihead.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
        writer.writeheader(); writer.writerows(all_records)
    print(f"\nData saved to {csv_path}")

    # ── Analysis ───────────────────────────────────────────────────────────────
    final = all_records[-1]
    acc_A = final['acc_add97']
    acc_B = final['acc_sub97']
    acc_C = final['acc_add53']

    steps_B = (grok_B - grok_A)              if grok_B is not None else None
    steps_C = (grok_C - grok_A - GROK_STEPS) if grok_C is not None else None

    multi_80  = acc_A >= 0.8 and acc_B >= 0.8 and acc_C >= 0.8
    multi_60  = acc_A >= 0.6 and acc_B >= 0.6 and acc_C >= 0.6
    replay_trap_broken = acc_B >= 0.5  # key test: did multi-head break the trap?

    print(f"\n── Compounding analysis (v6: multi-head) ────────────────────")
    print(f"  Final acc — A(add97):{acc_A:.3f} | B(sub97):{acc_B:.3f} | C(add53):{acc_C:.3f}")
    print(f"  λ₂ staircase (diagnostic): {lam2_after_A:.2f} → {lam2_after_B:.2f} → {lam2_after_C:.2f}")
    print(f"  Steps to grok: A={grok_A} | B={steps_B} | C={steps_C}")
    print(f"  All tasks ≥80%: {'YES ✓' if multi_80 else 'NO ✗'}")
    print(f"  All tasks ≥60%: {'YES ✓' if multi_60 else 'NO ✗'}")
    print(f"  Replay trap broken (B≥50%): {'YES ✓' if replay_trap_broken else 'NO ✗'}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    steps_  = [r['global_step'] for r in all_records]
    lam2s   = [r['lambda2']     for r in all_records]
    replays = [r['replay_frac'] for r in all_records]
    deficits= [r['deficit']     for r in all_records]
    acc_As  = [r['acc_add97']   for r in all_records]
    acc_Bs  = [r['acc_sub97']   for r in all_records]
    acc_Cs  = [r['acc_add53']   for r in all_records]

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
    phase_colours = {'B_grok': '#3498db', 'C_grok': '#e67e22'}
    B_start = grok_A
    C_start = grok_A + GROK_STEPS

    ax = axes[0]
    ax.plot(steps_, acc_As, color='#2ecc71', linewidth=1.5, label='Task A: (a+b) mod 97 [head 0]')
    ax.plot(steps_, acc_Bs, color='#3498db', linewidth=1.5, label='Task B: (a-b) mod 97 [head 1]')
    ax.plot(steps_, acc_Cs, color='#e67e22', linewidth=1.5, label='Task C: (a+b) mod 53 [head 2]')
    ax.axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.5, label='80% floor')
    ax.axhline(0.6, color='grey',  linestyle=':', linewidth=1, alpha=0.3, label='60%')
    for boundary, lbl in [(C_start, 'C')]:
        ax.axvline(boundary, color='black', linestyle='--', linewidth=1.2)
        ax.text(boundary + 200, 0.95, f'Phase {lbl}', fontsize=8)
    for gstep, col, lbl in [(grok_B,'#3498db','grok B'),(grok_C,'#e67e22','grok C')]:
        if gstep:
            ax.axvline(gstep, color=col, linestyle=':', linewidth=1.2, alpha=0.7)
            ax.text(gstep+200, 0.82, lbl, fontsize=7, color=col)
    ax.set_ylabel('Test accuracy'); ax.set_ylim(-0.05, 1.05)
    ax.set_title('Neural Structural Intelligence — Step 4 v6: Multi-Head Architecture\n'
                 'Separate fc2 per task eliminates output competition. Shared fc1 protected by accuracy-anchored controller.',
                 fontsize=11)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps_, lam2s, color='#2c3e50', linewidth=1.4)
    for b in [C_start]: ax.axvline(b, color='black', linestyle='--', linewidth=1.2)
    ax.annotate(f'λ₂={lam2_after_B:.1f}', xy=(C_start-1, lam2_after_B), fontsize=8,
                xytext=(C_start-5000, lam2_after_B+3), arrowprops=dict(arrowstyle='->',color='grey'))
    ax.annotate(f'λ₂={lam2_after_C:.1f}', xy=(steps_[-1], lam2_after_C), fontsize=8,
                xytext=(steps_[-1]-5000, lam2_after_C+3), arrowprops=dict(arrowstyle='->',color='grey'))
    ax.set_ylabel('λ₂ (Fiedler) — diagnostic'); ax.grid(alpha=0.3)

    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(steps_, deficits, color='#e74c3c', linewidth=1.0, alpha=0.7, label='accuracy deficit')
    ax.axhline(0, color='grey', linestyle=':', linewidth=0.8)
    ax2.plot(steps_, replays, color='#9b59b6', linewidth=1.2, alpha=0.8, label='replay_frac')
    ax2.set_ylabel('replay fraction', color='#9b59b6'); ax2.set_ylim(-0.05, 1.1)
    ax.set_ylabel('accuracy deficit')
    for b in [C_start]: ax.axvline(b, color='black', linestyle='--', linewidth=1.2)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc='upper left')
    ax.set_xlabel('Training step'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = 'results/compounding_v6_multihead.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    run()
