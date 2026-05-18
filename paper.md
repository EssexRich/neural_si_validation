# Structural Topology Monitoring Predicts and Steers Neural Network Phase Transitions

**Richard Benfield**

*Independent researcher. Southend-on-Sea, United Kingdom.*

*Patent application GB2611542.8, filed 18 May 2026 (pending).*

---

## Abstract

We present a method for real-time monitoring of neural network weight graph topology during training using graph spectral analysis and early warning indicators from complex systems science. The method constructs a weighted graph from the network's weight matrices, computes the Fiedler value (second-smallest eigenvalue of the graph Laplacian), and applies Scheffer critical slowing down indicators (AR1 autocorrelation, variance ratio) to the Fiedler trajectory. We demonstrate five capabilities on modular arithmetic tasks (2-layer MLP) and sequence prediction tasks (1-layer transformer): (1) detection of approaching grokking 21,000 training steps before the loss function, exceeding the success criterion by 200x; (2) classification of approaching transitions as beneficial (generalisation) or destructive (catastrophic forgetting) via distinct structural fingerprints in the Fiedler trajectory; (3) prevention of catastrophic forgetting with 91.7% knowledge retention versus 2.6% without intervention, using Fiedler-sensitivity-guided layer freezing; (4) compounding of three sequential grokking events with 48x acceleration and near-perfect simultaneous retention (100%, 100%, 97.5%) using multi-head architecture with structural monitoring; and (5) preemptive curriculum design using a structural compatibility score that correctly ranks task disruption risk, with bridging curriculum preserving 100% of prior knowledge versus 0% under direct introduction. On the two architectures and task domains tested, the framework applies identical mathematics without modification. All code and data are available for reproduction.

---

## 1. Introduction

Neural network training is performed without real-time structural monitoring of the weight graph. The primary feedback signal is the loss function, which measures output performance but provides no information about the internal structural state of the network. Phase transitions during training, including sudden generalisation (grokking; Power et al., 2022), catastrophic forgetting (McCloskey and Cohen, 1989; French, 1999), and mode collapse, are detected by the loss function only after they have occurred.

This paper introduces a structural monitoring framework that applies graph spectral analysis and complex systems early warning indicators to the weight graph of neural networks during training. On the tasks and architectures tested, the framework detects approaching phase transitions before they manifest in output metrics, classifies them as beneficial or destructive, and steers the training process to compound beneficial transitions while preventing destructive ones.

The method originates from cross-domain structural topology detection. The same mathematical framework has been applied to financial market correlation graphs, epidemiological monitoring, and geophysical systems (Benfield, 2025-2026), in each case detecting approaching phase transitions from the spectral properties of the system's correlation graph. Neural network weight graphs are a natural extension: the weights define a graph, the graph has spectral properties, and those properties change during training in ways that precede observable changes in output metrics.

The key insight is that generalisation events and catastrophic forgetting are both structural phase transitions in the weight graph, with distinct spectral signatures that can be detected, classified, and acted upon before they manifest in the loss function.

We validate the framework through five sequential experiments, each building on the previous:

1. **Detection**: structural metrics detect approaching grokking 21,000 steps before the loss function.
2. **Classification**: grokking and catastrophic forgetting produce distinct structural fingerprints distinguishable by rate and shape.
3. **Steering**: structurally-guided intervention preserves 91.7% of knowledge versus 2.6% without intervention.
4. **Compounding**: three sequential tasks retained simultaneously at near-perfect accuracy with 48x grokking acceleration.
5. **Preemptive curriculum**: structural compatibility scoring correctly ranks task disruption risk; bridging curriculum preserves 100% versus 0% under direct introduction.

---

## 2. Related Work

### 2.1 Grokking

Power et al. (2022) identified grokking as a delayed generalisation phenomenon where neural networks suddenly transition from memorisation to generalisation long after training loss has converged. Subsequent work has characterised grokking through information-theoretic measures (Clauw et al., 2024), spectral analysis of gradient dynamics (Chen et al., 2026), and critical phenomena analogies (Wang, 2026). None of these approaches monitor the weight graph Laplacian or apply Scheffer early warning indicators to predict the transition.

### 2.2 Catastrophic Forgetting

Catastrophic forgetting occurs when fine-tuning on new data destroys previously learned capabilities (McCloskey and Cohen, 1989). Mitigations include Elastic Weight Consolidation (Kirkpatrick et al., 2017), experience replay (Rolnick et al., 2019), and progressive neural networks (Rusu et al., 2016). These methods operate without knowledge of the weight graph topology and cannot predict which new data will cause forgetting before it occurs.

### 2.3 Graph Spectral Analysis in Neural Networks

Tam et al. (2020) used the Fiedler value as a regularisation penalty during training to control network connectivity. This is a static application: the Fiedler value is added to the loss function as a penalty term, not tracked over time as a dynamic indicator. Our method tracks the Fiedler trajectory and its derivatives as predictive signals, which is a fundamentally different use of the same mathematical object.

### 2.4 Critical Slowing Down

Scheffer et al. (2009) demonstrated that complex systems approaching tipping points exhibit characteristic early warning signals: rising autocorrelation (AR1), increasing variance (variance ratio), and flickering. These indicators have been validated in ecosystems, climate systems, and financial markets. Our contribution is applying these indicators to the spectral trajectory of neural network weight graphs, where the "system" is the weight topology and the "tipping point" is a phase transition in training dynamics.

---

## 3. Method

### 3.1 Weight Graph Construction

Given a neural network with L layers and weight matrices W_1, W_2, ..., W_L, we construct a weighted graph G = (V, E) where V is the set of all neurons across all layers, E is the set of all connections, and the weight of edge (i, j) is |w_ij|.

### 3.2 Spectral Monitoring

At intervals during training, we compute:

**Fiedler value** (lambda_2): the second-smallest eigenvalue of the graph Laplacian L = D - A, where A is the adjacency matrix of absolute weights and D is the degree matrix. Lambda_2 measures algebraic connectivity: how structurally cohesive the weight graph is.

**Fiedler sensitivity**: for each edge (i, j), FS(i,j) = (v_2[i] - v_2[j])^2 x |w(i,j)|, where v_2 is the Fiedler eigenvector. This measures each edge's contribution to algebraic connectivity, identifying structural bridges. In practice we use a finite-difference approximation: FS(i,j) ≈ |lambda_2(G with w_ij + epsilon) - lambda_2(G)| / epsilon, computed for the top-K edges by absolute weight magnitude.

### 3.3 Early Warning Indicators

On a rolling window of the lambda_2 trajectory:

**AR1**: lag-1 autocorrelation. Values approaching 1.0 indicate critical slowing down (Scheffer et al., 2009).

**Variance ratio (VR)**: Var(recent 20% of window) / Var(earliest 80%). Values exceeding 1.0 indicate pre-bifurcation variance inflation.

**Entropy (S)**: Shannon entropy of weight update magnitudes. Decreasing values indicate structure forming in the training dynamics.

**Order parameter (order_p)**: first singular value fraction from SVD of per-layer weight change matrices. Measures macro-scale organisation of weight updates.

### 3.4 Detection and Classification

A phase transition is detected when AR1 exceeds a calibrated threshold and VR > 1.0 with entropy decreasing. The transition direction is classified by the Fiedler trajectory shape:

- **Beneficial (generalisation)**: slow monotonic lambda_2 rise, rate < 0.002/step.
- **Destructive (catastrophic forgetting)**: rapid lambda_2 change with preceding dip signature, rate > 0.004/step.

These thresholds were calibrated on the experiments described in Section 4 and may require recalibration for different architectures or tasks.

Severity is assessed via lambda_2 delta at 100 steps post-detection: positive delta indicates recovery (mild); negative delta indicates continuing degradation (severe).

### 3.5 Steering Interventions

Proportional to classified severity:

1. **Learning rate reduction**: scaled to threat severity.
2. **Curriculum modification**: replay of prior task examples with fraction controlled by a proportional feedback loop based on accuracy deficit.
3. **Selective layer freezing**: layers containing high Fiedler-sensitivity edges (structural bridges) are frozen; low-sensitivity layers (task-specific readout) remain trainable.

The system operates as a continuous closed-loop controller, adjusting intervention based on the structural response measured at each step.

### 3.6 Structural Compatibility Scoring

For preemptive curriculum design, we compute a compatibility score for candidate new tasks before training begins. For each candidate task, we forward-pass a batch of examples through the frozen network, compute the gradient that training on those examples would produce, and measure the directional conflict between the gradient and the current weights, weighted by Fiedler sensitivity. High conflict on high-sensitivity edges indicates high disruption risk.

---

## 4. Experiments

All experiments use modular arithmetic tasks on 2-layer MLPs (128 or 512 hidden units) trained with AdamW (LR=1e-3, WD=1.0) unless otherwise stated. The transformer experiment uses a 1-layer transformer (embed_dim=64, 4 attention heads, approximately 40K parameters) on next-token prediction tasks.

### 4.1 Experiment 1: Detection

**Setup**: 2-layer MLP (194 input, 128 hidden, 97 output) on (a+b) mod 97 for 50,000 steps. Structural metrics computed at every step.

**Results**: The network grokked at step 21,151 (test accuracy crossed 90%). Lambda_2 climbed continuously from 4.03 to 35.84 (9x increase) throughout the memorisation phase. AR1 saturated at 0.999 from step 94. VR produced 58 spike events before the grok, the first at step 1,766.

**Lead time**: 21,051 steps before the loss function detected generalisation. The success criterion of 100 steps was exceeded by a factor of 200.

### 4.2 Experiment 2: Classification

**Setup**: Phase 1: train on (a+b) mod 97 for 25,000 steps. Phase 2: fine-tune on (a x b) mod 97 with LR=5e-3 and WD=0 for 10,000 steps.

**Results**: The two transition types produced distinct structural fingerprints. Grokking: lambda_2 rate 0.00128/step, monotonic climb. Catastrophic forgetting: lambda_2 rate 0.00471/step (3.7x faster), with an initial dip from 35.91 to 33.10 in 51 steps followed by explosive climb. Order parameter dropped from 0.22 to 0.12 at forgetting onset. The dip was detectable while addition accuracy was still at 69.6%.

The classification rule: slow monotonic rise = beneficial transition; rapid change with preceding dip = destructive transition. The lambda_2 delta at 100 steps post-detection cleanly separates the two regimes (+0.045 for grokking versus -1.694 for forgetting).

### 4.3 Experiment 3: Steering

**Setup**: Phase 1 checkpoint loaded. Two Phase 2 conditions from identical starting weights: unsteered (standard fine-tuning) and steered (lambda_2 slope monitoring with two-stage intervention).

**Results**: The intervention fired at step 25,013 with addition accuracy still at 87.5%. Lambda_2 slope exceeded the -0.003 threshold for 10 consecutive steps, triggering LR reduction and curriculum adjustment. Escalation to fc1 layer freezing (identified as the high Fiedler-sensitivity layer) occurred when lambda_2 rate exceeded 0.004/step.

| Condition | Addition accuracy (end) | Lambda_2 (end) |
|---|---|---|
| Unsteered | 2.6% | 86.46 |
| Steered | 91.7% | 35.94 |

Lambda_2 moved by 0.02 in the steered condition versus 50.5 in the unsteered condition, confirming that the structural violence of catastrophic forgetting was contained.

### 4.4 Experiment 4: Compounding

**Setup**: Three sequential tasks (addition mod 97, subtraction mod 97, addition mod 53) with accuracy-anchored controller. Six iterations explored controller designs and architectures.

Iterations v1-v5 on single-head MLPs (128 and 512 units) identified a replay trap: a self-reinforcing loop where accuracy deficits drove replay fractions to 90%, starving new tasks of gradient bandwidth. All single-head variants converged to approximately equal capacity shares per task (approximately 25-30%) regardless of controller configuration. This was diagnosed as output layer competition.

Iteration v6 introduced multi-head architecture: shared representation layer (fc1, 512 units), separate readout head (fc2) per task. One architectural change.

**v6 Results** (512-unit multi-head MLP):

| Task | Accuracy |
|---|---|
| (a+b) mod 97 | 100.0% |
| (a-b) mod 97 | 100.0% |
| (a+b) mod 53 | 97.5% |

Controller deficit: 0.000 throughout. Replay: 10% baseline throughout. Steps to grok: 4,869 (Task A), 110 (Task B), 101 (Task C). The 48x acceleration from Task A to Task C demonstrates structural priming: each grokking event builds foundations in the shared representation layer that accelerate subsequent tasks.

The controller correctly identified that multi-head architecture eliminates task competition and remained at baseline. This demonstrates the framework's dual capability: active intervention when structural conflicts exist, passive monitoring when they do not.

**The replay trap characterisation**: proportional replay fails on capacity-limited single-head networks because accuracy deficits drive replay up, which starves new tasks of gradient bandwidth, which keeps accuracy low, which keeps replay high. This is a structural incompatibility between single-head readout and multi-task learning, not a controller tuning problem.

### 4.5 Experiment 5: Preemptive Curriculum Design

**Setup**: 1-layer transformer on three next-token prediction tasks across two structural families: arithmetic ("3 + 5 = 8"), reversed arithmetic ("8 = 5 + 3"), and logical syllogisms ("if A then B, A, therefore B"). Task C (logic) introduced after Task A (arithmetic) grokked.

**Compatibility scoring**: directional gradient conflict weighted by Fiedler sensitivity correctly ranked logic as higher disruption risk than reversed arithmetic. The transformer's attention heads specialise across tasks, producing genuinely differentiated activation subspaces (unlike the MLP where activation overlap was 96%).

**Curriculum results** (Task C introduction, LR=2e-3, WD=0.3):

| Condition | Task A (addition) accuracy | Task C (logic) accuracy |
|---|---|---|
| Direct (no replay) | 0% (catastrophic forgetting) | 82% |
| Bridged (blend phase + 10% replay) | 100% (fully preserved) | 83% |

The bridging curriculum completely prevented catastrophic forgetting that was total without it.

**Lambda_2 finding**: the bridged condition produced a larger lambda_2 dip than the direct condition. Under hostile takeover, lambda_2 barely moves before the weight graph is overwritten. Under controlled bridging, the topology reorganises gradually, producing a measurable structural signature. Hostile destruction bypasses topology reorganisation; controlled reorganisation is visible in the structural metrics and can be steered.

---

## 5. Discussion

### 5.1 The Structural Correlate of Generalisation

The lambda_2 trajectory during grokking is a continuous, monotonic correlate of generalisation forming in the weight graph. The network's algebraic connectivity increases steadily for 21,000 steps while the loss function shows nothing. The grok is not a sudden event in the topology. It is the moment that sustained structural preparation manifests in output metrics.

This has practical implications: in this setting, lambda_2 serves as a training progress indicator that is independent of loss and accuracy. A rising lambda_2 indicates the weight graph is building the cross-cluster connectivity needed for generalisation, even when the loss curve is flat. Whether this holds at larger scale is an open question.

### 5.2 Grokking Versus Forgetting: Shape, Not Direction

The initial prediction was that lambda_2 would rise for grokking and fall for forgetting. The data revealed a more nuanced signature: both transitions involve rising lambda_2, but at dramatically different rates (0.00128/step versus 0.00471/step) and with different shapes (monotonic versus dip-then-explosive-rise). The classification is by shape and rate, not by direction. This is a stronger discriminant than originally hypothesised.

### 5.3 The Replay Trap

The v1-v5 iterations of Experiment 4 characterise a failure mode of proportional replay on capacity-limited single-head networks. This is a genuine contribution to the continual learning literature: the mechanism is precisely identified (accuracy deficit drives replay, which starves new-task gradients, which maintains the deficit), the root cause is architectural (output layer competition, not controller tuning), and the resolution is a one-structural-change fix (multi-head readout). The structural monitoring framework correctly diagnosed the problem as architectural rather than dynamic — a diagnosis that methods operating without structural visibility cannot make directly.

### 5.4 Structural Priming and Grokking Acceleration

The 4,869 to 110 to 101 step progression across three sequential tasks demonstrates that grokking events build structural foundations that accelerate subsequent learning. The 4.3x acceleration from 128 to 512 hidden units (21,151 to 4,869 steps on the same task) shows that this priming compounds with capacity. These findings suggest that the shared representation layer accumulates reusable structural scaffolding across tasks.

### 5.5 The Lambda_2 Surprise in Preemptive Curriculum Design

The observation that hostile takeover produces a smaller lambda_2 dip than controlled bridging is counterintuitive but mechanistically clear. Hostile destruction overwrites the weight graph so rapidly that there is no reorganisation phase. The weights are simply replaced, and lambda_2 responds to the new structure, not to a transition between structures. Controlled bridging produces a genuine structural transition that is visible, measurable, and steerable. This suggests that the most destructive training regimes are the ones that are structurally invisible, which has implications for training safety: if you cannot see the structural disruption, you cannot prevent it.

### 5.6 Limitations

All experiments use toy tasks (modular arithmetic, simple sequence prediction) on small architectures (128-512 unit MLPs, 40K-parameter transformer). Generalisation to production-scale training on language, vision, or multimodal tasks is unvalidated. The computational cost of Fiedler value computation scales as O(n^3) for exact methods, requiring approximate methods at scale. The curriculum scoring function was undiscriminating on MLPs due to activation saturation and only worked on the transformer where attention specialisation produced differentiated activation patterns. Scaling behaviour of the structural metrics beyond 512 units is not characterised.

---

## 6. Conclusion

We have demonstrated that the structural topology of neural network weight graphs, monitored via Fiedler value analysis and Scheffer early warning indicators, provides a predictive signal for phase transitions during training that precedes the loss function by orders of magnitude. The framework detects, classifies, prevents, compounds, and preemptively manages these transitions through five validated experimental steps.

The same mathematical framework applied across the two architectures (MLP and transformer) and two task domains (modular arithmetic and language-like sequence prediction) tested without modification. This is consistent with, though not yet evidence for, the universality of critical slowing down indicators observed in other complex systems.

The core finding is that shape is signal: the structural topology of the weight graph encodes information about approaching phase transitions that is not available in the loss function. Monitoring this topology in real time enables a class of training interventions that are impossible when the loss function is the only feedback signal.

Future work includes validation at production scale, integration with established continual learning methods, and exploration of the structural priming mechanism as a basis for curriculum-optimised training.

---

## 7. Reproducibility

All code, data, and checkpoints are available at https://github.com/EssexRich/neural_si_validation. Total computation time for all five experiments is under 24 hours on a consumer laptop CPU (Intel i7) with no GPU. The structural monitoring framework requires only standard scientific Python libraries (PyTorch, NumPy, SciPy, NetworkX).

---

## References

Chen, X., et al. (2026). Spectral Gating Networks. *arXiv:2602.07679*.

French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

Wang, P. (2026). Grokking as Dimensional Phase Transition in Neural Networks. *arXiv:2604.04655*.

McCloskey, M. and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

Power, A., et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv:2201.02177*.

Rolnick, D., et al. (2019). Experience replay for continual learning. *NeurIPS 2019*.

Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.

Clauw, K., Stramaglia, S., and Marinazzo, D. (2024). Information-Theoretic Progress Measures reveal Grokking is an Emergent Phase Transition. *arXiv:2408.08944*.

Scheffer, M., et al. (2009). Early-warning signals for critical transitions. *Nature*, 461(7260), 53-59.

Tam, E. and Dunson, D. (2020). Fiedler Regularization: Learning Neural Networks with Graph Sparsity. *Proceedings of Machine Learning Research*, 119. *arXiv:2003.00992*.
