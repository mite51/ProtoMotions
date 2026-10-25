# Pattern B ("Spinal Cord") Hierarchical Motor Control for SMPL Humanoid Tracking

**Branch:** `exteded_obs_training_experiment`  
**Simulator / Rig:** IsaacLab (`PhysX`), SMPL Humanoid (`24` rigid bodies, `23` 3-DOF spherical joints = `69` actuated DOFs)  
**Benchmark Dataset:** `amass_smpl_subset64.pt` (`64` non-fighting AMASS locomotion & whole-body motion clips, `606.3s` total duration)

---

## 1. Motivation & Core Hypothesis

In standard flat-MLP motion tracking (`examples/experiments/mimic/mlp.py`), a single policy network maps high-dimensional whole-body state and future reference poses directly to `69` PD joint position targets (`u_pd`). Consequently, the main policy must simultaneously discover:
1. **Local actuator & load compensation** (how far to lead a PD target given joint velocity $\dot{q}_j$, stiffness $k_{p,j}$, damping $k_{d,j}$, per-joint stamina $s_j$, gravitational torque $\boldsymbol{\tau}_{j,\text{grav}}$, and distal subtree inertia $I_{j,\text{eff}}$), and
2. **Global whole-body coordination & balance** (how to adjust desired posture $q_{\text{des}}$ to maintain center-of-mass balance and foot contacts).

During early PPO epochs (`Epochs 1–50`), a randomly initialized flat MLP has not yet learned gravity compensation or velocity lead, causing rapid falls (`~5.7` frames average survival in Epochs 1–10) and requiring tens of millions of steps just to stand and track basic trajectories. Furthermore, when a flat policy fails on a motion clip, it is difficult to diagnose whether the failure was caused by unmodeled external/contact forces, local joint tracking saturation, or global balance strategy.

### The Pattern B ("Spinal Cord") Factorization
We factor motor control into three explicit layers:
1. **Analytical Subtree Load Kernel (`Level 1`):** Closed-form, vectorized rigid-body subtree kinematics/statics that compute exact gravitational torque $\boldsymbol{\tau}_{j,\text{grav}}$, subtree angular momentum $\mathbf{L}_j$, static subtree mass $M_j$, and dynamic parallel-axis effective inertia $I_{j,\text{eff}}$ in each joint's local parent frame, plus an analytical PD velocity-lead & gravity-compensation baseline $u_{j,\text{base}}$.
2. **Frozen Per-Joint Shared Mini-Brain (`Level 2`):** A lightweight (`~21K` parameters) shared MLP applied identically across all `23` 3-DOF spherical joints (`[E, 23, 34] -> [E, 23, 3]`), pretrained offline in ~45 seconds across randomized postures, velocities, tilts, stamina scales (`[0.2, 1.0]`), and 1- or 2-DOF axis lockouts (`s_axis`).
3. **High-Level Main Brain (`Level 3`):** A PPO actor that observes whole-body state, future reference poses, and `joint_load_obs`, and outputs residual desired joint angles $\Delta q$ around the reference pose:
   $$q_{\text{des}} = q_{\text{ref}} + \text{scale} \cdot \tanh(\Delta q)$$
   which are passed into the frozen `SharedJointMiniBrain` to produce the final PD target $u_{\text{pd}}$.

---

## 2. Architecture & Code Map

### 2.1 Analytical Subtree Load Kernel & Exact CoM Lever Arms
* **Implementation:** `protomotions/envs/obs/joint_load_obs.py`
* **Observation Factories:** `protomotions/envs/component_factories.py` (`joint_load_obs_factory`, `minibrain_features_obs_factory`)
* **Unit Tests:** `tests/test_joint_load_obs.py`

For each of the `J = 23` non-root joints (`b_child = j + 1`), we precompute the static kinematic subtree mask $S_{j,b} \in \{0, 1\}^{23 \times 24}$, body masses $m_b$ (`estimate_mjcf_body_masses`), and local rigid-body Center-of-Mass (CoM) offsets $\mathbf{c}_{b,\text{local}}$ (`estimate_mjcf_body_com_offsets`, extracted from `mj_model.body_ipos[1:]`) in `SubtreeTopology` (`joint_load_obs.py:29–167`).

At runtime inside `@torch.jit.script` `compute_analytical_subtree_load` (`joint_load_obs.py:170–248`):
1. **True Rigid-Body CoM World Positions:**
   Both `PhysX` (`r.data.body_pos_w`) and `MuJoCo` (`xpos`) report each body's **joint pivot origin** $\mathbf{p}_b$. The gravitational force acts at the rigid body's true CoM:
   $$\mathbf{c}_b = \mathbf{p}_b + R_b \mathbf{c}_{b,\text{local}}$$
2. **Subtree Gravitational Torque in Local Parent Frame:**
   $$\boldsymbol{\tau}_{j,\text{grav}}^{\text{world}} = \sum_{b \in \text{subtree}(j)} (\mathbf{c}_b - \mathbf{p}_j) \times (m_b \mathbf{g}), \qquad \boldsymbol{\tau}_{j,\text{grav}}^{\text{local}} = R_{\text{parent}(j)}^\top \boldsymbol{\tau}_{j,\text{grav}}^{\text{world}}$$
   *Verification (`tests/test_joint_load_obs.py:224–273`):* Including `body_com_offset` reduces the discrepancy between `compute_analytical_subtree_load` and MuJoCo's exact Recursive Newton-Euler (`mj_data.qfrc_bias` at $\dot{q}=0$) from **`20.23 N·m` (`41.0%`) $\to$ `0.000010 N·m` (`0.000%`, float32 machine precision)** across randomized postures.
3. **Subtree Angular Momentum & Dynamic Effective Inertia:**
   $$\mathbf{L}_j^{\text{local}} = R_{\text{parent}(j)}^\top \sum_{b \in \text{subtree}(j)} m_b (\mathbf{c}_b - \mathbf{p}_j) \times (\mathbf{v}_b - \mathbf{v}_j)$$
   $$I_{j,\text{eff}} = \sum_{b \in \text{subtree}(j)} m_b \|\mathbf{c}_b - \mathbf{p}_j\|^2 + I_{\text{armature}}$$
4. **Analytical PD Lead Baseline (`compute_analytical_pd_lead_action`, `joint_load_obs.py:390–442`):**
   $$u_{j,\text{base}} = q_{j,\text{des}} + \frac{k_{d,\text{eff}}}{k_{p,\text{eff}}} \dot{q}_{j,\text{des}} - \frac{\boldsymbol{\tau}_{j,\text{grav}}^{\text{local}}}{k_{p,\text{eff}}}$$
   Note the negative sign on $\boldsymbol{\tau}_{j,\text{grav}}^{\text{local}} / k_{p,\text{eff}}$: because $\boldsymbol{\tau}_{j,\text{grav}}$ is the torque exerted *by* gravity on the subtree, the actuator must command a PD target opposing $\boldsymbol{\tau}_{j,\text{grav}}$ to hold equilibrium.

---

### 2.2 Per-Joint Shared Mini-Brain (`SharedJointMiniBrain`)
* **Implementation:** `protomotions/agents/common/minibrain.py`
* **Standalone Short-Horizon Trainer:** `scripts/train_smpl_minibrain.py`
* **Pretrained Weights:** `checkpoints/smpl_minibrain_v3.pt` (copied to `checkpoints/smpl_minibrain.pt`)
* **Unit Tests:** `tests/test_minibrain_module.py`, `tests/test_minibrain_trainer.py`

`SharedJointMiniBrain` processes all `23` joints in parallel with shape `[E, 23, 34]`:
- **Per-joint 34-D feature vector (`compute_minibrain_joint_features`, `joint_load_obs.py:302–387`):**
  - Local joint state: $q_j \in \mathbb{R}^3$, $\dot{q}_j \in \mathbb{R}^3$, tracking error $e_j = q_{j,\text{des}} - q_j \in \mathbb{R}^3$, desired velocity $\dot{q}_{j,\text{des}} \in \mathbb{R}^3$, previous action $u_{j,\text{prev}} \in \mathbb{R}^3$ (`15` dims)
  - Normalized effective PD gains $(k_{p,\text{eff}}/1000, k_{d,\text{eff}}/100)$ (`6` dims)
  - Analytical load features: $\boldsymbol{\tau}_{j,\text{grav}}^{\text{local}}/200$, $\mathbf{L}_j^{\text{local}}/50$, $M_j/80$, $I_{j,\text{eff}}/20$ (`8` dims)
  - Stamina scalar $s_j \in [0, 1]$ (`1` dim)
  - Per-axis availability mask $s_{\text{axis},j} \in [0, 1]^3$ (`3` dims; supports 1-DOF hinge or 2-DOF universal joint lockouts without retraining)
  - Topological depth $d_j / 6$ (`1` dim)
- **Zero-Initialized Residual + Physics Basis:**
  $$u_j = u_{j,\text{base}} + s_{\text{axis},j} \odot \left( \text{MLP}_{\theta}(\tilde{x}_j) + W_{\text{phys}} \begin{bmatrix} e_j \\ \dot{q}_{j,\text{des}} - \dot{q}_j \\ -\boldsymbol{\tau}_{j,\text{grav}}^{\text{local}} / k_{p,\text{eff}} \\ \mathbf{L}_j^{\text{local}} / (I_{j,\text{eff}} + \epsilon) \end{bmatrix} \right)$$
  Both the last layer of $\text{MLP}_{\theta}$ and $W_{\text{phys}}$ are initialized to zero so the Mini-Brain starts at the exact analytical baseline $u_{j,\text{base}}$ and learns nonlinear inertial/damping corrections plus a 1-step forward dynamics predictor $\hat{s}_{j,t+1} = [\hat{q}_{j,t+1}, \hat{\dot{q}}_{j,t+1}]$ (`dynamics_head`).

---

### 2.3 Hierarchical Main Brain (`HierarchicalPPOActor` & `HierarchicalPPO`)
* **Implementation:** `protomotions/agents/ppo/hierarchical_model.py`
* **Experiment Config:** `examples/experiments/mimic/smpl_hierarchical_mlp.py`
* **Unit Tests:** `tests/test_hierarchical_ppo.py`

Key architectural and training stability features:
1. **Reference-Relative Residual Pose Space (`hierarchical_model.py:252–278`):**
   The Main Brain trunk outputs `raw_delta_q` (`[E, 69]`), scaled by `init_main_brain_last_layer_scale = 0.01` at initialization so initial policy rollouts immediately track $q_{\text{ref}}$ via the Mini-Brain:
   $$q_{\text{des}} = q_{\text{ref}} + r_{\text{scale}} \tanh(\Delta q_{\text{raw}}), \qquad r_{\text{scale}} = 1.8\text{ rad}$$
2. **Normalized Action Space Alignment (`convert_to_normalized_action_space=True`, `hierarchical_model.py:293–305`):**
   Standard PPO in `ProtoMotions` applies `bounds_loss` (`weight=10.0`) whenever `|mean_action| > 1.0`. In `Hierarchical v1`, `mean_action` was emitted in raw radians (`u_pd`), which triggered false `bounds_loss` penalties whenever a joint angle exceeded $1.0\text{ rad}$ ($57.3^\circ$). In `Hierarchical v2` and `v3`:
   - `HierarchicalPPOActor` maps `u_pd` into normalized action space:
     $$a_{\text{norm}} = \frac{u_{\text{pd}} - \text{offset}}{\text{scale}} = \frac{u_{\text{pd}}}{\pi}$$
   - Paired with `make_pd_action_config(robot_cfg, action_transform="clamp", clamp_value=1.0)` in `smpl_hierarchical_mlp.py:139–143`, the environment reconstructs `processed_action = offset + scale * clamp(a_norm, -1, 1) == u_pd` without `tanh` saturation distortion or false `bounds_loss` penalties.
3. **Critic Warmup & Horizon Curriculum (`hierarchical_model.py:390–475`):**
   - `critic_warmup_epochs = 4`: Freezes actor updates for the first 4 epochs while the critic fits the high initial returns of the pretrained Mini-Brain.
   - `compute_curriculum_episode_length`: Ramps `max_episode_length` linearly from `300` to `1000` over the first `60` epochs.

---

### 2.4 Three-Level Edge-Case Collector & Diagnostic Triager
* **Implementation:** `protomotions/envs/diagnostics/edge_case_collector.py`
* **CLI Evaluator:** `scripts/evaluate_edge_cases.py`
* **Unit Tests:** `tests/test_edge_case_collector.py`

Every environment step or termination can be triaged into three actionable buckets:
- **Level 1 (`LEVEL1_ANALYTICAL_LOAD_MISMATCH`):** High 1-step dynamics prediction error $\|\Delta \dot{q}_{\text{actual}} - \Delta \dot{q}_{\text{pred}}\|_\infty > \tau_1$ caused by external contact impacts or unmodeled forces.
- **Level 2 (`LEVEL2_MINIBRAIN_TRACKING`):** Low dynamics surprise ($\le \tau_1$) but high local joint tracking error $\|q_{\text{des}} - q_{t+1}\|_\infty > \tau_2$, exported directly into `train_smpl_minibrain.py --edge-case-file` for targeted Mini-Brain fine-tuning.
- **Level 3 (`LEVEL3_MAINBRAIN_BALANCE`):** Local joints tracked $q_{\text{des}}$ accurately ($\le \tau_2$), yet the humanoid terminated or exceeded whole-body tracking thresholds (Main Brain posture/balance error).

---

### 2.5 Side-by-Side Live Policy Comparison & Training Resume Fixes
* **Side-by-Side Viewer (`protomotions/inference_agent.py`):**
  Added `--compare-checkpoint`, `--pair-spacing` (default `1.5m`), and `--max-compare-steps`. Even environments (`env_0`, `/World/envs/env_0/Robot`) run Policy A (`--checkpoint`) and odd environments (`env_1`, `/World/envs/env_1/Robot`) run Policy B (`--compare-checkpoint`) on synchronized motion clips (`motion_ids`, `motion_times`) offset by `+1.5m` in world $Y$. Automatically merges missing `observation_components` from Policy B into `env_config` and applies per-environment `action_config` processing (`tanh` vs. `clamp`).
* **Training Step Override on Resume (`protomotions/train_agent.py:665–680`):**
  Fixed `mode == "resume"` so passing `--training-max-steps` on the CLI updates `agent_config.training_max_steps` after `resolved_configs.pt` is loaded.

---

## 3. Experimental Results (`200M` Environment Steps / `1,525` Epochs in PhysX)

All four models were trained from scratch for **200,000,000 steps (`1,525` epochs)** on `amass_smpl_subset64.pt` (`4,096` envs, batch size `16,384`, IsaacLab PhysX):

1. **Baseline (`smpl_baseline_subset64`):** Standard flat MLP (`examples/experiments/mimic/mlp.py`).
2. **Hierarchical v1 (`smpl_hierarchical_subset64`):** Unnormalized radian action space (`action_transform=None`), `residual_pose_scale=0.7`, link-origin lever arms.
3. **Hierarchical v2 (`smpl_hierarchical_v2_subset64`):** Normalized action space (`convert_to_normalized_action_space=True` + `clamp`), `residual_pose_scale=1.8`, `0.01` last-layer init, foot contact match reward, link-origin lever arms.
4. **Hierarchical v3 (`smpl_hierarchical_v3_subset64`):** Identical to `v2`, plus **exact rigid-body CoM lever arms (`body_com_offset = mj_model.body_ipos[1:]`)** in `SubtreeTopology` and `SharedJointMiniBrain` (`checkpoints/smpl_minibrain_v3.pt`).

### 3.1 Sample Efficiency & Evaluation Success Rate (`%` and `Clips Succeeded / 64`)

| Training Stage | Steps | Baseline (`mlp.py`) | Hierarchical v1 | Hierarchical v2 (Normalized + Clamp) | **Hierarchical v3 (+ Exact CoM Lever Arm)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Early Survival (Epochs 1–10)** | `0–1.3M` | `5.71` frames | **`22.54` frames** | `22.18` frames | `22.23` frames |
| **Early Survival (Epochs 1–50)** | `0–6.5M` | `12.17` frames | `24.21` frames | `26.64` frames | **`26.75` frames** |
| **Epoch 200** | `26.2M` | `39.06%` (`25/64`) | `31.25%` (`20/64`) | `39.06%` (`25/64`) | **`39.06%` (`25/64`)** |
| **Epoch 400** | `52.4M` | `43.75%` (`28/64`) | `42.19%` (`27/64`) | `57.81%` (`37/64`) | **`60.94%` (`39/64`)** |
| **Epoch 600** | `78.6M` | `70.31%` (`45/64`) | `70.31%` (`45/64`) | **`71.88%` (`46/64`)** | **`71.88%` (`46/64`)** |
| **Epoch 800** | `104.9M` | `76.56%` (`49/64`) | `76.56%` (`49/64`) | `79.69%` (`51/64`) | **`82.81%` (`53/64`)** |
| **Epoch 1000** | `131.1M` | `85.94%` (`55/64`) | `79.69%` (`51/64`) | **`89.06%` (`57/64`)** | `87.50%` (`56/64`) |
| **Epoch 1200** | `157.3M` | **`95.31%` (`61/64`)** | `85.94%` (`55/64`) | `93.75%` (`60/64`) | `92.19%` (`59/64`) |
| **Epoch 1400** | `183.5M` | **`96.88%` (`62/64`)** | `90.62%` (`58/64`) | `95.31%` (`61/64`) | **`96.88%` (`62/64`)** |
| **Best / Final (`200M` steps)** | `200.0M` | `96.88%` (`62/64`) | `90.62%` (`58/64`) | **`98.44%` (`63/64`)** | **`96.88%` (`62/64`)** (`score_based`) / `95.31%` (`61/64`) (`last`) |

### 3.2 Final Full-Evaluation Tracking & Smoothness Metrics (`200M` Steps)

| Metric | Baseline (`mlp.py`, `last.ckpt`) | Hierarchical v1 (Ep 1400) | **Hierarchical v2 (`last.ckpt`)** | **Hierarchical v3 (`score_based.ckpt` / `last.ckpt`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Success Rate (`eval/success_rate`)** | `96.88%` (`62/64`) | `90.62%` (`58/64`) | **`98.44%` (`63/64`)** | `96.88%` (`62/64`) / `95.31%` (`61/64`) |
| **Global Position Error (`gt_error/mean`)** | `0.0765 m` (`7.65 cm`) | `0.1378 m` (`13.78 cm`) | **`0.0590 m` (`5.90 cm`)** | `0.0617 m` (`6.17 cm`) / `0.0621 m` (`6.21 cm`) |
| **Global Position Max Error (`gt_error/max`)** | `1.5391 m` | `2.8685 m` | **`0.9352 m`** | `1.2050 m` / `1.3762 m` |
| **Global Rotation Error (`gr_error/mean`)** | `0.2513 rad` (`14.40°`) | `0.2039 rad` (`11.68°`) | `0.1261 rad` (`7.22°`) | **`0.1226 rad` (`7.02°`)** / `0.1293 rad` (`7.41°`) |
| **Max Joint Error (`max_joint_error/mean`)** | `0.1668 rad` (`9.56°`) | `0.2078 rad` (`11.91°`) | `0.1116 rad` (`6.39°`) | **`0.1115 rad` (`6.39°`)** / `0.1141 rad` (`6.54°`) |
| **Normalized Jerk (`normalized_jerk_mean`)** | **`170.8`** | `222.8` | `266.7` | `271.6` / **`256.7`** |
| **High-Jerk Frames (`high_jerk_frame_%`)** | `3.95%` | `4.68%` | `5.12%` | `5.14%` / **`4.63%`** |
| **Action Smoothness (`action_delta_mean_deg`)** | **`0.283°`** | `1.494°` | `0.521°` | `0.523°` / **`0.518°`** |

---

## 4. Key Findings & Takeaways

1. **Immediate `3.9x` Early Survival Boost (`Epochs 1–10`):**
   Because the frozen `SharedJointMiniBrain` already knows how to compensate for gravity and joint velocity at step 0, hierarchical policies survive **`22.2–22.5` frames** on average in Epochs 1–10 compared to **`5.71` frames** for the flat MLP baseline.
2. **Accelerated Mid-Training Sample Efficiency (`Epochs 400–800`):**
   At **Epoch 400 (`52.4M` steps)**, **Hierarchical v3** (with exact CoM lever arms) achieves **`60.94%` (`39/64`)** success rate vs. **`43.75%` (`28/64`)** for the flat baseline (**`+17.2%`** absolute improvement). At **Epoch 800 (`104.9M` steps)**, **Hierarchical v3** reaches **`82.81%` (`53/64`)** vs. **`76.56%` (`49/64`)** for the flat baseline.
3. **~50% Lower Global Rotation Error & ~23% Lower Position Error at Convergence (`200M` Steps):**
   Both **Hierarchical v2** and **Hierarchical v3** cut the flat baseline's global rotation error in half (**`7.02°–7.22°`** vs. **`14.40°`**), reduce max-joint tracking error by **33%** (**`6.39°`** vs. **`9.56°`**), and reduce global position error from **`7.65 cm`** to **`5.90–6.17 cm`**, while **Hierarchical v2** achieves **`98.44%` (`63/64`)** clip completion.
4. **Why Action-Space Normalization (`v1 -> v2/v3`) Was Critical:**
   In `v1`, emitting raw radian PD targets (`[-pi, pi]`) into PPO's `bounds_loss` penalized any joint angle $> 1.0\text{ rad}$ ($57.3^\circ$) and capped residuals at `0.7 rad`. Normalizing `u_pd` by `pd_action_scale` (`pi`) with `action_transform="clamp"` eliminated the false `bounds_loss` penalty and cut action delta per step from `1.494°` (`v1`) to `0.518°` (`v3`).

---

## 5. Reproducing Evaluation & Live Side-by-Side Comparison

```bash
# Run all 29 unit tests (including exact RNE gravity torque verification)
PYTHONPATH=. conda run -n protomotion_isaaclab_12 pytest tests/ -v

# Live side-by-side viewport comparison:
# Env 0 (/World/envs/env_0/Robot, Left)  = Flat MLP Baseline
# Env 1 (/World/envs/env_1/Robot, Right) = Hierarchical v3 (score_based.ckpt)
PYTHONPATH=. conda run -n protomotion_isaaclab_12 python protomotions/inference_agent.py \
  --checkpoint results/smpl_baseline_subset64/last.ckpt \
  --compare-checkpoint results/smpl_hierarchical_v3_subset64/score_based.ckpt \
  --motion-file /home/jwylie/Dev/ProtomotionsAnimData/amass_smpl_subset64.pt \
  --simulator isaaclab \
  --num-envs 1 \
  --pair-spacing 1.5
```
