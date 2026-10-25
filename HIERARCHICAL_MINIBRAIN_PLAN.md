# Hierarchical Shared Mini-Brain & Analytical Subtree Load Plan (SMPL)

**Branch:** `exteded_obs_training_experiment` (forked from `extended_obs`)  
**Target Rig:** Base SMPL Humanoid (`protomotions/data/assets/mjcf/smpl_humanoid.xml`, `protomotions/robot_configs/smpl.py`)  
**Target Simulator / Conda Env:** IsaacLab (`protomotion_isaaclab_12`)

---

## 1. Motivation & Core Hypothesis

Standard PPO motion tracking from scratch wastes significant early training time falling over because a single monolithic network must simultaneously learn two coupled problems:
1. **Local Joint Execution (Spinal Reflexes):** How much PD target offset is needed to move each of the 23 3-DOF joints (`69` hinge DOFs) to a desired next-frame pose under varying gravity orientation, child-limb momentum, joint stamina/strength, and locked/unavailable DOFs.
2. **Global Whole-Body Balance & Impact Compensation (Motor Cortex / Cerebellum):** How to subtly adjust target poses ($\Delta q$) using full-body state and upcoming collider proximity (`collision_primitives`) so the unactuated root (`Pelvis`) remains balanced and survives impacts.

By factoring the controller into a **Shared Per-Joint Mini-Brain** (fed with analytical subtree load features and trained over short 1–4 frame horizons) beneath a **Main Brain** that outputs residual pose adjustments (**Pattern B**), the character tracks local poses accurately from step 0 of PPO, allowing the Main Brain to focus on balance and contact/impact adaptation.

---

## 2. System Architecture & Tensor Contracts

### 2.1 Pattern B Hierarchy ("Spinal Cord" Architecture)

```text
                       ┌───────────────────────────────────────────────┐
                       │                 MAIN BRAIN                    │
                       │  Inputs: max_coords_obs, mimic_target_poses,  │
                       │          collision_primitives, stamina_obs,   │
                       │          previous_actions, minibrain_state    │
                       └───────────────────────┬───────────────────────┘
                                               │
                         Outputs Residual Pose Adjustment: Δq [E, 23, 3]
                         (optional: Δdq [E, 23, 3], load_scale [E, 23, 1])
                                               │
                                               ▼
Ref Pose (q*_t, dq*_t) ───► (+) Adjusted Target: q_des = q*_t + Δq
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 SHARED MINI-BRAIN (Batched across [E, 23])                   │
│  Inputs per joint j (in Joint Local Frame):                                  │
│   • Joint state: q_j [3], dq_j [3]                                           │
│   • Target delta: wrap(q_des - q_j) [3], (dq_des - dq_j) [3]                 │
│   • Parent-to-child local rotation (6D) [6] & world gravity dir in joint [3] │
│   • Analytical Subtree Load: τ_grav [3], L_momentum [3], M_sub [1], I_eff [1]│
│   • 1-Step Velocity Prediction Error (Impact Detector): dq_err [3]           │
│   • Per-Axis Availability / Stamina Mask: s_axis [3] (0=locked/off, 1=full)  │
│   • Static Joint Params: [kp, kd, effort_limit, armature, limit_lo(3), hi(3)]│
└──────────────────────────────────────────────┬───────────────────────────────┘
                                               │
                         Outputs Masked PD Target Offset: Δu_pd [E, 23, 3]
                         Final PD Target: u = (q_des + Δu_pd * (s_axis > 0)) -> [E, 69]
                                               │
                                               ▼
                       ┌───────────────────────────────────────────────┐
                       │       Simulator (IsaacLab BUILT_IN_PD)        │
                       └───────────────────────────────────────────────┘
```

### 2.2 SMPL Rig Dimensions & Per-Axis DOF Availability (`s_axis`)
* **Rigid Bodies (`B`):** `24` (`Pelvis` at index 0 + `23` articulated child bodies).
* **Joints (`J`):** `23` spherical 3-DOF body joints (`L_Hip`, `L_Knee`, `L_Ankle`, `L_Toe`, `R_Hip`, `R_Knee`, `R_Ankle`, `R_Toe`, `Torso`, `Spine`, `Chest`, `Neck`, `Head`, `L_Thorax`, `L_Shoulder`, `L_Elbow`, `L_Wrist`, `L_Hand`, `R_Thorax`, `R_Shoulder`, `R_Elbow`, `R_Wrist`, `R_Hand`).
* **Actuated DOFs (`D`):** `23 * 3 = 69` (`_x`, `_y`, `_z` hinges per body).
* **Handling 1 or 2 Unavailable / Locked DOFs:**
  * Each joint receives a 3D availability/strength vector `s_axis: Tensor` of shape `[E, 23, 3]` with values in `[0.0, 1.0]`.
  * `s_axis[e, j, a] == 0.0` indicates that hinge axis `a` (`x`, `y`, or `z`) is unavailable (locked, broken, or zero stamina).
  * `s_axis` defaults to `body_stamina.unsqueeze(-1) * dof_active_mask` (`[E, 23, 3]`).
  * Output masking guarantees `Δu_pd = Δu_pd * (s_axis > 0.0)`, while feeding `s_axis` into the Mini-Brain input allows cross-axis compensation when 1 or 2 axes are unavailable.

---

## 3. Feature Breakdown & Independent Test Modules

### Component 1: Analytical Subtree Load & Per-Joint Feature Kernel
* **Files:**
  * `protomotions/envs/obs/joint_load_obs.py`
  * `tests/test_joint_load_obs.py`
* **Responsibilities:**
  1. **Static Subtree Topology (`build_subtree_topology(kinematic_info, body_masses)`):**
     * Builds the binary descendant mask $S \in \{0, 1\}^{23 \times 24}$ from `kinematic_info.parent_indices` where $S_{j, b} = 1$ iff body $b$ is in the subtree rooted at joint $j$'s child body.
     * Precomputes static subtree mass $M_j = \sum_{b} S_{j,b} m_b$ and static per-joint control parameters (`kp, kd, effort_limit, armature, dof_limits_lower, dof_limits_upper`).
  2. **Vectorized Analytical Load Kernel (`compute_joint_minibrain_features(...)`):**
     * Pure tensor function operating on `[E, 24, 3]` body positions/velocities and `[E, 69]` DOF states:
     * **Subtree Gravity Torque ($\boldsymbol{\tau}_{j,\text{grav}} \in \mathbb{R}^3$):**
       $$\boldsymbol{\tau}_{j,\text{grav}}^{\text{world}} = \sum_{b \in \text{Subtree}(j)} (\mathbf{p}_b - \mathbf{p}_j) \times (m_b \mathbf{g})$$
       rotated into joint $j$'s local frame.
     * **Subtree Linear & Angular Momentum ($\mathbf{L}_j \in \mathbb{R}^3$):**
       $$\mathbf{L}_j^{\text{world}} = \sum_{b \in \text{Subtree}(j)} (\mathbf{p}_b - \mathbf{p}_j) \times \big(m_b (\mathbf{v}_b - \mathbf{v}_j)\big)$$
       rotated into joint $j$'s local frame.
     * **Dynamic Effective Inertia ($I_{j,\text{eff}} \in \mathbb{R}^1$):**
       $$I_{j,\text{eff}} = \sum_{b \in \text{Subtree}(j)} m_b \|\mathbf{p}_b - \mathbf{p}_j\|^2$$
     * **1-Step Velocity Prediction Error ($\Delta \dot{q}_{\text{err}} \in \mathbb{R}^3$):**
       Difference between actual $\dot{q}_t$ and expected velocity from the previous step's analytical/PD prediction, serving as an instantaneous contact/impact indicator.
* **Standalone Tests (`tests/test_joint_load_obs.py`):**
  * Verify SMPL subtree mask $S \in \{0, 1\}^{23 \times 24}$ (e.g., `L_Hip` subtree contains `L_Hip`, `L_Knee`, `L_Ankle`, `L_Toe`; `L_Toe` subtree contains only `L_Toe`).
  * Verify $\boldsymbol{\tau}_{\text{grav}}$ increases when limbs are extended horizontally vs. hanging vertically.
  * Verify frame invariance when the character root is rotated in 3D space.
  * Verify $\mathbf{L}_j == \mathbf{0}$ when relative velocities are zero.

---

### Component 2: Shared Per-Joint Mini-Brain Network
* **Files:**
  * `protomotions/agents/common/minibrain.py`
  * `tests/test_minibrain_module.py`
* **Responsibilities:**
  * Implements `SharedJointMiniBrain(nn.Module)` and `SharedJointMiniBrainConfig`:
    * Accepts per-joint feature tensor `[E, 23, F_in]`.
    * Normalizes inputs with running mean/variance (`RunningMeanStd`).
    * Evaluates a compact shared MLP (`F_in -> 128 -> 128 -> 3`, SiLU activations) across all `E * 23` joints in a single batched forward pass.
    * Zero-initializes the output layer so at initialization $\Delta u_{\text{pd}} = \mathbf{0}$ and $u_{\text{pd}} = q_{\text{des}}$.
    * Applies `s_axis` availability mask to zero out locked/disabled DOFs.
    * Optionally includes a lightweight per-joint value head `V_joint: [E, 23, 1]` for short-horizon per-joint RL/bandit updates.
* **Standalone Tests (`tests/test_minibrain_module.py`):**
  * Verify output shapes `[E, 23, 3]` and flattened `[E, 69]`.
  * Verify zero-init identity property ($u_{\text{pd}} == q_{\text{des}}$ at step 0).
  * Verify strict masking when 1, 2, or 3 axes in `s_axis` are `0.0`.
  * Verify permutation/batching consistency across environments and joints.

---

### Component 3: Standalone Short-Horizon In-Simulator Mini-Brain Trainer
* **Files:**
  * `scripts/train_smpl_minibrain.py`
* **Responsibilities:**
  * Trains `SharedJointMiniBrain` inside IsaacLab (`protomotion_isaaclab_12`) on SMPL using a test motion file before Main Brain training.
  * **Short-Horizon Rollout Loop ($H \in \{1, 2, 4\}$ steps):**
    1. Sample reference frame $t-1$ from `MotionLib` (RSI).
    2. Reset SMPL to frame $t-1$ with configurable perturbations:
       * Joint position noise $\epsilon_q$, joint velocity noise $\epsilon_{\dot{q}}$, and root orientation noise.
       * Random per-body stamina `body_stamina` $\in [0.2, 1.0]$.
       * Random 1- or 2-axis DOF lockouts (`s_axis = 0` on a subset of axes, with matching simulator gain zeroing).
    3. Step $H$ frames in IsaacLab where the Mini-Brain attempts to drive each joint $j$ to $(q_{t+k,j}^*, \dot{q}_{t+k,j}^*)$.
    4. Optimize Mini-Brain parameters using:
       * **Mode A (Differentiable Surrogate / Supervised Action Refinement):** Learn a local per-joint 1-step forward dynamics model $\hat{f}(x_{t,j}, u_{t,j}) \rightarrow (\hat{q}_{t+1,j}, \hat{\dot{q}}_{t+1,j})$ online from simulator transitions and backpropagate tracking error $\| \hat{q}_{t+1,j} - q_{t+1,j}^* \|^2 + c_v \| \hat{\dot{q}}_{t+1,j} - \dot{q}_{t+1,j}^* \|^2$ directly into `SharedJointMiniBrain`.
       * **Mode B (Per-Joint Short-Horizon PPO):** Treat each `(env, joint)` pair as a parallel agent (`E * 23` parallel trajectories) optimizing local per-joint tracking reward over $H$ steps.
  * **Benchmark Output:**
    * Logs a side-by-side table comparing 1-step & 4-step joint position/velocity tracking error for:
      1. Standard PD (`u = q_des`)
      2. Analytical PD Lead (`u = q_des + (kd/kp) * dq_des`)
      3. Trained `SharedJointMiniBrain`
    * Saves frozen checkpoint `checkpoints/smpl_minibrain.pt`.

---

### Component 4: Hierarchical Main Brain + Frozen Mini-Brain Integration (Pattern B)
* **Files:**
  * `protomotions/agents/ppo/hierarchical_model.py`
  * `examples/experiments/mimic/smpl_hierarchical_mlp.py`
  * `tests/test_hierarchical_ppo.py`
* **Responsibilities:**
  1. **Observation Integration:**
     * Adds `joint_load_obs` (`[E, 23 * F_load]`) and `minibrain_base_cmd` (`[E, 69]`) to `EnvContext` and `observation_components` alongside `max_coords_obs`, `mimic_target_poses`, `previous_actions`, `collision_primitives`, and `stamina_obs`.
  2. **Pattern B Hierarchical Actor (`HierarchicalPPOActor`):**
     * Main Brain MLP outputs residual pose adjustment $\Delta q \in \mathbb{R}^{E \times 69}$ (scaled by `residual_pose_scale`, e.g. `0.5 rad`).
     * Forms adjusted desired pose $q_{\text{des}} = q_{\text{ref}, t+1}^* + \Delta q$.
     * Queries the frozen `SharedJointMiniBrain` to produce the final load-compensated PD action $u_{\text{pd}}$.
     * Computes Gaussian log-probabilities in the Main Brain's output space so PPO updates remain standard and well-conditioned.
  3. **Critic Warmup & Short-to-Long Horizon Curriculum:**
     * Supports `critic_warmup_epochs` (freezes Main Brain Actor for the first $N$ epochs while the Critic learns the value function of the Mini-Brain-stabilized rollout, preventing Critic warmup destruction).
     * Supports `horizon_curriculum` (ramps `max_episode_length` from short horizons like `16` or `32` frames up to `1000` frames as value estimates stabilize).

---

### Component 5: Multi-Level Edge-Case Collector & Diagnostic Triager
* **Files:**
  * `protomotions/envs/diagnostics/edge_case_collector.py`
  * `scripts/evaluate_edge_cases.py`
  * `tests/test_edge_case_collector.py`
* **Responsibilities:**
  * Maintains a lightweight GPU ring buffer of the last $K=8$ steps per environment (`ResetState`, `motion_id`, `motion_time`, `s_axis`, analytical load features, Main Brain $\Delta q$, Mini-Brain $u_{\text{pd}}$, achieved $(q_{t+1}, \dot{q}_{t+1})$, `collision_primitives`, and contact forces).
  * Whenever an episode terminates early (fall / tracking failure) or spikes above per-joint error thresholds, snapshots the window and **automatically classifies the failure into one of 3 diagnostic levels**:

| Diagnostic Level | Trigger Condition | Component Responsible | Automated Follow-Up Action |
| :--- | :--- | :--- | :--- |
| **Level 1: Analytical Load Mismatch** | High 1-step velocity prediction error ($\|\dot{q}_{t+1} - \dot{q}_{\text{pred}}\| > \theta_1$) while rigid-body contact forces are zero (`no external impact`) | `joint_load_obs.py` (unmodeled Coriolis / parent-acceleration coupling) | Saved to `edge_cases_level1_load.pt` for offline analytical vs. sim torque inspection |
| **Level 2: Mini-Brain Local Tracking Failure** | Low external contact force on subtree, valid $q_{\text{des}}$, but joint $j$ failed to track target ($\|q_{t+1,j} - q_{\text{des},j}\| > \theta_2$) | `SharedJointMiniBrain` (under/over-compensation for gravity, momentum, stamina, or locked DOFs) | Saved to `edge_cases_level2_minibrain.pt` and loaded directly as hard-example resets in `train_smpl_minibrain.py` |
| **Level 3: Main Brain Balance / Impact Failure** | Local joints tracked $q_{\text{des}}$ within tolerance ($\|q_{t+1,j} - q_{\text{des},j}\| \le \theta_2$) OR high external contact/collider impact occurred, followed by root fall/termination | Main Brain Policy (poor whole-body balance adjustment $\Delta q$ or poor collider impact bracing) | Saved to `edge_cases_level3_mainbrain.pt` for prioritized RSI reset sampling during Main Brain PPO |

---

## 4. Multi-Agent Implementation Schedule

### Wave 1: Parallel Core Modules (Zero File Overlap)
* **Agent A (Component 1 — Analytical Subtree Load):**
  * Implement `protomotions/envs/obs/joint_load_obs.py` and `tests/test_joint_load_obs.py`.
* **Agent B (Component 2 — Shared Joint Mini-Brain):**
  * Implement `protomotions/agents/common/minibrain.py` and `tests/test_minibrain_module.py`.
* **Agent C (Component 5 — Multi-Level Edge-Case Collector):**
  * Implement `protomotions/envs/diagnostics/edge_case_collector.py` and `tests/test_edge_case_collector.py`.

### Wave 2: Training Scripts & Pattern-B PPO Integration
* **Agent D (Component 3 — Standalone Mini-Brain Trainer & Edge-Case Evaluator):**
  * Implement `scripts/train_smpl_minibrain.py` and `scripts/evaluate_edge_cases.py`.
* **Agent E (Component 4 — Hierarchical PPO & SMPL Experiment Config):**
  * Implement `protomotions/agents/ppo/hierarchical_model.py`, `examples/experiments/mimic/smpl_hierarchical_mlp.py`, and `tests/test_hierarchical_ppo.py`.

### Wave 3: End-to-End Verification & Experimentation (`protomotion_isaaclab_12`)
1. Run unit test suite (`pytest tests/test_joint_load_obs.py tests/test_minibrain_module.py tests/test_edge_case_collector.py tests/test_hierarchical_ppo.py`).
2. Run `scripts/train_smpl_minibrain.py` on a small SMPL motion file; verify 1-step and 4-step tracking error improvement over standard PD across normal, low-stamina, and 1–2 locked-DOF conditions.
3. Run `examples/experiments/mimic/smpl_hierarchical_mlp.py` with the frozen Mini-Brain + Critic Warmup + Edge-Case Collector enabled; inspect collected Level 1 / Level 2 / Level 3 edge cases via `scripts/evaluate_edge_cases.py`.
