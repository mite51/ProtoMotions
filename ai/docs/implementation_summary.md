# ProtoMotions Implementation Summary

This document provides a high-level overview of the ProtoMotions architecture to help understand the inference and action application pipeline.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Inference Flow                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌───────────┐    ┌─────────┐ │
│  │   Env   │───▶│  Obs    │───▶│  Model  │───▶│  Actions  │───▶│Simulator│ │
│  │  Reset  │    │ TensorD │    │(PPO/etc)│    │ [N, 69]   │    │  Step   │ │
│  └─────────┘    └─────────┘    └─────────┘    └───────────┘    └─────────┘ │
│       │                                              │               │      │
│       └──────────────────────────────────────────────┴───────────────┘      │
│                              Loop                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Entry Points

| Script | Purpose |
|--------|---------|
| `inference_agent.py` | Run trained policy for visualization/testing |
| `train_agent.py` | Train a new policy |
| `export_model_to_onnx.py` | Export trained model to ONNX format |

### 2. Model Architecture (`protomotions/agents/ppo/model.py`)

The PPO model uses TensorDict for input/output:

```python
# Model Input (TensorDict)
obs_td = {
    "max_coords_obs": [batch, 358],      # Robot proprioception
    "mimic_target_poses": [batch, 577],  # Target motion poses
    # ... other observation keys
}

# Model Output
model_outs = {
    "action": [batch, 69],       # Sampled action (stochastic)
    "mean_action": [batch, 69],  # Deterministic action (mean)
    "neglogp": [batch],          # Negative log probability
    "value": [batch, 1],         # Value estimate (critic)
}
```

**Important**: The model's `in_keys` property defines which observations it actually uses. During ONNX export, only these keys are included as model inputs.

### 3. Environment (`protomotions/envs/base_env/env.py`)

The environment handles:
- **Observation computation** via observation callbacks (`self_obs_cb`, `terrain_obs_cb`, etc.)
- **Action processing** (clamping, scaling)
- **Reward computation**
- **Reset logic**

#### Step Flow:
```python
def step(self, actions):
    actions = self.process_actions(actions)      # Clamp actions
    self.simulator.step(actions, ...)            # Physics step
    self.post_physics_step()                     # Update observations, rewards
    obs = self.get_obs()
    return obs, rewards, dones, terminated, extras
```

### 4. Simulator (`protomotions/simulator/base_simulator/simulator.py`)

The simulator is an abstraction layer supporting multiple backends:
- IsaacGym
- IsaacLab
- Genesis
- Newton

#### Action Application Flow:

```
Actions [N, 69]
      │
      ▼
┌─────────────────────────────────────────┐
│  simulator.step(common_actions)         │
│    │                                    │
│    ├── _common_actions = actions        │
│    ├── _physics_step()                  │
│    │      │                             │
│    │      └── _apply_control()          │
│    │             │                      │
│    │             ├─ BUILT_IN_PD mode:   │
│    │             │   pd_targets = action_to_pd_targets(actions)
│    │             │   _apply_simulator_pd_targets(pd_targets)
│    │             │                      │
│    │             ├─ PROPORTIONAL mode:  │
│    │             │   pd_targets = action_to_pd_targets(actions)
│    │             │   torques = P*(target-pos) - D*vel
│    │             │   _apply_simulator_torques(torques)
│    │             │                      │
│    │             └─ TORQUE mode:        │
│    │                 torques = action_to_torque_targets(actions)
│    │                 _apply_simulator_torques(torques)
│    │                                    │
│    ├── _update_markers()                │
│    └── render()                         │
└─────────────────────────────────────────┘
```

### 5. Control Types

| Control Type | Description |
|--------------|-------------|
| `BUILT_IN_PD` | Uses simulator's native PD controller |
| `PROPORTIONAL` | Custom PD: `torque = Kp*(target-pos) - Kd*vel` |
| `TORQUE` | Direct torque control |

#### Action to PD Target Conversion:
```python
def _action_to_pd_targets(self, action):
    # action is in [-1, 1] range (or clamped range)
    pd_tar = pd_action_offset + pd_action_scale * action
    return pd_tar
```

The `pd_action_offset` is typically the default joint position, and `pd_action_scale` determines the range of motion.

## Data Flow Summary

### Inference Loop (from `mimic_evaluator.py`):

```python
# 1. Reset environment, get initial observations
obs, _ = env.reset(env_ids)
obs = agent.add_agent_info_to_obs(obs)
obs_td = agent.obs_dict_to_tensordict(obs)

# 2. Run model inference
model_outs = agent.model(obs_td)
actions = model_outs["mean_action"]  # or "action" for stochastic

# 3. Step environment with actions
obs, rewards, dones, terminated, extras = env.step(actions)
```

### Key Tensor Shapes (SMPL Humanoid Example):

| Tensor | Shape | Description |
|--------|-------|-------------|
| `max_coords_obs` | [N, 358] | Robot proprioceptive state |
| `mimic_target_poses` | [N, 577] | Target poses from motion library |
| `actions` | [N, 69] | Joint position targets (69 DOFs for SMPL) |
| `terrain` | [N, 256] | Terrain heightmap observations |

## ONNX Export

When exporting to ONNX (`scripts/export_model_to_onnx.py`):

1. The model's `in_keys` determines which observation tensors become ONNX inputs
2. The model's `out_keys` determines ONNX outputs
3. A metadata JSON file is saved with semantic key mappings

Example ONNX inputs for SMPL tracker:
- Input 0: `max_coords_obs` [1, 358]
- Input 1: `mimic_target_poses` [1, 577]

## Action Application for Unity

### Overview

The ONNX model outputs actions in the range **[-1, 1]**. These must be converted to **joint angle targets (radians)** and then applied via a **PD controller**.

### Step 1: Action to Joint Target Conversion

For SMPL humanoid, all joints are 3-DOF spherical joints (x, y, z hinge axes). The conversion is:

```csharp
// For SMPL: all joints have symmetric limits capped at π
float pd_action_offset = 0.0f;  // Center position (radians)
float pd_action_scale = Mathf.PI;  // π radians

// Convert action [-1, 1] to joint target (radians)
float joint_target = pd_action_offset + pd_action_scale * action;
// Simplified: joint_target = action * π  (radians)
```

### Step 2: PD Controller

Apply a PD (Proportional-Derivative) controller to compute torques:

```csharp
// Per-joint PD control
float torque = Kp * (target_position - current_position) - Kd * current_velocity;
torque = Mathf.Clamp(torque, -effort_limit, effort_limit);
```

### Step 3: SMPL Joint Configuration (69 DOFs)

The joints are ordered by body hierarchy. Each body has 3 DOFs (x, y, z rotation axes):

| Index | Body | Kp (stiffness) | Kd (damping) | Effort Limit |
|-------|------|----------------|--------------|--------------|
| 0-2 | L_Hip | 800 | 80 | 500 |
| 3-5 | L_Knee | 800 | 80 | 500 |
| 6-8 | L_Ankle | 800 | 80 | 500 |
| 9-11 | L_Toe | 500 | 50 | 500 |
| 12-14 | R_Hip | 800 | 80 | 500 |
| 15-17 | R_Knee | 800 | 80 | 500 |
| 18-20 | R_Ankle | 800 | 80 | 500 |
| 21-23 | R_Toe | 500 | 50 | 500 |
| 24-26 | Torso | 1000 | 100 | 500 |
| 27-29 | Spine | 1000 | 100 | 500 |
| 30-32 | Chest | 1000 | 100 | 500 |
| 33-35 | Neck | 500 | 50 | 500 |
| 36-38 | Head | 500 | 50 | 500 |
| 39-41 | L_Thorax | 500 | 50 | 500 |
| 42-44 | L_Shoulder | 500 | 50 | 500 |
| 45-47 | L_Elbow | 500 | 50 | 500 |
| 48-50 | L_Wrist | 300 | 30 | 500 |
| 51-53 | L_Hand | 300 | 30 | 500 |
| 54-56 | R_Thorax | 500 | 50 | 500 |
| 57-59 | R_Shoulder | 500 | 50 | 500 |
| 60-62 | R_Elbow | 500 | 50 | 500 |
| 63-65 | R_Wrist | 300 | 30 | 500 |
| 66-68 | R_Hand | 300 | 30 | 500 |

### Complete Unity Implementation Example

```csharp
public class SMPLPolicyController : MonoBehaviour
{
    // PD gains per joint group (index 0-68)
    private float[] Kp = new float[69];
    private float[] Kd = new float[69];
    private float effortLimit = 500f;

    // Action scaling
    private float actionScale = Mathf.PI;  // π radians

    void Start()
    {
        InitializePDGains();
    }

    void InitializePDGains()
    {
        // Hip, Knee, Ankle (indices 0-8, 12-20)
        SetGains(0, 9, 800, 80);    // L_Hip, L_Knee, L_Ankle
        SetGains(12, 21, 800, 80);  // R_Hip, R_Knee, R_Ankle

        // Toe (indices 9-11, 21-23)
        SetGains(9, 12, 500, 50);   // L_Toe
        SetGains(21, 24, 500, 50);  // R_Toe

        // Torso, Spine, Chest (indices 24-32)
        SetGains(24, 33, 1000, 100);

        // Neck, Head, Thorax, Shoulder, Elbow (indices 33-47, 54-62)
        SetGains(33, 48, 500, 50);  // Neck to L_Elbow
        SetGains(54, 63, 500, 50);  // R_Thorax to R_Elbow

        // Wrist, Hand (indices 48-53, 63-68)
        SetGains(48, 54, 300, 30);  // L_Wrist, L_Hand
        SetGains(63, 69, 300, 30);  // R_Wrist, R_Hand
    }

    void SetGains(int start, int end, float kp, float kd)
    {
        for (int i = start; i < end; i++)
        {
            Kp[i] = kp;
            Kd[i] = kd;
        }
    }

    public float[] ComputeTorques(float[] actions, float[] currentPos, float[] currentVel)
    {
        float[] torques = new float[69];

        for (int i = 0; i < 69; i++)
        {
            // Convert action to target position (radians)
            float targetPos = actions[i] * actionScale;

            // PD control
            float torque = Kp[i] * (targetPos - currentPos[i]) - Kd[i] * currentVel[i];

            // Clamp to effort limits
            torques[i] = Mathf.Clamp(torque, -effortLimit, effortLimit);
        }

        return torques;
    }
}
```

### Important Notes

1. **Simulation Frequency**: The policy was trained at 60 FPS (IsaacGym) or 120 FPS (IsaacLab). Match this in Unity.

2. **Coordinate System**: Verify your Unity rig matches the MJCF coordinate conventions.

3. **Joint Ordering**: The order must match exactly. The MJCF file defines the canonical order.

4. **Control Mode**:
   - `BUILT_IN_PD`: Simulator handles PD internally (just set targets)
   - `PROPORTIONAL`: You compute torques manually (as shown above)

5. **Units**:
   - Positions: radians
   - Velocities: radians/second
   - Torques: Newton-meters

### File References

- Robot config: `protomotions/robot_configs/smpl.py`
- MJCF model: `protomotions/data/assets/mjcf/smpl_humanoid.xml`
- Action conversion: `protomotions/simulator/base_simulator/utils.py`
- Control application: `protomotions/simulator/base_simulator/simulator.py` (see `_apply_control()`)
