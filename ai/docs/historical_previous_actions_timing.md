# Historical Previous Actions: 2-Frame Delay Explanation

## Summary

The observation key `historical_previous_actions` is **delayed by 2 frames** relative to the current decision step. At frame N, the model sees `historical_previous_actions` = the actions that were **output at frame N-2**, not frame N-1.

This is **not a bug in the debug exporter** — the captured data correctly reflects what the model receives. The 2-frame delay is inherent in the step/observation pipeline.

---

## Evidence from Debug Output

| Frame | `actions[0]` | `historical_previous_actions[0]` | Interpretation |
|-------|--------------|-----------------------------------|----------------|
| 0     | 0.5691...    | 0.0                               | Initial (zeros) |
| 1     | -0.1881...   | 0.0                               | Initial (zeros) |
| 2     | 0.0496...    | **0.5691...**                     | = Frame 0 actions |
| 3     | -0.1631...   | **-0.1881...**                    | = Frame 1 actions |
| 4     | -0.0432...   | **0.0496...**                     | = Frame 2 actions |

So: **`historical_previous_actions` at frame N = `actions` at frame N-2.**

---

## Why This Happens: Code Flow

### 1. Inference loop (e.g. `mimic_evaluator.py`)

For each step N:

1. Model receives `obs` (from the **previous** step’s `env.step()` return).
2. Model outputs `actions` = `actions_N`.
3. Debug capture records `obs` and `actions_N` (this is correct).
4. `env.step(actions_N)` is called.

### 2. Inside `env.step(actions_N)` (base_env)

- `simulator.step(actions_N, ...)` is called.
- Then `post_physics_step()` runs.
- Finally `get_obs()` is called and returned as the **next** step’s observation.

### 3. Inside `simulator.step(actions_N)` (base_simulator)

```python
# Store the previous actions (actions_{N-1})
self._previous_actions = self._common_actions.clone()

# Update to current (actions_N)
self._common_actions = common_actions.to(self.device)
self._physics_step()
# ...
```

So at this point: `_previous_actions` = actions from step N-1, `_common_actions` = actions_N.

### 4. Inside `post_physics_step()` (humanoid_obs)

- `previous_actions_hist_buf.rotate()` — shift history by one timestep.
- Later, `compute_observations()` is called.

### 5. Inside `compute_observations()` / history update (humanoid_obs)

When the action history buffer is updated (e.g. in `reset_hist_buf` or equivalent path that sets current frame):

- `previous_actions_hist_buf.set_curr(self.env.simulator.get_previous_actions(env_ids), env_ids)`

So the buffer’s “current” slot is filled with `_previous_actions`, i.e. **actions_{N-1}**.

### 6. When the next observation is read (step N+1)

- `get_obs()` returns `historical_previous_actions` = contents of the action history buffer.
- That buffer currently holds **actions_{N-1}** (the action that was just used in the step we just completed).

So at **decision step N+1**, the model sees:

- `historical_previous_actions` = **actions_{N-1}**.

That is **one step behind** the action we just applied (actions_N). But relative to “the action that led to the state we’re looking at,” we’re actually **two steps behind**:

- The state at step N+1 is the result of applying **actions_N**.
- The observation at step N+1 includes **actions_{N-1}**, not **actions_N**.

Hence: **2-frame delay** between “the action we took” and “the action we see in the observation.”

---

## Conclusion

- **Debug capture is correct:** It records the same `obs` (including `historical_previous_actions`) that the model sees.
- **Simulator debug prints are correct:** `_previous_actions` is indeed one step behind `_common_actions` at the simulator level.
- **The 2-frame delay is in the model input:** The observation is built after the step, using `_previous_actions`, and that observation is only used at the **next** inference step, so the model never sees its most recent action in `historical_previous_actions`.

If the policy was **trained** with this same pipeline (same env step and observation construction), then inference is consistent with training. If the goal is for the model to see the **immediately** previous action (N-1 at step N), the observation pipeline would need to use the current action (e.g. `get_current_actions()`) for the “current” slot of the action history buffer instead of `get_previous_actions()`, and the exact point where that is written into the buffer would need to be aligned with the desired semantics.
