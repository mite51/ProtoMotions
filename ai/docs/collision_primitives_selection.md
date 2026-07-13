# Collision-Primitive Observation: Discovery, Priority & Selection

This document describes how collidable entities are discovered and chosen for
the fixed-width `collision_primitives` observation.

## Selection overview

All discovered collider categories are treated identically:

1. `BaseEnv._build_collision_primitives()` assembles ground, active
   projectiles, and opponent key bodies into one temporary candidate set.
2. If that set is wider than `EnvConfig.max_collision_primitives` (`M`), the
   same contact-priority score used by the observation kernel compacts it to
   `M`. Slot/category order does not decide which category survives.
3. `compute_collision_primitives_obs()` range-gates and scores the `M`
   candidates, selects the highest-priority `K`, transforms them into the
   character's egocentric frame, and zero-pads missing entries.

`K` and the emitted feature width remain unchanged. Physical mass is internal
selection metadata; it is not added to the policy observation.

## Eligibility range

A candidate is eligible when it is valid and its center lies within
`collision_selection_range` of at least one character body. Distance is
measured against all body parts, not only the root. Candidates outside the
range receive a sentinel score and are zero-padded if selected only because
fewer than `K` eligible candidates exist.

## Category-neutral priority score

For each collider, the kernel computes:

- `d`: minimum center distance to any character body;
- `v`: maximum positive closing speed toward any body, using collider velocity
  relative to that body's velocity;
- `m`: non-negative physical/effective mass.

The bounded components are:

```text
proximity = distance_scale / (d + distance_scale)
closing   = v / (v + speed_scale)
mass      = m / (m + mass_scale)
score     = distance_weight * proximity
          + closing_speed_weight * closing
          + mass_weight * mass
```

Receding and tangential colliders receive no closing-speed bonus. Ties prefer
the nearer collider and then the earlier candidate index for deterministic
selection.

### Mass sources

- Projectile mass is computed from configured density and actual box, sphere,
  or capsule volume.
- Opponent-body mass comes from the simulator's physical articulation masses.
  Backends without a runtime mass query use a neutral 1 kg fallback.
- Static ground/obstacles use `static_collider_effective_mass`, a finite value
  that avoids infinities while representing an immovable contact.

`damage` remains a policy-visible semantic feature, but it does not affect
selection. This keeps physical contact likelihood separate from learned threat
meaning.

## Candidate assembly and capacity

The temporary raw width is at least:

```text
1 ground + projectile pool size + (N - 1) * opponent key-body count
```

It is padded to at least `M`. If the raw width exceeds `M`, all categories are
scored together before `topk(M)`. The final observation then independently
applies the same score before `topk(K)`.

In multi-character scenes, each flattened row sees the projectiles in its
physical scene and key bodies from every sibling character, excluding itself.

## Per-primitive feature layout

Every selected primitive still emits 17 values:

| index | name | description |
|---|---|---|
| 0:3 | `rel_pos` | Position relative to the hips |
| 3:6 | `rel_lin_vel` | Velocity relative to the hips |
| 6:9 | `local_tan` | Primitive local X-axis |
| 9:12 | `local_norm` | Primitive local Z-axis |
| 12 | `radius` | Sphere/capsule radius |
| 13 | `extent_z` | Box height or capsule cylinder length |
| 14 | `damage` | Policy-visible threat scalar |
| 15:16 | `shape` | `[is_box, is_sphere]`; capsule is `[0, 0]` |

Output shape remains `[num_envs, K * 17]`.

## Configuration

Relevant `EnvConfig` fields:

- `max_collision_primitives`
- `collision_selection_range`
- `collision_distance_weight`
- `collision_closing_speed_weight`
- `collision_mass_weight`
- `collision_distance_scale`
- `collision_speed_scale`
- `collision_mass_scale`
- `static_collider_effective_mass`

`num_obs_primitives` (`K`) remains a static parameter of
`collision_primitives_obs_factory`.
