# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for the base environment.

This module defines the configuration dataclasses for environment settings,
rewards, terminations, and observation components.
"""

from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

from protomotions.envs.obs.scene_obs import SceneObsConfig
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.control.base import ControlComponentConfig

if TYPE_CHECKING:
    from protomotions.envs.mdp_component import MdpComponent


@dataclass
class RecoveryResetConfig:
    """Reset-time sampling from a cache of physically settled fall poses."""

    recovery_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability that a reset uses a fall pose.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    fall_sim_steps: int = field(
        default=150,
        metadata={
            "help": "Physics steps used to generate settled fall poses.",
            "min": 0,
        },
    )


@dataclass
class EnvConfig:
    """Main environment configuration."""

    max_episode_length: int = field(
        default=300,
        metadata={"help": "Maximum steps per episode before automatic reset.", "min": 1}
    )
    reset_grace_period: int = field(
        default=5,
        metadata={"help": "Steps after reset where grace period applies (for zeroing unreliable rewards).", "min": 0}
    )
    num_state_history_steps: int = field(
        default=0,
        metadata={"help": "Number of historical state steps to store. 0 = no history.", "min": 0}
    )

    _target_: str = "protomotions.envs.base_env.env.BaseEnv"

    scene_obs: SceneObsConfig = field(
        default_factory=SceneObsConfig,
        metadata={"help": "Scene observation configuration."}
    )

    motion_manager: MotionManagerConfig = field(
        default_factory=MotionManagerConfig,
        metadata={"help": "Motion manager for reference motion handling."}
    )

    ref_respawn_offset: float = field(
        default=0.05,
        metadata={"help": "Height offset for respawning relative to reference.", "min": 0.0}
    )
    ref_object_respawn_offset: float = field(
        default=0.0,
        metadata={"help": "Height offset for object respawning."}
    )
    ref_contact_smooth_window: int = field(
        default=0,
        metadata={"help": "Window length for smoothing contact labels. 0 = no smoothing.", "min": 0}
    )
    recovery_reset: RecoveryResetConfig = field(
        default_factory=RecoveryResetConfig,
        metadata={"help": "Optional reset sampling from generated fall poses."},
    )
    skip_correct_terrain_height_on_flat: bool = field(
        default=True,
        metadata={"help": "Skip terrain height correction when terrain is flat (optimization)."}
    )

    show_terrain_markers: bool = field(
        default=False,
        metadata={"help": "Show terrain markers during evaluation. Uses significant memory in IsaacGym."}
    )

    max_collision_primitives: int = field(
        default=16,
        metadata={
            "help": (
                "Capacity (M) of the per-env collision-primitive candidate buffer that "
                "feeds the collision_primitives observation. Ground, scene obstacles, "
                "thrown projectiles, and other characters' key bodies are written into "
                "this fixed-size buffer; the obs kernel selects the top-K priorities. Must "
                "be >= the obs K (num_obs_primitives). FROZEN once a tier is trained -- "
                "changing it changes the ONNX input shape."
            ),
            "min": 1,
        },
    )
    collision_selection_range: float = field(
        default=8.0,
        metadata={
            "help": (
                "Maximum center distance (m) from any character body for a "
                "collider to be eligible."
            ),
            "min": 0.0,
        },
    )
    collision_distance_weight: float = field(
        default=1.0,
        metadata={"help": "Priority weight for collider proximity.", "min": 0.0},
    )
    collision_closing_speed_weight: float = field(
        default=1.0,
        metadata={
            "help": "Priority weight for velocity toward any character body.",
            "min": 0.0,
        },
    )
    collision_mass_weight: float = field(
        default=0.25,
        metadata={"help": "Priority weight for physical collider mass.", "min": 0.0},
    )
    collision_distance_scale: float = field(
        default=2.0,
        metadata={"help": "Distance normalization scale (m).", "min": 1.0e-6},
    )
    collision_speed_scale: float = field(
        default=10.0,
        metadata={"help": "Closing-speed normalization scale (m/s).", "min": 1.0e-6},
    )
    collision_mass_scale: float = field(
        default=10.0,
        metadata={"help": "Mass normalization scale (kg).", "min": 1.0e-6},
    )
    static_collider_effective_mass: float = field(
        default=100.0,
        metadata={
            "help": "Finite effective mass (kg) used to rank static ground/obstacles.",
            "min": 0.0,
        },
    )
    ground_primitive_damage: float = field(
        default=0.0,
        metadata={
            "help": (
                "Baseline 'damage' value assigned to the ground collision primitive "
                "(index 14 of the 17-float layout). Ground is environmental, not a "
                "threat, so defaults to 0.0."
            )
        },
    )
    projectile_baseline_damage: float = field(
        default=0.1,
        metadata={
            "help": (
                "Baseline 'damage' for non-dangerous (passive) thrown projectiles. "
                "Matches the spec's static-prop baseline."
            )
        },
    )
    projectile_damage_range: Tuple[float, float] = field(
        default=(0.3, 1.0),
        metadata={
            "help": (
                "Range from which the 'damage' of the randomly-selected dangerous "
                "projectiles is sampled each episode."
            )
        },
    )
    projectile_dangerous_fraction_range: Tuple[float, float] = field(
        default=(0.0, 1.0),
        metadata={
            "help": (
                "Per-episode, a random fraction (sampled from this range) of the "
                "projectile pool is designated 'dangerous' and given randomized "
                "damage; the rest keep the baseline. This realizes 'randomize damage "
                "on a random number of colliders'."
            )
        },
    )
    randomize_body_stamina: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, per-episode randomize per-body 'stamina' in "
                "``body_stamina_range``. Stamina is that body's joint-stiffness "
                "scale: IsaacLab scales stiffness and torque capacity linearly, damping by sqrt(strength). "
                "Under BUILT_IN_PD the scaled gains are written into the engine on "
                "reset (``Simulator.set_joint_gain_scale``); it is also exposed as "
                "an observation. Off in early tiers (stamina == 1.0 == nominal)."
            )
        },
    )
    body_stamina_range: Tuple[float, float] = field(
        default=(0.75, 1.0),
        metadata={
            "help": (
                "Range from which per-body stamina (the joint-stiffness scale) is "
                "sampled when ``randomize_body_stamina`` is True. 1.0 == nominal "
                "gains; <1 weaker/more compliant, >1 stiffer."
            )
        },
    )
    # --- Multi-character self-play (Phase 5) -------------------------------------
    # The number of characters per scene (N) is set on the *simulator* config
    # (SimulatorConfig.num_characters); the env reads it from the simulator. These
    # fields tune the per-character geometry and opponent observation/reward. All
    # are no-ops when num_characters == 1.
    character_spawn_radius: float = field(
        default=1.0,
        metadata={
            "help": (
                "Radius (m) of the circle on which the N characters of a scene are "
                "spawned (multi-character self-play). Large enough to avoid initial "
                "interpenetration, small enough that characters can interact."
            )
        },
    )
    character_interaction_lookahead: float = field(
        default=1.0,
        metadata={
            "help": (
                "Seconds of sampled root motion used to predict an interaction "
                "point."
            ),
            "min": 0.0,
        },
    )
    character_spawn_radius_variance: float = field(
        default=0.2,
        metadata={
            "help": (
                "Fractional random variation applied to multi-character spawn "
                "distance."
            ),
            "min": 0.0,
        },
    )
    character_interaction_target_radius: float = field(
        default=0.2,
        metadata={
            "help": (
                "Per-character target jitter radius around the shared "
                "interaction point."
            ),
            "min": 0.0,
        },
    )
    character_min_spawn_separation: float = field(
        default=0.4,
        metadata={
            "help": "Minimum desired root separation at a multi-character reset.",
            "min": 0.0,
        },
    )
    opponent_key_body_names: List[str] = field(
        default_factory=lambda: [
            "Head",
            "Pelvis",
            ".*_Shoulder",
            ".*_Elbow",
            ".*_Knee",
            ".*_Ankle",
        ],
        metadata={
            "help": (
                "Regex patterns (fullmatch against robot body names) selecting which "
                "of an opponent's bodies are exposed as collision primitives to other "
                "characters. Defaults cover head/pelvis/arms/forearms/shins/feet."
            )
        },
    )
    opponent_body_radius: float = field(
        default=0.12,
        metadata={
            "help": (
                "Sphere radius (m) used to represent opponent key bodies in the "
                "collision-primitive observation."
            )
        },
    )
    striking_body_names: List[str] = field(
        default_factory=lambda: [".*_Hand", ".*_Ankle", ".*_Toe"],
        metadata={
            "help": (
                "Regex patterns (fullmatch) selecting the character's own bodies that "
                "earn the opponent-impact reward when they strike an opponent "
                "(hands/feet). These are the bodies excluded from the impact PENALTY."
            )
        },
    )
    opponent_strike_radius: float = field(
        default=0.3,
        metadata={
            "help": (
                "Distance (m) within which a striking body counts as 'impacting' an "
                "opponent key body for the opponent-impact reward."
            )
        },
    )

    sanitize_non_finite_state: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, tolerate rare non-finite simulator state (PhysX solver "
                "blowups from hard impacts) by sanitizing it and letting the normal "
                "termination path reset the affected env, instead of asserting and "
                "crashing. Intended for impact-heavy training (fighting tiers). "
                "Default False keeps strict fail-fast validation."
            )
        },
    )

    save_dir: str = field(
        default="",
        metadata={"help": "Directory for saving evaluation outputs."}
    )

    reward_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of named reward components. Each is a MdpComponent."}
    )
    
    control_components: Dict[str, ControlComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateful task/control managers."}
    )
    
    termination_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of termination functions. Each is a MdpComponent."}
    )
    
    observation_components: Dict[str, "MdpComponent"] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of observation functions. Each is a MdpComponent."}
    )

    action_config: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Single action processing config dict with 'fn' key. Use make_pd_action_config() helper."}
    )

    # Odometer corruption parameters.  Used by odom_offset_factory to
    # simulate per-session calibration error in the G1 leg-kinematics odometer.
    # The env samples odom_scale and a yaw_bias angle once per episode at reset.
    # Identity defaults mean no corruption when the factory is not used.
    odom_scale_range: tuple = field(
        default=(1.0, 1.0),
        metadata={
            "help": (
                "Per-episode odometer scale factor range (lo, hi) drawn from Uniform. "
                "Default (1.0, 1.0) = no scale corruption. "
                "Recommended for odom experiments: (0.7, 1.3)."
            )
        },
    )
    odom_yaw_range_deg: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-episode odometer yaw bias magnitude in degrees. "
                "The actual bias is drawn from Uniform(-deg, +deg). "
                "Default 0.0 = no yaw corruption. "
                "Recommended for odom experiments: 6.0."
            )
        },
    )
    odom_log_noise_std: float = field(
        default=0.0,
        metadata={
            "help": (
                "Per-step odometer noise std in log(1+mag) space. "
                "Default 0.0 = no per-step noise. "
                "Recommended for odom experiments: 0.12."
            )
        },
    )
    odom_soft_threshold: float = field(
        default=0.15,
        metadata={
            "help": (
                "Smooth noise ramp characteristic length in metres. "
                "Noise weight = mag / (mag + threshold). "
                "Default 0.15."
            )
        },
    )
