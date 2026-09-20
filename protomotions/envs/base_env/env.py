# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base environment implementation for reinforcement learning.

This module provides the foundational environment class for all RL tasks. It integrates
the simulator, handles robot state management, computes observations and rewards, manages
episode resets, and coordinates with terrain and scene systems.

Key Classes:
    - BaseEnv: Core environment class that all tasks inherit from

Key Features:
    - Multi-simulator support (IsaacGym, IsaacLab, Genesis)
    - Terrain integration for complex ground surfaces
    - Scene management for object interaction
    - Motion library integration for reference motions
    - Modular observation components

## BaseEnv

| Member | Type | Why Kept |
|--------|------|----------|
| `config` | `EnvConfig` | Core config, used everywhere |
| `robot_config` | `RobotConfig` | Core config, used everywhere |
| `device` | `torch.device` | Required for tensor creation |
| `terrain` | `Terrain` | Core dependency for terrain queries |
| `scene_lib` | `SceneLib` | Core dependency for scene/object handling |
| `motion_lib` | `MotionLib` | Core dependency for reference motions |
| `simulator` | `Simulator` | Core dependency for physics |
| `num_envs` | `int` | Frequently accessed, avoiding repeated `simulator.num_envs` |
| `max_episode_length` | `int` | Mutable - modified by agent for curriculum learning |
| `dt` | `float` | Frequently accessed, avoiding repeated `simulator.dt` |
| `rew_buf` | `Tensor` | Mutable buffer - accumulates rewards each step |
| `reset_buf` | `Tensor` | Mutable buffer - tracks which envs need reset |
| `progress_buf` | `Tensor` | Mutable buffer - tracks episode progress |
| `terminate_buf` | `Tensor` | Mutable buffer - tracks terminations |
| `extras` | `dict` | Mutable - collects per-step logging data |
| `respawn_root_offset` | `Tensor` | Mutable state - tracks spawn position offsets |
| `skip_height_correction` | `bool` | Performance optimization flag (read-only after init) |
| `motion_manager` | `MotionManager` | Core component for motion sampling |
| `motion_manager_disable_resample` | `bool` | Mutable flag - controlled by evaluator |
| `terrain_obs_cb` | `TerrainObs` | Observation component |
| `scene_obs_cb` | `SceneObs` | Observation component |

"""

from dataclasses import fields
from functools import cached_property
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

import torch
from torch import Tensor
from protomotions.utils.hydra_replacement import get_class

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    ObjectState,
    ResetState,
)
from protomotions.envs.terminations import check_max_length_term
from protomotions.envs.context_views import (
    EnvContext,
    CurrentStateView,
    HistoricalView,
    TerrainContext,
    SceneSurfaceContext,

    CollisionPrimitivesView,
)
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    apply_observation_noise,
    apply_reset_noise,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.scene_obs import SceneObs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.control.manager import ControlManager

# Component infrastructure for MdpComponent-based configs
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.base_env.utils import (
    combine_rewards,
    combine_terminations,
    compute_smooth_realign_offset,
)
from protomotions.components.pose_lib import build_body_ids_tensor

from protomotions.robot_configs.base import RobotConfig, ControlType

if TYPE_CHECKING:
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.motion_lib import MotionLib



class BaseEnv:
    """Base class for all reinforcement learning environments.

    Provides core functionality for robot simulation including:
    - Simulator integration (IsaacGym, IsaacLab, Genesis)
    - Terrain management
    - Scene and object handling
    - Motion library integration
    - Observation and reward computation
    - Episode management and resets

    Subclasses should implement task-specific reward functions and
    observation spaces by overriding compute_reward() and compute_observations().

    Attributes:
        simulator: The physics simulator instance.
        num_envs: Number of parallel environments.
        device: PyTorch device for computations.
        terrain: Terrain instance for complex ground surfaces.
        scene_lib: Library of object scenes for interaction tasks.
        motion_lib: Library of reference motions for imitation tasks.

    Example:
        >>> config = SteeringEnvConfig()
        >>> robot_config = G1Config()
        >>> env = Steering(config, robot_config, simulator_config, device)
        >>> obs, _ = env.reset()
        >>> next_obs, rewards, dones, info = env.step(action_dict)
    """

    num_characters = 1
    _extended_observations_enabled = False

    def __init__(
        self,
        config: EnvConfig,
        robot_config: RobotConfig,
        device: torch.device,
        terrain: "Terrain",
        simulator: Simulator,
        scene_lib: "SceneLib",
        motion_lib: "MotionLib",
        *args,
        **kwargs,
    ):
        """Initialize BaseEnv.

        Args:
            config: Environment configuration
            robot_config: Robot configuration
            device: Device for computation
            terrain: Pre-created Terrain object (always provided, can be None for visualizers)
            simulator: Pre-created Simulator shell (not yet initialized, will be initialized by env)
            scene_lib: Pre-created SceneLib (always provided, empty if no scenes)
            motion_lib: Pre-created MotionLib (always provided, empty if no motions)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.config = config
        # Pickled resolved configs predate newly-added dataclass fields and bypass
        # dataclass initialization when unpickled. Backfill defaults so old fighting
        # checkpoints remain usable with newer environment code.
        default_config = EnvConfig()
        for config_field in fields(EnvConfig):
            if not hasattr(self.config, config_field.name):
                setattr(
                    self.config,
                    config_field.name,
                    getattr(default_config, config_field.name),
                )
        self.robot_config = robot_config
        self.device = device
        self.terrain = terrain
        self.scene_lib = scene_lib
        self.motion_lib = None
        self.simulator = simulator
        from protomotions.simulator.base_simulator.simulator_state import set_sanitize_non_finite_state
        set_sanitize_non_finite_state(self.config.sanitize_non_finite_state)
        # Multi-character self-play: ``num_envs`` is the flattened per-character row
        # count (E * N) seen by the RL agent; ``num_physical_envs`` (E) counts shared
        # physical scenes / projectile pools. N == 1 makes them equal (legacy).
        self.num_envs = simulator.num_envs
        self.num_physical_envs = getattr(simulator, "num_physical_envs", self.num_envs)
        self.num_characters = getattr(simulator, "num_characters", 1)
        self._extended_observations_enabled = any(
            key in config.observation_components
            for key in ("collision_primitives", "stamina_obs")
        )

        self.max_episode_length = self.config.max_episode_length

        # Buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.respawn_root_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )
        self._fall_reset_states = None

        # Per-episode odometer corruption parameters.
        # Sampled once at episode reset; held constant within the episode.
        # Identity values (scale=1, yaw_bias=0) until first reset.
        self.odom_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.odom_yaw_cos_sin = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.odom_yaw_cos_sin[:, 0] = 1.0  # cos(0) = 1
        # odom_start_xy: robot anchor XY at episode start, for distance-from-start.
        # odom_start_heading_inv: inverse heading quat at episode start.
        self.odom_start_xy = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.odom_start_heading_inv = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device
        )
        self.odom_start_heading_inv[:, 3] = 1.0

        # Magnitude (m) the reference XY offset was shifted this step by smooth
        # re-anchoring. Penalized (realign_penalty reward) so the policy treats
        # re-anchoring as a safety net rather than a crutch. Stays 0 when re-anchoring
        # is disabled (e.g. the Tier 1 base), making the penalty a no-op there.
        self._realign_offset_delta = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        # Contact force tracking for impact penalty rewards
        # Initialized properly after simulator init when we know num_bodies
        self.prev_contact_force_magnitudes = None

        # Action buffers (current step only; previous actions come from state_history)
        num_actions = robot_config.number_of_actions
        self._current_raw_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )
        self._current_processed_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )

        # Per-DOF "stamina" scale applied to PD gains (1.0 == full strength).
        # Default of ones reproduces fixed-gain behaviour; components (e.g. the
        # stamina manager in a later tier) overwrite this per-env, per-DOF.
        self._dof_stamina_scale = torch.ones(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )
        # Runtime kwargs merged into the action function each step (see
        # _process_action). Keys must match the action config's declared params.
        self._runtime_action_inputs: Dict[str, Tensor] = {}

        # Global context cache - built once per step in post_physics_step
        # and reused by observations, rewards, and terminations
        self._current_context: Dict[str, Any] = None

        # Noisy observation cache - computed once in post_physics_step,
        # reused by both state_history and _build_global_context
        self._current_noisy_obs = None

        self.skip_height_correction = (
            self.config.skip_correct_terrain_height_on_flat and self.terrain.is_flat()
        )

        self.dt = self.simulator.dt
        self.install_motion_lib(motion_lib)
        self.initialize_simulator()

    def initialize_simulator(self):
        """Initialize simulator with task-specific visualization markers.

        Called at the end of __init__ to finalize simulator setup after visualization
        markers have been created (potentially by child env class override).
        """
        if (
            hasattr(self.robot_config, "kinematic_info")
            and self.robot_config.kinematic_info is not None
        ):
            self.robot_config.kinematic_info.to(self.device)

        # Initialize contact force buffer now that we know num_bodies
        num_bodies = self.robot_config.kinematic_info.num_bodies
        self.prev_contact_force_magnitudes = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.float, device=self.device
        )

        if self.config.num_state_history_steps > 0:
            # Check if observation noise is configured - if so, allocate noisy buffers
            store_noisy = (
                self.simulator.config.domain_randomization is not None
                and self.simulator.config.domain_randomization.observation_noise
                is not None
                and self.simulator.config.domain_randomization.observation_noise.has_noise()
            )
            self.state_history = StateHistoryBuffer(
                num_envs=self.num_envs,
                num_history_steps=self.config.num_state_history_steps,
                num_bodies=num_bodies,
                num_dofs=self.robot_config.kinematic_info.num_dofs,
                action_dim=self.robot_config.number_of_actions,
                num_contact_bodies=len(self.contact_body_ids),
                anchor_body_index=self.robot_config.anchor_body_index,
                device=self.device,
                store_noisy=store_noisy,
            )
        else:
            self.state_history = None

        self.terrain_obs_cb = TerrainObs(self.terrain.config, self)
        self.scene_obs_cb = SceneObs(self.config.scene_obs, self)

        self._key_bindings = self.simulator.user_interface.scope("env")
        self._key_bindings.register("R", "reset", "Reset all environments")
        self.control_manager = ControlManager(self.config.control_components, self)

        visualization_markers = self.create_visualization_markers(
            self.simulator.headless
        )
        self.simulator._initialize_with_markers(visualization_markers)

        # Component infrastructure for MdpComponent
        self._component_manager = ComponentManager(self.device)
        collision_component = self.config.observation_components.get(
            "collision_primitives"
        )
        if collision_component is not None:
            collision_component.dynamic_vars["primitive_mass"] = (
                EnvContext.collision_primitives.mass
            )
            collision_component.static_params.update(
                {
                    "selection_range": self.config.collision_selection_range,
                    "distance_weight": self.config.collision_distance_weight,
                    "closing_speed_weight": self.config.collision_closing_speed_weight,
                    "mass_weight": self.config.collision_mass_weight,
                    "distance_scale": self.config.collision_distance_scale,
                    "speed_scale": self.config.collision_speed_scale,
                    "mass_scale": self.config.collision_mass_scale,
                }
            )
        self._observation_buffer: Dict[str, Tensor] = {}

        recovery_config = getattr(self.config, "recovery_reset", None)
        if recovery_config is not None and recovery_config.recovery_prob > 0:
            self._generate_fall_reset_states()

        if self._extended_observations_enabled:
            # Projectile bookkeeping for the collision-primitive layer: per-env, per-
            # projectile "damage" metadata (randomized per episode) and the previous
            # positions used to derive projectile velocity by finite difference. Must be
            # set up before _initialize_observations() so the collision-primitive and
            # stamina observation components see valid buffers on their first compute.
            # Projectiles are a per-physical-scene resource, so these buffers are sized by
            # num_physical_envs (E), not the flattened per-character count.
            self._num_projectiles = self.simulator.num_projectiles
            self._projectile_damage = torch.full(
                (self.num_physical_envs, self._num_projectiles),
                self.config.projectile_baseline_damage,
                device=self.device,
            )
            self._prev_projectile_pos = torch.zeros(
                self.num_physical_envs, self._num_projectiles, 3, device=self.device
            )
            self._prev_projectile_active = torch.zeros(
                self.num_physical_envs, self._num_projectiles, device=self.device
            )
            self._projectile_velocity = torch.zeros_like(self._prev_projectile_pos)
            self._randomize_projectile_damage(
                torch.arange(self.num_physical_envs, device=self.device)
            )

            self._init_body_stamina()
            self._init_multi_character()

        # Initialize observations
        self._initialize_observations()

    def _validate_motion_lib_compatibility(self, motion_lib=None):
        """Validate that the motion file is compatible with the robot config."""
        if motion_lib is None:
            motion_lib = self.motion_lib
        ki = self.robot_config.kinematic_info
        expected_dofs = ki.num_dofs
        expected_bodies = ki.num_bodies

        sample_state = motion_lib.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.zeros(1, device=self.device),
        )
        motion_dofs = sample_state.dof_pos.shape[1]
        motion_bodies = sample_state.rigid_body_pos.shape[1]

        if motion_dofs != expected_dofs or motion_bodies != expected_bodies:
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"MOTION FILE / ROBOT MISMATCH\n"
                f"{'=' * 70}\n"
                f"Motion file has {motion_dofs} DOFs and {motion_bodies} bodies,\n"
                f"but robot '{type(self.robot_config).__name__}' expects "
                f"{expected_dofs} DOFs and {expected_bodies} bodies.\n\n"
                f"The motion file was likely generated for a different robot.\n"
                f"Make sure --motion-file matches the robot in your "
                f"checkpoint/config.\n"
                f"{'=' * 70}"
            )

    ###############################################################
    # Getters
    ###############################################################
    def is_simulation_running(self):
        """Check if the physics simulation is running.

        Returns:
            Boolean indicating simulation state
        """
        return self.simulator.is_simulation_running()

    def get_obs(self):
        """Gather observations from all components.

        Returns:
            Dictionary of observation tensors from humanoid, terrain, scene,
            and dynamic observation components
        """
        obs = {}
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        if self.scene_lib.num_scenes() > 0 and self.config.scene_obs.enabled:
            scene_obs = self.scene_obs_cb.get_obs()
            obs.update(scene_obs)

        # Get dynamic observations
        dynamic_obs = {
            name: tensor.clone() for name, tensor in self._observation_buffer.items()
        }
        obs.update(dynamic_obs)

        return obs

    def get_action_size(self):
        """Get the dimensionality of the action space.

        Returns:
            Number of action dimensions
        """
        return self.simulator.num_act

    def consume_reset_request(self) -> bool:
        """Return and consume a user-interface reset request."""
        return self._key_bindings.reset.consume()

    ###############################################################
    # Component Processing
    ###############################################################
    def _initialize_observations(self):
        """Initialize observation buffers."""
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._process_observations(self.context, all_env_ids)

    def _process_observations(self, context: EnvContext, env_ids: Tensor):
        """Process observations using MdpComponent."""
        raw_obs = self._component_manager.execute_all(
            components=self.config.observation_components,
            ctx=context,
        )

        # Update observation buffer with results
        context_env_ids = getattr(context, "env_ids", None)
        if context_env_ids is not None and not torch.equal(context_env_ids, env_ids):
            raise ValueError(
                "Observation context env_ids must match the destination env_ids"
            )

        for name, obs_value in raw_obs.items():
            expected_rows = (
                len(env_ids) if context_env_ids is not None else self.num_envs
            )
            if obs_value.dim() == 0 or obs_value.shape[0] != expected_rows:
                raise ValueError(
                    f"Observation component '{name}' must return {expected_rows} rows "
                    f"for a {'subset' if context_env_ids is not None else 'global'} "
                    f"context; got {obs_value.shape[0] if obs_value.dim() else 0}"
                )

            if name not in self._observation_buffer:
                self._observation_buffer[name] = torch.zeros(
                    self.num_envs,
                    *obs_value.shape[1:],
                    dtype=obs_value.dtype,
                    device=self.device,
                )
            rows = obs_value if context_env_ids is not None else obs_value[env_ids]
            self._observation_buffer[name][env_ids] = rows

    def _process_rewards(
        self, context: EnvContext, grace_mask: Optional[Tensor] = None
    ):
        """Process rewards using MdpComponent."""
        raw_rewards = self._component_manager.execute_all(
            components=self.config.reward_components,
            ctx=context,
        )

        return combine_rewards(
            raw_rewards=raw_rewards,
            configs=self.config.reward_components,
            grace_mask=grace_mask,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _process_terminations(self, context: EnvContext):
        """Process terminations using MdpComponent."""
        raw_terms = self._component_manager.execute_all(
            components=self.config.termination_components,
            ctx=context,
        )

        return combine_terminations(
            raw_terms=raw_terms,
            configs=self.config.termination_components,
            num_envs=self.num_envs,
            device=self.device,
        )

    _action_config_device_ready: bool = False

    def set_dof_stamina_scale(
        self, stamina: Tensor, env_ids: Optional[Tensor] = None
    ) -> None:
        """Set the per-DOF stamina scale applied to PD gains (legacy PROPORTIONAL).

        Stamina scales the joint drive (``1.0`` == nominal). The value is forwarded
        to the action function via ``_runtime_action_inputs`` under the ``"stamina"``
        key, so it only takes effect when the active action config declares a
        ``stamina`` parameter (e.g. ``make_pd_stamina_action_config``) and the robot
        uses ``ControlType.PROPORTIONAL``. For the default BUILT_IN_PD path stamina
        is applied via ``Simulator.set_joint_gain_scale`` instead (see
        ``_apply_body_stamina_to_gains``).

        Args:
            stamina: Per-DOF stamina, shape [num_envs, num_actions] or
                [len(env_ids), num_actions].
            env_ids: Optional subset of environments to update.
        """
        if env_ids is None:
            self._dof_stamina_scale[:] = stamina
        else:
            self._dof_stamina_scale[env_ids] = stamina
        self._runtime_action_inputs["stamina"] = self._dof_stamina_scale

    def _process_action(self, action: Tensor, context: EnvContext) -> Dict[str, Tensor]:
        """Process action using single action config dict.

        action_config is a single dict with "fn" key and parameters.
        """
        if self.config.action_config is None:
            return {"processed_action": action}

        # Lazy device migration on first call
        if not self._action_config_device_ready:
            for key, val in self.config.action_config.items():
                if isinstance(val, torch.Tensor):
                    self.config.action_config[key] = val.to(action.device)
            self._action_config_device_ready = True

        fn = self.config.action_config["fn"]
        # Extract all params except "fn"
        params = {k: v for k, v in self.config.action_config.items() if k != "fn"}
        # Runtime overrides injected by components (e.g. per-env, per-DOF "stamina").
        # Only override keys the action config already declares, so we never pass an
        # unexpected kwarg to the action function.
        for key, value in getattr(self, "_runtime_action_inputs", {}).items():
            if key in params:
                params[key] = value
        params["action"] = action
        return fn(**params)

    ###############################################################
    # Cached Properties
    ###############################################################
    @cached_property
    def contact_body_ids(self) -> torch.Tensor:
        """Body indices for contact sensing."""
        return build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.robot_config.contact_bodies,
            self.device,
        )

    @cached_property
    def non_termination_contact_body_ids(self) -> torch.Tensor:
        """Body indices that don't trigger termination on contact."""
        body_names = self.robot_config.kinematic_info.body_names
        if self.robot_config.non_termination_contact_bodies == "all":
            return build_body_ids_tensor(body_names, body_names, self.device)
        else:
            return build_body_ids_tensor(
                body_names,
                self.robot_config.non_termination_contact_bodies,
                self.device,
            )

    @cached_property
    def default_reset_state(self) -> ResetState:
        """Default robot reset state from simulator."""
        return self.simulator.get_default_robot_reset_state()

    @cached_property
    def default_object_state(self) -> ObjectState:
        """Default object state (empty if no scenes)."""
        return self.scene_lib.get_default_object_state(self.device)

    def update_respawn_root_offset_by_env_ids(
        self,
        env_ids,
        ref_state: Optional[RobotState] = None,
        sample_flat: bool = False,
    ) -> torch.Tensor:
        """
        Samples a new starting position for the environment.
        And obtains the root translation offset relative to the reference state.

        This method considers both scene and terrain requirements.

        When a scene is required for obj interaction,
        the character is spawned relative to the scene's position.

        For environments without a scene, a random valid coordinate is sampled,
        and non-negative vertical offset is added based on terrain height.

        During co-training, scene groups use flat terrain, but during
        inference the resolved terrain may be complex (with negative heights
        that get normalised).  Height correction is applied to both scene
        and non-scene envs unless the terrain is entirely flat.

        """

        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)

        # Get boolean masks for scene vs non-scene envs
        scene_mask, non_scene_mask = self.get_scene_non_scene_mask(env_ids)

        if scene_mask.any():
            scene_pos = self.scene_lib.get_scene_positions(self.terrain, self.device)
            respawn_offset[scene_mask, :2] = scene_pos[env_ids[scene_mask], :2]

            # Scene envs also need terrain height correction — the object
            # playground is flat at height-field 0, but terrain normalisation
            # (shifting min height to z=0) can raise the playground above
            # world z=0.  Without correction the agent spawns underground.
            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[scene_mask, 2] = terrain_heights

        if non_scene_mask.any():
            num_non_scene = non_scene_mask.sum().item()
            respawn_position_xy = self.terrain.sample_valid_locations(
                num_envs=num_non_scene, sample_flat=sample_flat
            )

            if ref_state is None:
                ref_root = torch.zeros((num_non_scene, 2), device=self.device)
            else:
                ref_root = ref_state.root_pos[non_scene_mask, :2]
            respawn_offset[non_scene_mask, :2] = respawn_position_xy - ref_root

            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[non_scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        non_scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[non_scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[non_scene_mask, 2] = terrain_heights

        respawn_offset[:, 2] += self.config.ref_respawn_offset

        # Multi-character self-play: the N characters sharing a physical scene must
        # spawn at the SAME sampled location (they are then separated only by the
        # small per-character spawn circle). Broadcast each scene's first-row offset
        # to all of that scene's character rows so they end up co-located and able to
        # interact, rather than scattered to independent random terrain locations.
        if self.num_characters > 1:
            phys = env_ids // self.num_characters
            uniq, inverse = torch.unique(phys, return_inverse=True, sorted=True)
            order = torch.arange(env_ids.numel(), device=self.device)
            first_idx = torch.full(
                (uniq.numel(),),
                env_ids.numel(),
                dtype=torch.long,
                device=self.device,
            )
            first_idx = first_idx.scatter_reduce(
                0, inverse, order, reduce="amin", include_self=True
            )
            shared_offset = respawn_offset[first_idx[inverse]]
            respawn_offset[:, :2] = shared_offset[:, :2]

        self.respawn_root_offset[env_ids] = respawn_offset

    def _place_multi_character_reference_states(
        self, env_ids: Tensor, ref_state: RobotState
    ) -> None:
        """Translate sampled roots so their near-future paths approach one point."""
        if self.num_characters <= 1 or env_ids.numel() == 0:
            return

        N = self.num_characters
        if env_ids.numel() % N != 0:
            raise ValueError("Multi-character reference placement requires full scenes")
        rows = env_ids.view(-1, N)
        num_scenes = rows.shape[0]

        lookahead_jitter = self.config.character_spawn_radius_variance
        scene_lookahead = self.config.character_interaction_lookahead * (
            1.0
            + (torch.rand(num_scenes, 1, device=self.device) * 2.0 - 1.0)
            * lookahead_jitter
        )
        lookahead = scene_lookahead.expand(-1, N).reshape(-1).clamp_min(0.0)
        future_times = self.motion_manager.motion_times[env_ids] + lookahead
        motion_lengths = self.motion_lib.get_motion_length(
            self.motion_manager.motion_ids[env_ids]
        )
        future_times = torch.minimum(future_times, motion_lengths)
        future_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids], future_times
        )

        current_xy = ref_state.root_pos[:, :2]
        displacement = future_state.root_pos[:, :2] - current_xy
        displacement_norm = displacement.norm(dim=-1, keepdim=True)

        from protomotions.utils.rotations import calc_heading

        heading = calc_heading(ref_state.root_rot, w_last=True)
        facing = torch.stack((torch.cos(heading), torch.sin(heading)), dim=-1)
        fallback_distance = torch.full_like(
            displacement_norm, self.config.character_spawn_radius
        )
        approach = torch.where(
            displacement_norm > 0.05,
            displacement,
            facing * fallback_distance,
        )

        max_spawn_distance = self.config.character_spawn_radius * (
            1.0 + self.config.character_spawn_radius_variance
        )
        approach_norm = approach.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        approach = approach * torch.clamp(
            max_spawn_distance / approach_norm, max=1.0
        )

        # The first sampled row already identifies the valid terrain anchor.
        anchored_xy = current_xy + self.respawn_root_offset[env_ids, :2]
        interaction_center = anchored_xy.view(num_scenes, N, 2)[:, 0, :]

        char_ids = torch.arange(N, device=self.device, dtype=torch.float)
        phase = torch.rand(num_scenes, 1, device=self.device) * (2.0 * torch.pi)
        angles = phase + char_ids.unsqueeze(0) * (2.0 * torch.pi / N)
        ring = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        chord_factor = max(
            2.0 * torch.sin(torch.tensor(torch.pi / N)).item(), 1.0e-6
        )
        min_target_radius = self.config.character_min_spawn_separation / chord_factor
        target_radius = max(
            self.config.character_interaction_target_radius, min_target_radius
        )
        target_xy = interaction_center.unsqueeze(1) + ring * target_radius

        desired_xy = target_xy.reshape(-1, 2) - approach
        self.respawn_root_offset[env_ids, :2] = desired_xy - current_xy

    def align_motion_with_humanoid(self, env_ids, root_pos):
        """Compute XY offset between humanoid spawn position and reference motion data.

        Args:
            env_ids: Environment indices to align
            root_pos: Desired root positions [len(env_ids), 3]
        """
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )

        self.respawn_root_offset[env_ids, :2] = (
            root_pos[:, :2] - ref_state.rigid_body_pos[:, 0, :2]
        )

    def update_smooth_motion_alignment(self, env_ids, dt: float):
        """Smoothly re-anchor the reference XY offset toward the character.

        Unlike ``align_motion_with_humanoid`` (an instant XY snap used at reset), this
        blends ``respawn_root_offset`` toward the snap target with a strength
        proportional to the "unexpected" root XY velocity (character vs. reference clip
        velocity). See ``compute_smooth_realign_offset`` and the fighting-mimic
        deployment doc for the exact contract.

        Args:
            env_ids: Environment indices to re-anchor.
            dt: Control timestep (s), used to cap offset drift per step.
        """
        cfg = self.motion_manager.config
        root_state = self.simulator.get_root_state(env_ids)
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )

        prev_offset_xy = self.respawn_root_offset[env_ids, :2].clone()
        new_offset_xy = compute_smooth_realign_offset(
            current_root_xy=root_state.root_pos[:, :2],
            ref_root_xy=ref_state.rigid_body_pos[:, 0, :2],
            current_root_xy_vel=root_state.root_vel[:, :2],
            ref_root_xy_vel=ref_state.rigid_body_vel[:, 0, :2],
            prev_offset_xy=prev_offset_xy,
            alpha_min=cfg.realign_alpha_min,
            alpha_max=cfg.realign_alpha_max,
            vel_err_low=cfg.realign_vel_err_low,
            vel_err_high=cfg.realign_vel_err_high,
            max_xy_speed=cfg.realign_max_xy_speed,
            dt=dt,
        )
        self.respawn_root_offset[env_ids, :2] = new_offset_xy
        # Record how far the reference was shifted this step (fed to realign_penalty).
        self._realign_offset_delta[env_ids] = torch.linalg.norm(
            new_offset_xy - prev_offset_xy, dim=-1
        )

    def get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        self, target_pos: Tensor, env_ids: Optional[Tensor] = None
    ) -> Tensor:
        """Compute spawn offset with terrain height correction for reference poses.

        Used by motion tracking tasks to correctly position reference poses in the environment,
        accounting for both XY spawn offset and terrain height.

        Args:
            target_pos: Reference body positions [num_envs, num_bodies, 3]
                       without spawning offset applied.
            env_ids: Environment indices [num_envs]. If None, uses all envs.

        Returns:
            Offset to add to target_pos [num_envs, num_bodies, 3].

        Note:
            - For XY offset: all bodies share the same respawn_root_offset
            - For Z offset: all bodies share the same offset computed from
              the body furthest below terrain
            - This preserves the rigid body structure during spawning
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        new_offset = torch.zeros_like(target_pos)
        new_offset[:, :, :2] = self.respawn_root_offset[env_ids, :2][:, None, :]

        if not self.skip_height_correction:
            target_pos_spawned = target_pos.clone() + new_offset
            z_offset = self.terrain.find_terrain_height_for_max_below_body(
                target_pos_spawned
            )
            new_offset[:, :, 2] = z_offset.unsqueeze(1)

        return new_offset

    def get_scene_non_scene_mask(self, env_ids):
        """
        Returns boolean masks indicating which envs require a scene and which don't.

        Args:
            env_ids: Environment IDs to check

        Returns:
            scene_mask: Boolean tensor (len(env_ids),) - True for scene envs
            non_scene_mask: Boolean tensor (len(env_ids),) - True for non-scene envs

        Note: For now assumes either all or none require a scene
        """
        num_envs = len(env_ids)
        if self.scene_lib.num_scenes() > 0:
            scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        else:
            scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        return scene_mask, non_scene_mask

    def get_markers_state(self):
        """Compute visualization marker positions for rendering.

        Returns:
            Dictionary mapping marker names to MarkerState objects
        """
        if self.simulator.headless:
            return {}

        markers_state = {}

        # Update terrain markers
        if self.config.show_terrain_markers:
            height_maps = self.terrain.get_height_maps(
                self.simulator.get_root_state(), None, return_all_dims=True
            ).view(self.num_envs, -1, 3)
            markers_state["terrain_markers"] = MarkerState(
                translation=height_maps,
                orientation=torch.zeros(
                    self.num_envs, height_maps.shape[1], 4, device=self.device
                ),
            )

        # Merge markers from control components
        control_markers_state = self.control_manager.get_markers_state()
        markers_state.update(control_markers_state)

        return markers_state

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, action: Tensor):
        """Step the environment forward one timestep.

        Args:
            action: Raw action tensor from the policy [num_envs, num_actions]

        Returns:
            obs, rewards, dones, terminated, extras
        """
        self.extras = {}

        # Invalidate cached context - will be rebuilt after physics in post_physics_step
        self._current_context = None
        self._current_noisy_obs = None

        # Store current actions
        self._current_raw_action[:] = action

        # Process action
        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"]
        self._current_processed_action[:] = processed_action

        # Forward per-step PD gains to the simulator. These are only consumed by
        # PROPORTIONAL control (ignored otherwise), and enable runtime per-env,
        # per-DOF gain modulation (the "stamina" feature).
        gain_kwargs = {}
        if "stiffness_targets" in action_dict:
            gain_kwargs["stiffness"] = action_dict["stiffness_targets"]
            gain_kwargs["damping"] = action_dict["damping_targets"]
        self.simulator.step(
            processed_action, markers_callback=self.get_markers_state, **gain_kwargs
        )

        self.post_physics_step()

        if self.consume_reset_request():
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras

    def on_epoch_end(self, current_epoch: int):
        """Hook called at end of each training epoch. Override in subclasses if needed.

        Args:
            current_epoch: Current epoch number
        """
        pass

    def post_physics_step(self):
        """Update environment state after physics simulation step.

        Increments progress counter, updates motion manager, computes observations and rewards,
        checks for resets, and stores raw robot state in extras for logging.
        """
        self.progress_buf += 1
        if self._extended_observations_enabled and self._num_projectiles > 0:
            self._update_projectile_velocity()

        if self.state_history is not None:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[:, self.robot_config.anchor_body_index]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[
                :, self.contact_body_ids
            ].bool()

            # Compute noisy versions if observation noise is configured and history stores noisy data
            noisy_kwargs = {}
            if self.state_history.store_noisy:
                obs_noise_cfg = (
                    self.simulator.config.domain_randomization.observation_noise
                )

                # Single source of truth: uniform noise via apply_observation_noise
                noisy = apply_observation_noise(
                    obs_noise_cfg=obs_noise_cfg,
                    robot_state=current_state,
                    anchor_idx=self.robot_config.anchor_body_index,
                    ground_heights=ground_heights,
                )
                self._current_noisy_obs = noisy

                # Extract noisy tensors for history buffer
                noisy_kwargs["noisy_rigid_body_pos"] = noisy.rigid_body_pos
                noisy_kwargs["noisy_rigid_body_rot"] = noisy.rigid_body_rot
                noisy_kwargs["noisy_rigid_body_vel"] = noisy.rigid_body_vel
                noisy_kwargs["noisy_rigid_body_ang_vel"] = noisy.rigid_body_ang_vel
                noisy_kwargs["noisy_dof_pos"] = noisy.dof_pos
                noisy_kwargs["noisy_dof_vel"] = noisy.dof_vel
                noisy_kwargs["noisy_ground_heights"] = noisy.ground_heights

            self.state_history.rotate_and_update(
                rigid_body_pos=current_state.rigid_body_pos,
                rigid_body_rot=current_state.rigid_body_rot,
                rigid_body_vel=current_state.rigid_body_vel,
                rigid_body_ang_vel=current_state.rigid_body_ang_vel,
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                actions=self._current_raw_action,
                ground_heights=ground_heights,
                body_contacts=body_contacts,
                processed_actions=self._current_processed_action,
                **noisy_kwargs,
            )

        if self.motion_manager is not None and hasattr(
            self.motion_manager, "post_physics_step"
        ):
            self.motion_manager.post_physics_step()

        self.control_manager.step()

        if (
            self.motion_manager is not None
            and self.motion_manager.config.realign_motion_with_humanoid_on_each_step
        ):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if getattr(self.motion_manager.config, "smooth_realign_enabled", False):
                self.update_smooth_motion_alignment(env_ids, dt=self.simulator.dt)
            else:
                self.align_motion_with_humanoid(env_ids, self.simulator.get_root_state().root_pos)

        # Build context once and reuse for observations, rewards, and terminations
        self._current_context = self._build_global_context()

        self.compute_observations(context=self._current_context)
        self.compute_reward(context=self._current_context)
        self.reset_buf[:], self.terminate_buf[:] = self.check_resets_and_terminations(
            context=self._current_context
        )

        self.extras["terminate"] = self.terminate_buf

        rbs: RobotState = self.simulator.get_robot_state()
        for k, _ in rbs.get_shape_mapping(flattened=True).items():
            self.extras[f"raw/{k}"] = rbs.flatten_bodies(k)

        # Preserve main's contact-force reward inputs.
        self.prev_contact_force_magnitudes[:] = torch.norm(
            rbs.rigid_body_contact_forces, dim=-1
        )

    def user_reset(self):
        """Force environments to reset on next check (triggered by user input)."""
        self.progress_buf[:] = 100000000000

    def compute_observations(self, env_ids=None, context: EnvContext = None):
        """Compute observations for specified environments.

        Args:
            env_ids: Environment indices to update (None = all environments)
            context: Pre-built EnvContext from self.context property.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if context is None:
            raise ValueError("context is required - use self.context to build it")

        # Process dynamic observations
        self._process_observations(context, env_ids)

        self.terrain_obs_cb.compute_observations(env_ids)
        if self.scene_lib.num_scenes() > 0:
            self.scene_obs_cb.compute_observations(env_ids)

    def _expand_to_physical_scenes(self, env_ids: Tensor) -> Tensor:
        """Expand flattened character rows to every sibling in their scenes."""
        if self.num_characters <= 1 or env_ids.numel() == 0:
            return env_ids
        physical_ids = torch.unique(env_ids // self.num_characters, sorted=True)
        character_ids = torch.arange(
            self.num_characters, device=self.device, dtype=torch.long
        )
        return (
            physical_ids.unsqueeze(1) * self.num_characters
            + character_ids.unsqueeze(0)
        ).reshape(-1)

    def _couple_scene_flags(self, flags: Tensor) -> Tensor:
        """Make a per-row flag true for all siblings when any scene row is true."""
        if self.num_characters <= 1:
            return flags
        scene_flags = flags.view(
            self.num_physical_envs, self.num_characters
        ).any(dim=1, keepdim=True)
        return scene_flags.expand(-1, self.num_characters).reshape(-1)

    def check_resets_and_terminations(self, context: EnvContext):
        """Check reset and termination conditions.

        Only handles max episode length directly. All other terminations
        (including height/fall termination) should be configured via:
        - termination_components (dynamic termination system)
        - control_components (task-specific terminations)

        Args:
            context: Pre-built context from self.context property.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        max_length_reached = check_max_length_term(
            self.progress_buf, self.max_episode_length
        )
        reset_buf = max_length_reached.clone()
        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        comp_reset, comp_terminate = (
            self.control_manager.check_resets_and_terminations()
        )
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate

        # Process terminations
        comp_reset, comp_terminate, term_logging = self._process_terminations(context)
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate
        reset_buf = self._couple_scene_flags(reset_buf)
        terminated = self._couple_scene_flags(terminated)
        self.extras.update(term_logging)

        return reset_buf, terminated

    ###############################################################
    # Dynamic Reward System
    ###############################################################
    @property
    def context(self) -> EnvContext:
        """Get global context for observation/reward/termination evaluation.

        Returns cached context from _current_context if set (after post_physics_step),
        otherwise builds a fresh context.

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        if self._current_context is None:
            self._current_context = self._build_global_context()
        return self._current_context

    def _select_noisy_observations(
        self, noisy: NoisyObservations, env_ids: Optional[Tensor]
    ) -> NoisyObservations:
        if env_ids is None:
            return noisy
        return noisy.select(env_ids)

    @staticmethod
    def _select_context_tensor(
        tensor: Optional[Tensor], env_ids: Optional[Tensor]
    ) -> Optional[Tensor]:
        if tensor is None or env_ids is None:
            return tensor
        return tensor[env_ids]

    def _build_context(self, env_ids: Optional[Tensor] = None) -> EnvContext:
        """Build a fresh full or subset context.

        Creates typed EnvContext with view wrappers around existing data structures.
        Controllers populate their task-specific views via populate_context().

        When observation noise is configured:
        - noisy views have noise applied
        - current views contain clean data

        When no observation noise is configured:
        - Both point to the same tensors (memory efficient)

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        if env_ids is not None:
            env_ids = env_ids.to(self.device)
        current_state = self.simulator.get_robot_state(env_ids)
        anchor_idx = self.robot_config.anchor_body_index

        ground_heights = self.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, anchor_idx]
        ).squeeze(-1)

        body_contacts = current_state.rigid_body_contacts[
            :, self.contact_body_ids
        ].bool()

        # Preserve main's contact-force reward inputs.
        current_contact_force_magnitudes = torch.norm(
            current_state.rigid_body_contact_forces, dim=-1
        )

        # Use cached noisy obs from post_physics_step when available.
        # During init/reset the cache is None — use clean (no-noise) fallback.
        current_view = CurrentStateView(current_state, anchor_idx)
        if self._current_noisy_obs is not None:
            noisy = self._select_noisy_observations(self._current_noisy_obs, env_ids)
            noisy_view = CurrentStateView(noisy, anchor_idx)
            noisy_ground_heights = noisy.ground_heights
        else:
            noisy_view = current_view
            noisy_ground_heights = ground_heights

        context_num_envs = current_state.root_pos.shape[0]
        scene_surface_context = self._build_scene_surface_context(
            env_ids, context_num_envs=context_num_envs
        )
        terrain_context = None
        if hasattr(self.terrain, "height_points") and hasattr(
            self.terrain, "height_samples"
        ):
            terrain_context = TerrainContext(
                self._select_context_tensor(self.terrain.height_points, env_ids),
                self.terrain.height_samples,
            )

        collision_primitives = None
        if self._extended_observations_enabled:
            # Build candidates using every character, then select context rows.
            # Opponents may be outside the subset currently being reset.
            full_state = current_state if env_ids is None else self.simulator.get_robot_state()
            full_heights = ground_heights if env_ids is None else self.terrain.get_ground_heights(
                full_state.rigid_body_pos[:, anchor_idx]
            ).squeeze(-1)
            collision_primitives = self._build_collision_primitives(full_state, full_heights)
            self._opponent_impact = self._compute_opponent_impact(full_state)
            if env_ids is not None:
                collision_primitives = CollisionPrimitivesView(**{
                    key: getattr(collision_primitives, key)[env_ids]
                    for key in ("pos", "rot", "lin_vel", "radius", "extent_z", "damage", "mass", "shape", "valid")
                })

        # Build context with view wrappers
        ctx = EnvContext(
            # Core state views (wrap RobotState without copying)
            current=current_view,
            noisy=noisy_view,
            # Historical views (wrap StateHistoryBuffer without copying)
            historical=HistoricalView(
                self.state_history, use_noisy=False, env_ids=env_ids
            )
            if self.state_history
            else None,
            noisy_historical=HistoricalView(
                self.state_history, use_noisy=True, env_ids=env_ids
            )
            if self.state_history
            else None,
            # Actions (historical)
            current_processed_action=self._select_context_tensor(
                self._current_processed_action, env_ids
            ),
            previous_action=self._select_context_tensor(
                self.state_history.actions[:, 1], env_ids
            )
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            previous_processed_action=self._select_context_tensor(
                self.state_history.processed_actions[:, 1], env_ids
            )
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            # Environment state
            env_ids=env_ids,
            ground_heights=ground_heights,
            noisy_ground_heights=noisy_ground_heights,
            terrain=terrain_context,
            scene=scene_surface_context,
            body_contacts=body_contacts,
            current_contact_force_magnitudes=current_contact_force_magnitudes,
            prev_contact_force_magnitudes=self._select_context_tensor(
                self.prev_contact_force_magnitudes, env_ids
            ),

            incoming_damage=self._select_context_tensor(getattr(self, "_incoming_damage", None), env_ids),
            body_stamina=self._select_context_tensor(getattr(self, "_body_stamina", None), env_ids),
            opponent_impact=self._select_context_tensor(getattr(self, "_opponent_impact", None), env_ids),
            realign_offset_delta=self._select_context_tensor(getattr(self, "_realign_offset_delta", None), env_ids),
            dt=self.dt,
            progress_buf=self._select_context_tensor(self.progress_buf, env_ids),
            # Contact tracking
            contact_body_ids=self.contact_body_ids,
            non_termination_contact_body_ids=self.non_termination_contact_body_ids,
            # Per-episode odometer corruption parameters
            odom_scale=self._select_context_tensor(self.odom_scale, env_ids),
            odom_yaw_cos_sin=self._select_context_tensor(
                self.odom_yaw_cos_sin, env_ids
            ),

            # Collision primitives (ground + obstacles/projectiles/characters)
            collision_primitives=collision_primitives,
        )

        # Controllers populate their task-specific views
        self.control_manager.populate_context(ctx)

        # Compute the corrupted odometer displacement once per step.
        #
        # The corruption models the G1's leg-kinematics odometer: noise is
        # proportional to the robot's displacement from episode start (distance
        # walked), NOT to the displacement to the reference.  Only the raw
        # sensor fields (odom_disp_start_{corrupt,clean}, odom_start_xy,
        # odom_start_heading_inv) are stored here -- the heading-local offset to
        # reference is derived on demand by consumers from these ingredients via
        # obs.target_poses.compute_odom_offset_local.
        #
        # At deployment: odom_position comes from rt/odommodestate directly.
        if ctx.mimic is not None:
            odom_start_xy = self._select_context_tensor(self.odom_start_xy, env_ids)
            odom_start_heading_inv = self._select_context_tensor(
                self.odom_start_heading_inv, env_ids
            )
            from protomotions.utils.odom_corruption import apply_odom_corruption_torch
            from protomotions.utils import rotations as rot_utils

            cur_anchor_xy = current_state.rigid_body_pos[:, anchor_idx, :2]

            # Corrupt the robot's position (displacement from episode start),
            # expressed in the episode-start heading frame. This matches real
            # deployment odometry: zero at motion start, facing robot-front.
            robot_disp_world = cur_anchor_xy - odom_start_xy
            robot_disp_world_3d = torch.cat(
                [robot_disp_world, torch.zeros_like(robot_disp_world[:, :1])],
                dim=-1,
            )
            robot_disp_from_start = rot_utils.quat_rotate(
                odom_start_heading_inv, robot_disp_world_3d, w_last=True
            )[:, :2]
            corrupted_disp = apply_odom_corruption_torch(
                robot_disp_from_start, ctx.odom_scale, ctx.odom_yaw_cos_sin,
                log_noise_std=self.config.odom_log_noise_std,
                soft_threshold=self.config.odom_soft_threshold,
            )
            # Store only the raw odometer sensor fields. The heading-local offset
            # to reference (formerly odom_offset_local_{corrupt,clean}) is derived
            # on demand by consumers from these ingredients, so the stochastic
            # corruption stays confined to this sensor boundary.
            ctx.odom_start_xy = odom_start_xy
            ctx.odom_start_heading_inv = odom_start_heading_inv
            ctx.odom_disp_start_clean = robot_disp_from_start
            ctx.odom_disp_start_corrupt = corrupted_disp
        else:
            ctx.odom_start_xy = torch.zeros(
                context_num_envs, 2, device=self.device, dtype=torch.float
            )
            ctx.odom_start_heading_inv = torch.zeros(
                context_num_envs, 4, device=self.device, dtype=torch.float
            )
            ctx.odom_start_heading_inv[:, 3] = 1.0
            ctx.odom_disp_start_clean = torch.zeros(
                context_num_envs, 2, device=self.device, dtype=torch.float
            )
            ctx.odom_disp_start_corrupt = torch.zeros(
                context_num_envs, 2, device=self.device, dtype=torch.float
            )

        return ctx

    def _build_scene_surface_context(
        self, env_ids: Optional[Tensor] = None, context_num_envs: Optional[int] = None
    ) -> SceneSurfaceContext:
        """Build scene-object surface tensors for component observations.

        Nearest-surface observations bind these fields unconditionally. Envs
        without object pointclouds receive empty tensors, which lets the compute
        kernel naturally fall back to terrain-only behavior.
        """
        if env_ids is not None:
            env_ids = env_ids.to(self.device)
        if context_num_envs is None:
            context_num_envs = self.num_envs if env_ids is None else env_ids.shape[0]
        scene_lib = getattr(self, "scene_lib", None)
        num_objects_per_scene = (
            getattr(scene_lib, "num_objects_per_scene", 0)
            if scene_lib is not None
            else 0
        )
        has_object_pointclouds = (
            getattr(scene_lib, "_object_pointclouds", None) is not None
        )
        if num_objects_per_scene <= 0 or not has_object_pointclouds:
            object_pos = torch.zeros(context_num_envs, 0, 3, device=self.device)
            object_rot = torch.zeros(context_num_envs, 0, 4, device=self.device)
            neutral_pointclouds = torch.zeros(
                context_num_envs, 0, 0, 3, device=self.device
            )
            object_valid_mask = torch.zeros(
                context_num_envs, 0, dtype=torch.bool, device=self.device
            )
            return SceneSurfaceContext(
                object_pos=object_pos,
                object_rot=object_rot,
                neutral_pointclouds=neutral_pointclouds,
                object_valid_mask=object_valid_mask,
            )

        scene_env_ids = (
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            if env_ids is None
            else env_ids
        )
        object_state = self.simulator.get_object_root_state(env_ids)
        return SceneSurfaceContext(
            object_pos=object_state.root_pos,
            object_rot=object_state.root_rot,
            neutral_pointclouds=scene_lib.get_scene_neutral_pointcloud(scene_env_ids),
            object_valid_mask=scene_lib.get_per_object_valid_mask(scene_env_ids),
        )

    def _build_global_context(self) -> EnvContext:
        return self._build_context()

    def _randomize_projectile_damage(self, env_ids: Tensor) -> None:
        """Randomize per-projectile 'damage' for the given envs (per episode).

        For each env, a random fraction of the projectile pool is designated
        'dangerous' (damage sampled from ``projectile_damage_range``); the rest keep
        ``projectile_baseline_damage``. This realizes "randomize damage on a random
        number of colliders" so the policy learns to discriminate threats rather
        than treating every collider identically.
        """
        num_e = env_ids.numel()
        if num_e == 0 or self._num_projectiles == 0:
            return
        n = self._num_projectiles
        device = self.device

        frac_lo, frac_hi = self.config.projectile_dangerous_fraction_range
        fraction = torch.rand(num_e, 1, device=device) * (frac_hi - frac_lo) + frac_lo
        # Random per-(env, projectile) priority; the lowest `count` become dangerous.
        priority = torch.rand(num_e, n, device=device)
        count = torch.round(fraction * n)  # [num_e, 1]
        rank = torch.argsort(torch.argsort(priority, dim=1), dim=1).to(priority.dtype)
        dangerous_mask = rank < count  # [num_e, n] bool

        dmg_lo, dmg_hi = self.config.projectile_damage_range
        dangerous_damage = torch.rand(num_e, n, device=device) * (dmg_hi - dmg_lo) + dmg_lo
        baseline = torch.full(
            (num_e, n), self.config.projectile_baseline_damage, device=device
        )
        self._projectile_damage[env_ids] = torch.where(
            dangerous_mask, dangerous_damage, baseline
        )

    def _init_body_stamina(self) -> None:
        """Set up per-body 'stamina' and the body->DOF mapping used to scale gains.

        Stamina is one scalar per actuated body (a body that has hinge DOFs). It
        scales the PD gains of that body's parent joint and is exposed as an obs.
        DOFs are emitted in ascending-body-index traversal order, so iterating
        bodies in index order and counting their hinge DOFs reproduces the DOF order.
        """
        kin = self.robot_config.kinematic_info
        hinge_axes_map = kin.hinge_axes_map
        stamina_body_indices = sorted(hinge_axes_map.keys())
        self._stamina_body_indices = stamina_body_indices
        self._num_stamina_bodies = len(stamina_body_indices)

        body_to_col = {b: c for c, b in enumerate(stamina_body_indices)}
        dof_to_stamina_col = []
        for body_idx in range(kin.num_bodies):
            if body_idx in hinge_axes_map:
                n_dofs = len(hinge_axes_map[body_idx])
                dof_to_stamina_col.extend([body_to_col[body_idx]] * n_dofs)
        self._dof_to_stamina_col = torch.tensor(
            dof_to_stamina_col, dtype=torch.long, device=self.device
        )

        self._body_stamina = torch.ones(
            self.num_envs, self._num_stamina_bodies, device=self.device
        )
        if self.config.randomize_body_stamina:
            self._randomize_body_stamina(torch.arange(self.num_envs, device=self.device))

    def _init_multi_character(self) -> None:
        """Set up per-character spawn offsets and opponent-body bookkeeping.

        For single-character (N == 1) this is a no-op beyond allocating a trivial
        spawn offset, so legacy behavior is unchanged. For N > 1 it precomputes:
          - ``_character_spawn_offset`` [N, 2]: scene-local XY spawn positions.
          - ``_opponent_key_body_ids`` [K]: body indices of opponent key bodies
            (head/pelvis/limbs) that are exposed as collision primitives.
          - ``_opponent_char_idx`` [N, N-1]: for each character, the indices of the
            other characters in its scene.
        """
        from protomotions.simulator.base_simulator.utils import (
            character_spawn_offsets,
        )
        from protomotions.simulator.base_simulator.config import (
            get_matching_indices,
        )

        offsets = character_spawn_offsets(
            self.num_characters, self.config.character_spawn_radius
        )
        self._character_spawn_offset = torch.tensor(
            offsets, dtype=torch.float, device=self.device
        )  # [N, 2]

        self._opponent_impact = torch.zeros(self.num_envs, device=self.device)
        self._robot_body_masses = self.simulator.get_robot_body_masses()

        projectile_masses = []
        for spec in self.simulator._proj_config.get_shape_specs():
            if spec.shape_type == "box":
                volume = spec.extent_z**3
            elif spec.shape_type == "sphere":
                volume = 4.0 * torch.pi * spec.radius**3 / 3.0
            else:
                volume = (
                    torch.pi * spec.radius**2 * spec.extent_z
                    + 4.0 * torch.pi * spec.radius**3 / 3.0
                )
            projectile_masses.append(
                float(volume * self.simulator._proj_config.density)
            )
        self._projectile_masses = torch.tensor(
            projectile_masses, dtype=torch.float, device=self.device
        )

        if self.num_characters <= 1:
            self._opponent_key_body_ids = torch.zeros(
                0, dtype=torch.long, device=self.device
            )
            self._striking_body_ids = torch.zeros(
                0, dtype=torch.long, device=self.device
            )
            self._opponent_char_idx = None
            return

        body_names = self.robot_config.kinematic_info.body_names
        key_ids = get_matching_indices(
            body_names, names_to_match=list(self.config.opponent_key_body_names)
        )
        self._opponent_key_body_ids = torch.tensor(
            sorted(key_ids), dtype=torch.long, device=self.device
        )
        strike_ids = get_matching_indices(
            body_names, names_to_match=list(self.config.striking_body_names)
        )
        self._striking_body_ids = torch.tensor(
            sorted(strike_ids), dtype=torch.long, device=self.device
        )

        N = self.num_characters
        opp = [[c2 for c2 in range(N) if c2 != c] for c in range(N)]
        self._opponent_char_idx = torch.tensor(
            opp, dtype=torch.long, device=self.device
        )  # [N, N-1]

    def _compute_opponent_impact(self, current_state) -> Tensor:
        """Per-character 'striking' signal for the opponent-impact reward.

        For each character, finds the maximum closing speed of any of its striking
        bodies (hands/feet) toward any opponent key body that is within
        ``opponent_strike_radius``. Returns ``[num_envs]`` (0 when N == 1). This
        rewards landing fast limb strikes on opponents (closing speed, gated by
        proximity) rather than mere contact.
        """
        if (
            self.num_characters <= 1
            or self._striking_body_ids.numel() == 0
            or self._opponent_key_body_ids.numel() == 0
        ):
            return torch.zeros(self.num_envs, device=self.device)

        E = self.num_physical_envs
        N = self.num_characters
        B = current_state.rigid_body_pos.shape[1]
        body_pos = current_state.rigid_body_pos.view(E, N, B, 3)
        body_vel = current_state.rigid_body_vel.view(E, N, B, 3)

        strike_pos = body_pos[:, :, self._striking_body_ids]  # [E, N, S, 3]
        strike_vel = body_vel[:, :, self._striking_body_ids]
        kb_pos = body_pos[:, :, self._opponent_key_body_ids]  # [E, N, K, 3]
        kb_vel = body_vel[:, :, self._opponent_key_body_ids]

        radius = self.config.opponent_strike_radius
        impact = torch.zeros(E, N, device=self.device)
        for c in range(N):
            sp = strike_pos[:, c]  # [E, S, 3]
            sv = strike_vel[:, c]
            others = self._opponent_char_idx[c]  # [N-1]
            op = kb_pos[:, others].reshape(E, -1, 3)  # [E, O, 3]
            ov = kb_vel[:, others].reshape(E, -1, 3)
            diff = op[:, None, :, :] - sp[:, :, None, :]  # [E, S, O, 3]
            dist = diff.norm(dim=-1).clamp_min(1e-6)  # [E, S, O]
            rel_vel = sv[:, :, None, :] - ov[:, None, :, :]  # [E, S, O, 3]
            closing = (rel_vel * (diff / dist.unsqueeze(-1))).sum(-1)  # [E, S, O]
            within = dist < radius
            val = closing.clamp_min(0.0) * within
            impact[:, c] = val.amax(dim=(1, 2))
        return impact.reshape(E * N)

    def set_body_stamina(self, strength: Tensor, env_ids: Optional[Tensor] = None) -> None:
        """Set effective strength per actuated body for damage/fatigue at runtime.

        Columns follow sorted ``kinematic_info.hinge_axes_map`` body indices.
        Values in [0, 1] combine damage and fatigue into effective strength; zero
        disables active drive, one restores nominal engine properties. Call between
        control steps, never from inside a physics substep.
        """
        if not self._extended_observations_enabled:
            raise RuntimeError("Body stamina requires the extended-observation experiment")
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        strength = torch.as_tensor(strength, device=self.device, dtype=self._body_stamina.dtype)
        expected = (env_ids.numel(), self._num_stamina_bodies)
        if strength.shape != expected or not torch.isfinite(strength).all():
            raise ValueError(f"strength must be finite with shape {expected}")
        if torch.any((strength < 0) | (strength > 1)):
            raise ValueError("strength must be in [0, 1]")
        self._body_stamina[env_ids] = strength
        self._apply_body_stamina_to_gains(env_ids)
        self._current_context = None
        if "stamina_obs" in self._observation_buffer:
            self._observation_buffer["stamina_obs"][env_ids] = strength

    def _apply_body_stamina_to_gains(self, env_ids: Tensor) -> None:
        """Expand per-body stamina to per-DOF and write the resulting PD gains.

        Under BUILT_IN_PD the engine owns the PD loop, so stamina cannot be a
        per-step action scaling; instead it multiplies the nominal per-DOF gains
        and is written into the simulator for the given envs (a no-op on backends
        that do not support runtime gain writes). ``env_ids`` are flattened RL
        rows; the simulator routes them to physical scenes / characters.
        """
        if self._dof_to_stamina_col.numel() == 0:
            return
        self._stamina_gains_modified = True
        per_dof = self._body_stamina[env_ids][:, self._dof_to_stamina_col]
        if self.robot_config.control.control_type == ControlType.PROPORTIONAL:
            # Legacy path: per-step gain scaling through the action pipeline
            # (custom PD computed in Python). Kept for backward compatibility.
            self.set_dof_stamina_scale(per_dof, env_ids)
        else:
            # BUILT_IN_PD (default): the engine owns the PD loop, so write the
            # scaled per-DOF gains into the simulator instead.
            self.simulator.set_joint_gain_scale(per_dof, env_ids)

    def _randomize_body_stamina(self, env_ids: Tensor) -> None:
        """Per-episode randomize per-body stamina (Tier-4 curriculum), then re-map."""
        if env_ids.numel() == 0 or self._num_stamina_bodies == 0:
            return
        if self.config.randomize_body_stamina:
            lo, hi = self.config.body_stamina_range
            self._body_stamina[env_ids] = (
                torch.rand(
                    env_ids.numel(), self._num_stamina_bodies, device=self.device
                )
                * (hi - lo)
                + lo
            )
        else:
            self._body_stamina[env_ids] = 1.0
        if self.config.randomize_body_stamina or getattr(self, "_stamina_gains_modified", False):
            self._apply_body_stamina_to_gains(env_ids)

    def _update_projectile_velocity(self) -> None:
        """Sample finite differences once per physics step, never on context reads."""
        state = self.simulator.get_active_projectile_states()
        pos, active = state["positions"], state["active"]
        valid = (self._prev_projectile_active > 0.5) & (active > 0.5)
        velocity = (pos - self._prev_projectile_pos) / self.dt
        self._projectile_velocity.copy_(
            torch.where(valid.unsqueeze(-1), velocity, torch.zeros_like(velocity))
        )
        self._prev_projectile_pos.copy_(pos)
        self._prev_projectile_active.copy_(active)

    def _build_collision_primitives(
        self, current_state, ground_heights: Tensor
    ) -> CollisionPrimitivesView:
        """Assemble the fixed-capacity collision-primitive candidate buffer.

        Writes all discovered primitives into a temporary WORLD-frame tensor.
        When the raw set exceeds M, all categories use the same contact-priority
        score before compaction. Padding slots carry ``valid == 0``.

        The initial comparison populates only ground (slot 0). Optional tiers add
        thrown projectiles and other characters' key bodies. Scene obstacles
        are not yet populated here.
        """
        num_envs = self.num_envs
        capacity = self.config.max_collision_primitives
        device = self.device
        N = self.num_characters
        num_opponent_candidates = (
            (N - 1) * self._opponent_key_body_ids.numel() if N > 1 else 0
        )
        raw_capacity = max(
            capacity, 1 + self._num_projectiles + num_opponent_candidates
        )

        pos = torch.zeros(num_envs, raw_capacity, 3, device=device)
        rot = torch.zeros(num_envs, raw_capacity, 4, device=device)
        rot[..., 3] = 1.0  # identity quaternion (xyzw)
        lin_vel = torch.zeros(num_envs, raw_capacity, 3, device=device)
        radius = torch.zeros(num_envs, raw_capacity, device=device)
        extent_z = torch.zeros(num_envs, raw_capacity, device=device)
        damage = torch.zeros(num_envs, raw_capacity, device=device)
        mass = torch.zeros(num_envs, raw_capacity, device=device)
        shape = torch.zeros(num_envs, raw_capacity, 2, device=device)
        valid = torch.zeros(num_envs, raw_capacity, device=device)

        # Slot 0: ground box directly beneath the root. rel_pos.z then encodes how
        # high the character is above ground (a richer replacement for root-height).
        root_pos = current_state.rigid_body_pos[:, 0, :]
        pos[:, 0, 0] = root_pos[:, 0]
        pos[:, 0, 1] = root_pos[:, 1]
        pos[:, 0, 2] = ground_heights
        shape[:, 0, 0] = 1.0  # is_box
        damage[:, 0] = self.config.ground_primitive_damage
        mass[:, 0] = self.config.static_collider_effective_mass
        valid[:, 0] = 1.0

        # Slots 1..: active thrown projectiles (boxes). Velocity is derived by
        # finite difference on positions (backend-agnostic), zeroed on the first
        # active step to avoid the spike from the hidden->thrown teleport. Projectile
        # state is per physical scene ([E, P, ...]); for multi-character self-play it
        # is expanded so all N characters in a scene see the same projectiles.
        incoming_damage = torch.zeros(num_envs, device=device)
        num_proj = getattr(self, "_num_projectiles", 0)
        next_slot = 1
        if num_proj > 0:
            proj = self.simulator.get_active_projectile_states()
            proj_pos = proj["positions"][:, :num_proj]  # [E, P, 3]
            proj_rot = proj["rotations"][:, :num_proj]  # [E, P, 4]
            proj_active = proj["active"][:, :num_proj]  # [E, P]
            # Per-projectile shape encoding (box/sphere/capsule). Per pool index, so
            # independent of the character dimension (no N-expansion needed).
            proj_radius = proj["radius"][:num_proj]  # [P]
            proj_extent_z = proj["extent_z"][:num_proj]  # [P]
            proj_shape = proj["shape"][:num_proj]  # [P, 2]

            proj_vel = self._projectile_velocity[:, :num_proj]

            proj_damage = self._projectile_damage[:, :num_proj]
            # Per-scene incoming threat = max damage among active projectiles.
            incoming_damage_phys = (proj_damage * proj_active).max(dim=1).values  # [E]

            if N > 1:
                proj_pos = proj_pos.repeat_interleave(N, dim=0)
                proj_rot = proj_rot.repeat_interleave(N, dim=0)
                proj_vel = proj_vel.repeat_interleave(N, dim=0)
                proj_active = proj_active.repeat_interleave(N, dim=0)
                proj_damage = proj_damage.repeat_interleave(N, dim=0)
                incoming_damage = incoming_damage_phys.repeat_interleave(N, dim=0)
            else:
                incoming_damage = incoming_damage_phys

            pos[:, 1 : 1 + num_proj] = proj_pos
            rot[:, 1 : 1 + num_proj] = proj_rot
            lin_vel[:, 1 : 1 + num_proj] = proj_vel
            # Per-projectile primitive shape: box -> radius=0, extent_z=full height,
            # shape=[1,0]; sphere -> radius=r, extent_z=0, shape=[0,1]; capsule ->
            # radius=r, extent_z=cyl length, shape=[0,0]. Broadcast [P] -> [E, P].
            radius[:, 1 : 1 + num_proj] = proj_radius.unsqueeze(0)
            extent_z[:, 1 : 1 + num_proj] = proj_extent_z.unsqueeze(0)
            shape[:, 1 : 1 + num_proj] = proj_shape.unsqueeze(0)
            damage[:, 1 : 1 + num_proj] = proj_damage
            mass[:, 1 : 1 + num_proj] = self._projectile_masses[
                :num_proj
            ].unsqueeze(0)
            valid[:, 1 : 1 + num_proj] = proj_active
            next_slot = 1 + num_proj

        self._incoming_damage = incoming_damage

        # Other characters' key bodies (multi-character self-play).
        # Each character observes the opponents in its physical scene as sphere
        # primitives so it can perceive, avoid, and strike them. Bodies are written
        # in world frame; the shared priority function handles any capacity cut.
        if N > 1 and self._opponent_key_body_ids.numel() > 0:
            self._write_opponent_primitives(
                current_state, next_slot, pos, lin_vel, radius, mass, shape, valid
            )

        if raw_capacity > capacity:
            from protomotions.envs.obs.collision_primitives import (
                compute_collision_priority,
            )

            priority = compute_collision_priority(
                body_pos=current_state.rigid_body_pos,
                body_vel=current_state.rigid_body_vel,
                primitive_pos=pos,
                primitive_lin_vel=lin_vel,
                primitive_mass=mass,
                primitive_valid=valid,
                selection_range=self.config.collision_selection_range,
                distance_weight=self.config.collision_distance_weight,
                closing_speed_weight=self.config.collision_closing_speed_weight,
                mass_weight=self.config.collision_mass_weight,
                distance_scale=self.config.collision_distance_scale,
                speed_scale=self.config.collision_speed_scale,
                mass_scale=self.config.collision_mass_scale,
            )
            keep = torch.topk(priority, capacity, dim=1, largest=True).indices

            def _select(x: Tensor) -> Tensor:
                if x.dim() == 2:
                    return torch.gather(x, 1, keep)
                idx = keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                return torch.gather(x, 1, idx)

            pos = _select(pos)
            rot = _select(rot)
            lin_vel = _select(lin_vel)
            radius = _select(radius)
            extent_z = _select(extent_z)
            damage = _select(damage)
            mass = _select(mass)
            shape = _select(shape)
            valid = _select(valid)

        return CollisionPrimitivesView(
            pos=pos,
            rot=rot,
            lin_vel=lin_vel,
            radius=radius,
            extent_z=extent_z,
            damage=damage,
            mass=mass,
            shape=shape,
            valid=valid,
        )

    def _write_opponent_primitives(
        self,
        current_state,
        next_slot: int,
        pos: Tensor,
        lin_vel: Tensor,
        radius: Tensor,
        mass: Tensor,
        shape: Tensor,
        valid: Tensor,
    ) -> None:
        """Write opponents' key bodies into the collision-primitive buffers.

        For each character row, the key bodies (head/pelvis/limbs) of the other
        characters in the same physical scene are written as sphere primitives in
        world frame, starting at slot ``next_slot``. The raw buffer is sized for
        all opponents before category-neutral capacity selection.
        """
        E = self.num_physical_envs
        N = self.num_characters
        key_ids = self._opponent_key_body_ids
        K = key_ids.numel()
        capacity = pos.shape[1]

        num_bodies = current_state.rigid_body_pos.shape[1]
        body_pos = current_state.rigid_body_pos.view(E, N, num_bodies, 3)
        body_vel = current_state.rigid_body_vel.view(E, N, num_bodies, 3)

        kb_pos = body_pos[:, :, key_ids]  # [E, N, K, 3]
        kb_vel = body_vel[:, :, key_ids]  # [E, N, K, 3]
        body_mass = self._robot_body_masses.view(E, N, num_bodies)
        kb_mass = body_mass[:, :, key_ids]

        # Gather the opponents (N-1 of them) for each character, then flatten the
        # (E, N) leading dims into the E*N row layout (row == e * N + c).
        opp_idx = self._opponent_char_idx  # [N, N-1]
        opp_pos = kb_pos[:, opp_idx].reshape(E * N, (N - 1) * K, 3)
        opp_vel = kb_vel[:, opp_idx].reshape(E * N, (N - 1) * K, 3)
        opp_mass = kb_mass[:, opp_idx].reshape(E * N, (N - 1) * K)

        num_opp = opp_pos.shape[1]
        n_write = min(num_opp, capacity - next_slot)
        if n_write <= 0:
            return
        sl = slice(next_slot, next_slot + n_write)
        pos[:, sl] = opp_pos[:, :n_write]
        lin_vel[:, sl] = opp_vel[:, :n_write]
        radius[:, sl] = self.config.opponent_body_radius
        mass[:, sl] = opp_mass[:, :n_write]
        shape[:, sl, 1] = 1.0  # sphere one-hot ([is_box, is_sphere] == [0, 1])
        valid[:, sl] = 1.0

    def get_has_reset_grace(self):
        """Check if environments are in the grace period after reset.

        Grace period is useful for zeroing rewards that are unreliable immediately
        after reset (e.g., power consumption, contact changes).

        Returns:
            Boolean tensor indicating which environments are within reset_grace_period steps of last reset.
            Returns None if reset_grace_period is 0 or negative.
        """
        if self.config.reset_grace_period <= 0:
            return None
        return self.progress_buf <= self.config.reset_grace_period

    def compute_reward(self, context: EnvContext):
        """Compute base rewards using the dynamic reward component system.

        Args:
            context: Pre-built EnvContext from self.context property.

        Subclasses should override this to add task-specific rewards, calling super().compute_reward() first.
        """
        grace_mask = self.get_has_reset_grace()

        # Process rewards
        combined_reward, reward_logging = self._process_rewards(context, grace_mask)

        self.rew_buf[:] = combined_reward
        self.extras.update(reward_logging)
        self.extras["total_env_reward"] = combined_reward

    ###############################################################
    # Handle Resets
    ###############################################################
    def move_reset_robot_obj_states_to_respawn_position(
        self,
        env_ids,
        new_states: ResetState,
        new_object_states: ObjectState,
        apply_character_offset: bool = True,
    ) -> Tuple[ResetState, ObjectState]:
        new_states.root_pos += self.respawn_root_offset[env_ids]
        if self.scene_lib.num_scenes() > 0:
            new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)

        # Multi-character self-play: separate the N characters within a shared scene
        # by their fixed per-character XY offset so they spawn apart but close enough
        # to interact (character of row r == r % N).
        if self.num_characters > 1 and apply_character_offset:
            char_idx = env_ids % self.num_characters
            new_states.root_pos[:, :2] += self._character_spawn_offset[char_idx]

        return new_states, new_object_states

    def compute_default_reset_state(
        self, env_ids, sample_flat: bool = False
    ) -> Tuple[ResetState, ObjectState]:
        """Reset environments to default state."""

        new_states = self.default_reset_state[env_ids].clone()
        new_object_states = self.default_object_state[env_ids].clone()

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=None,
            sample_flat=sample_flat,
        )

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def compute_ref_reset_state(
        self,
        env_ids,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        sample_flat: bool = False,
    ) -> Tuple[ResetState, ObjectState]:
        """Compute reset state from reference motion data.

        Args:
            env_ids: Environment indices to reset
            motion_ids: Motion IDs to use [len(env_ids)]
            motion_times: Start times for each motion [len(env_ids)]
            sample_flat: If True, spawn on flat terrain

        Returns:
            Tuple of (reset_state, object_reset_state)
        """

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)
        new_states = ResetState.from_robot_state(ref_state).clone()

        new_object_states = self.scene_lib.get_scene_pose(
            env_ids, motion_times, respawn_offset=self.config.ref_object_respawn_offset
        )
        new_object_states.root_vel = torch.zeros_like(new_object_states.root_pos)
        new_object_states.root_ang_vel = torch.zeros_like(new_object_states.root_pos)

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=ref_state,
            sample_flat=sample_flat,
        )
        self._place_multi_character_reference_states(env_ids, ref_state)

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids,
            new_states,
            new_object_states,
            apply_character_offset=False,
        )

    def _generate_fall_reset_states(self) -> None:
        """Cache physically settled fall poses for reset sampling."""
        config = self.config.recovery_reset
        if config.fall_sim_steps <= 0:
            self._fall_reset_states = self.default_reset_state.clone()
            return

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        snapshot = self.save_state()
        try:
            fall_states = self.default_reset_state.clone()
            flat_xy = self.terrain.sample_valid_locations(
                num_envs=self.num_envs, sample_flat=True
            )
            ground_heights = self.terrain.get_ground_heights(
                torch.cat(
                    [
                        flat_xy,
                        torch.zeros(self.num_envs, 1, device=self.device),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
            fall_states.root_pos[:, :2] = flat_xy
            fall_states.root_pos[:, 2] += ground_heights

            random_quat = torch.randn(self.num_envs, 4, device=self.device)
            random_quat = random_quat / random_quat.norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            fall_states.root_rot = random_quat
            fall_states.root_vel.zero_()
            fall_states.root_ang_vel.zero_()
            if fall_states.dof_vel is not None:
                fall_states.dof_vel.zero_()

            self.simulator.reset_envs(
                fall_states, self.default_object_state, env_ids
            )

            dof_lower = self.robot_config.kinematic_info.dof_limits_lower.to(
                self.device
            )
            dof_upper = self.robot_config.kinematic_info.dof_limits_upper.to(
                self.device
            )
            random_targets = (
                torch.rand(
                    self.num_envs,
                    self.robot_config.number_of_actions,
                    dtype=torch.float,
                    device=self.device,
                )
                * (dof_upper - dof_lower)
                + dof_lower
            )
            for _ in range(config.fall_sim_steps):
                self.simulator.step(random_targets, markers_callback=None)

            self._fall_reset_states = ResetState.from_robot_state(
                self.simulator.get_robot_state()
            ).clone()
            self._fall_reset_states.root_vel.zero_()
            self._fall_reset_states.root_ang_vel.zero_()
            if self._fall_reset_states.dof_vel is not None:
                self._fall_reset_states.dof_vel.zero_()
        finally:
            self.restore_state(snapshot)

    def _apply_recovery_reset_states(
        self, env_ids: Tensor, reset_states: ResetState
    ) -> Tuple[ResetState, Tensor]:
        """Replace a sampled subset with cached fall poses and return the result."""
        recovery_mask = torch.zeros(
            len(env_ids), dtype=torch.bool, device=self.device
        )
        config = getattr(self.config, "recovery_reset", None)
        if config is None or config.recovery_prob <= 0:
            return reset_states, recovery_mask
        if self._fall_reset_states is None:
            raise RuntimeError(
                "recovery_prob is positive but fall reset states were not generated"
            )

        recovery_mask = (
            torch.rand(len(env_ids), device=self.device) < config.recovery_prob
        )
        if not recovery_mask.any():
            return reset_states, recovery_mask

        local_ids = torch.nonzero(recovery_mask, as_tuple=False).flatten()
        random_ids = torch.randint(
            0,
            self._fall_reset_states.root_pos.shape[0],
            (len(local_ids),),
            device=self.device,
        )
        fall_states = self._fall_reset_states[random_ids]

        reset_states.root_rot[local_ids] = fall_states.root_rot
        reset_states.root_vel[local_ids] = 0.0
        reset_states.root_ang_vel[local_ids] = 0.0
        reset_states.dof_pos[local_ids] = fall_states.dof_pos
        if reset_states.dof_vel is not None and fall_states.dof_vel is not None:
            reset_states.dof_vel[local_ids] = 0.0

        env_subset = env_ids[local_ids]
        reset_states.root_pos[local_ids, 2] = (
            self.respawn_root_offset[env_subset, 2] + fall_states.root_pos[:, 2]
        )
        return reset_states, recovery_mask

    def reset(
        self,
        env_ids=None,
        sample_flat=False,
        force_default_mask=None,
        disable_motion_resample=False,
    ):
        """Reset environments and return observations.

        - auto if no motion_lib: reset from default state
        - auto if motion_lib exists: reset from reference motion
        - force_default_mask: optional boolean mask [len(env_ids)] to force specific envs
            ref_prob = 0.5
            mask = torch.bernoulli(torch.full((len(env_ids),), 1-ref_prob)).bool()
            env.reset(env_ids, force_default_mask=mask)

        Args:
            env_ids: Environment IDs to reset, or None to reset all
            sample_flat: If True, spawn on flat terrain (useful for evaluation)
            force_default_mask: Optional boolean mask [len(env_ids)] to force specific envs
                               to use default reset instead of reference motion reset.
                               Only used if motion_lib exists.
            disable_motion_resample: If True, skip resampling motions (use existing motion_ids/times).
                               Useful for evaluation when you want to replay specific motions.

        Returns:
            obs: Dictionary of observation tensors
            info: Dictionary containing reset metadata (currently empty)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if len(env_ids) == 0:
            return self.get_obs(), {}

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        # A physical multi-character scene is one episode. Resetting only one
        # flattened row strands its opponents at the old anchor and motion time.
        if self.num_characters > 1:
            requested_env_ids = env_ids
            expanded_env_ids = self._expand_to_physical_scenes(requested_env_ids)
            if (
                force_default_mask is not None
                and expanded_env_ids.numel() != env_ids.numel()
            ):
                requested_mask = torch.as_tensor(
                    force_default_mask, device=self.device, dtype=torch.bool
                )
                scene_default = torch.zeros(
                    self.num_physical_envs, device=self.device, dtype=torch.long
                )
                scene_default.scatter_reduce_(
                    0,
                    requested_env_ids // self.num_characters,
                    requested_mask.to(torch.long),
                    reduce="amax",
                    include_self=True,
                )
                force_default_mask = (
                    scene_default[expanded_env_ids // self.num_characters] > 0
                )
            env_ids = expanded_env_ids

        # STEP 1: Reset motion manager and determine which envs need reference motion reset
        # This calls motion_manager.sample_motions() internally
        ref_env_ids, motion_ids, motion_times = self._get_ref_reset_envs(
            env_ids, force_default_mask, disable_motion_resample
        )

        default_mask = ~torch.isin(env_ids, ref_env_ids)
        default_indices = default_mask.nonzero(as_tuple=True)[0]

        if ref_env_ids.numel() == 0:
            new_states, new_object_states = self.compute_default_reset_state(
                env_ids, sample_flat
            )
        elif default_indices.numel() == 0:
            new_states, new_object_states = self.compute_ref_reset_state(
                ref_env_ids, motion_ids, motion_times, sample_flat
            )
        else:
            new_states = self.default_reset_state[env_ids].clone()
            new_object_states = self.default_object_state[env_ids].clone()

            default_env_ids = env_ids[default_indices]
            default_states, default_object_states = self.compute_default_reset_state(
                default_env_ids, sample_flat
            )
            new_states[default_indices] = default_states
            new_object_states[default_indices] = default_object_states

            ref_states, ref_object_states = self.compute_ref_reset_state(
                ref_env_ids, motion_ids, motion_times, sample_flat
            )
            ref_indices = torch.isin(env_ids, ref_env_ids).nonzero(as_tuple=True)[0]
            new_states[ref_indices] = ref_states
            new_object_states[ref_indices] = ref_object_states

        new_states, recovery_mask = self._apply_recovery_reset_states(
            env_ids, new_states
        )

        if self.robot_config.reset_noise is not None:
            apply_reset_noise(
                reset_state=new_states,
                config=self.robot_config.reset_noise,
                dof_limits_lower=self.robot_config.kinematic_info.dof_limits_lower,
                dof_limits_upper=self.robot_config.kinematic_info.dof_limits_upper,
            )

        self.simulator.reset_envs(new_states, new_object_states, env_ids)

        current_state_history_mask = ~torch.isin(env_ids, ref_env_ids)
        history_ref_env_ids = ref_env_ids
        history_motion_ids = motion_ids
        history_motion_times = motion_times
        if recovery_mask.any():
            current_state_history_mask = current_state_history_mask | recovery_mask
            if len(ref_env_ids) > 0:
                ref_recovery = torch.isin(ref_env_ids, env_ids[recovery_mask])
                history_ref_env_ids = ref_env_ids[~ref_recovery]
                if motion_ids is not None:
                    history_motion_ids = motion_ids[~ref_recovery]
                    history_motion_times = motion_times[~ref_recovery]

        if self.state_history is not None:
            self._reset_state_history(
                env_ids,
                current_state_history_mask,
                history_ref_env_ids,
                history_motion_ids,
                history_motion_times,
            )

        # Reset control components after motion_manager has been reset
        self.control_manager.reset(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.terminate_buf[env_ids] = False
        self.prev_contact_force_magnitudes[env_ids] = 0.0
        if hasattr(self, "_realign_offset_delta"):
            self._realign_offset_delta[env_ids] = 0.0
        self._current_raw_action[env_ids] = 0.0
        self._current_processed_action[env_ids] = 0.0

        # Resample per-episode odometer corruption parameters.
        # These remain constant within an episode and are used by
        # odom_offset_factory when present in observation components.
        n = len(env_ids)
        self.odom_scale[env_ids] = torch.empty(n, device=self.device).uniform_(
            self.config.odom_scale_range[0], self.config.odom_scale_range[1]
        )
        yaw_bias = torch.empty(n, device=self.device).uniform_(
            -self.config.odom_yaw_range_deg, self.config.odom_yaw_range_deg
        ) * (3.14159265358979 / 180.0)
        self.odom_yaw_cos_sin[env_ids, 0] = torch.cos(yaw_bias)
        self.odom_yaw_cos_sin[env_ids, 1] = torch.sin(yaw_bias)
        # Record robot XY at episode start (after simulator reset) for distance-from-start.
        anchor_idx = self.robot_config.anchor_body_index
        reset_state = self.simulator.get_robot_state()
        self.odom_start_xy[env_ids] = reset_state.rigid_body_pos[env_ids, anchor_idx, :2]
        from protomotions.utils import rotations as rot_utils

        self.odom_start_heading_inv[env_ids] = rot_utils.calc_heading_quat_inv(
            reset_state.rigid_body_rot[env_ids, anchor_idx], w_last=True
        )

        if self._extended_observations_enabled:
            # Re-randomize projectile damage and clear finite-difference velocity state.
            # Projectile buffers are per physical scene, so collapse character rows.
            if self.num_characters > 1:
                phys_ids = torch.unique(env_ids // self.num_characters)
            else:
                phys_ids = env_ids
            self._randomize_projectile_damage(phys_ids)
            self._prev_projectile_active[phys_ids] = 0.0
            self._projectile_velocity[phys_ids] = 0.0

            # Re-randomize per-body stamina (and re-map to per-DOF gains) for this episode.
            # Stamina is per character, so it uses the (flattened) character rows directly.
            self._randomize_body_stamina(env_ids)

        # Update cached noisy obs for the reset envs with fresh noise
        if self._current_noisy_obs is not None:
            current_state = self.simulator.get_robot_state(env_ids)
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[:, self.robot_config.anchor_body_index]
            ).squeeze(-1)
            obs_noise_cfg = self.simulator.config.domain_randomization.observation_noise
            noisy_subset = apply_observation_noise(
                obs_noise_cfg=obs_noise_cfg,
                robot_state=current_state,
                anchor_idx=self.robot_config.anchor_body_index,
                ground_heights=ground_heights,
            )
            self._current_noisy_obs.update_subset(env_ids, noisy_subset)

        # Recompute observations after reset to reflect new control component state
        # Invalidate the cached all-env context, but use a temporary subset
        # context so partial resets only read and process reset environments.
        self._current_context = None
        self.compute_observations(env_ids, context=self._build_context(env_ids))

        return self.get_obs(), {}

    def _get_ref_reset_envs(
        self, env_ids, force_default_mask, disable_motion_resample=False
    ):
        """Determine which envs should use reference motion reset and reset motion manager.

        This method is responsible for resetting the motion_manager by calling
        motion_manager.sample_motions(). Control components should be reset AFTER
        this method is called so they have access to fresh motion_ids and motion_times.

        Args:
            env_ids: Environment IDs to check
            force_default_mask: Boolean mask to force default reset
            disable_motion_resample: If True, use existing motion_ids/times instead of resampling

        Returns:
            ref_env_ids: Environments to reset with reference motion
            motion_ids: Motion IDs for ref resets (or None)
            motion_times: Motion times for ref resets (or None)
        """
        # No motions - no ref resets
        if self.motion_lib.num_motions() == 0:
            empty_ids = torch.tensor([], device=self.device, dtype=torch.long)
            return empty_ids, None, None

        if force_default_mask is not None:
            assert (
                len(force_default_mask) == len(env_ids)
            ), f"force_default_mask length {len(force_default_mask)} != env_ids length {len(env_ids)}"
            ref_env_ids = env_ids[~force_default_mask]
        else:
            ref_env_ids = env_ids

        if len(ref_env_ids) > 0:
            if not disable_motion_resample:
                self.motion_manager.sample_motions(ref_env_ids)
            motion_ids = self.motion_manager.motion_ids[ref_env_ids]
            motion_times = self.motion_manager.motion_times[ref_env_ids]
        else:
            motion_ids = None
            motion_times = None

        return ref_env_ids, motion_ids, motion_times

    def _reset_state_history(
        self,
        env_ids: Tensor,
        current_state_history_mask: Tensor,
        ref_env_ids: Tensor,
        motion_ids: Optional[Tensor],
        motion_times: Optional[Tensor],
    ):
        """Reset state history buffer for specified environments.

        For current-state history, repeat the simulated state across all slots.
        For ref reset: query motion_lib at t-dt, t-2*dt, ... to get historical states.

        Args:
            env_ids: All environment indices being reset.
            current_state_history_mask: Environments that repeat their current
                simulated state across the history buffer.
            ref_env_ids: Environment indices using reference motion reset.
            motion_ids: Motion IDs for ref envs (or None).
            motion_times: Motion times for ref envs (or None).
        """
        current_state_history_env_ids = env_ids[current_state_history_mask]
        num_history_steps = self.state_history.num_history_steps
        # Buffer stores current + history, so total slots = num_history_steps + 1
        buffer_size = num_history_steps + 1

        # Default and recovery resets repeat the current simulator state.
        if len(current_state_history_env_ids) > 0:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[
                    current_state_history_env_ids, self.robot_config.anchor_body_index
                ]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[
                current_state_history_env_ids
            ][:, self.contact_body_ids].bool()
            self.state_history.reset_from_single_state(
                env_ids=current_state_history_env_ids,
                rigid_body_pos=current_state.rigid_body_pos[
                    current_state_history_env_ids
                ],
                rigid_body_rot=current_state.rigid_body_rot[
                    current_state_history_env_ids
                ],
                rigid_body_vel=current_state.rigid_body_vel[
                    current_state_history_env_ids
                ],
                rigid_body_ang_vel=current_state.rigid_body_ang_vel[
                    current_state_history_env_ids
                ],
                dof_pos=current_state.dof_pos[current_state_history_env_ids],
                dof_vel=current_state.dof_vel[current_state_history_env_ids],
                ground_heights=ground_heights,
                body_contacts=body_contacts,
            )

        # Reference reset: fill buffer with current state at index 0 and historical states at index 1+
        # This ensures historical_* properties (which return [:, 1:]) give exactly num_history_steps elements
        if len(ref_env_ids) > 0 and motion_ids is not None and motion_times is not None:
            # motion_ids shape: [len(ref_env_ids)]
            # motion_times shape: [len(ref_env_ids)]
            num_ref_envs = len(ref_env_ids)

            # Create time offsets: [0, -dt, -2*dt, ..., -N*dt] for buffer_size slots
            # Index 0 = current (t), Index 1..N = historical (t-dt, t-2*dt, ..., t-N*dt)
            time_offsets = -self.dt * torch.arange(buffer_size, device=self.device)

            # Expand for batch query: [num_ref_envs, buffer_size]
            expanded_motion_ids = motion_ids.unsqueeze(1).expand(-1, buffer_size)
            expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(
                0
            )

            # Clamp times to valid range
            motion_lengths = self.motion_lib.motion_lengths[motion_ids]
            expanded_motion_times = expanded_motion_times.clamp(min=0.0)
            expanded_motion_times = torch.min(
                expanded_motion_times,
                motion_lengths.unsqueeze(1).expand(-1, buffer_size),
            )

            # Flatten for motion_lib query
            flat_motion_ids = expanded_motion_ids.reshape(-1)
            flat_motion_times = expanded_motion_times.reshape(-1)

            # Query motion library
            historical_state = self.motion_lib.get_motion_state(
                flat_motion_ids, flat_motion_times
            )

            # Motion library data is recorded on flat terrain (height = 0)
            # Only simulator-based states need terrain height queries
            historical_ground_heights = torch.zeros(
                num_ref_envs, buffer_size, device=self.device
            )

            # Get contacts from motion library if available, otherwise zeros
            if historical_state.rigid_body_contacts is not None:
                flat_contacts = historical_state.rigid_body_contacts[
                    :, self.contact_body_ids
                ].bool()
                historical_body_contacts = flat_contacts.view(
                    num_ref_envs, buffer_size, -1
                )
            else:
                historical_body_contacts = torch.zeros(
                    num_ref_envs,
                    buffer_size,
                    len(self.contact_body_ids),
                    dtype=torch.bool,
                    device=self.device,
                )

            # Reshape back to [num_ref_envs, buffer_size, ...]
            self.state_history.reset_from_states(
                env_ids=ref_env_ids,
                rigid_body_pos=historical_state.rigid_body_pos.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_rot=historical_state.rigid_body_rot.view(
                    num_ref_envs, buffer_size, -1, 4
                ),
                rigid_body_vel=historical_state.rigid_body_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_ang_vel=historical_state.rigid_body_ang_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                dof_pos=historical_state.dof_pos.view(num_ref_envs, buffer_size, -1),
                dof_vel=historical_state.dof_vel.view(num_ref_envs, buffer_size, -1),
                ground_heights=historical_ground_heights,
                body_contacts=historical_body_contacts,
                actions=None,  # Zero actions for historical reset
            )

    ###############################################################
    # Motion and Visualization Helpers
    ###############################################################
    def install_motion_lib(self, motion_lib: "MotionLib") -> None:
        """Install a prebuilt motion library and rebuild its environment state."""
        if motion_lib.num_motions() > 0:
            self._validate_motion_lib_compatibility(motion_lib)

        if (
            motion_lib.num_motions() > 0
            and self.config.ref_contact_smooth_window > 0
        ):
            motion_lib.smooth_contacts(self.config.ref_contact_smooth_window)

        self.motion_lib = motion_lib
        if motion_lib.num_motions() > 0:
            self.create_motion_manager()
        else:
            self.motion_manager = None
        self._current_context = None

    def create_motion_manager(self):
        """Instantiate motion manager from configuration."""
        MotionManagerClass = get_class(self.config.motion_manager._target_)

        fixed_motion_ids = None
        if self.scene_lib.num_scenes() > 0:
            humanoid_motion_ids = self.scene_lib.get_humanoid_motion_ids()
            if humanoid_motion_ids is not None:
                fixed_motion_ids = torch.tensor(
                    humanoid_motion_ids, dtype=torch.long, device=self.device
                )

        self.motion_manager = MotionManagerClass(
            config=self.config.motion_manager,
            num_envs=self.num_envs,
            env_dt=self.dt,
            device=self.device,
            motion_lib=self.motion_lib,
            fixed_motion_ids_per_env=fixed_motion_ids,
        )

    def create_visualization_markers(self, headless: bool):
        """Create visualization markers based on headless flag.

        Args:
            headless: If True, no markers are created (empty dict).
                      If False, creates markers according to config.

        Returns:
            Dict of visualization markers.
        """
        if headless:
            return {}

        visualization_markers = {}

        if self.config.show_terrain_markers:
            terrain_markers = []
            for _ in range(self.terrain.num_height_points):
                terrain_markers.append(MarkerConfig(size="small"))
            terrain_markers_cfg = VisualizationMarkerConfig(
                type="sphere", color=(0.008, 0.345, 0.224), markers=terrain_markers
            )
            visualization_markers["terrain_markers"] = terrain_markers_cfg

        # Merge markers from control components
        control_markers = self.control_manager.create_visualization_markers(headless)
        visualization_markers.update(control_markers)

        return visualization_markers

    def get_state_dict(self):
        """Get environment state for checkpointing.

        Returns:
            Dictionary containing motion manager state
        """
        if self.motion_manager is not None:
            return {"motion_manager": self.motion_manager.get_state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        """Load environment state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        if self.motion_manager is not None:
            self.motion_manager.load_state_dict(state_dict["motion_manager"])

    def get_task_id(self):
        """Get task identifier for logging and checkpointing.

        Returns:
            String identifier (motion file name or 'null')
        """
        if self.motion_manager is not None:
            return self.motion_lib.motion_file.split("/")[-1]
        return "null"

    @staticmethod
    def apply_motion_weights_to_scene_weights(
        save_dir: Optional[str], motion_file: Optional[str], device: torch.device
    ) -> Optional[list]:
        """Apply motion weights from checkpoint as scene weights for curriculum learning.

        Loads motion weights from a previous training checkpoint and uses them as
        scene replication weights, allowing over-sampling of scenes corresponding to
        failed motions in curriculum learning.

        IMPORTANT: Assumes 1:1 correspondence between scenes and motions,
        where scene[i].humanoid_motion_id == i.

        Args:
            save_dir: Directory where checkpoints are saved (or None)
            motion_file: Motion file path to identify checkpoint (or None)
            device: PyTorch device

        Returns:
            List of scene weights from motion training or None if not available
        """
        from pathlib import Path

        if not save_dir or not motion_file:
            return None

        try:
            evaluated_motions = motion_file.split("/")[-1]
            checkpoint_path = Path(save_dir) / f"env_{evaluated_motions}.ckpt"

            if not checkpoint_path.exists():
                return None

            print(f"Loading motion weights from checkpoint: {checkpoint_path}")
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            if "motion_manager" not in checkpoint_data:
                print(
                    "No motion_manager found in checkpoint, using uniform scene weights."
                )
                return None

            motion_weights = checkpoint_data["motion_manager"]["motion_weights"]
            print(f"Applying {len(motion_weights)} motion weights as scene weights")
            print(
                "WARNING: Assumes 1:1 scene-to-motion correspondence (scene[i].humanoid_motion_id == i)"
            )
            return motion_weights.cpu().tolist()

        except Exception as e:
            print(f"Error applying motion weights to scene weights: {e}")
            return None

    def save_state(self) -> dict:
        """Save all mutable env state for later restoration.

        Snapshots the current state of the environment including robot state,
        simulator state, progress/reset/terminate buffers, and state history.
        This is useful for temporarily interrupting normal training to run
        evaluation episodes, then restoring to continue training from where
        it left off.

        Returns:
            Dictionary containing cloned copies of all mutable state tensors
        """
        snapshot = {
            "robot_state": self.simulator.get_robot_state(),
            "markers_state": self.get_markers_state(),
            "actions": self.simulator.get_current_actions(),
            "progress_buf": self.progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "terminate_buf": self.terminate_buf.clone(),
            "respawn_root_offset": self.respawn_root_offset.clone(),
            "odom_scale": self.odom_scale.clone(),
            "odom_yaw_cos_sin": self.odom_yaw_cos_sin.clone(),
            "odom_start_xy": self.odom_start_xy.clone(),
            "odom_start_heading_inv": self.odom_start_heading_inv.clone(),
        }
        if self.state_history is not None:
            snapshot["state_history"] = self.state_history.save_state()
        if self._current_noisy_obs is not None:
            from dataclasses import fields as dc_fields

            noisy = self._current_noisy_obs
            snapshot["_current_noisy_obs"] = NoisyObservations(
                **{f.name: getattr(noisy, f.name).clone() for f in dc_fields(noisy)}
            )
        if self.scene_lib.num_objects_per_scene > 0:
            snapshot["object_state"] = self.simulator.get_object_root_state()
        return snapshot

    def restore_state(self, snapshot: dict) -> None:
        """Restore env state from a previous save_state() snapshot.

        Restores all mutable state that was captured by save_state(),
        including robot positions/velocities, buffers, and state history.

        Args:
            snapshot: Dictionary from save_state() containing state tensors
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.reset_envs(
            snapshot["robot_state"], snapshot.get("object_state"), env_ids
        )

        if "state_history" in snapshot and self.state_history is not None:
            self.state_history.load_state(snapshot["state_history"])

        self.progress_buf.copy_(snapshot["progress_buf"])
        self.reset_buf.copy_(snapshot["reset_buf"])
        self.terminate_buf.copy_(snapshot["terminate_buf"])
        self.respawn_root_offset.copy_(snapshot["respawn_root_offset"])
        if "odom_scale" in snapshot:
            self.odom_scale.copy_(snapshot["odom_scale"])
            self.odom_yaw_cos_sin.copy_(snapshot["odom_yaw_cos_sin"])
        if "odom_start_xy" in snapshot:
            self.odom_start_xy.copy_(snapshot["odom_start_xy"])
        if "odom_start_heading_inv" in snapshot:
            self.odom_start_heading_inv.copy_(snapshot["odom_start_heading_inv"])
        self._current_noisy_obs = snapshot.get("_current_noisy_obs")
        self._current_context = None

        # IsaacGym needs an extra step after state restore to sync internal state
        if "isaacgym" in self.simulator.config._target_.lower():
            self.simulator.step(snapshot["actions"], markers_callback=None)

    def close(self) -> None:
        """Release control-component and env-owned UI handles, then close
        the simulator. Safe to call multiple times."""
        control_manager = getattr(self, "control_manager", None)
        if control_manager is not None:
            for component in control_manager.components.values():
                component.close()

        ui = getattr(self, "_key_bindings", None)
        if ui is not None:
            ui.unregister_all()
            self._key_bindings = None

        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()
