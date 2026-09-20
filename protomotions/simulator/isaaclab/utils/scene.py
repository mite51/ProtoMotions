# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from protomotions.components.terrains.terrain import Terrain
from protomotions.robot_configs.base import RobotConfig
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from protomotions.simulator.isaaclab.utils.usd_utils import TrimeshTerrainImporter
from protomotions.simulator.isaaclab.utils.actuator_groups import (
    build_isaaclab_joint_name_map,
    resolve_actuator_specs_for_control_type,
)
from protomotions.simulator.isaaclab.utils.mjcf_to_usd import convert_robot_mjcf_to_usd
from protomotions.simulator.isaaclab.utils.usd_body_paths import (
    contact_sensor_prim_path,
    resolve_robot_prim_paths,
)
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
from protomotions.simulator.base_simulator.config import ProjectileConfig
from protomotions.robot_configs.base import ControlType


@configclass
class TrimeshTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = TrimeshTerrainImporter

    terrain_type: str = "trimesh"
    terrain_vertices: list = None
    terrain_faces: list = None


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(
        self,
        config: IsaacLabSimulatorConfig,
        robot_config: RobotConfig,
        terrain: Optional[Terrain] = None,
        scene_cfgs=None,
        projectile_config: Optional[ProjectileConfig] = None,
        pretty=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        activate_contact_sensors = robot_config.contact_bodies is not None

        # lights
        if True:  # pretty:
            # This is way prettier, but also slower to render
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=750.0,
                    texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
                ),
            )
        else:
            self.light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)
                ),
            )

        num_objects_per_scene = 0
        if scene_cfgs is not None:
            num_objects_per_scene = len(scene_cfgs)
            for obj_idx, obj_configs in enumerate(scene_cfgs):
                spawn_cfg = sim_utils.MultiAssetSpawnerCfg(
                    activate_contact_sensors=activate_contact_sensors,
                    assets_cfg=obj_configs,
                    random_choice=False,
                )
                # Rigid Object
                object = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
                setattr(self, f"object_{obj_idx}", object)

                # Object contact sensors are used to detect collisions between objects.
                object_contact_paths = ["/World/ground/terrain/mesh"]
                for i in range(num_objects_per_scene):
                    if i != obj_idx:
                        object_contact_paths.append(f"/World/envs/env_.*/Object_{i}")
                if activate_contact_sensors:
                    object_sensor_cfg = ContactSensorCfg(
                        prim_path=f"/World/envs/env_.*/Object_{obj_idx}",
                        # debug_vis=True,
                        filter_prim_paths_expr=object_contact_paths,
                        history_length=config.sim.decimation,
                    )
                    setattr(self, f"object_{obj_idx}_contact_sensor", object_sensor_cfg)

        # Projectile rigid objects (always created, independent of scene objects).
        # Each pool index gets a primitive shape (box/sphere/capsule) from the config;
        # only the spawn geometry differs, all share the same physics/visual props.
        if projectile_config is not None:
            proj_specs = projectile_config.get_shape_specs()
            shared_props = dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    enable_gyroscopic_forces=True,
                ),
                mass_props=sim_utils.MassPropertiesCfg(
                    density=projectile_config.density
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.02,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.8, 0.1, 0.1)
                ),
            )
            for proj_idx in range(projectile_config.num_projectiles):
                spec = proj_specs[proj_idx]
                if spec.shape_type == "sphere":
                    spawn_cfg = sim_utils.SphereCfg(radius=spec.radius, **shared_props)
                elif spec.shape_type == "capsule":
                    # extent_z is the capsule cylinder length; default axis is Z.
                    spawn_cfg = sim_utils.CapsuleCfg(
                        radius=spec.radius, height=spec.extent_z, **shared_props
                    )
                else:  # "box"
                    spawn_cfg = sim_utils.CuboidCfg(
                        size=(spec.extent_z, spec.extent_z, spec.extent_z),
                        **shared_props,
                    )
                proj_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Projectile_{proj_idx}",
                    spawn=spawn_cfg,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(
                            float(proj_idx),
                            float(proj_idx),
                            projectile_config.hidden_z_for_index(proj_idx),
                        )
                    ),
                )
                setattr(self, f"projectile_{proj_idx}", proj_cfg)

        actuators = {}
        ActuatorConfig = (
            ImplicitActuatorCfg
            if robot_config.control.control_type == ControlType.BUILT_IN_PD
            else IdealPDActuatorCfg
        )
        joint_names = build_isaaclab_joint_name_map(robot_config.kinematic_info)
        isaaclab_control_info = {
            joint_names.semantic_to_backend[name]: control_info
            for name, control_info in robot_config.control.control_info.items()
        }
        actuator_groups = resolve_actuator_specs_for_control_type(
            isaaclab_control_info, robot_config.control.control_type
        )
        for actuator_group in actuator_groups:
            actuators[actuator_group.name] = ActuatorConfig(
                joint_names_expr=list(actuator_group.joint_names_expr),
                **actuator_group.params,
            )

        # Derive USD from the robot MJCF via IsaacLab 3 MjcfConverter.
        robot_usd_path = convert_robot_mjcf_to_usd(robot_config.asset)
        contact_body_names = (
            robot_config.contact_bodies if activate_contact_sensors else []
        )
        articulation_root_prim_path, body_prim_paths = resolve_robot_prim_paths(
            robot_usd_path,
            contact_body_names,
        )

        # articulation(s). For multi-character self-play (num_characters > 1) we spawn
        # N identical articulations per env (Robot_0..Robot_{N-1}) that physically
        # interact within the scene. For N == 1 we keep the legacy prim name "Robot"
        # and attribute "robot" so single-character behavior is byte-for-byte identical.
        from protomotions.simulator.base_simulator.utils import (
            character_spawn_offsets,
        )

        num_characters = getattr(config, "num_characters", 1)
        spawn_offsets = character_spawn_offsets(
            num_characters, getattr(config, "character_spawn_radius", 1.0)
        )

        def _robot_name(idx: int) -> str:
            return "Robot" if num_characters == 1 else f"Robot_{idx}"

        def _attr_name(idx: int) -> str:
            return "robot" if num_characters == 1 else f"robot_{idx}"

        for char_idx in range(num_characters):
            robot_name = _robot_name(char_idx)
            offset_x, offset_y = spawn_offsets[char_idx]
            robot_cfg = ArticulationCfg(
                prim_path=f"/World/envs/env_.*/{robot_name}",
                articulation_root_prim_path=articulation_root_prim_path,
                spawn=sim_utils.UsdFileCfg(
                    usd_path=robot_usd_path,
                    activate_contact_sensors=activate_contact_sensors,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=robot_config.asset.disable_gravity,
                        retain_accelerations=False,
                        linear_damping=robot_config.asset.linear_damping,
                        angular_damping=robot_config.asset.angular_damping,
                        max_linear_velocity=robot_config.asset.max_linear_velocity,
                        max_angular_velocity=robot_config.asset.max_angular_velocity,
                        max_depenetration_velocity=config.sim.physx.max_depenetration_velocity,
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=robot_config.asset.self_collisions,
                        solver_position_iteration_count=config.sim.physx.num_position_iterations,
                        solver_velocity_iteration_count=config.sim.physx.num_velocity_iterations,
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=config.sim.physx.contact_offset,
                        rest_offset=config.sim.physx.rest_offset,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.9, 0.9), metallic=0.5
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(offset_x, offset_y, robot_config.default_root_height),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators=actuators,
            )

            # Apply disable_gravity setting for all robot types if specified
            if (
                hasattr(robot_config.asset, "disable_gravity")
                and robot_config.asset.disable_gravity
            ):
                # Only modify disable_gravity field, keeping all other settings
                new_rigid_props = robot_cfg.spawn.rigid_props.replace(
                    disable_gravity=True
                )
                robot_cfg.spawn = robot_cfg.spawn.replace(rigid_props=new_rigid_props)

            setattr(self, _attr_name(char_idx), robot_cfg)

            if activate_contact_sensors:
                sensing_filter = ["/World/ground/terrain/mesh"]
                for obj_idx in range(num_objects_per_scene):
                    sensing_filter.append(f"/World/envs/env_.*/Object_{obj_idx}")
                # Body prim root for this character: substitute the robot name into
                # the configured "/Robot/" path. Inter-character contact forces are
                # captured by net_forces_w regardless of the filter list, so we do
                # not add other characters to the filter (opponent-impact reward is
                # computed geometrically, not via contact-object resolution).
                for body_name in robot_config.contact_bodies:
                    contact_sensor_cfg = ContactSensorCfg(
                        prim_path=contact_sensor_prim_path(body_name, body_prim_paths).replace("/Robot/", f"/{robot_name}/"),
                        filter_prim_paths_expr=sensing_filter,
                        history_length=config.sim.decimation,
                    )
                    if num_characters == 1:
                        sensor_attr = f"contact_sensor_{body_name}"
                    else:
                        sensor_attr = (
                            f"contact_sensor_{_attr_name(char_idx)}_{body_name}"
                        )
                    setattr(self, sensor_attr, contact_sensor_cfg)

        if terrain is not None:
            terrain_physics_material = sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=terrain.sim_config.combine_mode.value,
                restitution_combine_mode=terrain.sim_config.combine_mode.value,
                static_friction=terrain.sim_config.static_friction,
                dynamic_friction=terrain.sim_config.dynamic_friction,
                restitution=terrain.sim_config.restitution,
            )
            terrain_visual_material = sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            )

            vertices = terrain.vertices
            height_offset = terrain.sim_config.height_offset
            vertices[..., 2] += height_offset

            self.terrain = TrimeshTerrainImporterCfg(
                prim_path="/World/ground",
                # Pass the mesh data instead of the mesh object
                terrain_vertices=vertices.tolist(),
                terrain_faces=terrain.triangles,
                collision_group=-1,
                visual_material=terrain_visual_material,
                physics_material=terrain_physics_material,
            )
        else:
            self.terrain = None
