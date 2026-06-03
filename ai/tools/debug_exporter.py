# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""Debug data exporter for Unity deployment verification.

This module provides frame-by-frame data export for comparing Python simulation
with Unity deployment. Used to debug discrepancies in observation computation,
action application, and physics simulation.

Example usage:
    exporter = DebugExporter(output_path="debug_output.json", simulator=sim, env=env)
    
    # In simulation loop:
    exporter.capture_frame(frame_idx=step, obs_dict=obs, actions=actions, ...)
    
    # After simulation:
    exporter.save()
"""

import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict


def tensor_to_list(t: torch.Tensor) -> List:
    """Convert tensor to nested Python list for JSON serialization."""
    if t is None:
        return None
    return t.detach().cpu().numpy().tolist()


def ensure_list(value, expected_length: int) -> List:
    """Ensure value is a list of expected length. Expands scalars to uniform lists."""
    if value is None:
        return [0.0] * expected_length
    if isinstance(value, (int, float)):
        return [float(value)] * expected_length
    if isinstance(value, list):
        if len(value) == expected_length:
            return value
        elif len(value) == 1:
            return value * expected_length
    return value


@dataclass
class SimulationParams:
    """Simulation parameters that remain constant throughout the run."""
    dt: float
    decimation: int
    physics_fps: float
    policy_fps: float
    motion_length: float
    total_expected_frames: int
    physics_substeps_per_frame: int
    control_type: str
    pd_action_offset: List[float]
    pd_action_scale: List[float]
    p_gains: List[float]
    d_gains: List[float]
    effort_limits: List[float]
    dof_names: List[str]
    body_names: List[str]
    num_bodies: int
    num_dofs: int
    respawn_root_offset: List[float]  # [3] x,y,z in sim coordinates for exported env
    motion_offset: List[float]  # [3] pelvis x,y,z at motion time 0 (sim coordinates)


@dataclass
class FrameData:
    """Data captured for a single simulation frame."""
    frame_idx: int
    simulation_time: float
    motion_time: float
    
    # Robot state before action (in simulator coordinates - Z-up for IsaacLab)
    rigid_body_pos: List[List[float]]  # [num_bodies, 3]
    rigid_body_rot: List[List[float]]  # [num_bodies, 4] quaternion wxyz
    rigid_body_vel: List[List[float]]  # [num_bodies, 3]
    rigid_body_ang_vel: List[List[float]]  # [num_bodies, 3]
    dof_pos: List[float]  # [num_dofs]
    dof_vel: List[float]  # [num_dofs]
    
    # Observations (as sent to model)
    max_coords_obs: List[float]  # [358]
    mimic_target_poses: List[float]  # [577]
    historical_previous_actions: List[float]  # [69]
    
    # Model output
    actions: List[float]  # [69] raw model output in [-1, 1]
    
    # Control computation
    pd_targets: List[float]  # [69] offset + scale * action (the position targets sent to PD controller)
    
    # Control action targets from robot._data (what IsaacLab's actuator model receives)
    # These are used to compute effort: stiffness*(pos_target-pos) + damping*(vel_target-vel) + effort_target
    joint_pos_target: Optional[List[float]] = None  # [69] position targets from robot._data.joint_pos_target
    joint_vel_target: Optional[List[float]] = None  # [69] velocity targets (typically 0)
    joint_effort_target: Optional[List[float]] = None  # [69] feedforward effort targets (typically 0)
    
    # Articulation cache data
    dof_acc: Optional[List[float]] = None  # [num_dofs] joint accelerations from articulation cache
    
    # Target motion data (ground truth from motion lib)
    target_body_pos: Optional[List[List[float]]] = None
    target_body_rot: Optional[List[List[float]]] = None
    target_body_vel: Optional[List[List[float]]] = None
    target_body_ang_vel: Optional[List[List[float]]] = None


@dataclass 
class DebugExportData:
    """Complete debug export containing metadata, params, and all frames."""
    metadata: Dict[str, Any]
    simulation_params: SimulationParams
    frames: List[FrameData] = field(default_factory=list)


class DebugExporter:
    """Exports frame-by-frame debug data for Unity verification."""
    
    def __init__(self, output_path: str, simulator, env, max_frames: int = 1000, env_idx: int = 0):
        self.output_path = Path(output_path)
        self.simulator = simulator
        self.env = env
        self.max_frames = max_frames
        self.env_idx = env_idx
        self.frames: List[FrameData] = []
        self._initialized = False
        self._simulation_params: Optional[SimulationParams] = None
        
    def _initialize_params(self) -> SimulationParams:
        """Extract simulation parameters from simulator and env."""
        sim = self.simulator
        robot_cfg = sim.robot_config
        
        motion_length = 0.0
        if hasattr(self.env, 'motion_lib') and self.env.motion_lib is not None:
            motion_lengths = self.env.motion_lib.get_motion_length(None)
            if motion_lengths is not None and len(motion_lengths) > 0:
                motion_length = motion_lengths[0].item()
        
        total_expected_frames = int(motion_length / sim.dt) if motion_length > 0 else self.max_frames
        num_dofs = robot_cfg.kinematic_info.num_dofs
        
        # Get raw values (may be scalars or arrays)
        pd_offset_raw = tensor_to_list(sim._common_pd_action_offset[0]) if hasattr(sim, '_common_pd_action_offset') else 0.0
        pd_scale_raw = tensor_to_list(sim._common_pd_action_scale[0]) if hasattr(sim, '_common_pd_action_scale') else 3.14159
        p_gains_raw = tensor_to_list(sim._common_p_gains[0]) if hasattr(sim, '_common_p_gains') else 0.0
        d_gains_raw = tensor_to_list(sim._common_d_gains[0]) if hasattr(sim, '_common_d_gains') else 0.0
        effort_limits_raw = tensor_to_list(sim._torque_limits_common[0]) if hasattr(sim, '_torque_limits_common') else 500.0
        
        # Ensure all are lists of correct length (expands scalars to uniform arrays)
        pd_offset = ensure_list(pd_offset_raw, num_dofs)
        pd_scale = ensure_list(pd_scale_raw, num_dofs)
        p_gains = ensure_list(p_gains_raw, num_dofs)
        d_gains = ensure_list(d_gains_raw, num_dofs)
        effort_limits = ensure_list(effort_limits_raw, num_dofs)
        
        if hasattr(self.env, 'respawn_root_offset') and self.env.respawn_root_offset is not None:
            respawn_root_offset = tensor_to_list(self.env.respawn_root_offset[self.env_idx])
        else:
            respawn_root_offset = [0.0, 0.0, 0.0]
        
        try:
            if (
                hasattr(self.env, 'motion_manager')
                and hasattr(self.env, 'motion_lib')
                and self.env.motion_lib is not None
                and getattr(self.env.motion_manager, 'motion_ids', None) is not None
            ):
                device = getattr(self.env, 'device', None) or getattr(sim, 'device', None)
                if device is not None:
                    motion_ids = self.env.motion_manager.motion_ids[self.env_idx : self.env_idx + 1]
                    time = torch.zeros(1, device=device, dtype=torch.float)
                    frame0_state = self.env.motion_lib.get_motion_state(motion_ids, time)
                    motion_offset = tensor_to_list(frame0_state.rigid_body_pos[0, 0])
                else:
                    motion_offset = [0.0, 0.0, 0.0]
            else:
                motion_offset = [0.0, 0.0, 0.0]
        except Exception:
            motion_offset = [0.0, 0.0, 0.0]
        
        return SimulationParams(
            dt=sim.dt,
            decimation=sim.decimation,
            physics_fps=sim.config.sim.fps,
            policy_fps=sim.config.sim.fps / sim.decimation,
            motion_length=motion_length,
            total_expected_frames=total_expected_frames,
            physics_substeps_per_frame=sim.decimation,
            control_type=str(sim.control_type.name),
            pd_action_offset=pd_offset,
            pd_action_scale=pd_scale,
            p_gains=p_gains,
            d_gains=d_gains,
            effort_limits=effort_limits,
            dof_names=robot_cfg.kinematic_info.dof_names,
            body_names=robot_cfg.kinematic_info.body_names,
            num_bodies=robot_cfg.kinematic_info.num_bodies,
            num_dofs=robot_cfg.kinematic_info.num_dofs,
            respawn_root_offset=respawn_root_offset,
            motion_offset=motion_offset,
        )
    
    def capture_frame(self, frame_idx: int, simulation_time: float, motion_time: float,
                      obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor,
                      robot_state=None, target_state=None) -> bool:
        """Capture data for a single frame. Returns False if max frames reached."""
        if len(self.frames) >= self.max_frames:
            return False
            
        if not self._initialized:
            self._simulation_params = self._initialize_params()
            self._initialized = True
        
        idx = self.env_idx
        
        if robot_state is None:
            robot_state = self.simulator.get_robot_state()
        
        max_coords_obs = tensor_to_list(obs_dict.get('max_coords_obs', torch.zeros(358))[idx])
        mimic_target_poses = tensor_to_list(obs_dict.get('mimic_target_poses', torch.zeros(577))[idx])
        historical_actions = tensor_to_list(obs_dict.get('historical_previous_actions', torch.zeros(69))[idx])
        
        actions_single = actions[idx]
        pd_offset = self.simulator._common_pd_action_offset[idx] if hasattr(self.simulator, '_common_pd_action_offset') else torch.zeros_like(actions_single)
        pd_scale = self.simulator._common_pd_action_scale[idx] if hasattr(self.simulator, '_common_pd_action_scale') else torch.ones_like(actions_single) * 3.14159
        pd_targets = pd_offset + pd_scale * actions_single
        
        # Get control action targets from robot._data (what IsaacLab's actuator model receives)
        # These values are used in the actuator model: effort = stiffness*(pos_target-pos) + damping*(vel_target-vel) + effort_target
        joint_pos_target = None
        joint_vel_target = None
        joint_effort_target = None
        
        if hasattr(self.simulator, '_robot') and hasattr(self.simulator._robot, '_data'):
            robot_data = self.simulator._robot._data
            if hasattr(robot_data, 'joint_pos_target') and robot_data.joint_pos_target is not None:
                joint_pos_target = tensor_to_list(robot_data.joint_pos_target[idx])
            if hasattr(robot_data, 'joint_vel_target') and robot_data.joint_vel_target is not None:
                joint_vel_target = tensor_to_list(robot_data.joint_vel_target[idx])
            if hasattr(robot_data, 'joint_effort_target') and robot_data.joint_effort_target is not None:
                joint_effort_target = tensor_to_list(robot_data.joint_effort_target[idx])
        
        # Get joint velocity from articulation cache
        dof_vel = None
        if hasattr(self.simulator, '_robot') and hasattr(self.simulator._robot, '_data'):
            robot_data = self.simulator._robot._data
            if hasattr(robot_data, 'joint_vel') and robot_data.joint_vel is not None:
                dof_vel = tensor_to_list(robot_data.joint_vel[idx])
        if dof_vel is None:
            print(f"*** dof_vel is None")

        # Get joint acceleration from articulation cache
        dof_acc = None
        if hasattr(self.simulator, '_robot') and hasattr(self.simulator._robot, '_data'):
            robot_data = self.simulator._robot._data
            if hasattr(robot_data, 'joint_acc') and robot_data.joint_acc is not None:
                dof_acc = tensor_to_list(robot_data.joint_acc[idx])
        
        target_body_pos = tensor_to_list(target_state.rigid_body_pos[idx]) if target_state else None
        target_body_rot = tensor_to_list(target_state.rigid_body_rot[idx]) if target_state else None
        target_body_vel = tensor_to_list(target_state.rigid_body_vel[idx]) if target_state and hasattr(target_state, 'rigid_body_vel') and target_state.rigid_body_vel is not None else None
        target_body_ang_vel = tensor_to_list(target_state.rigid_body_ang_vel[idx]) if target_state and hasattr(target_state, 'rigid_body_ang_vel') and target_state.rigid_body_ang_vel is not None else None
        
        frame = FrameData(
            frame_idx=frame_idx,
            simulation_time=simulation_time,
            motion_time=motion_time,
            rigid_body_pos=tensor_to_list(robot_state.rigid_body_pos[idx]),
            rigid_body_rot=tensor_to_list(robot_state.rigid_body_rot[idx]),
            rigid_body_vel=tensor_to_list(robot_state.rigid_body_vel[idx]),
            rigid_body_ang_vel=tensor_to_list(robot_state.rigid_body_ang_vel[idx]),
            dof_pos=tensor_to_list(robot_state.dof_pos[idx]),
            dof_vel=dof_vel,#tensor_to_list(robot_state.dof_vel[idx]),
            max_coords_obs=max_coords_obs,
            mimic_target_poses=mimic_target_poses,
            historical_previous_actions=historical_actions,
            actions=tensor_to_list(actions_single),
            pd_targets=tensor_to_list(pd_targets),
            joint_pos_target=joint_pos_target,
            joint_vel_target=joint_vel_target,
            joint_effort_target=joint_effort_target,
            dof_acc=dof_acc,
            target_body_pos=target_body_pos,
            target_body_rot=target_body_rot,
            target_body_vel=target_body_vel,
            target_body_ang_vel=target_body_ang_vel,
        )
        
        self.frames.append(frame)
        return True

    def set_action(self, frame_idx: int, action: torch.Tensor) -> None:
        """Set the action to the capture frame date."""
        self.frames[frame_idx].actions = tensor_to_list(action)
    
    def save(self) -> str:
        """Save captured data to JSON file."""
        if not self._initialized:
            self._simulation_params = self._initialize_params()
        
        metadata = {
            "export_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "motion_file": getattr(self.env.motion_lib.config, 'motion_file', 'unknown') if hasattr(self.env, 'motion_lib') else 'unknown',
            "num_frames_captured": len(self.frames),
            "env_idx": self.env_idx,
        }
        
        export_data = DebugExportData(metadata=metadata, simulation_params=self._simulation_params, frames=self.frames)
        export_dict = asdict(export_data)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            json.dump(export_dict, f, indent=2)
        
        print(f"[DebugExporter] Saved {len(self.frames)} frames to {self.output_path}")
        return str(self.output_path)
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def is_full(self) -> bool:
        return len(self.frames) >= self.max_frames
