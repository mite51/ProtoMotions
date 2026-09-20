"""Regression tests for the main-based observation ablation and runtime weakness."""
import argparse
from dataclasses import asdict

import pytest
import torch

from examples.experiments.mimic import fight, fight_stamina, mlp
from protomotions.envs.base_env.env import BaseEnv
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
from protomotions.envs.rewards.regularization import compute_contact_match_rew
from protomotions.utils.config_utils import clean_dict_for_storage


def _configs(experiment):
    args = argparse.Namespace(motion_file="baseline.pt", scenes_file=None,
                              batch_size=32, training_max_steps=2048)
    robot = robot_config("smpl")
    sim = IsaacLabSimulatorConfig(headless=True, num_envs=32, experiment_name="test")
    experiment.configure_robot_and_simulator(robot, sim, args)
    env = experiment.env_config(robot, args)
    agent = experiment.agent_config(robot, env, args)
    return robot, sim, env, agent


def test_fighting_baseline_preserves_training_and_physics():
    robot, sim, env, agent = _configs(mlp)
    ext_robot, ext_sim, ext, ext_agent = _configs(fight)
    assert asdict(sim) == asdict(ext_sim)
    assert asdict(robot.control) == asdict(ext_robot.control)
    assert asdict(env.motion_manager) == asdict(ext.motion_manager)
    assert env.action_config.keys() == ext.action_config.keys()
    for key in env.action_config:
        a, b = env.action_config[key], ext.action_config[key]
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    assert env.reward_components.keys() == ext.reward_components.keys()
    for key in env.reward_components:
        a, b = env.reward_components[key], ext.reward_components[key]
        assert a.compute_func is b.compute_func
        if key != "contact_match_rew":
            assert a.static_params == b.static_params
            assert str(a.dynamic_vars) == str(b.dynamic_vars)
    # Every agent setting must match after replacing the added input keys.
    for comp, base in ((ext_agent.model, agent.model),
                       (ext_agent.model.actor, agent.model.actor),
                       (ext_agent.model.actor.mu_model, agent.model.actor.mu_model),
                       (ext_agent.model.critic, agent.model.critic)):
        assert comp.in_keys == fight.OBS_IN_KEYS
        comp.in_keys = list(base.in_keys)
    assert clean_dict_for_storage(asdict(agent)) == clean_dict_for_storage(asdict(ext_agent))
    assert not ext.randomize_body_stamina
    assert not ext.sanitize_non_finite_state


def test_contact_reward_remains_feet_only_with_extra_sensors():
    robot, _, env, _ = _configs(fight)
    ids = env.reward_components["contact_match_rew"].static_params["contact_body_ids"]
    assert [robot.kinematic_info.body_names[i] for i in ids] == ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
    sim = torch.zeros(2, 24)
    ref = sim.clone()
    sim[:, robot.kinematic_info.body_names.index("L_Hand")] = 1
    reward = compute_contact_match_rew(sim, ref, ids)
    assert torch.equal(reward, torch.zeros(2))


def test_stamina_stage_changes_only_strength_randomization():
    _, sim, base, agent = _configs(fight)
    _, weak_sim, weak, weak_agent = _configs(fight_stamina)
    assert weak.randomize_body_stamina
    assert weak.body_stamina_range == (0.75, 1.0)
    assert clean_dict_for_storage(asdict(agent)) == clean_dict_for_storage(asdict(weak_agent))
    assert asdict(sim) == asdict(weak_sim)
    assert base.reward_components.keys() == weak.reward_components.keys()
    assert asdict(base.motion_manager) == asdict(weak.motion_manager)


def test_runtime_damage_updates_only_selected_character_and_observation():
    env = BaseEnv.__new__(BaseEnv)
    env.device = torch.device("cpu")
    env.num_envs, env._num_stamina_bodies = 3, 2
    env._extended_observations_enabled = True
    env._body_stamina = torch.ones(3, 2)
    env._observation_buffer = {"stamina_obs": torch.ones(3, 2)}
    calls = []
    env._apply_body_stamina_to_gains = lambda ids: calls.append((ids.clone(), env._body_stamina[ids].clone()))
    env._current_context = object()
    env.set_body_stamina(torch.tensor([[0., 0.5]]), torch.tensor([1]))
    assert torch.equal(env._body_stamina, torch.tensor([[1., 1.], [0., .5], [1., 1.]]))
    assert torch.equal(env._observation_buffer["stamina_obs"], env._body_stamina)
    assert env._current_context is None
    assert calls[0][0].tolist() == [1]
    with pytest.raises(ValueError):
        env.set_body_stamina(torch.tensor([[float("nan"), 0.5]]), torch.tensor([1]))
    with pytest.raises(ValueError):
        env.set_body_stamina(torch.tensor([[1.1, 0.5]]), torch.tensor([1]))


def test_contact_smoothing_restores_training_autotuning():
    from protomotions.components.motion_lib import MotionLib
    lib = MotionLib.empty()
    lib.contacts = torch.tensor([[0.], [1.], [0.], [1.], [1.], [1.]])
    lib.length_starts = torch.tensor([0, 3])
    lib.motion_num_frames = torch.tensor([3, 3])
    lib.motion_lengths = torch.tensor([1., 1.])
    original = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.benchmark = True
        lib.smooth_contacts(3)
        assert torch.backends.cudnn.benchmark
        torch.testing.assert_close(lib.contacts[:, 0], torch.tensor([1/3, 1/3, 1/3, 1., 1., 1.]))
    finally:
        torch.backends.cudnn.benchmark = original


def test_projectile_velocity_is_stable_across_context_reads():
    from types import SimpleNamespace
    env = BaseEnv.__new__(BaseEnv)
    env.device = torch.device('cpu')
    env.num_envs = env.num_characters = env._num_projectiles = 1
    env.dt = 0.1
    env.config = SimpleNamespace(max_collision_primitives=2,
                                 ground_primitive_damage=0., static_collider_effective_mass=100.)
    env._prev_projectile_pos = torch.zeros(1, 1, 3)
    env._prev_projectile_active = torch.zeros(1, 1)
    env._projectile_velocity = torch.zeros(1, 1, 3)
    env._projectile_damage = torch.ones(1, 1)
    env._projectile_masses = torch.ones(1)
    projectile = dict(positions=torch.zeros(1, 1, 3),
                      rotations=torch.tensor([[[0., 0., 0., 1.]]]), active=torch.ones(1, 1),
                      radius=torch.zeros(1), extent_z=torch.ones(1), shape=torch.tensor([[1., 0.]]))
    env.simulator = SimpleNamespace(get_active_projectile_states=lambda: projectile)
    env._update_projectile_velocity()
    assert env._projectile_velocity.count_nonzero() == 0
    projectile['positions'][..., 0] += 1.
    env._update_projectile_velocity()
    state = SimpleNamespace(rigid_body_pos=torch.zeros(1, 1, 3))
    first = env._build_collision_primitives(state, torch.zeros(1))
    second = env._build_collision_primitives(state, torch.zeros(1))
    torch.testing.assert_close(first.lin_vel, second.lin_vel)
    assert second.lin_vel[0, 1, 0] == 10.
