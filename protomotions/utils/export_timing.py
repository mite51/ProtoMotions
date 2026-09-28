# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timing contract shared by the simulator-free tracker exporters.

Training cadence comes only from the checkpoint's saved training config.
An explicit target conversion is a deployment choice, not training evidence.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path


def _get(config, name):
    return config.get(name) if isinstance(config, dict) else getattr(config, name, None)


def _cadence(simulator, future_steps, source):
    sim = _get(simulator, "sim")
    fps, decimation = _get(sim, "fps"), _get(sim, "decimation")
    try:
        if isinstance(fps, bool) or isinstance(decimation, bool):
            raise ValueError
        fps, decimation = float(fps), float(decimation)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError
        if not math.isfinite(decimation) or decimation <= 0 or not decimation.is_integer():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"Invalid or missing simulator.sim.fps / decimation in {source}; "
            "export requires saved timing, with positive finite fps and positive "
            "integer decimation. No simulator preset fallback is used."
        ) from exc
    physics_dt = 1.0 / fps
    control_dt = physics_dt * int(decimation)
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError(f"Invalid derived control_dt in {source}")
    return {
        "simulator": _get(simulator, "_target_"),
        "physics_fps": fps,
        "dt": control_dt,
        "control_dt": control_dt,
        "physics_dt": physics_dt,
        "decimation": int(decimation),
        "future_step_indices": list(future_steps),
        "future_dt_seconds": [step * control_dt for step in future_steps],
    }


def _provenance(path):
    return {
        "config_path": str(path.resolve()),
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fields": ["simulator.sim.fps", "simulator.sim.decimation"],
    }


def resolve_export_timing(checkpoint, resolved, resolved_path, future_steps, target_simulator=None):
    """Read training timing independently of the config used to build the graph.

    Require resolved_configs.pt even when only an inference config is available:
    inference overrides can change timing, so it cannot prove training cadence.
    Preserve both cadence records when the caller explicitly requests conversion.
    """
    import torch

    training_path = Path(checkpoint).parent / "resolved_configs.pt"
    if not training_path.is_file():
        raise FileNotFoundError(
            f"Saved training config required for timing: {training_path}. "
            "An inference config or simulator preset cannot establish training cadence."
        )
    training = (
        resolved if training_path.resolve() == Path(resolved_path).resolve()
        else torch.load(training_path, map_location="cpu", weights_only=False)
    )
    training_cadence = _cadence(training.get("simulator"), future_steps, training_path)
    training_cadence["provenance"] = _provenance(training_path)
    target_cadence = None
    if target_simulator is not None:
        from protomotions.simulator.factory import update_simulator_config_for_test

        # The conversion mutates BOTH simulator and robot configs. Keep graph
        # configuration and the saved training evidence untouched.
        robot = deepcopy(resolved["robot"])
        simulator = deepcopy(resolved.get("simulator"))
        if simulator is None:
            raise ValueError("Target conversion requires an export simulator config")
        params = _get(robot, "simulation_params")
        has_preset = hasattr(params, target_simulator)
        converted = update_simulator_config_for_test(simulator, target_simulator, robot)
        target_cadence = _cadence(converted, future_steps, f"target simulator {target_simulator}")
        target_cadence["provenance"] = {
            **_provenance(Path(resolved_path)),
            "fields": [f"robot.simulation_params.{target_simulator}"] if has_preset else [],
            "conversion": "protomotions.simulator.factory.update_simulator_config_for_test",
            "requested_simulator": target_simulator,
            "parameters_source": "saved_robot_preset" if has_preset else "simulator_class_defaults",
        }

    selected = target_cadence if target_cadence is not None else training_cadence
    export_provenance = _provenance(Path(resolved_path))
    export_provenance["fields"] = ["env.control_components.mimic.future_steps"]
    return {
        "dt": selected["dt"],
        "control_dt": selected["control_dt"],
        "physics_dt": selected["physics_dt"],
        "decimation": selected["decimation"],
        "basis": "target_simulator" if target_cadence is not None else "training",
        "training": training_cadence,
        "target_simulator": target_cadence,
        "export_config": export_provenance,
    }


def future_dt_seconds(timing, future_steps):
    """Use the same policy interval as dt and control_dt, without rounding drift."""
    return [step * timing["control_dt"] for step in future_steps]
