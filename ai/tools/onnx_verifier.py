# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""ONNX side-by-side verification for comparing PyTorch and ONNX inference.

This module provides the ONNXVerifier class that loads an exported ONNX model
and compares its outputs frame-by-frame against the live PyTorch model during
inference. Used to verify that the ONNX export is numerically correct.

Example usage in inference_agent.py::

    python protomotions/inference_agent.py \\
        --checkpoint data/pretrained_models/motion_tracker/smpl/last.ckpt \\
        --motion-file data/yaml_files/amass_test_single.yaml \\
        --simulator isaaclab \\
        --verify-onnx onnx/smpl_policy/model.onnx
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tensordict import TensorDict


class ONNXVerifier:
    """Runs ONNX inference side-by-side with PyTorch and compares outputs.

    Loads an ONNX model (and its companion metadata JSON), then on each frame:
    1. Prepares the same observation inputs for ONNX
    2. Runs ONNX inference via onnxruntime
    3. Compares each deterministic output with the PyTorch result
    4. Accumulates per-frame error statistics

    Args:
        onnx_path: Path to the .onnx model file. Expects a companion .json
            metadata file at the same location (with input/output key mappings).
    """

    # Outputs that involve random sampling -- report but don't fail on mismatch
    STOCHASTIC_KEYS = {"action", "neglogp"}

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.onnx_path = Path(onnx_path)
        meta_path = self.onnx_path.with_suffix(".json")

        # Load ONNX model
        print(f"\n=== Loading ONNX model for verification ===")
        print(f"  ONNX file: {self.onnx_path}")
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )

        # Get ONNX input/output names
        self.onnx_input_names = [inp.name for inp in self.session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.session.get_outputs()]

        # Load metadata for semantic key mapping
        self.metadata: Dict = {}
        self.semantic_in_keys: List[str] = []
        self.semantic_out_keys: List[str] = []

        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.semantic_in_keys = self.metadata.get("in_keys", [])
            self.semantic_out_keys = self.metadata.get("out_keys", [])
            print(f"  Metadata file: {meta_path}")
        else:
            print(f"  WARNING: No metadata file found at {meta_path}")
            print(f"  Will attempt to match ONNX names to observation keys directly.")

        # Build output index mapping: semantic_key -> index in ort_output list
        self._out_key_to_idx: Dict[str, int] = {}
        if self.semantic_out_keys:
            for i, key in enumerate(self.semantic_out_keys):
                self._out_key_to_idx[key] = i
        else:
            # Fallback: use onnx output names as keys
            for i, name in enumerate(self.onnx_output_names):
                self._out_key_to_idx[name] = i

        # Per-frame error tracking
        # Each entry: (frame_idx, {key: (max_err, mean_err)})
        self._frame_errors: List[Dict] = []
        self._num_frames = 0
        self._num_mismatches = 0  # frames where a deterministic output exceeds tolerance
        self._tolerance = 1e-4  # absolute tolerance for PASS/FAIL

    def print_info(self):
        """Print ONNX model info at startup."""
        print(f"\n=== ONNX Side-by-Side Verification ===")
        print(f"  ONNX model:    {self.onnx_path}")
        print(f"  Input keys:    {self.semantic_in_keys or self.onnx_input_names}")
        print(f"  Output keys:   {self.semantic_out_keys or self.onnx_output_names}")
        print(f"  ONNX inputs:   {self.onnx_input_names}")
        print(f"  ONNX outputs:  {self.onnx_output_names}")
        print(f"  Tolerance:     {self._tolerance}")
        print(f"  Stochastic (skip strict check): {self.STOCHASTIC_KEYS}")
        print()

    def _prepare_onnx_input(self, obs_td: TensorDict) -> Dict[str, np.ndarray]:
        """Convert observation TensorDict to ONNX input dict.

        Maps semantic observation keys to ONNX input tensor names using the
        metadata mapping.
        """
        onnx_input = {}
        if self.semantic_in_keys:
            # Use metadata mapping: semantic keys -> ONNX names
            for onnx_name, semantic_key in zip(
                self.onnx_input_names, self.semantic_in_keys
            ):
                if semantic_key in obs_td.keys():
                    onnx_input[onnx_name] = (
                        obs_td[semantic_key].detach().cpu().numpy()
                    )
                else:
                    raise KeyError(
                        f"Observation key '{semantic_key}' required by ONNX model "
                        f"not found in obs_td. Available keys: {list(obs_td.keys())}"
                    )
        else:
            # Fallback: try to match ONNX names directly to obs keys
            for inp in self.session.get_inputs():
                if inp.name in obs_td.keys():
                    onnx_input[inp.name] = (
                        obs_td[inp.name].detach().cpu().numpy()
                    )
                else:
                    raise KeyError(
                        f"ONNX input '{inp.name}' not found in obs_td and no "
                        f"metadata mapping available."
                    )
        return onnx_input

    def compare_frame(
        self,
        frame_idx: int,
        obs_td: TensorDict,
        pytorch_outs: TensorDict,
    ) -> bool:
        """Run ONNX inference and compare with PyTorch outputs for one frame.

        Args:
            frame_idx: Current frame index (for logging).
            obs_td: The observation TensorDict fed to the PyTorch model.
            pytorch_outs: The TensorDict output from the PyTorch model.

        Returns:
            True if all deterministic outputs match within tolerance.
        """
        # Prepare ONNX input
        onnx_input = self._prepare_onnx_input(obs_td)

        # Run ONNX inference
        ort_outputs = self.session.run(self.onnx_output_names, onnx_input)

        # Compare each output
        frame_ok = True
        frame_errors: Dict[str, Dict] = {}
        out_keys = self.semantic_out_keys or self.onnx_output_names

        for i, key in enumerate(out_keys):
            onnx_val = ort_outputs[i]

            # Get corresponding PyTorch output
            if key in pytorch_outs.keys():
                pytorch_val = pytorch_outs[key].detach().cpu().numpy()
            else:
                # Key not in pytorch outputs -- skip
                continue

            abs_diff = np.abs(pytorch_val.astype(np.float64) - onnx_val.astype(np.float64))
            max_err = float(abs_diff.max())
            mean_err = float(abs_diff.mean())

            is_stochastic = key in self.STOCHASTIC_KEYS
            frame_errors[key] = {
                "max_err": max_err,
                "mean_err": mean_err,
                "stochastic": is_stochastic,
            }

            if not is_stochastic and max_err > self._tolerance:
                frame_ok = False

        self._frame_errors.append({"frame_idx": frame_idx, "errors": frame_errors})
        self._num_frames += 1
        if not frame_ok:
            self._num_mismatches += 1

        # Print per-frame info (compact)
        parts = []
        for key, err in frame_errors.items():
            if err["stochastic"]:
                continue  # Don't clutter output with stochastic keys
            status = "OK" if err["max_err"] <= self._tolerance else "MISMATCH"
            parts.append(f"{key}: max_err={err['max_err']:.6e} {status}")
        status_str = " | ".join(parts)
        frame_label = "PASS" if frame_ok else "FAIL"
        print(f"  [ONNX] Frame {frame_idx:4d}: {status_str}  [{frame_label}]")

        return frame_ok

    def print_summary(self):
        """Print aggregate verification summary."""
        if self._num_frames == 0:
            print("\n=== ONNX Verification Summary ===")
            print("  No frames were compared.")
            return

        print(f"\n{'=' * 60}")
        print(f"  ONNX Verification Summary ({self._num_frames} frames)")
        print(f"{'=' * 60}")

        # Aggregate per-key statistics
        key_stats: Dict[str, Dict] = {}
        out_keys = self.semantic_out_keys or self.onnx_output_names

        for key in out_keys:
            max_errs = []
            mean_errs = []
            for frame_data in self._frame_errors:
                if key in frame_data["errors"]:
                    err = frame_data["errors"][key]
                    max_errs.append(err["max_err"])
                    mean_errs.append(err["mean_err"])
            if max_errs:
                is_stochastic = key in self.STOCHASTIC_KEYS
                key_stats[key] = {
                    "worst_max_err": max(max_errs),
                    "avg_max_err": sum(max_errs) / len(max_errs),
                    "avg_mean_err": sum(mean_errs) / len(mean_errs),
                    "stochastic": is_stochastic,
                    "passed": is_stochastic or max(max_errs) <= self._tolerance,
                }

        all_passed = True
        for key, stats in key_stats.items():
            if stats["stochastic"]:
                tag = "(stochastic -- skipped)"
                status = ""
            else:
                status = "PASS" if stats["passed"] else "FAIL"
                tag = ""
                if not stats["passed"]:
                    all_passed = False

            print(
                f"  {key:25s}: worst_max_err={stats['worst_max_err']:.6e}, "
                f"avg_max_err={stats['avg_max_err']:.6e}  {status} {tag}"
            )

        print()
        if all_passed:
            print(
                f"  VERDICT: ONNX model matches PyTorch -- the export is CORRECT "
                f"(tolerance={self._tolerance})"
            )
        else:
            print(
                f"  VERDICT: ONNX model DOES NOT MATCH PyTorch! "
                f"({self._num_mismatches}/{self._num_frames} frames had mismatches)"
            )
            print(
                f"  Deterministic outputs exceed tolerance ({self._tolerance}). "
                f"Check the export process or model architecture."
            )
        print(f"{'=' * 60}\n")
