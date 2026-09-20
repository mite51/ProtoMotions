# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rename converted ``.motion`` files on disk so they match the names
referenced by a motion-list YAML (e.g. ``amass_smpl_train.yaml``).

AMASS releases sometimes use different suffixes for the same clip:
``_poses.npz`` (older) vs ``_stageii.npz`` / ``_stagei.npz`` (newer).
After conversion these become ``..._poses.motion`` / ``..._stageii.motion``
respectively, but the training YAMLs reference the ``_poses`` variant.

For every entry in the YAML this script:
  1. Builds the expected absolute path under ``--amass-root``.
  2. If that file already exists, it is left alone.
  3. Otherwise it searches the same directory for files matching the
     same prefix but any trailing token (the trailing ``_poses`` is
     replaced with a wildcard).
  4. If exactly one candidate is found it is renamed to the expected
     name (only when ``--apply`` is passed; otherwise a dry-run report
     is printed).
  5. Misses (no candidate, multiple candidates, collisions) are written
     to a log file.

Example:
    python data/scripts/rename_motions_to_match_yaml.py \\
        --yaml data/yaml_files/amass_smpl_train.yaml \\
        --amass-root C:/Git/AMASS \\
        --log-file rename_motions.log \\
        --apply
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
import yaml
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_enable=False, add_completion=False)


# Tokens that look like the AMASS suffix we want to normalise away. The
# regex ``_poses\.motion$`` is replaced with ``_<group>.motion``; any of
# these (or any other token, since the group is ``[^/\\]+``) will match.
KNOWN_SUFFIXES = ("poses", "stageii", "stagei", "mosh", "mosh_stageii")


def build_candidate_pattern(expected_name: str) -> Optional[re.Pattern[str]]:
    """Return a regex that matches sibling files with the same stem but
    a different trailing suffix. ``None`` if the expected name doesn't
    end with a recognised ``_<token>.motion`` suffix."""
    m = re.match(r"^(?P<prefix>.+)_(?P<token>[^_]+)\.motion$", expected_name)
    if m is None:
        return None
    prefix = re.escape(m.group("prefix"))
    return re.compile(rf"^{prefix}_[^/\\]+\.motion$", re.IGNORECASE)


def load_expected_paths(yaml_path: Path) -> List[str]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    motions = data.get("motions", []) if isinstance(data, dict) else []
    paths: List[str] = []
    for m in motions:
        f = m.get("file") if isinstance(m, dict) else None
        if isinstance(f, str):
            paths.append(f)
    return paths


@app.command()
def main(
    yaml_file: Path = typer.Option(
        ...,
        "--yaml",
        "-y",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Motion-list YAML (e.g. data/yaml_files/amass_smpl_train.yaml).",
    ),
    amass_root: Path = typer.Option(
        ...,
        "--amass-root",
        "-r",
        exists=True,
        file_okay=False,
        help="Root directory containing converted .motion files (e.g. C:/Git/AMASS).",
    ),
    log_file: Path = typer.Option(
        Path("rename_motions.log"),
        "--log-file",
        "-l",
        help="Where to write warnings (no match / multiple matches / collisions).",
    ),
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Actually perform the renames. Without this flag the script is a dry-run.",
    ),
    prefer_shortest: bool = typer.Option(
        False,
        "--prefer-shortest",
        help=(
            "When multiple candidates match, automatically pick the one with the "
            "shortest filename (typically the 'plain' variant, e.g. "
            "'foo_stageii.motion' over 'foo_t2_stageii.motion'). If multiple "
            "candidates share the shortest length, the alphabetically first one "
            "is picked and logged as 'tiebreak-shortest-first'."
        ),
    ),
):
    expected_paths = load_expected_paths(yaml_file)
    typer.echo(f"Loaded {len(expected_paths)} entries from {yaml_file}")

    counts: dict = defaultdict(int)
    log_lines: list[str] = []

    def warn(category: str, message: str) -> None:
        counts[category] += 1
        log_lines.append(f"[{category}] {message}")

    for rel_path in tqdm(expected_paths, desc="Scanning"):
        # Normalise to OS path and resolve under the AMASS root.
        rel = Path(rel_path)
        expected_abs = amass_root / rel
        directory = expected_abs.parent
        expected_name = expected_abs.name

        if not directory.exists():
            warn("missing-dir", f"{rel_path}  (parent dir not found: {directory})")
            counts["already-correct"] += 0  # touch key for ordering
            continue

        if expected_abs.exists():
            counts["already-correct"] += 1
            continue

        pattern = build_candidate_pattern(expected_name)
        if pattern is None:
            warn(
                "unparsable-name",
                f"{rel_path}  (couldn't parse '<prefix>_<token>.motion' from name)",
            )
            continue

        try:
            siblings = [p for p in directory.iterdir() if p.is_file()]
        except OSError as e:
            warn("io-error", f"{rel_path}  (could not list {directory}: {e})")
            continue

        candidates = [p for p in siblings if pattern.match(p.name)]

        if len(candidates) == 0:
            warn("no-match", f"{rel_path}  (no candidate in {directory})")
            continue

        if len(candidates) > 1:
            chosen = None
            if prefer_shortest:
                min_len = min(len(p.name) for p in candidates)
                shortest = sorted(
                    (p for p in candidates if len(p.name) == min_len),
                    key=lambda p: p.name,
                )
                chosen = shortest[0]
                tag = (
                    "tiebreak-shortest"
                    if len(shortest) == 1
                    else "tiebreak-shortest-first"
                )
                counts[tag] += 1
                log_lines.append(
                    f"[{tag}] {rel_path}  (chose {chosen.name} "
                    f"over {len(candidates) - 1} other candidate(s))"
                )
            if chosen is None:
                names = ", ".join(sorted(p.name for p in candidates))
                warn(
                    "multiple-matches",
                    f"{rel_path}  (candidates: {names})",
                )
                continue
            candidates = [chosen]

        src = candidates[0]
        dst = expected_abs

        if src.name == dst.name:
            # Should already be caught by the exists() check, but be safe.
            counts["already-correct"] += 1
            continue

        if dst.exists():
            warn(
                "collision",
                f"{rel_path}  (would overwrite existing target: {dst.name})",
            )
            continue

        action = "RENAME" if apply else "DRY-RUN rename"
        log_lines.append(f"[{action.lower()}] {src.name} -> {dst.name}  (in {directory})")
        counts["to-rename"] += 1

        if apply:
            try:
                src.rename(dst)
                counts["renamed"] += 1
            except OSError as e:
                warn("rename-failed", f"{src} -> {dst}  ({e})")

    header = [
        f"# rename_motions_to_match_yaml run @ {datetime.now().isoformat(timespec='seconds')}",
        f"# yaml      : {yaml_file}",
        f"# amass_root: {amass_root}",
        f"# apply     : {apply}",
        "# --- summary ---",
    ]
    summary_keys = [
        "to-rename",
        "renamed",
        "already-correct",
        "tiebreak-shortest",
        "tiebreak-shortest-first",
        "no-match",
        "multiple-matches",
        "collision",
        "unparsable-name",
        "missing-dir",
        "rename-failed",
        "io-error",
    ]
    for k in summary_keys:
        header.append(f"# {k:<18}: {counts.get(k, 0)}")
    header.append("# --- entries ---")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(log_lines) + ("\n" if log_lines else ""))

    typer.echo("")
    typer.echo("Summary:")
    for k in summary_keys:
        typer.echo(f"  {k:<18}: {counts.get(k, 0)}")
    typer.echo(f"\nLog written to: {log_file}")
    if not apply and counts.get("to-rename", 0) > 0:
        typer.echo("\n(dry-run) Re-run with --apply to perform the renames.")


if __name__ == "__main__":
    app()
