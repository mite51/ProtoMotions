# SPDX-License-Identifier: Apache-2.0
"""Convert a SMPL .motion or directory tree to Unity bakes, or import bakes.

python data/scripts/convert_motion.py motion-to-baked input.motion output_baked.json
python data/scripts/convert_motion.py motion-to-baked input_directory output_directory
python data/scripts/convert_motion.py baked-to-motion input_baked.json output.motion
Existing outputs require --overwrite. Directory conversion preserves relative paths.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from protomotions.simulator.base_simulator.simulator_state import RobotState, StateConversion
from protomotions.utils.baked_animation import motion_to_baked, baked_to_motion, write_baked


def convert_one(mode, source, output, clip_name=None):
    if mode == 'motion-to-baked':
        state = RobotState.from_dict(torch.load(source, weights_only=False, map_location='cpu'),
                                     state_conversion=StateConversion.COMMON)
        data = motion_to_baked(state, clip_name or source.stem)
        write_baked(data, output)
    else:
        state = baked_to_motion(json.loads(source.read_text(encoding='utf-8-sig')))
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state.to_dict(), output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['motion-to-baked', 'baked-to-motion'])
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--clip-name', help='Metadata name for a single motion export')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)
    if args.input.is_dir():
        if args.clip_name:
            parser.error('--clip-name requires a single motion')
        suffix = '.motion' if args.mode == 'motion-to-baked' else '.json'
        jobs = []
        for source in sorted(args.input.rglob('*' + suffix)):
            relative = source.relative_to(args.input)
            name = source.stem + '_baked.json' if args.mode == 'motion-to-baked' else source.stem.removesuffix('_baked') + '.motion'
            jobs.append((source, args.output / relative.parent / name))
    else:
        jobs = [(args.input, args.output)]
    if not jobs:
        parser.error('No input files found')
    outputs = [out.resolve() for _, out in jobs]
    if len(set(outputs)) != len(outputs):
        parser.error('Multiple inputs map to the same output')
    for source, output in jobs:
        if not source.is_file():
            parser.error(f'Input does not exist: {source}')
        if source.resolve() == output.resolve():
            parser.error('Input and output must differ')
        if output.exists() and not args.overwrite:
            parser.error(f'Output exists; use --overwrite: {output}')
    for source, output in jobs:
        convert_one(args.mode, source, output, args.clip_name)
        print(f'{source} -> {output}', flush=True)
    print(f'Converted {len(jobs)} motion(s).')


if __name__ == '__main__':
    with torch.no_grad():
        main()
