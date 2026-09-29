#!/usr/bin/env python3
"""Single-threaded game-trace accuracy checks. Never starts training or benchmarks."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRACES = ROOT / 'docs/traces'
CASES = {
    'a01': ('rally_a01_trained_scene', 'rally_a01_contact_reference'),
    'a05': ('rally_a05_trained_scene', 'rally_a05_trained'),
    'a06': ('rally_a06_trained_scene', 'rally_a06_trained'),
    'a07': ('rally_a07_contact_scene', 'rally_a07_contact_reference'),
    'a07_old': ('rally_a07_trained_scene', 'rally_a07_trained'),
    'b06': ('rally_b06_scene', 'rally_b06_reference'),
    'formula': ('formula_a01_scene', 'formula_a01_throttle'),
}

def float32(value):
    if isinstance(value, list):
        return [float32(x) for x in value]
    return struct.unpack('f', struct.pack('f', value))[0]

def normalize_trace(source, destination, scene=None):
    trace = json.loads(source.read_text())
    # These reference captures use GameFacade.RecordTrace's default reset=true.
    # Preserve that provenance explicitly rather than guessing in the simulator.
    if source.stem in ('rally_a01_contact_reference', 'rally_a01_network_reference',
                       'rally_a07_contact_reference', 'rally_b06_reference',
                       'rally_a05_trained', 'rally_a06_trained'):
        trace['frames'][0]['reset_transform'] = True
        if scene is not None:
            rotation = json.loads(scene.read_text())['track'].get('curve', {}).get('reset_rotation')
            if rotation is not None:
                trace['frames'][0]['reset_rotation'] = rotation
    # System.Text.Json emits shortest-roundtrip *float* decimals. Parsing those
    # as f64 is not the original body state. Neural inputs/outputs remain double.
    for frame in trace['frames']:
        for key in ('position', 'rotation', 'velocity', 'angular_velocity',
                    'cached_position', 'cached_velocity', 'cached_acceleration',
                    'boost_energy', 'wheel_angles'):
            if key in frame:
                frame[key] = float32(frame[key])
    destination.write_text(json.dumps(trace))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=HERE / 'target/release/altd-sim')
    parser.add_argument('--label', required=True)
    parser.add_argument('--scene-directory', type=Path, default=HERE / 'scenes_exact')
    parser.add_argument('--model', type=Path, default=HERE / 'rally_trained_model_exact.json')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--closed-loop', action='store_true')
    parser.add_argument('--assert-closed-loop-exact', action='store_true')
    parser.add_argument('--sensors', action='store_true')
    parser.add_argument('--assert-ray-error', type=float)
    parser.add_argument('--assert-free-position', type=float)
    parser.add_argument('--assert-position', type=float)
    parser.add_argument('--assert-contact-position', type=float)
    args = parser.parse_args()
    os.nice(19)
    out = HERE / 'reports'
    out.mkdir(exist_ok=True)
    results = {}
    def scene_path(name):
        candidate = args.scene_directory / (name + '.json')
        return candidate if candidate.exists() else TRACES / (name + '.json')
    with tempfile.TemporaryDirectory(prefix='altd-fidelity-') as temp:
        def run(name, scene, trace, closed=False):
            normalized = Path(temp) / (trace + '.json')
            normalize_trace(TRACES / (trace + '.json'), normalized, scene_path(scene))
            report = out / f'{args.label}_{name}.json'
            cmd = [str(args.binary.resolve()), '--threads', '1',
                   'compare-closed-loop' if closed else 'compare-one-step',
                   str(scene_path(scene).resolve()), str(normalized)]
            if closed:
                cmd += [str(TRACES / 'rally_a01_live_network.json'),
                        str(args.model.resolve())]
            subprocess.run(cmd + [str(report)], check=True, stdout=subprocess.DEVNULL)
            data = json.loads(report.read_text())
            results[name] = {k: v for k, v in data.items() if k != 'rows'}
            if closed:
                print(name, 'first >1px:', data['first_position_error_over_1_px'],
                      'first 500 max:', data['first_500_frames']['position_error_px']['max'])
            else:
                print(name, 'position:', data['all']['position_error_px'],
                      'velocity:', data['all']['velocity_error_px_s'],
                      'contact mismatches:', len(data['contact_mismatches']) if data['contacts'] or data['noncontacts'] else 'unavailable')
        for name in args.cases:
            run(name, *CASES[name])
            if args.sensors:
                scene, trace = CASES[name]
                context = Path(temp) / 'sensor_context.json'
                context.write_text('{"start_index": 0}')
                report = out / f'{args.label}_{name}_sensors.json'
                cmd = [str(args.binary.resolve()), '--threads', '1', 'compare-sensors', '--all-frames',
                       str(scene_path(scene).resolve()), str(Path(temp) / (trace + '.json')),
                       str(args.model.resolve()), str(context), str(report)]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
                data = json.loads(report.read_text())
                results[name + '_sensors'] = data
                print(name, 'rays:', data['vision_all'], 'VelF:', data['sensors']['VelF'], 'VelS:', data['sensors']['VelS'])
        if args.closed_loop:
            run('a01_closed', 'rally_a01_trained_scene', 'rally_a01_network_reference', True)
            run('a07_closed', 'rally_a07_contact_scene', 'rally_a07_contact_reference', True)
            run('b06_closed', 'rally_b06_scene', 'rally_b06_reference', True)
            run('a05_closed', 'rally_a05_trained_scene', 'rally_a05_trained', True)
            run('a06_closed', 'rally_a06_trained_scene', 'rally_a06_trained', True)
    (out / f'{args.label}_summary.json').write_text(json.dumps(results, indent=2) + '\n')
    if args.assert_closed_loop_exact:
        assert args.closed_loop, '--assert-closed-loop-exact requires --closed-loop'
        for name in ('a01_closed', 'a07_closed', 'b06_closed'):
            for field, errors in results[name]['full'].items():
                assert errors['max'] == 0, (name, field, errors)
    if args.assert_position is not None:
        for name in args.cases:
            assert results[name]['all']['position_error_px']['max'] <= args.assert_position, (name, results[name]['all'])
            assert not results[name]['contact_mismatches'], (name, results[name]['contact_mismatches'])
    if args.assert_ray_error is not None:
        assert args.sensors, '--assert-ray-error requires --sensors'
        for name in args.cases:
            rays = results[name + '_sensors']['vision_all']
            assert rays['max'] <= args.assert_ray_error, (name, rays)
    if args.assert_contact_position is not None:
        for name in args.cases:
            contacts = results[name]['contacts']
            assert contacts, f'{name}: direct contact data required'
            assert contacts['position_error_px']['max'] <= args.assert_contact_position, (name, contacts)
    if args.assert_free_position is not None:
        for name in args.cases:
            free = results[name]['noncontacts']
            if free:
                assert free['position_error_px']['max'] <= args.assert_free_position, (name, free)
            elif name in ('a05', 'a06'):
                # These captures omit direct contacts. Check every transition.
                assert results[name]['all']['position_error_px']['max'] <= args.assert_free_position, name
            assert not results[name]['contact_mismatches'], (name, results[name]['contact_mismatches'])

if __name__ == '__main__':
    main()
