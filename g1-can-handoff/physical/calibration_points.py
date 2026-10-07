"""Extract marker-free reference points from an archived, exact RGB-D frame.

Input point rows contain pixel_uv, world_xyz_m and validation. Coordinates in
world_xyz_m must be independently measured, never generated from a guessed
camera transform. This produces a candidate only; it cannot enable motion.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from fit_camera_calibration import fit


def extract(frame, measurements):
    depth = np.asarray(frame['depth_m'], dtype=float)
    rays = np.asarray(frame['rays'], dtype=float)
    if depth.ndim != 2 or rays.shape != (*depth.shape, 3):
        raise ValueError('Aligned depth and rays must have matching dimensions')
    metadata = json.loads(str(frame['metadata_json']))
    if metadata.get('camera_serial') != measurements.get('camera_serial'):
        raise ValueError('Measurement and capture camera identities differ')
    if metadata.get('frame_id') != measurements.get('frame_id'):
        raise ValueError('Measurements must name the exact captured frame')
    camera, world, held, pixels = [], [], [], set()
    for row in measurements['points']:
        pixel = np.asarray(row['pixel_uv'], dtype=float)
        if pixel.shape != (2,) or not np.isfinite(pixel).all() or np.any(pixel != np.round(pixel)):
            raise ValueError('Each pixel_uv must be an integer [u,v] pair')
        u,v = map(int, pixel)
        if (u,v) in pixels:
            raise ValueError('Reference pixels must be distinct')
        pixels.add((u,v))
        if not (1 <= u < depth.shape[1]-1 and 1 <= v < depth.shape[0]-1):
            raise ValueError('Reference pixel is outside the usable image')
        # A depth discontinuity can place an apparent table corner on the
        # background. Refuse uncertain points instead of filling missing depth.
        patch = depth[v-1:v+2,u-1:u+2]
        if not np.isfinite(patch).all() or np.any(patch <= .1) or np.any(patch >= 2.5):
            raise ValueError('Reference point has missing or out-of-range depth')
        if np.ptp(patch) > .01:
            raise ValueError('Reference point crosses a depth edge; select a measured point on the surface')
        xyz = rays[v,u] * depth[v,u]
        if not np.isfinite(xyz).all():
            raise ValueError('Reference ray is invalid')
        if row.get('validation') is True:
            held.append(len(camera))
        camera.append(xyz.tolist())
        world.append(row['world_xyz_m'])
    data = dict(camera_points_m=camera,world_points_m=world,validation_indices=held,
                source_frame_id=metadata['frame_id'],camera_serial=metadata['camera_serial'],
                source_measurements=measurements)
    candidate = fit(data)
    candidate.update(camera_serial=metadata['camera_serial'],source_frame_id=metadata['frame_id'])
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frame_npz')
    parser.add_argument('measurements_json')
    parser.add_argument('candidate_json')
    args = parser.parse_args()
    with np.load(args.frame_npz,allow_pickle=False) as frame:
        candidate = extract(frame,json.loads(Path(args.measurements_json).read_text()))
    Path(args.candidate_json).write_text(json.dumps(candidate,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__': main()
