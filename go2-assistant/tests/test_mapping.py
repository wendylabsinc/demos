"""Sensor-only tests: geometry is used to generate rays, never given to the mapper."""
import math

import numpy as np
import pytest

from go2_assistant.mapping import OccupancyMapper


ANGLES = np.linspace(-math.pi, math.pi, 360, endpoint=False)
WALLS = [((-4., -3.), (4., -3.)), ((4., -3.), (4., 2.5)),
         ((4., 2.5), (-4., 2.5)), ((-4., 2.5), (-4., -3.))]


def scan(segments=WALLS, pose=(0., 0., 0.), angles=ANGLES, max_range=8., mount=(0., 0.)):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    origin = np.array((pose[0] + mount[0] * c - mount[1] * s,
                       pose[1] + mount[0] * s + mount[1] * c))
    direction = np.column_stack((np.cos(angles + pose[2]), np.sin(angles + pose[2])))
    result = np.full(len(angles), max_range)
    for first, second in segments:
        offset, edge = np.asarray(first) - origin, np.asarray(second) - np.asarray(first)
        denominator = direction[:, 0] * edge[1] - direction[:, 1] * edge[0]
        usable = np.abs(denominator) > 1e-8
        distance = np.divide(offset[0] * edge[1] - offset[1] * edge[0], denominator,
                             out=np.full(len(angles), math.inf), where=usable)
        fraction = np.divide(offset[0] * direction[:, 1] - offset[1] * direction[:, 0], denominator,
                             out=np.full(len(angles), math.inf), where=usable)
        hits = usable & (distance >= .05) & (fraction >= 0.) & (fraction <= 1.)
        result[hits] = np.minimum(result[hits], distance[hits])
    return result


def mapper(radius=.22):
    return OccupancyMapper(robot_radius=radius, half_extent_m=10.)


def confirm_reload(restored, ranges, angles=ANGLES, *, timestamp=1.):
    result = None
    for index in range(3):
        result = restored.update_scan(ranges, angles, timestamp=timestamp + index * .1)
    return result


def test_starts_unknown_and_sees_only_measured_wall():
    m = mapper()
    assert m.status()['observed_cells'] == 0 and not m.free(0., 0.)
    assert m.plan(1., 0.) == [] and m.frontiers() == []
    angles = np.linspace(-.7, .7, 81)
    m.update_scan(scan([((3., -3.), (3., 3.))], angles=angles), angles)
    wall = m._cell(3., 0.)
    hidden = m._cell(4., 0.)
    behind = m._cell(-1., 0.)
    assert m.observed[wall[1], wall[0]] and m.log_odds[wall[1], wall[0]] >= .65
    assert not m.observed[hidden[1], hidden[0]] and not m.observed[behind[1], behind[0]]
    assert not m.free(4., 0.) and m.plan(4., 0.) == []


def test_max_range_marks_free_space_without_inventing_walls():
    m = mapper()
    m.update_scan(np.full(360, math.inf), ANGLES, max_range=3.)
    assert m.status()['occupied_cells'] == 0
    assert m.free(0., 0.) and m.free(1., 0.)
    assert not m.free(3.2, 0.) and m.plan(3.2, 0.) == []
    assert m.frontiers() and all(m.plan(*goal) for goal in m.frontiers())


def test_sparse_far_beams_still_have_reachable_frontier_vantages():
    m = mapper(radius=.4)
    m.update_scan(np.full(360, 8.), ANGLES)
    # Distant beam gaps are unknown. The robot must be able to move to a
    # reachable near viewpoint and observe them, rather than declare a full map.
    goals = m.frontiers(limit=4)
    assert goals and all(m.plan(*goal) for goal in goals)
    assert all(m.free(*goal) for goal in goals)


def test_layout_translation_and_rotation_do_not_leak_world_pose():
    angle, translation = .43, np.array((17.5, -22.))
    rotation = np.array(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    transformed = [(tuple(rotation @ first + translation), tuple(rotation @ second + translation))
                   for first, second in WALLS]
    original = scan()
    moved = scan(transformed, (*translation, angle))
    np.testing.assert_allclose(original, moved, atol=1e-12)
    a, b = mapper(), mapper()
    a.update_scan(original, ANGLES); b.update_scan(moved, ANGLES)
    assert a.pose() == b.pose() == {'x': 0., 'y': 0., 'yaw': 0.}
    # Cell rounding may differ on exact half-cell endpoints by machine epsilon.
    assert np.count_nonzero(a.observed != b.observed) <= 4
    assert a.status()['source'] == 'measured lidar rays'


def test_scan_matching_corrects_biased_incremental_odometry():
    m = mapper()
    m.update_scan(scan(), ANGLES)
    true = (.45, .18, .06)
    biased = (.63, .28, .10)
    result = m.update_scan(scan(pose=true), ANGLES, biased)
    pose = m.pose()
    assert result['localization']['method'] == 'scan_match'
    assert math.hypot(pose['x'] - true[0], pose['y'] - true[1]) < .09
    assert abs(pose['yaw'] - true[2]) < .035
    assert result['localization']['correction_m'] > .1


def test_lidar_mount_is_applied_without_shifting_base_frame():
    m = mapper()
    m.update_scan(scan(mount=(.28, 0.)), ANGLES, lidar_origin=(.28, 0.))
    assert m.pose()['x'] == 0.
    wall = m._cell(4., 0.)
    assert m.log_odds[wall[1], wall[0]] >= .65
    assert m.plan(2., 0.)


def test_path_stays_in_observed_space_and_inflates_robot_footprint():
    m = mapper(radius=.35)
    m.update_scan(scan(), ANGLES)
    path = m.plan(2., 0.)
    assert path and all(m.line_free(first, second) for first, second in zip(path, path[1:]))
    assert not m.free(3.8, 0.) and not m.free(4.5, 0.)
    assert m.plan(3.8, 0.) == [] and m.plan(4.5, 0.) == []


def test_new_obstruction_closes_route_on_first_return_and_replans():
    m = mapper()
    m.update_scan(scan(), ANGLES)
    original = m.plan(2., 0.)
    assert original and m.line_free((0., 0.), (2., 0.))
    obstacle = [((.9, -.6), (.9, .6)), ((.9, .6), (1.1, .6)),
                ((1.1, .6), (1.1, -.6)), ((1.1, -.6), (.9, -.6))]
    m.update_scan(scan(WALLS + obstacle), ANGLES)
    assert not m.line_free((0., 0.), (2., 0.))
    changed = m.plan(2., 0.)
    assert changed and changed != original
    assert any(abs(point[1]) > .7 for point in changed)
    assert all(m.line_free(first, second) for first, second in zip(changed, changed[1:]))


def test_frontiers_are_reachable_and_unknown_is_never_a_goal():
    m = mapper()
    # An open doorway gives rays to an outer room; only observed free vantages
    # near its unknown boundary may be proposed.
    walls = [((-3., -3.), (3., -3.)), ((3., -3.), (3., 3.)),
             ((-3., -3.), (-3., 3.)), ((-3., 3.), (-.6, 3.)), ((.6, 3.), (3., 3.))]
    m.update_scan(scan(walls, max_range=5.), ANGLES, max_range=5.)
    frontiers = m.frontiers(limit=8)
    assert frontiers and len(frontiers) <= 8
    for goal in frontiers:
        cell = m._cell(*goal)
        assert m.observed[cell[1], cell[0]] and m.free(*goal) and m.plan(*goal)


def test_closed_observed_walls_do_not_become_frontiers():
    m = mapper()
    # Resolve every wall cell. At low angular resolution, genuine unobserved
    # gaps at distant walls can still warrant a closer sensor observation.
    angles = np.linspace(-math.pi, math.pi, 1440, endpoint=False)
    m.update_scan(scan(angles=angles), angles)
    assert m.frontiers() == []


def test_persistence_requires_matching_scan_and_keeps_map_identity(tmp_path):
    path = tmp_path / 'learned-map.bin'
    m = mapper(); m.update_scan(scan(), ANGLES); m.save(path)
    restored = OccupancyMapper.load(path)
    assert restored.map_id == m.map_id and not restored.localization['valid']
    assert restored.plan(2., 0.) == []
    result = confirm_reload(restored, scan())
    assert result['localization']['valid'] and restored.plan(2., 0.)
    np.testing.assert_array_equal(restored.observed, m.observed)


def test_persisted_map_relocalizes_at_initial_anchor_after_travel(tmp_path):
    m = mapper()
    m.update_scan(scan(), ANGLES)
    for x in np.arange(.3, 2.11, .3):
        m.update_scan(scan(pose=(float(x), 0., 0.)), ANGLES, (.3, 0., 0.))
    assert m.pose()['x'] > 2.
    path = tmp_path / 'map.bin'; m.save(path)
    restored = OccupancyMapper.load(path)
    result = confirm_reload(restored, scan())
    assert result['localization']['valid']
    assert math.hypot(restored.pose()['x'], restored.pose()['y']) < .1


def test_unrelated_reload_scan_cannot_repaint_saved_map(tmp_path):
    m = mapper(); m.update_scan(scan(), ANGLES)
    path = tmp_path / 'map.bin'; m.save(path)
    restored = OccupancyMapper.load(path)
    before = restored.log_odds.copy()
    result = restored.update_scan(np.full(360, 1.), ANGLES)
    assert not result['localization']['valid'] and restored.plan(.2, 0.) == []
    np.testing.assert_array_equal(restored.log_odds, before)


def test_missing_invalid_stale_input_is_rejected_without_map_changes():
    m = mapper()
    for ranges, angles, kwargs in [(np.full(360, np.nan), ANGLES, {}),
                                   (np.ones(7), ANGLES[:7], {}),
                                   (scan(), ANGLES, {'odometry_delta': (math.inf, 0., 0.)})]:
        with pytest.raises(ValueError): m.update_scan(ranges, angles, **kwargs)
    assert m.status()['observed_cells'] == 0
    m.update_scan(scan(), ANGLES, timestamp=10.)
    before = m.log_odds.copy()
    with pytest.raises(ValueError): m.update_scan(scan(), ANGLES, timestamp=10.)
    np.testing.assert_array_equal(m.log_odds, before)


def test_status_and_map_memory_are_bounded():
    m = mapper()
    m.update_scan(scan(), ANGLES)
    assert len(m.status()) < 20 and 'cells' not in m.status()
    assert len(m.snapshot()['cells']) == m.size <= 601
    with pytest.raises(ValueError): OccupancyMapper(resolution=.05, half_extent_m=25.)
    with pytest.raises(ValueError): m.frontiers(1000)


def test_vectorized_ray_walk_matches_scalar_observed_and_occupied_cells():
    rng = np.random.default_rng(174)
    angles = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    ranges = rng.uniform(.1, 8., 360)
    ranges[::7] = 8.
    m = OccupancyMapper(robot_radius=.22, half_extent_m=5.)
    origin = (.28, 0.)
    start = m._cell(*origin)
    expected_free = np.zeros_like(m.observed)
    expected_occupied = np.zeros_like(m.observed)
    for distance, angle in zip(ranges, angles):
        end = m._cell(origin[0] + distance * math.cos(angle), origin[1] + distance * math.sin(angle))
        steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        x = np.rint(np.linspace(start[0], end[0], steps + 1)).astype(int)
        y = np.rint(np.linspace(start[1], end[1], steps + 1)).astype(int)
        usable = (x >= 0) & (y >= 0) & (x < m.size) & (y < m.size)
        if distance < 7.98 and m._inside(end):
            expected_occupied[end[1], end[0]] = True
            usable[-1] = False
        expected_free[y[usable], x[usable]] = True
    expected_free &= ~expected_occupied
    m.update_scan(ranges, angles, lidar_origin=origin)
    np.testing.assert_array_equal(m.observed, expected_free | expected_occupied)
    np.testing.assert_array_equal(m.log_odds >= .65, expected_occupied)
    np.testing.assert_array_equal(m.log_odds < -.2, expected_free)


def test_likelihood_field_cache_changes_only_when_occupied_geometry_changes():
    m = mapper(); m.update_scan(scan(), ANGLES)
    original = m._score_field()
    m.update_scan(scan(), ANGLES)
    assert m._score_cache is original
    obstacle = [((.9, -.6), (.9, .6))]
    m.update_scan(scan(WALLS + obstacle), ANGLES)
    assert m._score_cache is None
    assert m._score_field() is not original


def test_ten_minutes_of_stationary_scans_resist_biased_odometry_and_preserve_reload(tmp_path):
    m = mapper()
    ranges = scan()
    m.update_scan(ranges, ANGLES, timestamp=1.)
    initially_occupied = m.status()['occupied_cells']
    # One measured scan per simulated second, with the exact injected gyro and
    # lateral biases from the simulator. The sensor ranges remain stationary;
    # no idle command or external pose is supplied to localization.
    for elapsed_second in range(1, 601):
        result = m.update_scan(ranges, ANGLES, (0., .0005, .0008), timestamp=elapsed_second + 1.)
        assert result['localization']['valid']
    pose = m.pose()
    assert math.hypot(pose['x'], pose['y']) < .04
    assert abs(pose['yaw']) < .03
    assert m.status()['occupied_cells'] <= initially_occupied * 1.05
    path = tmp_path / 'stationary-map.npz'
    m.save(path)
    restored = OccupancyMapper.load(path)
    assert restored.map_id == m.map_id
    result = confirm_reload(restored, ranges, timestamp=602.)
    assert result['localization']['valid']
    assert math.hypot(restored.pose()['x'], restored.pose()['y']) < .06
    assert abs(restored.pose()['yaw']) < .03


def test_reload_requires_three_consistent_scans_before_pose_or_map_changes(tmp_path):
    m = mapper(); ranges = scan(); m.update_scan(ranges, ANGLES)
    path = tmp_path / 'map.npz'; m.save(path)
    restored = OccupancyMapper.load(path)
    before_grid, before_pose, before_revision = restored.log_odds.copy(), restored.pose(), restored.revision
    for index in range(2):
        result = restored.update_scan(ranges, ANGLES, timestamp=1. + index * .1)
        assert not result['localization']['valid']
        assert result['localization']['confirmed_scans'] == index + 1
        assert restored.pose() == before_pose and restored.revision == before_revision
        np.testing.assert_array_equal(restored.log_odds, before_grid)
        assert restored.plan(1., 0.) == []
    result = restored.update_scan(ranges, ANGLES, timestamp=1.2)
    assert result['localization']['valid'] and restored.plan(1., 0.)


def test_reload_scores_all_candidates_before_merging_nearby_estimates(tmp_path):
    m = mapper(); m.update_scan(scan(), ANGLES)
    m.update_scan(scan(pose=(.08, 0., .04)), ANGLES, (.08, 0., .04))
    path = tmp_path / 'map.npz'; m.save(path)
    restored = OccupancyMapper.load(path)

    def match(predicted, points, *, trim=False):
        # Nearby anchors can reach different optima; retain the strongest one.
        if predicted[0] > .04:
            return np.array((.08, 0., .04)), .70, True
        return np.zeros(3), .97, True

    restored._match = match
    result = confirm_reload(restored, scan())
    assert result['localization']['valid'] and result['localization']['score'] == .97
    assert restored.pose() == {'x': 0., 'y': 0., 'yaw': 0.}


def test_reload_does_not_hide_distinct_close_pose_ambiguity(tmp_path):
    m = mapper(); m.update_scan(scan(), ANGLES)
    m.update_scan(scan(pose=(.25, 0., .10)), ANGLES, (.25, 0., .10))
    path = tmp_path / 'map.npz'; m.save(path)
    restored = OccupancyMapper.load(path)
    before = restored.log_odds.copy()

    def match(predicted, points, *, trim=False):
        if predicted[0] > .1:
            return np.array((.25, 0., .1)), .95, True
        return np.zeros(3), .94, True

    restored._match = match
    for index in range(4):
        result = restored.update_scan(scan(), ANGLES, timestamp=1. + index * .1)
        assert not result['localization']['valid']
        assert result['localization']['method'] == 'ambiguous_relocalization'
        assert restored._reload_count == 0
    np.testing.assert_array_equal(restored.log_odds, before)


def test_reload_confirmation_resets_after_mismatch_or_measurement_gap(tmp_path):
    m = mapper(); ranges = scan(); m.update_scan(ranges, ANGLES)
    path = tmp_path / 'map.npz'; m.save(path)
    restored = OccupancyMapper.load(path)
    result = restored.update_scan(ranges, ANGLES, timestamp=1.)
    assert result['localization']['confirmed_scans'] == 1
    result = restored.update_scan(np.full(360, 1.), ANGLES, timestamp=1.1)
    assert not result['localization']['valid'] and restored._reload_count == 0
    result = restored.update_scan(ranges, ANGLES, timestamp=1.2)
    assert result['localization']['confirmed_scans'] == 1
    result = restored.update_scan(ranges, ANGLES, timestamp=3.)
    assert result['localization']['confirmed_scans'] == 1
    result = restored.update_scan(ranges, ANGLES, timestamp=3.1)
    assert result['localization']['confirmed_scans'] == 2
    result = restored.update_scan(ranges, ANGLES, timestamp=3.2)
    assert result['localization']['valid']
