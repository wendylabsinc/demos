"""One offline lab for the October 8 failures and current perception mode."""
import html
import json


COVERAGE = [
    ('Recognition and tracking', ['test_color_depth_acquisition', 'test_perception_policy'],
     'Exact failed RGB-D frames 115503 and 115616; recorded empty YOLO result 97662.',
     'Color/depth must acquire both failed views without YOLO; stale or missing targets still stop. Phase mode must not schedule periodic classification.'),
    ('Camera freshness and handoff', ['test_stale_trial_replay', 'test_latest_trial_replay'],
     'Nine archived exposure stops and the delayed observation handoff.',
     'Reproduce the old 400 ms stops; verify the recorded 500 ms handoff and reject truly stale, future, uncertain or cancelled observations.'),
    ('Table geometry', ['test_table_geometry_acceptance'],
     'Two recorded table-height failures with exposure-paired robot poses.',
     'Frozen calibration reproduces the failure; exposure-aligned correction accepts the same frames without relaxing the 5 mm gate.'),
    ('Pickup area', ['test_pickup_area_acceptance'],
     'Three captured placements, including the can 50.2 cm forward.',
     'Reachable captures pass full path checks; the genuinely out-of-range placement stays rejected.'),
    ('Control handover', ['test_control_settling_acceptance'],
     'Two native body/hand feedback traces.',
     'Stable traces settle under the current strictly-below-1-rad/s bound. Moving, faulty, stale and duplicate feedback stays rejected.'),
    ('Planning and collision', ['test_planning_acceptance', 'test_clearance_planning'],
     'Recorded rejected grasp/staging scenes, sampled arm paths and head-camera offset.',
     'Replan feasible scenes with positive clearance. Preserve genuine collision and >40 mm tracking rejections.'),
    ('Position changes', ['test_tracking_position_replay'],
     'Recorded ownership-induced position shift.',
     'Plan from the measured owned pose; never silently rebase later can movement.'),
    ('Start, stop and overlay', ['test_runtime_lifecycle', 'test_operator_panel', 'test_placement_overlay'],
     'Recorded failure lifecycle plus deterministic fake HTTP/controller inputs.',
     'Keep the original error, latch stop, prevent duplicate starts, and retain explicitly labelled reference overlays.'),
    ('Current pickup dynamics', ['test_sim_color_depth', 'test_sim_runtime'],
     'Synthetic MuJoCo RGB-D, contact physics and simulated actuator feedback.',
     'Use the real color/depth tracker and physical pickup sequence. Simulate a pickup, lost target and a stale alignment frame with zero YOLO calls.'),
]
MODULES = [module for _, modules, _, _ in COVERAGE for module in modules]


def dashboard(report, destination, report_filename='latest.json'):
    """Render only local results; this page has no control or network API."""
    esc = html.escape
    modules = {item['module']: item for item in report['modules']}
    cards = []
    for title, names, evidence, expectation in COVERAGE:
        selected = [modules[name] for name in names if name in modules]
        verdict = ('FAIL' if any(not m['successful'] for m in selected) else
                   'PASS' if len(selected) == len(names) else 'NOT RUN')
        cases = [case for module in selected for case in module['cases']]
        details = ''.join('<li><b>'+esc(case['status'])+'</b> '+esc(case['test'])+
                          ('<pre>'+esc(case['detail'])+'</pre>' if case.get('detail') else '')+'</li>' for case in cases)
        for module in selected:
            if module.get('error'):
                details += '<li><pre>'+esc(module['error'])+'</pre></li>'
        cards.append(f'<section class="case {verdict.lower().replace(" ","-")}"><h2>{esc(title)} '
                     f'<span>{verdict} · {sum(m["tests_run"] for m in selected)} tests</span></h2>'
                     f'<p>{esc(evidence)}</p><p>{esc(expectation)}</p><details><summary>Individual outcomes</summary><ul>{details}</ul></details></section>')
    animations = []
    simulation_dir = destination.parent / report.get('simulation_artifact_dir', 'simulations')
    for result_path in sorted(simulation_dir.glob('*/result.json')):
        result = json.loads(result_path.read_text())
        picture = result_path.with_suffix('.gif')
        outcome = 'Simulated lift verified' if result['lift_verified'] else result.get('error', 'Stopped')
        image = f'<img loading="lazy" src="{esc(picture.relative_to(destination.parent).as_posix())}" alt="MuJoCo simulated pickup">' if picture.exists() else ''
        stopped_frame = result_path.with_suffix('.png')
        if result.get('fault'):
            image += '<p>Synthetic fault injected before alignment; this is not a camera recording from the robot.</p>'
        if stopped_frame.exists():
            image += '<details><summary>Camera image at the stop</summary><img loading="lazy" src="'+esc(stopped_frame.relative_to(destination.parent).as_posix())+'" alt="Synthetic camera image at failure"></details>'
        animations.append(f'<section><h2>{esc(result_path.parent.name)}</h2>{image}<p>{esc(outcome)}</p>'
                          f'<p>Simulated time: {result["simulated_time_s"]:.2f}s · YOLO calls: {result["perception"]["yolo_calls"]}</p>'
                          '<p>Playback is sampled at 5 frames/s; it is not a Jetson timing benchmark.</p>'
                          '<a href="'+esc(result_path.relative_to(destination.parent).as_posix())+'">Simulation event log</a></section>')
    payload = '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>G1 offline demo lab</title><style>
body{font:16px system-ui;background:#10161f;color:#e5edf8;max-width:1100px;margin:40px auto;padding:0 24px;line-height:1.6}
h1{font-size:36px}h2{font-size:21px;margin-top:0}section{padding:22px;background:#1a2533;border:1px solid #34465c;border-radius:14px;margin:18px 0}
h2 span{float:right;font-size:14px}.pass h2 span{color:#7cdbad}.fail h2 span{color:#ff9a91}.note{border-left:4px solid #efbe6d;padding:14px 20px;background:#242b32}
summary{cursor:pointer;color:#9bc8ff}li{padding:5px 0;overflow-wrap:anywhere}pre{white-space:pre-wrap;color:#ffb4ac}img{max-width:100%;border-radius:10px}
button{background:#314b68;color:white;border:0;padding:10px 18px;border-radius:7px;cursor:pointer;margin:10px 8px 0 0}a{color:#9bc8ff}
</style></head><body><h1>G1 offline demo lab</h1>'''
    payload += f'<p><strong>{"PASS" if report["successful"] else "FAIL"} · {report["tests_run"]} tests · {report["evidence_verified"]} verified source hashes · {report["seconds"]:.1f}s on this computer</strong></p>'
    payload += '<p class="note">No robot, DDS or external network access. PASS means the stated expectation passed, including expected safe stops. It does not mean every captured situation permits a pickup. Camera transport, GPU latency, real actuator dynamics, friction and physical success still require hardware validation.</p>'
    payload += '<p>The contact simulation uses a supported robot base, an untextured red cylinder, synthetic healthy feedback and approximate camera geometry. Real images and recorded timing are tested separately below.</p>'
    payload += '<button onclick="document.querySelectorAll(\'.case.pass\').forEach(e=>e.hidden=!e.hidden)">Show/hide passing groups</button>'
    payload += '<a href="'+esc(report_filename,quote=True)+'">Raw test report</a>'
    payload += ''.join(animations)+''.join(cards)+'</body></html>'
    destination.write_text(payload)
