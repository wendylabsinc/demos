"""Summarize the complete evaluation and populate its conversation visual."""

import argparse, csv, json, hashlib
from pathlib import Path
import mujoco
from . import bar_film as bar

OUTPUT = bar.PROJECT / "results" / "generalization-physics"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", type=Path)
    args = parser.parse_args()
    results = json.loads((OUTPUT / "results.json").read_text())
    assert len(results) == 39 and len({r["id"] for r in results}) == 39
    assert all(
        r["support_constraints"] == 0 and r["mocap_bodies"] == 0 for r in results
    )
    assert all(r["max_furniture_contact_force_N"] == 0 for r in results)
    full = [
        json.loads((OUTPUT / name / "result.json").read_text())
        for name in ("full75", "full95")
    ]
    original = json.loads(
        (
            bar.PROJECT / "results" / "bar-delivery-physics" / "simulation.json"
        ).read_text()
    )
    original.update(
        id="original85",
        kind="full_route",
        scenario=dict(height=0.85, x=0.315, y=0.19, delivery_height=0.75),
        previous_run=True,
    )
    full.insert(1, original)
    groups = {
        kind: dict(
            passed=sum(r["success"] for r in results if r["kind"] == kind),
            total=sum(r["kind"] == kind for r in results),
        )
        for kind in ("grid", "heldout", "boundary")
    }
    report = dict(
        summary=groups,
        full_route=dict(
            passed=sum(r["success"] for r in full),
            total=len(full),
            includes_one_previous_run=True,
        ),
        results=results,
        full=full,
        manifest=json.loads((OUTPUT / "manifest.json").read_text()),
        mujoco_version=mujoco.__version__,
        additional_source_hashes={
            name: hashlib.sha256(
                (bar.PROJECT / "g1_can_handoff" / name).read_bytes()
            ).hexdigest()
            for name in ("bar_delivery.py", "bar_delivery_physics.py")
        },
        scope=dict(
            grid_height_cm=[75, 85, 95],
            grid_x_cm=[28, 31.5, 35],
            grid_y_cm=[10, 19, 28],
            mass_kg=0.37,
            friction=1.0,
            depth_noise_m=0,
            ground="flat",
            camera="fixed calibrated RGB-D",
            balance_state="simulator state",
            navigation="known furniture geometry",
            destination_height_cm=75,
            full_route_tested_for_every_case=False,
            hardware_validated=False,
        ),
        limitations=[
            "Finite samples do not prove every continuous position.",
            "Short carry success does not establish full delivery.",
            "Physical hardware is untested.",
        ],
    )
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2))
    fields = [
        "id",
        "kind",
        "height",
        "x",
        "y",
        "success",
        "stage",
        "failure",
        "max_furniture_contact_force_N",
        "min_pelvis_height",
        "min_upright_cosine",
    ]
    with (OUTPUT / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    if args.visual:
        compact = [
            {
                key: r[key]
                for key in (
                    "id",
                    "kind",
                    "height",
                    "x",
                    "y",
                    "success",
                    "stage",
                    "max_furniture_contact_force_N",
                )
            }
            | {"failure": r.get("failure", "")}
            for r in results
        ]
        short_full = [
            {
                key: r[key]
                for key in (
                    "id",
                    "scenario",
                    "success",
                    "max_robot_furniture_contact_force_N",
                )
            }
            | {"failure": r.get("failure", "")}
            for r in full
        ]
        fragment = args.visual.read_text()
        data = json.dumps(dict(results=compact, full=short_full), separators=(",", ":"))
        start = fragment.index(
            '<script id="g1-measured-data" type="application/json">'
        ) + len('<script id="g1-measured-data" type="application/json">')
        end = fragment.index("</script>", start)
        args.visual.write_text(fragment[:start] + data + fragment[end:])
    print(json.dumps(dict(summary=groups, full_route=report["full_route"]), indent=2))


if __name__ == "__main__":
    main()
