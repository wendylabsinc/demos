"""Plain-language presentation of controller state; no control decisions."""

ERROR_CASES = [
    dict(
        triggers=["left temperature", "right temperature"],
        title="Hand temperature limit reached",
        summary="A hand temperature reading reached the configured guard, or was invalid.",
        action="Check live hand temperatures and fault codes below before retrying.",
        source="physical_core.py · snapshot_guard",
    ),
    dict(
        triggers=["left motor fault", "right motor fault"],
        title="Hand reports a motor fault",
        summary="A hand reported a nonzero fault code, or fault feedback was missing.",
        action="Review the hand feedback below. Release is not a confirmed hardware fault reset.",
        source="physical_core.py · snapshot_guard",
    ),
    dict(
        triggers=["Grasp orientation search time limit"],
        title="Planning took too long",
        summary="The grasp search exceeded its configured time limit.",
        action="Review the trial log and planner timing before another attempt.",
        source="g1_can_handoff/task.py",
    ),
    dict(
        triggers=["Waist moved beyond", "Body attitude exceeds"],
        title="Robot posture changed",
        summary="The robot moved beyond the camera alignment limits after calibration.",
        action="Keep the robot stationary and refresh the scene before retrying.",
        source="pose_alignment.py",
    ),
    dict(
        triggers=["exposure stale", "exposure timestamp invalid/stale"],
        title="Camera observation is too old",
        summary="The latest observation did not meet the freshness limit.",
        action="Check that the live camera is advancing before another attempt.",
        source="camera_input.py",
    ),
    dict(
        triggers=["Coke YOLO target absent/ambiguous"],
        title="Coke identity could not be confirmed",
        summary="YOLO could not identify one unambiguous Coke can.",
        action="Give the detector a clear view of the can before retrying.",
        source="yolo_client.py",
    ),
    dict(
        triggers=["Color/depth can target absent/ambiguous"],
        title="Can position could not be confirmed",
        summary="Color/depth tracking could not isolate one can from the image and depth.",
        action="Give the camera an unobstructed view of one upright can inside the placement area.",
        source="target_tracker.py",
    ),
    dict(
        triggers=["Passive recorder health gate failed"],
        title="Trial recorder is not ready",
        summary="The recorder did not pass its health check, so the attempt could not proceed.",
        action="Check recorder health and storage before retrying.",
        source="runtime.py",
    ),
    dict(
        triggers=["Target has too few", "No sufficiently visible red can"],
        title="Can detection needs a clearer view",
        summary="The detector could not obtain an unambiguous can position.",
        action="Place the upright can inside the placement area with an unobstructed view.",
        source="camera_input.py",
    ),
    dict(
        triggers=["Can is outside the left-hand pickup area"],
        title="Can is outside the pickup area",
        summary="The measured can position is outside the supported reach.",
        action="Move the can base inside the placement outline, then retry.",
        source="g1_can_handoff/task.py",
    ),
    dict(
        triggers=["Too few yellow tape depth points"],
        title="Table markers are not visible enough",
        summary="There are too few valid depth points on the yellow tape.",
        action="Restore the marked table to the camera view before refreshing the scene.",
        source="factory_table_candidate.py",
    ),
    dict(
        triggers=["Control handover did not settle"],
        title="Arm control did not settle",
        summary="The arms did not stabilize during control handover.",
        action="Support the arms and can, release control, then reset the setup.",
        source="control_settle.py",
    ),
    dict(
        triggers=["intersects", "collides", "clearance"],
        title="Planned path is blocked",
        summary="The arm or finger path does not have the required clearance.",
        action="Reposition the can within the pickup area and replan with clearance.",
        source="physical_core.py",
    ),
    dict(
        triggers=["Continuous Cartesian lift IK"],
        title="Lift path could not be solved",
        summary="The planner could not maintain the grasp through the lift.",
        action="Save the trial log and check the lift plan before retrying.",
        source="physical_core.py",
    ),
    dict(
        triggers=["HTTP Error 423", "423: Locked"],
        title="Camera is busy",
        summary="Camera is temporarily busy. The last image stays visible while the feed retries.",
        action="Wait for that operation to finish. The placement outline is a reference meanwhile.",
        source="camera_input.py",
    ),
    dict(
        triggers=["Operator cancelled"],
        title="Demo stopped",
        summary="The operator stopped this attempt.",
        action="Check the setup before starting another attempt.",
        source="runtime.py",
    ),
]


def describe_error(message):
    return next(
        (
            rule
            for rule in ERROR_CASES
            if any(trigger in message for trigger in rule["triggers"])
        ),
        dict(
            title="Demo stopped",
            summary="The attempt stopped. Technical details contain the full error.",
            action="Check the reported problem before retrying.",
            source="Not identified",
        ),
    )


def describe_state(state):
    owned = state.get("adapter_initialized")
    phase = state.get("phase") or "UNKNOWN"
    running = bool(state.get("demo_running"))
    # Start clears demo_error immediately, but the worker sets its next phase
    # later. Conversely, a new demo_error marks a stopped worker in cleanup.
    stopped = phase in ("FROZEN", "BLOCKED") and (
        not running or bool(state.get("demo_error"))
    )
    active = running and not stopped
    # A previous failure can remain in backend fields until the worker replaces
    # it. It must not describe a newly accepted attempt as already stopped.
    failure = (state.get("demo_error") or state.get("error")) if not active else None
    summary = describe_error(failure)["summary"] if failure else None
    result = state.get("demo_result") or {}
    owner = (
        "Demo controls arms + fingers"
        if owned
        else (
            "Upper-body control released to Unitree"
            if owned is False
            else "Control ownership unknown"
        )
    )
    if state.get("deployment_in_progress"):
        title, tone, next_action = (
            "Updating demo",
            "busy",
            "Wait for deployment verification.",
        )
    elif stopped:
        if owned:
            title, tone, next_action = (
                "Stopped — arms held",
                "hold",
                "Support the can and arms, then release control before resetting.",
            )
        elif owned is False:
            title, tone, next_action = (
                "Pickup blocked — control released",
                "blocked",
                "Correct the reported problem, reset the setup, then start again.",
            )
        else:
            title, tone, next_action = (
                "Attempt stopped — ownership unknown",
                "unknown",
                "Check the controller connection before resetting.",
            )
        if running:
            next_action += " The attempt is finishing cleanup."
    elif owned and (
        phase == "HOLDING_CAN" or (not running and result.get("physical_lift_verified"))
    ):
        title, tone, next_action = (
            "Holding the can",
            "hold",
            "Support the can and raised arms before releasing control.",
        )
    elif running:
        if phase in ("PASSIVE", "FROZEN", "BLOCKED", "UNKNOWN"):
            title = "Starting pickup"
        elif phase in ("CONTROL_ACQUIRING", "CONTROL_SETTLING"):
            title = "Taking arm control"
        elif phase in (
            "staging_segment",
            "target",
            "approach",
            "align",
            "reach",
            "close",
            "lift",
        ):
            title = {
                "staging_segment": "Moving to pickup",
                "target": "Moving the arm",
                "approach": "Approaching the can",
                "align": "Aligning the hand",
                "reach": "Reaching the can",
                "close": "Closing fingers",
                "lift": "Lifting the can",
            }[phase]
        else:
            title = "Planning pickup"
        tone, next_action = (
            "busy",
            "One attempt is running. Stop and hold remains available.",
        )
    elif owned:
        title, tone, next_action = (
            "Stopped — arms held",
            "hold",
            "Support the can and arms, then release control before resetting.",
        )
    elif owned is None:
        title, tone, next_action = (
            "Robot state unavailable",
            "unknown",
            "Check the controller connection; ownership is unknown.",
        )
    elif state.get("telemetry_healthy") is False:
        title, tone, next_action = (
            "Robot feedback unavailable",
            "blocked",
            "Wait for healthy robot feedback before starting.",
        )
    elif phase == "BLOCKED":
        title, tone, next_action = (
            "Pickup blocked — control released",
            "blocked",
            "Correct the reported problem, reset the setup, then start again.",
        )
    else:
        title, tone, next_action = (
            "Control released · ready to start",
            "ready",
            "Place the can base inside the green area, then start one demo.",
        )
    activity = (
        "Attempt stopped"
        if stopped
        else (
            "Attempt complete"
            if phase == "HOLDING_CAN"
            else "Attempt running" if running else "No attempt running"
        )
    )
    return dict(
        title=title,
        tone=tone,
        ownership=owner,
        next_action=next_action,
        activity=activity,
        failure=failure,
        failure_summary=summary,
        failure_label=(
            "Last attempt" if not owned and not running else "Current problem"
        ),
        phase=phase,
    )
