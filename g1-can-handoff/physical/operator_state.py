"""Plain-language presentation of controller state; no control decisions."""

ERROR_CASES=[
    dict(triggers=['Grasp orientation search time limit'],title='Planning took too long',
         summary='The grasp search exceeded its configured time limit.',
         action='Review the trial log and planner timing before another attempt.',source='g1_can_handoff/task.py'),
    dict(triggers=['Waist moved beyond','Body attitude exceeds'],title='Robot posture changed',
         summary='The robot moved beyond the camera alignment limits after calibration.',
         action='Keep the robot stationary and refresh the scene before retrying.',source='pose_alignment.py'),
    dict(triggers=['exposure stale','exposure timestamp invalid/stale'],title='Camera observation is too old',
         summary='The latest observation did not meet the freshness limit.',
         action='Check that the live camera is advancing before another attempt.',source='camera_input.py'),
    dict(triggers=['Coke YOLO target absent/ambiguous'],title='Coke identity could not be confirmed',
         summary='YOLO could not identify one unambiguous Coke can.',
         action='Give the detector a clear view of the can before retrying.',source='yolo_client.py'),
    dict(triggers=['Target has too few','No sufficiently visible red can'],title='Can detection needs a clearer view',
         summary='The detector could not obtain an unambiguous can position.',
         action='Place the upright can inside the placement area with an unobstructed view.',source='camera_input.py'),
    dict(triggers=['Can is outside the left-hand pickup area'],title='Can is outside the pickup area',
         summary='The measured can position is outside the supported reach.',
         action='Move the can base inside the placement outline, then retry.',source='g1_can_handoff/task.py'),
    dict(triggers=['Too few yellow tape depth points'],title='Table markers are not visible enough',
         summary='There are too few valid depth points on the yellow tape.',
         action='Restore the marked table to the camera view before refreshing the scene.',source='factory_table_candidate.py'),
    dict(triggers=['Control handover did not settle'],title='Arm control did not settle',
         summary='The arms did not stabilize during control handover.',
         action='Support the arms and can, release control, then reset the setup.',source='control_settle.py'),
    dict(triggers=['intersects','collides','clearance'],title='Planned path is blocked',
         summary='The arm or finger path does not have the required clearance.',
         action='Reposition the can within the pickup area and replan with clearance.',source='physical_core.py'),
    dict(triggers=['Continuous Cartesian lift IK'],title='Lift path could not be solved',
         summary='The planner could not maintain the grasp through the lift.',
         action='Save the trial log and check the lift plan before retrying.',source='physical_core.py'),
    dict(triggers=['HTTP Error 423','423: Locked'],title='Camera is busy',
         summary='Another camera operation currently holds the camera lease.',
         action='Wait for that operation to finish. The placement outline is a reference meanwhile.',source='camera_input.py'),
    dict(triggers=['Operator cancelled'],title='Demo stopped',summary='The operator stopped this attempt.',
         action='Check the setup before starting another attempt.',source='runtime.py')]


def describe_error(message):
    return next((rule for rule in ERROR_CASES if any(trigger in message for trigger in rule['triggers'])),
                dict(title='Demo stopped',summary='The attempt stopped. Technical details contain the full error.',
                     action='Check the reported problem before retrying.',source='Not identified'))


def describe_state(state):
    owned = state.get('adapter_initialized')
    phase = state.get('phase') or 'UNKNOWN'
    running = bool(state.get('demo_running'))
    failure = state.get('demo_error') or state.get('error')
    summary = describe_error(failure)['summary'] if failure else None
    result = state.get('demo_result') or {}
    owner = 'Demo controls arms + fingers' if owned else 'Upper-body control released to Unitree' if owned is False else 'Control ownership unknown'
    if state.get('deployment_in_progress'):
        title, tone, next_action = 'Updating demo', 'busy', 'Wait for deployment verification.'
    elif owned and (phase == 'HOLDING_CAN' or result.get('physical_lift_verified')):
        title, tone, next_action = 'Holding the can', 'hold', 'Support the can and raised arms before releasing control.'
    elif running:
        if phase == 'CONTROL_SETTLING':
            title = 'Taking arm control'
        elif phase in ('staging_segment', 'target', 'close', 'lift'):
            title = {'staging_segment':'Moving to pickup', 'target':'Moving the arm', 'close':'Closing fingers', 'lift':'Lifting the can'}[phase]
        else:
            title = 'Planning pickup'
        tone, next_action = 'busy', 'One attempt is running. Stop and hold remains available.'
    elif owned:
        title, tone, next_action = 'Stopped — arms held', 'hold', 'Support the can and arms, then release control before resetting.'
    elif owned is None:
        title, tone, next_action = 'Robot state unavailable', 'unknown', 'Check the controller connection; ownership is unknown.'
    elif state.get('telemetry_healthy') is False:
        title, tone, next_action = 'Robot feedback unavailable', 'blocked', 'Wait for healthy robot feedback before starting.'
    elif phase == 'BLOCKED':
        title, tone, next_action = 'Pickup blocked — control released', 'blocked', 'Correct the reported problem, reset the setup, then start again.'
    else:
        title, tone, next_action = 'Control released · ready to start', 'ready', 'Place the can base inside the green area, then start one demo.'
    return dict(title=title, tone=tone, ownership=owner, next_action=next_action,
                failure=failure, failure_summary=summary,
                failure_label='Last attempt' if not owned and not running else 'Current problem', phase=phase)
