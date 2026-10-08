"""Plain-language presentation of controller state; no control decisions."""


def describe_state(state):
    owned = state.get('adapter_initialized')
    phase = state.get('phase') or 'UNKNOWN'
    running = bool(state.get('demo_running'))
    failure = state.get('demo_error') or state.get('error')
    summary = failure
    for trigger, message in [
        ('Grasp orientation search time limit', 'The pickup planner could not find a valid grasp in time.'),
        ('Coke YOLO target absent/ambiguous', 'The Coke can could not be identified unambiguously.'),
        ('Control handover did not settle', 'The arms did not settle after the demo took control.'),
        ('intersects robot/table', 'The planned arm or finger path intersects the robot or table.'),
        ('exposure stale', 'The camera observation is too old for movement.'),
        ('Continuous Cartesian lift IK', 'The lift cannot preserve the planned grasp.'),
        ('Operator cancelled', 'The operator stopped the attempt.')]:
        if failure and trigger in failure:
            summary = message; break
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
