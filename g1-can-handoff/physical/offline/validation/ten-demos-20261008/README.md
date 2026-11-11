# Ten supervised regular demo attempts

Target: G1 Nx 2, 10.10.20.238:50052. Controller source is checked against the
verified compact-tracking release before every start. No controller guards are
bypassed. The experiment does not automatically release arms/fingers.

Run one attempt at a time:

```sh
python supervised_demo_trials.py --directory offline/validation/ten-demos-20261008 --index 1 --reset-confirmed
```

For attempts 2–10, replace the index only after the human has reset the setup and
confirmed readiness. `--reset-confirmed` records that acknowledgement; it cannot
permit retained arm ownership, a running prior trial, unhealthy feedback,
changed source/model, stale camera exposure or unhealthy recording. Idle BLOCKED
without ownership is admissible through the regular controller Start endpoint.

The runner reserves the attempt before sending Start. If the response is lost,
it preserves the reservation and will not blindly issue Start again. Each run
has a 180 second supervisor bound; exceeding it requests Stop and allows 15
seconds for terminal cleanup. Stop retains a measured hold until the operator
supports the robot/can and uses Release.

Each terminal attempt plays three local Glass alert sounds, records complete
controller/recorder status and time interval, and exports checksum-verified
native events and selected exact RGB-D frames locally. The existing DDS recorder
continues capturing all callback-delivered feedback and commands; its session,
file and sealed segment hashes are in the attempt record. Upstream DDS loss is
unknown. Large raw DDS archives stay local/on the robot.

`preflight-errors.jsonl` records problems that did not issue a Start and do not
count as a demo. `trial-NN.json` is the durable status record, and the associated
`.tar.gz` is its local replay artifact. The ten-run request is complete only
when ten numbered attempts have terminal results and operator reset/release is
verified afterward.
