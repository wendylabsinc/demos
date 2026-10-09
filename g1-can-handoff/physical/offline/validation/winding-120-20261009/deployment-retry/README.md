# Successful deployment retry

Deployed and verified `0.3.25-winding-120`, source digest
`ee038c3838c9e15774883bd73be562d85170ab3105d8e6ffb07a3eb0606b0343`,
to G1 Nx2 via `unitree-g1-nx-2.local:50052`. The initial literal IPv4 connection
failed; the authenticated hostname check succeeded and reported the expected
Nx2 network interfaces and Wendy Studio organization. Updated deployment and
operator transport selectors to that hostname; ownership preflight still
checks both select the same device. Existing physical_core.py changes were
preserved.

31 focused temperature, deployment and operator-transport tests passed. Native
deployment-wrapper audit passed. Deployment
`ce40e86b-1950-4979-8067-9aece0b3f1e6` passed configured functional readiness
and exact-version readback. Runtime reports casing 85 C and winding 120 C,
healthy telemetry and all-zero left-hand fault feedback. At the final read,
body max sensor temperature was 44 C, phase PASSIVE, arm adapter absent,
demo not running, and motion command count zero. No physical trial was started
by this deployment retry.

The earlier blocked-readiness evidence remains historical; the new readback
and timing JSON in this directory establish that the configuration is now live.
