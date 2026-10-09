# Body temperature trace — NX2 Coke demo

Recorded demo: 2026-10-09 19:24:31 UTC. Failure: 19:26:11 UTC, 100.115756 seconds after Start was accepted.

The independent recorder's session is 20261009T191749Z-f6bb3fab. The sealed source segments 006–008 were SHA-256 verified before decoding. The derived trace contains 150,423 rt/lowstate rows, covering five seconds before Start through five seconds after supported release. There are no empty 250 ms display buckets. Recorder callback times are used to align controller phases; they are not hardware sensor timestamps. Upstream DDS loss cannot be ruled out.

The first sample at or above the guard arrived 100.098453 seconds after Start: body motor index 16, left shoulder roll, reported sensors [53, 85] degrees C. The primary failure was recorded 17.3 ms later by the watchdog, 205.5 ms after close began. The other joints were at or below 54 degrees C in that threshold-crossing sample. The shoulder's hotter sensor rose from about 55 degrees C at 95 seconds to 85 degrees C, then cooled after the stop. These are reported values; the cause of the rapid sensor rise is not established by this graph.

- body-temperature-demo.png and .pdf: overview plus all 29 body-joint traces as a heatmap.
- body-temperatures.csv: both sensor minima and maxima per joint in 250 ms bins.
- segment-006.json through segment-008.json: decoded summaries, source hashes, exact first threshold-crossing sample, peak records and sample counts.
- events.json: relevant native controller phases and primary error.
- summary.json: coverage and hottest-joint ranking.
- plot_body_temperatures.py: local rerender script.
- extract_recorded_temperatures.py: read-only sealed-file extraction, no DDS subscribers or publishers. The extraction run used orjson.loads when available for faster JSON parsing; decoding and aggregation were otherwise identical.

Rerender locally:

```sh
/Users/olivertaylor/Documents/Wendy/demos-g1-nx2-yolo/.venv/bin/python /Users/olivertaylor/Documents/Wendy/demos-g1-nx2-trial-fixes/g1-can-handoff/physical/offline/validation/body-temperature-demo-20261009T192431Z/plot_body_temperatures.py
```

No controller code, guard, gains or robot motion were changed for this analysis.

