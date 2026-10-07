Calibration data collected 2026-10-07

This is an archived capture, frame 2889, not a live capture.
Camera: RealSense D435i, serial 338622073335.
Camera metadata contains factory intrinsics and depth scale.
capture.npz contains aligned color_bgr, depth_m, rays and metadata_json.

Operator setup: tabletop 0.762 m above floor; belly surface to table 0.2032 m.
Can dimensions: nominal 0.066 m diameter, 0.122 m height, not measured.

No independently measured camera-to-robot transform is available.
calibration.incomplete.json is NOT ready for deployment. Missing transforms
and validation values are deliberately null; commissioned remains false.

To obtain the transform without printed markers, measure six identifiable
points relative to one defined robot reference frame, and pair each with
its [u,v] pixel in the matching RGB-D capture. Use metres, forward X, left Y,
floor-up Z. Four non-collinear points fit the transform; two additional
points validate it. Select surface pixels with valid depth, not depth edges.
If the setup moved since this capture, capture a new frame first.

Also needed for commissioning: pelvis pose in that same frame, table center
and extent, arm limits, operator stop and support setup. The belly gap alone
does not give the pelvis-to-table distance.
