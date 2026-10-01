# Person detector

`yolox_tiny.onnx` is the official COCO YOLOX-Tiny model from the
[YOLOX ONNX Runtime release](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md).
The [download](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx)
is covered by `LICENSE.YOLOX`.

SHA-256: `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`.

The model takes raw BGR pixels letterboxed to 416 by 416. The app decodes
predictions, selects the COCO person class, applies a 0.35 confidence threshold
and nonmaximum suppression, then samples registered depth. Camera detections
without valid depth have no measured position. Simulator actor labels are never
inputs to the model.
