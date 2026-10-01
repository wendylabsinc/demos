# Third-party components

The viewer uses [three.js](https://github.com/mrdoob/three.js), MIT licensed, at commit `0af9729d0c143a86a1d725d6e2c3ad83301f3f34`. `assets.lock.json` pins module files and the license by SHA-256. The drone and yard geometry are generated locally.

`models.lock.json` pins the MIT-licensed [Systran faster-whisper tiny.en conversion](https://huggingface.co/Systran/faster-whisper-tiny.en), Apache2.0 [Qwen3-0.6B GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF), and MIT-licensed [llama.cpp](https://github.com/ggml-org/llama.cpp) runtime release b9265. Whisper and Silero VAD inference use faster-whisper's packaged model/runtime interfaces.

The final incident detector uses a locally authored RGB CNN with three convolution layers (16/32/64 channels) and 2×2 pooled features, trained from scratch on actual MuJoCo camera crops rendered with Linux OSMesa. `yard_drone/incident_cnn.onnx` contains these learned weights; the paired JSON records training and validation metrics, crop-domain limits, dataset hashes and the model SHA-256. It uses no pretrained incident-classification weights. [ONNX Runtime](https://github.com/microsoft/onnxruntime), MIT licensed, runs it on CPU.

Training and export use development-only [PyTorch 2.8.0](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE), under its BSD-style license and bundled third-party notices, and [ONNX 1.18.0](https://github.com/onnx/onnx/blob/v1.18.0/LICENSE), Apache2.0. They are not required in the deployed application.

For optional/reference tools, `vision-models.json` pins the [Xenova MobileCLIP-S0 ONNX export](https://huggingface.co/Xenova/mobileclip_s0) and [YOLOS-tiny ONNX export](https://huggingface.co/Xenova/yolos-tiny) by immutable revision and SHA-256. MobileCLIP is used by the earlier reference probe in `train_incident_classifier.py`. The current `generate_rgb_incidents.py` dataset generator and final incident CNN do not use it. MobileCLIP originates in [Apple's research repository](https://github.com/apple-aiml-research/ml-mobileclip); its bundled LICENSE contains its applicable terms. YOLOS supports optional COCO person detection. The [original YOLOS model](https://huggingface.co/hustvl/yolos-tiny) uses Apache2.0.

Dependencies retain their respective licenses. Model provisioning happens at build time; sensor processing and inference run locally inside the application.
