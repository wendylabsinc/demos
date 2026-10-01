# Learned vision models

The object-query DETR ResNet-50 INT8 ONNX model is converted by Xenova from facebook/detr-resnet-50 under Apache-2.0. The converted weights, preprocessing and class metadata are pinned at commit 8be7ab59ff663484ee9ba2e8d8f267330d5ad03e. The original model card and Apache-2.0 license ship in assets/licenses.

- Conversion: https://huggingface.co/Xenova/detr-resnet-50/tree/8be7ab59ff663484ee9ba2e8d8f267330d5ad03e
- Original model and attribution: https://huggingface.co/facebook/detr-resnet-50

The passive-awareness YOLOS tiny INT8 ONNX is converted by Xenova from hustvl/yolos-tiny. The original model weights declare Apache-2.0. The YOLOS implementation is MIT. This app loads the pinned converted weights with ONNX Runtime and reads their trained class names from config.json.

- Conversion: https://huggingface.co/Xenova/yolos-tiny/tree/e2f9c7673f0fa61849efe2b56a0d7774779ebb9d
- Original weights and license declaration: https://huggingface.co/hustvl/yolos-tiny/blob/95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de/README.md
- Implementation license: assets/licenses/YOLOS.LICENSE, fetched and verified by scripts/prepare_assets.py

COCO class labels describe the trained detector vocabulary. No class label identifies a simulator fixture or its position.
