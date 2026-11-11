"""Fine-tune a candidate; never replaces deployed weights or contacts a robot.

Retain the original Roboflow train/validation split. Add the manually labelled
NX2 missed view to training only, with an explicit sampling weight. Archived
G1 camera fixtures remain held out. Repeated copies are not independent views.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image
import yaml


def prepare(dataset, image, box, output, positive_weight=24):
    root = output/'dataset'
    for split, source in [('train', 'train'), ('val', 'valid'), ('test', 'test')]:
        for kind in ('images', 'labels'):
            target = root/split/kind
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted((dataset/source/kind).glob('*')):
                dest = target/path.name
                if not dest.exists():
                    dest.symlink_to(path.resolve())
    width, height = Image.open(image).size
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError('Invalid manually reviewed annotation')
    label = f'0 {(x1+x2)/2/width} {(y1+y2)/2/height} {(x2-x1)/width} {(y2-y1)/height}\n'
    if positive_weight < 1:
        raise ValueError('Positive sampling weight must be at least one')
    for i in range(positive_weight):
        shutil.copyfile(image, root/'train/images'/f'nx2-weighted-{i}.jpg')
        (root/'train/labels'/f'nx2-weighted-{i}.txt').write_text(label)
    # Explicit empty tabletop/cable regions from the inspected frame. The can
    # is below row 244, so neither negative crop contains the labelled object.
    rgb = Image.open(image).convert('RGB')
    regions = [(0, 0, width, 180), (350, 180, width, height)]
    for region in regions:
        a, b, c, d = region
        if not (0 <= a < c <= width and 0 <= b < d <= height):
            raise ValueError('Negative regions require the inspected 640x480 NX2 view')
        if max(a, x1) < min(c, x2) and max(b, y1) < min(d, y2):
            raise ValueError('Negative region overlaps the annotated can')
    for i, region in enumerate(regions):
        for j in range(8):
            name = f'nx2-empty-{i}-{j}'
            rgb.crop(region).save(root/'train/images'/f'{name}.jpg')
            (root/'train/labels'/f'{name}.txt').write_text('')
    data = dict(path=str(root.resolve()), train='train/images', val='val/images',
                test='test/images', nc=1, names=['Coca-Cola'])
    path = output/'data.yaml'
    path.write_text(yaml.safe_dump(data))
    record = dict(original_dataset=str(dataset), original_train_images=len(list((dataset/'train/images').glob('*'))),
                  added_unique_positive_views=1, positive_sampling_weight=positive_weight,
                  added_unique_negative_regions=2, negative_sampling_weight=8,
                  negative_regions_xyxy=regions,
                  added_image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                  bbox_xyxy=box, held_out_original_validation_images=42,
                  held_out_original_test_images=24, seed=7)
    (output/'adaptation.json').write_text(json.dumps(record, indent=2)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--box', type=float, nargs=4, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--stop-after', type=int, help='Stop after this many completed epochs while retaining the planned learning-rate schedule')
    parser.add_argument('--model', type=Path, default=Path(__file__).parent/'model/coke-can.pt')
    parser.add_argument('--positive-weight', type=int, default=24)
    parser.add_argument('--fixed-view', action='store_true', help='Disable geometric augmentations for the adaptation continuation')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'dataset').exists():
        raise ValueError('Use a fresh output directory to avoid stale training samples')
    data = prepare(args.dataset, args.image, args.box, args.output, args.positive_weight)
    import torch
    from ultralytics import YOLO
    model = YOLO(args.model)
    def classification_only(trainer):
        # Preserve the parent features and bounding-box calibration, including
        # BN running statistics. Only the detection head's class branch adapts.
        layers = trainer.model.model
        for layer in list(layers)[:-1]:
            layer.eval()
        head = layers[-1]
        head.cv2.eval()
        for parameter in head.cv2.parameters():
            parameter.requires_grad_(False)
    # Ultralytics calls model.train() after its epoch-start callback. Apply the
    # freeze after that reset, before every batch, so box-head BN cannot drift.
    model.add_callback('on_train_batch_start', classification_only)
    if args.stop_after is not None:
        if not 1 <= args.stop_after <= args.epochs:
            raise ValueError('stop-after must be between one and planned epochs')
        def stop_after(trainer):
            if trainer.epoch+1 >= args.stop_after:
                trainer.stop=True
        model.add_callback('on_fit_epoch_end', stop_after)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.train(data=str(data), epochs=args.epochs, imgsz=args.size, batch=8,
                device=device, workers=0, amp=False, freeze=22, optimizer='AdamW',
                lr0=.0001, lrf=.1, seed=7, patience=12, plots=False, save=True,
                degrees=0 if args.fixed_view else 10,
                translate=0 if args.fixed_view else .15,
                scale=0 if args.fixed_view else .6,
                mosaic=0 if args.fixed_view else .5,
                close_mosaic=0 if args.fixed_view else 3,
                project=str(args.output/'runs'), name='nx2-adaptation', exist_ok=False)
    print('Candidate only: '+str(args.output/'runs/nx2-adaptation/weights/best.pt'))


if __name__ == '__main__':
    main()
