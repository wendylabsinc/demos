"""Bounded candidate-only class branch rehearsal for the 10% semantic floor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from ultralytics import YOLO
import yaml


def main():
    root=Path(__file__).parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--model',type=Path,default=root/'model/coke-can.pt')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--negative-weight',type=int,default=48)
    args=parser.parse_args()
    if not 1<=args.epochs<=5 or not 1<=args.negative_weight<=96:
        parser.error('Bounded candidate requires epochs1..5 and negativeweight1..96')
    args.output=args.output.resolve()
    if args.output.exists():raise ValueError('Fresh candidate directory required')
    args.output.mkdir(parents=True)
    dataset=args.output/'dataset'
    for split,source in [('train','train'),('val','valid'),('test','test')]:
        for kind in ('images','labels'):
            destination=dataset/split/kind;destination.mkdir(parents=True)
            for path in sorted((args.dataset/source/kind).glob('*')):
                if path.suffix in ('.jpg','.jpeg','.png','.txt'):(destination/path.name).symlink_to(path.resolve())
    positive=root/'fixtures/nx2-missed.jpg'
    annotation='0 0.3578125 0.6489583333333333 0.140625 0.28125\n'
    for index in range(24):
        shutil.copyfile(positive,dataset/'train/images'/f'nx2-rehearsal-{index}.jpg')
        (dataset/'train/labels'/f'nx2-rehearsal-{index}.txt').write_text(annotation)
    for number in range(2):
        negative=root/f'fixtures/nx2-empty-{number}.jpg'
        for index in range(args.negative_weight):
            name=f'nx2-negative-{number}-{index}'
            shutil.copyfile(negative,dataset/'train/images'/f'{name}.jpg')
            (dataset/'train/labels'/f'{name}.txt').write_text('')
    data=args.output/'data.yaml';data.write_text(yaml.safe_dump({'path':str(dataset),'train':'train/images',
        'val':'val/images','test':'test/images','nc':1,'names':['Coca-Cola']}))
    model=YOLO(args.model);before={key:value.clone() for key,value in model.model.state_dict().items()}
    def classification_only(trainer):
        layers=trainer.model.model
        for layer in list(layers)[:-1]:layer.eval()
        layers[-1].cv2.eval()
        for parameter in layers[-1].cv2.parameters():parameter.requires_grad_(False)
    model.add_callback('on_train_batch_start',classification_only)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    model.train(data=str(data),epochs=args.epochs,imgsz=640,batch=8,device=device,workers=0,amp=False,
        freeze=22,optimizer='AdamW',lr0=.00005,lrf=.1,seed=7,patience=5,plots=False,save=True,
        degrees=0,translate=0,scale=0,mosaic=0,close_mosaic=0,fliplr=0,hsv_h=.01,hsv_s=.3,hsv_v=.15,
        project=str(args.output/'runs'),name='negative-rehearsal',exist_ok=False)
    reports=[]
    for name in ('last','best'):
        path=args.output/f'runs/negative-rehearsal/weights/{name}.pt'
        current=YOLO(path).model.state_dict()
        changed=[key for key,value in before.items() if not torch.equal(value,current[key])]
        unexpected=[key for key in changed if not key.startswith('model.22.cv3.')]
        if unexpected:raise RuntimeError('Frozen features/box state changed: '+repr(unexpected))
        reports.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'only_class_branch_changed':True,'changed_tensors':changed})
    report={'parent_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),'epochs':args.epochs,
            'original_train_images':453,'unique_positive_rehearsal_views':1,'positive_sampling_weight':24,
            'unique_negative_regions':2,'negative_sampling_weight':args.negative_weight,
            'held_out_val_images':42,'held_out_test_images':24,'training_bitwise_reproducibility_claimed':False,
            'candidates':reports,'candidate_only':True,'hardware_io':False}
    (args.output/'adaptation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
