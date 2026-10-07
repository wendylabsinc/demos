"""Loopback-only, inference-only Coke YOLO service for JetPack 6.2 / G1 NX2.

Accepts BGR bytes from the handoff's leased RGB-D frame. It never opens a
camera, reads DDS, constructs a motor adapter, or substitutes a later image.
"""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from preprocessing import letterbox_bgr
from postprocessing import decode_one_class_candidates

MODEL_SHA256 = '87fe92dc461f56077e007974045d67acd7a0acdaab00530b9e2fe0c5ec35bfc7'
SCHEMA = 'wendy.handoff.coke-yolo.v1'
MAX_BYTES = 1920 * 1080 * 3


def decode_frame(source, raw):
    shape = source['shape']
    if (len(shape) != 3 or any(type(v) is not int for v in shape) or
            not 1 <= shape[0] <= 1080 or not 1 <= shape[1] <= 1920 or shape[2] != 3 or
            len(raw) != int(np.prod(shape)) or len(raw) > MAX_BYTES):
        raise ValueError('Invalid BGR image dimensions or body length')
    if type(source['frame_id']) is not int or source['frame_id'] < 0 or not source['camera_serial']:
        raise ValueError('Invalid camera/frame identity')
    if hashlib.sha256(raw).hexdigest() != source['image_sha256']:
        raise ValueError('BGR checksum mismatch')
    if not 0 <= time.monotonic() - source['timestamp'] <= .3:
        raise ValueError('Stale/invalid exposure timestamp')
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


class Detector:
    def __init__(self, model_path):
        import torch
        from ultralytics import YOLO
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError('Coke checkpoint checksum mismatch')
        if not torch.cuda.is_available():
            raise RuntimeError('G1 NX2 Coke detector requires CUDA; no CPU fallback')
        torch.set_num_threads(1)
        self.model = YOLO(model_path)
        if self.model.names != {0: 'Coca-Cola'}:
            raise RuntimeError('Expected trained one-class Coca-Cola checkpoint')
        self.torch=torch
        self.network=self.model.model.fuse(verbose=False).eval().to('cuda:0').half()
        self.lock = threading.Lock()
        self.predict(np.zeros((480, 640, 3), dtype=np.uint8))
        if next(self.model.model.parameters()).device.type != 'cuda':
            raise RuntimeError('YOLO warm-up did not place model on CUDA')
        self.device = torch.cuda.get_device_name(0)
        self.qualify()

    def predict(self, image):
        # Reuse the bartending detector's fixed-shape preprocessing and small
        # one-class CPU NMS instead of generalized Ultralytics GPU NMS.
        with self.lock, self.torch.inference_mode():
            chw,scale,pad_x,pad_y=letterbox_bgr(image,640)
            tensor=self.torch.from_numpy(chw).to('cuda:0').half().div_(255).unsqueeze(0)
            raw=self.network(tensor)
            if isinstance(raw,(tuple,list)):raw=raw[0]
            if tuple(raw.shape)!=(1,5,8400):raise RuntimeError('Unexpected one-class YOLO output shape')
            prediction=raw[0]
            values,indices=prediction[4].topk(129,largest=True,sorted=True)
            candidates=prediction[:,indices[:128]].transpose(0,1).float().cpu().numpy()
            saturated=bool(values[-1].item()>.30)
            boxes=decode_one_class_candidates(candidates,confidence=.15,nms_iou=.7,max_detections=21,
                frame_width=image.shape[1],frame_height=image.shape[0],scale=scale,pad_x=pad_x,pad_y=pad_y)
            return [dict(xyxy=[b['x1'],b['y1'],b['x2'],b['y2']],confidence=b['conf'],
                         class_id=0,class_name='Coca-Cola') for b in boxes],saturated

    def qualify(self):
        # A real saved Coke frame must retain its semantic result through the
        # optimized path before the service binds its readiness port.
        from PIL import Image
        image=np.asarray(Image.open('/fixtures/g1-coke-raw.jpg').convert('RGB'))[:,:,::-1].copy()
        boxes,saturated=self.predict(image)
        expected=np.asarray([256,76,323,199],dtype=float)
        good=[b for b in boxes if b['confidence']>.30]
        if len(good)!=1 or saturated:raise RuntimeError('Saved Coke recognition qualification failed')
        observed=np.asarray(good[0]['xyxy'])
        intersection=np.maximum(0,np.minimum(observed[2:],expected[2:])-np.maximum(observed[:2],expected[:2])).prod()
        union=np.prod(observed[2:]-observed[:2])+np.prod(expected[2:]-expected[:2])-intersection
        if intersection/union<.5:raise RuntimeError('Saved Coke box qualification failed')

    def detect(self, source, raw):
        image = decode_frame(source, raw)
        start = time.monotonic()
        boxes,saturated = self.predict(image)
        if not 0 <= time.monotonic() - source['timestamp'] <= .3:
            raise ValueError('Exposure became stale during inference')
        return dict(schema=SCHEMA, source=source, model_sha256=MODEL_SHA256,
                    backend='pytorch_cuda', detections=boxes[:20], truncated=len(boxes)>20 or saturated,
                    inference_ms=round((time.monotonic()-start)*1000, 3))

    def health(self):
        return dict(healthy=True, backend='pytorch_cuda', model_sha256=MODEL_SHA256,
                    class_names=self.model.names, gpu=self.device, camera_owned=False,
                    motion_commands_sent=0)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(2)

    def reply(self, result, status=200):
        body = json.dumps(result, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            self.reply(self.server.detector.health())
        else:
            self.reply({'error': 'not found'}, 404)

    def do_POST(self):
        if self.path != '/detect':
            self.reply({'error': 'not found'}, 404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BYTES:
                raise ValueError('Invalid body length')
            if self.headers.get('Content-Type') != 'application/octet-stream':
                raise ValueError('Expected raw BGR bytes')
            source = json.loads(self.headers['X-Frame-Identity'])
            self.reply(self.server.detector.detect(source, self.rfile.read(length)))
        except Exception as exc:
            self.reply({'error': str(exc)}, 422)

    def log_message(self, *_args):
        pass


def main():
    # Binding happens only after CUDA/model warm-up; TCP readiness cannot pass
    # for a failed model, and all endpoints remain local to the NX2 host.
    detector = Detector(os.getenv('YOLO_MODEL_PATH', '/models/coke-can.pt'))
    # Keep inference on the thread warmed above. A new CUDA worker thread
    # per request recreates cuDNN handles and breaks the exposure-age budget.
    server = HTTPServer(('127.0.0.1', 8112), Handler)
    server.detector = detector
    print(json.dumps(detector.health()), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
