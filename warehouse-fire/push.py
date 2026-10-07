"""Persistent Web Push subscriptions, independent of an open dashboard tab."""
import base64
import json
import queue
import threading
from urllib.parse import urlparse
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from pywebpush import webpush, WebPushException

class PushDelivery:
    def __init__(self,directory,record):
        self.directory=directory; self.record=record; self.lock=threading.Lock()
        self.key=directory/'vapid.key'; self.subfile=directory/'subscriptions.json'
        if not self.key.exists():
            private=ec.generate_private_key(ec.SECP256R1())
            self.key.write_bytes(private.private_bytes(serialization.Encoding.PEM,serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
            self.key.chmod(0o600)
        private=serialization.load_pem_private_key(self.key.read_bytes(),password=None)
        public=private.public_key().public_bytes(serialization.Encoding.X962,serialization.PublicFormat.UncompressedPoint)
        self.public_key=base64.urlsafe_b64encode(public).decode().rstrip('=')
        self.subscriptions=json.loads(self.subfile.read_text()) if self.subfile.exists() else []
        self.jobs=queue.Queue()
        threading.Thread(target=self.worker,daemon=True).start()

    def subscribe(self,subscription):
        endpoint=subscription.get('endpoint',''); url=urlparse(endpoint)
        # A local registration endpoint must not become an arbitrary outbound URL proxy.
        allowed=('fcm.googleapis.com','push.services.mozilla.com','web.push.apple.com','notify.windows.com')
        if url.scheme!='https' or not any(url.hostname==host or (url.hostname or '').endswith('.'+host) for host in allowed):
            raise ValueError('Unsupported push service endpoint')
        if not subscription.get('keys',{}).get('p256dh') or not subscription.get('keys',{}).get('auth'):
            raise ValueError('Missing push encryption keys')
        with self.lock:
            self.subscriptions=[s for s in self.subscriptions if s['endpoint']!=endpoint]+[subscription]
            self.subfile.write_text(json.dumps(self.subscriptions))
        return len(self.subscriptions)

    def send(self,payload):
        self.jobs.put(payload)

    def worker(self):
        while True:
            payload=self.jobs.get()
            with self.lock: subscriptions=list(self.subscriptions)
            if not subscriptions:
                self.record('push_unsubscribed',incident_id=payload.get('incident_id'),message='Enable notifications in the dashboard to subscribe a browser')
            for subscription in subscriptions:
                try:
                    webpush(subscription_info=subscription,data=json.dumps(payload),vapid_private_key=str(self.key),vapid_claims={'sub':'https://wendy.dev'},timeout=8)
                    self.record('push_sent',incident_id=payload.get('incident_id'))
                except WebPushException as exc:
                    self.record('push_failed',incident_id=payload.get('incident_id'),error=str(exc))
                    if exc.response is not None and exc.response.status_code in (404,410):
                        with self.lock:
                            self.subscriptions=[s for s in self.subscriptions if s['endpoint']!=subscription['endpoint']]
                            self.subfile.write_text(json.dumps(self.subscriptions))
                except Exception as exc: self.record('push_failed',error=str(exc))
            self.jobs.task_done()
