"""App-owned DNS-SD. Wendy Net projects records across private app bridges."""
import fcntl
import ipaddress
import os
import queue
import socket
import struct
import threading
import time
import uuid
from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, ServiceStateChange, Zeroconf

SERVICE = "_ranger._tcp.local."
MESH = ipaddress.ip_network("10.99.0.0/16")


def interface_ipv4():
    if os.environ.get("MDNS_ADDRESS"):
        return os.environ["MDNS_ADDRESS"]
    interface = os.environ.get("MDNS_INTERFACE", "eth0")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        packed = struct.pack("256s", interface.encode()[:15])
        return socket.inet_ntoa(fcntl.ioctl(sock.fileno(), 0x8915, packed)[20:24])


def resolved_peer(info, node, simulate):
    properties = info.decoded_properties
    if (properties.get("demo") != "mesh-ranger-v2" or properties.get("node") == node
            or properties.get("mode") != ("simulation" if simulate else "inference")):
        return None
    addresses = info.parsed_addresses(IPVersion.V4Only)
    if not addresses:
        return None
    addresses.sort(key=lambda address: ipaddress.ip_address(address) not in MESH)
    address = addresses[0]
    # origin TXT is evidence for the UI only. Never use it as a dial target.
    return {"node": properties.get("node"), "name": info.name, "hostname": info.server,
            "origin": properties.get("origin", "unknown"),
            "endpoint": f"http://{address}:{info.port}", "addresses": addresses,
            "mesh_address": ipaddress.ip_address(address) in MESH,
            "records": ["PTR", "SRV", "TXT", "A"], "observed_at": time.time()}


class Discovery:
    def __init__(self, station, loop, port):
        self.station, self.loop = station, loop
        self.address = interface_ipv4()
        self.events = queue.Queue(maxsize=256)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.zc = self.browser = self.thread = None
        self.published = False
        self.info = ServiceInfo(SERVICE, f"Ranger {uuid.uuid4().hex[:12]}.{SERVICE}",
                                addresses=[socket.inet_aton(self.address)], port=port,
                                properties={"demo": "mesh-ranger-v2", "node": station.node,
                                            "mode": "simulation" if station.simulate else "inference",
                                            "origin": f"{self.address}:{port}"},
                                server=f"ranger-{uuid.uuid4().hex[:12]}.local.")

    def start(self):
        self.zc = Zeroconf(interfaces=[self.address], ip_version=IPVersion.V4Only)
        self.browser = ServiceBrowser(self.zc, SERVICE, handlers=[self.changed])
        self.thread = threading.Thread(target=self.resolve, daemon=True)
        self.thread.start()
        self.publish(True)

    def publish(self, enabled):
        with self.lock:
            if enabled == self.published:
                return
            if enabled:
                self.zc.register_service(self.info, allow_name_change=True)
            else:
                self.zc.unregister_service(self.info)
            self.published = enabled
        self.loop.call_soon_threadsafe(self.record, "announced" if enabled else "withdrawn", self.info.name, None)

    def changed(self, zeroconf, service_type, name, state_change):
        # Do not block the callback thread on resolution or HTTP.
        try:
            self.events.put_nowait((name, state_change))
        except queue.Full:
            self.loop.call_soon_threadsafe(self.record, "overflow", name, None)

    def resolve(self):
        while not self.stop.is_set():
            try:
                name, change = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            if change == ServiceStateChange.Removed:
                self.loop.call_soon_threadsafe(self.record, "removed", name, None)
                continue
            info = self.zc.get_service_info(SERVICE, name, timeout=2000)
            if info:
                peer = resolved_peer(info, self.station.node, self.station.simulate)
                if peer:
                    self.loop.call_soon_threadsafe(self.record, change.name.lower(), name, peer)

    def record(self, action, name, peer):
        if action == "removed":
            self.station.fleet.peers.pop(name, None)
            self.station.discovered.pop(name, None)
        elif peer:
            self.station.fleet.peers[name] = peer["endpoint"]
            self.station.discovered[name] = peer
        self.station.discovery_log.append({"time": time.time(), "action": action,
                                           "name": name, "peer": peer})

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        if self.browser:
            self.browser.cancel()
        if self.zc:
            if self.published:
                self.publish(False)
            self.zc.close()
