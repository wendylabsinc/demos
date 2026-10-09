"""Fail closed on accidental network/DDS use, including spawned test children."""

import os

if os.environ.get("HANDOFF_OFFLINE") == "1":
    import importlib.abc
    import socket
    import sys

    local_ports = set()
    original_bind = socket.socket.bind

    def bind_ephemeral(self, address):
        if self.family in (socket.AF_INET, socket.AF_INET6):
            if address[0] not in ("127.0.0.1", "::1") or address[1] != 0:
                raise RuntimeError(
                    "Offline regressions may bind only ephemeral loopback test servers"
                )
            result = original_bind(self, address)
            local_ports.add(self.getsockname()[1])
            return result
        return original_bind(self, address)

    socket.socket.bind = bind_ephemeral

    def block_network(original):
        def checked(self, *args, **kwargs):
            if self.family in (socket.AF_INET, socket.AF_INET6):
                address = (
                    args[0]
                    if original.__name__ in ("connect", "connect_ex") and args
                    else None
                )
                if (
                    not address
                    or address[0] not in ("127.0.0.1", "::1")
                    or address[1] not in local_ports
                ):
                    raise RuntimeError(
                        "Offline regression attempted network IO outside its fake HTTP server"
                    )
            return original(self, *args, **kwargs)

        return checked

    for method in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, method, block_network(getattr(socket.socket, method)))

    class NoDDS(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "unitree_sdk2py":
                raise ImportError(
                    "Real Unitree SDK is disabled in offline regressions; use fake transport"
                )

    sys.meta_path.insert(0, NoDDS())
