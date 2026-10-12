# WendyOS teleop protocol snapshot

Source repository: <https://github.com/wendylabsinc/WendyOS>
Revision: `b9f30bd7c97af9379dafe4c611e3a220bd0387ad`
Source contract: [unitree_teleop_service.proto](https://github.com/wendylabsinc/WendyOS/blob/b9f30bd7c97af9379dafe4c611e3a220bd0387ad/Proto/wendy/agent/services/v2/unitree_teleop_service.proto)

`proto/wendy/agent/services/v2/unitree_teleop_service.proto` is copied unchanged.
SHA-256: `18f8cf3829795c1d27f46e548b5d4cc0d6a6414300756e59f8468a477c390858`.
Bindings are generated, not edited manually. `LICENSE` and `NOTICE` are retained
from that repository (Apache-2.0); unrelated upstream notices are preserved as
received, not a statement that all those components ship in this demo.

Generator versions: protoc 36.2, protoc-gen-go 1.36.11 (built from the pinned
module; reports `v1.36.11-devel`), protoc-gen-go-grpc 1.6.1. Reproduce from the
`go2-teleop` directory with these plugins on PATH:

```sh
protoc -I proto \
  --go_out=. --go_opt=module=github.com/wendylabsinc/demos/go2-teleop \
  '--go_opt=Mwendy/agent/services/v2/unitree_teleop_service.proto=github.com/wendylabsinc/demos/go2-teleop/internal/agentpb;agentpb' \
  --go-grpc_out=. --go-grpc_opt=module=github.com/wendylabsinc/demos/go2-teleop \
  '--go-grpc_opt=Mwendy/agent/services/v2/unitree_teleop_service.proto=github.com/wendylabsinc/demos/go2-teleop/internal/agentpb;agentpb' \
  proto/wendy/agent/services/v2/unitree_teleop_service.proto
```

This import-path override avoids pulling the entire WendyOS agent module into
the bridge. Coordinate protocol updates with the agent and regenerate both
bindings; do not change a field number or reinterpret an existing field.
