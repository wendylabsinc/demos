package main

import (
	"context"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	pb "github.com/wendylabsinc/demos/go2-teleop/internal/agentpb"
	"google.golang.org/grpc"
)

// Offline private-socket fixture, not a running Wendy agent or robot.
type fixtureAgent struct {
	pb.UnimplementedWendyUnitreeTeleopServiceServer
	mu     sync.Mutex
	frames []*pb.UnitreeTeleopFrame
	ended  chan struct{}
}

func (a *fixtureAgent) Control(s grpc.BidiStreamingServer[pb.UnitreeTeleopFrame, pb.UnitreeTeleopReady]) error {
	defer close(a.ended)
	for {
		f, err := s.Recv()
		if err != nil {
			return err
		}
		a.mu.Lock()
		a.frames = append(a.frames, f)
		a.mu.Unlock()
		if f.Stop {
			return nil
		}
		if err = s.Send(&pb.UnitreeTeleopReady{Challenge: nonce()[:32], TimeoutMs: 350}); err != nil {
			return err
		}
	}
}
func privateAgentFixture(t *testing.T) (agentBackend, *fixtureAgent) {
	t.Helper()
	dir, err := os.MkdirTemp("", "go2-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	socket := filepath.Join(dir, "agent.sock")
	listener, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}
	a := &fixtureAgent{ended: make(chan struct{})}
	server := grpc.NewServer()
	pb.RegisterWendyUnitreeTeleopServiceServer(server, a)
	go server.Serve(listener)
	b, conn, err := newAgentBackend(socket, "GO2TEST1")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { conn.Close(); server.Stop(); listener.Close() })
	return b, a
}
func TestPrivateAgentBridgeFramesAndExplicitStop(t *testing.T) {
	b, a := privateAgentFixture(t)
	a.mu.Lock()
	started := len(a.frames) > 0
	a.mu.Unlock()
	if started {
		t.Fatal("constructing hardware backend sent command")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	c, err := b.Open(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	r := c.Initial()
	if r.Mode != b.Mode() {
		t.Fatal("hardware target missing from ready")
	}
	r, err = c.Exchange(ctx, command{Challenge: r.Challenge, Forward: .1})
	if err != nil {
		t.Fatal(err)
	}
	r, err = c.Exchange(ctx, command{Challenge: r.Challenge})
	if err != nil {
		t.Fatal(err)
	}
	_, err = c.Exchange(ctx, command{Challenge: r.Challenge, Stop: true})
	if err != io.EOF {
		t.Fatalf("explicit stop did not end stream: %v", err)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.frames) != 4 || a.frames[0].Challenge != "" || a.frames[0].Forward != 0 || a.frames[1].Forward != .1 || a.frames[2].Forward != 0 || !a.frames[3].Stop {
		t.Fatal("incorrect agent frames")
	}
	for _, f := range a.frames {
		if f.Serial != "GO2TEST1" {
			t.Fatal("target serial changed")
		}
	}
}
func TestPrivateAgentBridgeDisconnectCancelsControl(t *testing.T) {
	b, a := privateAgentFixture(t)
	c, err := b.Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	c.Close()
	select {
	case <-a.ended:
	case <-time.After(time.Second):
		t.Fatal("bridge disconnect did not end agent control stream")
	}
}
